use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

use ofposets::{
    Embedding, FramedPoset, FramedPosetSubset, RandomFramedPosetGenerator, Renderer, Sign,
    boundary, embedding_to_dot, iterated_boundary, to_dot,
};
use rand::rngs::{OsRng, SmallRng};
use rand::{SeedableRng, TryRngCore};

const CELL_COUNT: usize = 10;
const SIGN_PAIRS: [(Sign, Sign); 4] = [
    (Sign::Input, Sign::Input),
    (Sign::Input, Sign::Output),
    (Sign::Output, Sign::Input),
    (Sign::Output, Sign::Output),
];

struct SearchMatch {
    worker: usize,
    candidate: usize,
    shape: Arc<FramedPoset>,
    sign_0: Sign,
    sign_1: Sign,
    iterated: Embedding,
    intersection: Embedding,
}

fn main() -> std::io::Result<()> {
    let seed = OsRng.try_next_u64().map_err(std::io::Error::other)?;
    let worker_count = thread::available_parallelism()?.get();
    let generator = RandomFramedPosetGenerator::new(2, CELL_COUNT);
    let found = Arc::new(AtomicBool::new(false));
    let attempts = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::channel();

    let matched = thread::scope(|scope| {
        for worker in 0..worker_count {
            let generator = &generator;
            let found = Arc::clone(&found);
            let attempts = Arc::clone(&attempts);
            let sender = sender.clone();
            scope.spawn(move || {
                search_worker(worker, seed, generator, found, attempts, sender);
            });
        }
        drop(sender);

        loop {
            match receiver.recv_timeout(Duration::from_secs(1)) {
                Ok(matched) => break Ok(matched),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    println!("samples taken: {}", attempts.load(Ordering::Relaxed));
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break Err(std::io::Error::other(
                        "all search workers stopped without a match",
                    ));
                }
            }
        }
    })?;
    let candidate_count = attempts.load(Ordering::Relaxed);

    let output_dir = Path::new("visualizations/random_10_cells_intersection_subset_search");
    if output_dir.exists() {
        fs::remove_dir_all(output_dir)?;
    }
    fs::create_dir_all(output_dir)?;
    write_match(output_dir, seed, worker_count, candidate_count, &matched)?;
    println!(
        "found a matching cubular OFP with seed {seed:#018x} after checking {candidate_count} candidates with {worker_count} workers; worker {} found ({}, {})",
        matched.worker + 1,
        sign_name(matched.sign_0),
        sign_name(matched.sign_1)
    );
    Ok(())
}

fn search_worker(
    worker: usize,
    seed: u64,
    generator: &RandomFramedPosetGenerator,
    found: Arc<AtomicBool>,
    attempts: Arc<AtomicUsize>,
    sender: mpsc::Sender<SearchMatch>,
) {
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(worker as u64));

    while !found.load(Ordering::Acquire) {
        let candidate = attempts.fetch_add(1, Ordering::Relaxed) + 1;
        let shape = Arc::new(generator.generate(&mut rng));
        if !is_cubular(&shape) {
            continue;
        }

        for (sign_0, sign_1) in SIGN_PAIRS {
            let (iterated, intersection) = boundary_intersection_embeddings(&shape, sign_0, sign_1);
            if Embedding::equal(&iterated, &intersection) {
                continue;
            }

            let intersection_subset = FramedPosetSubset::from_embedding(&intersection);
            let iterated_subset = FramedPosetSubset::from_embedding(&iterated);
            if !intersection_subset.is_subset_of(&iterated_subset) {
                continue;
            }

            if found
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let _ = sender.send(SearchMatch {
                    worker,
                    candidate,
                    shape,
                    sign_0,
                    sign_1,
                    iterated,
                    intersection,
                });
            }
            return;
        }
    }
}

fn is_cubular(shape: &Arc<FramedPoset>) -> bool {
    for (sign_0, sign_1) in SIGN_PAIRS {
        let (_, zero_then_one) = iterated_boundary(&[(sign_0, 0), (sign_1, 1)], shape);
        let (_, one_then_zero) = iterated_boundary(&[(sign_1, 1), (sign_0, 0)], shape);

        if !Embedding::equal(&zero_then_one, &one_then_zero) {
            return false;
        }
    }

    true
}

fn boundary_intersection_embeddings(
    shape: &Arc<FramedPoset>,
    sign_0: Sign,
    sign_1: Sign,
) -> (Embedding, Embedding) {
    let (_, boundary_0) = boundary(sign_0, 0, shape);
    let (_, boundary_1) = boundary(sign_1, 1, shape);
    let intersection = Embedding::intersection(&boundary_0, &boundary_1).into_codomain;
    let (_, iterated) = iterated_boundary(&[(sign_1, 1), (sign_0, 0)], shape);

    (iterated, intersection)
}

fn write_match(
    output_dir: &Path,
    seed: u64,
    worker_count: usize,
    candidate_count: usize,
    matched: &SearchMatch,
) -> std::io::Result<()> {
    fs::write(
        output_dir.join("sample.dot"),
        to_dot(matched.shape.as_ref(), Renderer::CompassSpring),
    )?;
    fs::write(
        output_dir.join("sample_graded.dot"),
        to_dot(matched.shape.as_ref(), Renderer::Ranked),
    )?;

    let serialized =
        serde_json::to_string_pretty(matched.shape.as_ref()).map_err(std::io::Error::other)?;
    fs::write(
        output_dir.join("sample.ofp.json"),
        format!("{serialized}\n"),
    )?;

    let sign_0 = sign_file_name(matched.sign_0);
    let sign_1 = sign_file_name(matched.sign_1);
    let iterated_name = format!("{sign_0}_0_{sign_1}_1");
    let intersection_name = format!("{sign_0}_0_intersection_{sign_1}_1");

    write_embedding_layouts(output_dir, &iterated_name, &matched.iterated)?;
    write_embedding_layouts(output_dir, &intersection_name, &matched.intersection)?;
    fs::write(
        output_dir.join("match.txt"),
        format!(
            "seed\t{seed:#018x}\ncandidate_ticket\t{}\ncandidates_checked\t{candidate_count}\nworker\t{}\nworkers\t{worker_count}\nsign_0\t{sign_0}\nsign_1\t{sign_1}\n",
            matched.candidate,
            matched.worker + 1
        ),
    )
}

fn write_embedding_layouts(
    output_dir: &Path,
    name: &str,
    embedding: &Embedding,
) -> std::io::Result<()> {
    fs::write(
        output_dir.join(format!("{name}.dot")),
        embedding_to_dot(embedding, Renderer::CompassSpring),
    )?;
    fs::write(
        output_dir.join(format!("{name}_graded.dot")),
        embedding_to_dot(embedding, Renderer::Ranked),
    )
}

fn sign_name(sign: Sign) -> &'static str {
    match sign {
        Sign::Input => "input",
        Sign::Output => "output",
    }
}

fn sign_file_name(sign: Sign) -> &'static str {
    match sign {
        Sign::Input => "minus",
        Sign::Output => "plus",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_has_equal_iterated_boundaries_and_intersections() {
        let square = Arc::new(FramedPoset::from_faces(
            vec![
                vec![vec![], vec![], vec![], vec![]],
                vec![vec![0], vec![0], vec![1], vec![1]],
                vec![vec![0, 1]],
            ],
            vec![
                vec![vec![], vec![], vec![], vec![]],
                vec![vec![0], vec![2], vec![0], vec![1]],
                vec![vec![0, 2]],
            ],
            vec![
                vec![vec![], vec![], vec![], vec![]],
                vec![vec![1], vec![3], vec![2], vec![3]],
                vec![vec![1, 3]],
            ],
        ));

        assert!(is_cubular(&square));
        for (sign_0, sign_1) in SIGN_PAIRS {
            let (iterated, intersection) =
                boundary_intersection_embeddings(&square, sign_0, sign_1);
            assert!(Embedding::equal(&iterated, &intersection));

            let iterated_subset = FramedPosetSubset::from_embedding(&iterated);
            let intersection_subset = FramedPosetSubset::from_embedding(&intersection);
            assert!(intersection_subset.is_subset_of(&iterated_subset));
            assert!(iterated_subset.is_subset_of(&intersection_subset));
        }
    }
}
