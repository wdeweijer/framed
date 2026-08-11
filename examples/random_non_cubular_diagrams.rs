use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

use ofposets::{
    BoundaryMode, Embedding, FramedPoset, FramedPosetSubset, RandomFramedPosetGenerator, Renderer,
    Sign, boundary, embedding_to_dot, to_dot,
};
use rand::rngs::{OsRng, SmallRng};
use rand::{SeedableRng, TryRngCore};

const CELL_COUNT: usize = 4;
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
    boundary_0_after_1: Embedding,
    boundary_1_after_0: Embedding,
    intersection: Embedding,
    boundary_0_after_1_is_subset: bool,
    boundary_1_after_0_is_subset: bool,
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
                        "all search workers stopped without a counterexample",
                    ));
                }
            }
        }
    })?;
    let candidate_count = attempts.load(Ordering::Relaxed);

    let output_dir = Path::new("visualizations/random_10_cells_non_cubular_boundary_subset_search");
    if output_dir.exists() {
        fs::remove_dir_all(output_dir)?;
    }
    fs::create_dir_all(output_dir)?;
    write_match(output_dir, seed, worker_count, candidate_count, &matched)?;
    println!(
        "found a non-cubular counterexample with seed {seed:#018x} after checking {candidate_count} candidates with {worker_count} workers; worker {} found ({}, {})",
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
    let mut rng = SmallRng::seed_from_u64(worker_seed(seed, worker));

    while !found.load(Ordering::Acquire) {
        let candidate = attempts.fetch_add(1, Ordering::Relaxed) + 1;
        let shape = Arc::new(generator.generate(&mut rng));
        if is_cubular(&shape) {
            continue;
        }

        for (sign_0, sign_1) in SIGN_PAIRS {
            let (boundary_0_after_1, boundary_1_after_0, intersection) =
                boundary_comparison(&shape, sign_0, sign_1);
            let intersection_subset = FramedPosetSubset::from_embedding(&intersection);
            let boundary_0_after_1_subset = FramedPosetSubset::from_embedding(&boundary_0_after_1);
            let boundary_1_after_0_subset = FramedPosetSubset::from_embedding(&boundary_1_after_0);
            let boundary_0_after_1_is_subset =
                boundary_0_after_1_subset.is_subset_of(&intersection_subset);
            let boundary_1_after_0_is_subset =
                boundary_1_after_0_subset.is_subset_of(&intersection_subset);

            if boundary_0_after_1_is_subset && boundary_1_after_0_is_subset {
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
                    boundary_0_after_1,
                    boundary_1_after_0,
                    intersection,
                    boundary_0_after_1_is_subset,
                    boundary_1_after_0_is_subset,
                });
            }
            return;
        }
    }
}

fn worker_seed(seed: u64, worker: usize) -> u64 {
    seed ^ (worker as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

fn is_cubular(shape: &Arc<FramedPoset>) -> bool {
    SIGN_PAIRS.into_iter().all(|(sign_0, sign_1)| {
        let (boundary_0_after_1, boundary_1_after_0, _) =
            boundary_comparison(shape, sign_0, sign_1);
        Embedding::equal(&boundary_0_after_1, &boundary_1_after_0)
    })
}

fn boundary_comparison(
    shape: &Arc<FramedPoset>,
    sign_0: Sign,
    sign_1: Sign,
) -> (Embedding, Embedding, Embedding) {
    let (_, boundary_0) = boundary(BoundaryMode::Plain, sign_0, 0, shape);
    let (_, boundary_1) = boundary(BoundaryMode::Plain, sign_1, 1, shape);
    let intersection = Embedding::intersection(&boundary_0, &boundary_1).into_codomain;
    let boundary_0_after_1 = iterated_boundary(shape, sign_1, 1, sign_0, 0);
    let boundary_1_after_0 = iterated_boundary(shape, sign_0, 0, sign_1, 1);

    (boundary_0_after_1, boundary_1_after_0, intersection)
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
    let boundary_0_after_1_name = format!("{sign_0}_0_after_{sign_1}_1");
    let boundary_1_after_0_name = format!("{sign_1}_1_after_{sign_0}_0");
    let intersection_name = format!("{sign_0}_0_intersection_{sign_1}_1");

    write_embedding_layouts(
        output_dir,
        &boundary_0_after_1_name,
        &matched.boundary_0_after_1,
    )?;
    write_embedding_layouts(
        output_dir,
        &boundary_1_after_0_name,
        &matched.boundary_1_after_0,
    )?;
    write_embedding_layouts(output_dir, &intersection_name, &matched.intersection)?;
    fs::write(
        output_dir.join("match.txt"),
        format!(
            "seed\t{seed:#018x}\ncandidate_ticket\t{}\ncandidates_checked\t{candidate_count}\nworker\t{}\nworkers\t{worker_count}\nsign_0\t{sign_0}\nsign_1\t{sign_1}\nboundary_0_after_1_is_subset\t{}\nboundary_1_after_0_is_subset\t{}\n",
            matched.candidate,
            matched.worker + 1,
            matched.boundary_0_after_1_is_subset,
            matched.boundary_1_after_0_is_subset,
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

fn iterated_boundary(
    shape: &Arc<FramedPoset>,
    first_sign: Sign,
    first_direction: usize,
    second_sign: Sign,
    second_direction: usize,
) -> Embedding {
    let (first_boundary, first_embedding) =
        boundary(BoundaryMode::Plain, first_sign, first_direction, shape);
    let (_, second_embedding) = boundary(
        BoundaryMode::Plain,
        second_sign,
        second_direction,
        &first_boundary,
    );
    Embedding::compose(&second_embedding, &first_embedding)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_is_cubular_and_both_composites_are_subsets_of_the_intersection() {
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
            let (boundary_0_after_1, boundary_1_after_0, intersection) =
                boundary_comparison(&square, sign_0, sign_1);
            let intersection_subset = FramedPosetSubset::from_embedding(&intersection);

            assert!(
                FramedPosetSubset::from_embedding(&boundary_0_after_1)
                    .is_subset_of(&intersection_subset)
            );
            assert!(
                FramedPosetSubset::from_embedding(&boundary_1_after_0)
                    .is_subset_of(&intersection_subset)
            );
        }
    }
}
