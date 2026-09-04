//! Search for rigid OFPs whose boundary pasting is not rigid.
//!
//! For every retained rigid shape, this indexes its signed directional
//! boundaries by canonical normal form. Whenever an output boundary matches
//! an input boundary, it forms the unique boundary isomorphism and tests the
//! resulting pushout. Random candidate generation is parallel; indexing and
//! gluing remain sequential so every compatible ordered pair is tested once.

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ofposets::pushout::{Pushout, pushout};
use ofposets::{
    Embedding, FramedPoset, RandomFramedPosetGenerator, Renderer, Sign, boundary, embedding_to_dot,
    isomorphisms, normalize, to_dot,
};
use rand::rngs::{OsRng, SmallRng};
use rand::{SeedableRng, TryRngCore};
use rayon::prelude::*;
use serde::Serialize;

const DEFAULT_DIMENSION: usize = 2;
const DEFAULT_CELL_COUNT: usize = 9;
const CANDIDATES_PER_WORKER: u64 = 16;
const REPORT_INTERVAL: Duration = Duration::from_secs(5);
const FAILURE_ROOT: &str = "visualizations/non_rigid_boundary_pastings";

struct Options {
    dimension: usize,
    cell_count: usize,
    worker_count: usize,
    seed: u64,
}

#[derive(Clone)]
struct BoundaryOccurrence {
    sample: u64,
    into_shape: Embedding,
}

#[derive(Default)]
struct BoundaryClass {
    inputs: Vec<BoundaryOccurrence>,
    outputs: Vec<BoundaryOccurrence>,
}

#[derive(Default)]
struct SearchState {
    rigid_shapes: HashSet<Arc<FramedPoset>>,
    boundary_classes: HashMap<(usize, Arc<FramedPoset>), BoundaryClass>,
    rigid_candidates: u64,
    tested_pastings: u64,
}

struct Candidate {
    sample: u64,
    shape: Arc<FramedPoset>,
}

struct Failure {
    gluing: u64,
    direction: usize,
    first: BoundaryOccurrence,
    second: BoundaryOccurrence,
    boundary_isomorphism: Embedding,
    boundary_into_second: Embedding,
    pushout: Pushout,
}

#[derive(Serialize)]
struct FailureReport<'a> {
    seed: String,
    dimension: usize,
    cell_count: usize,
    worker_count: usize,
    generated_candidates: u64,
    rigid_candidates: u64,
    distinct_rigid_shapes: usize,
    tested_pastings: u64,
    direction: usize,
    first_sample: u64,
    second_sample: u64,
    first_and_second_are_equal: bool,
    boundary_isomorphism_map: &'a [Vec<usize>],
    pushout_sizes: Vec<usize>,
    pushout_connected: bool,
}

fn main() -> io::Result<()> {
    let options = arguments()?;
    let generator = RandomFramedPosetGenerator::new(options.dimension, options.cell_count);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.worker_count)
        .build()
        .map_err(io::Error::other)?;
    let batch_size = u64::try_from(options.worker_count)
        .map_err(|_| invalid_input("worker count does not fit in u64"))?
        .checked_mul(CANDIDATES_PER_WORKER)
        .ok_or_else(|| invalid_input("candidate batch size overflow"))?;

    println!(
        "searching random rigid {}-dimensional, {}-cell OFPs on {} threads (seed {:#018x})",
        options.dimension, options.cell_count, options.worker_count, options.seed
    );
    println!(
        "only nonempty output/input boundary pastings are considered; artifacts will be written below {}",
        failure_root().display()
    );

    let started = Instant::now();
    let mut next_report = started + REPORT_INTERVAL;
    let mut generated = 0u64;
    let mut state = SearchState::default();

    loop {
        let end = generated
            .checked_add(batch_size)
            .ok_or_else(|| io::Error::other("candidate counter overflow"))?;
        let candidates = pool.install(|| {
            (generated..end)
                .into_par_iter()
                .filter_map(|sample| rigid_candidate(&generator, options.seed, sample + 1))
                .collect::<Vec<_>>()
        });
        generated = end;

        for candidate in candidates {
            state.rigid_candidates += 1;
            if let Some(failure) = inspect_candidate(&mut state, candidate)? {
                let output = write_failure(&options, generated, &state, &failure)?;
                println!(
                    "found a non-rigid boundary pasting after {generated} generated candidates and {} tested pastings ({:.1?}); wrote {}",
                    state.tested_pastings,
                    started.elapsed(),
                    output.display()
                );
                return Ok(());
            }
        }

        if Instant::now() >= next_report {
            report_progress(generated, &state, started);
            next_report = Instant::now() + REPORT_INTERVAL;
        }
    }
}

fn rigid_candidate(
    generator: &RandomFramedPosetGenerator,
    seed: u64,
    sample: u64,
) -> Option<Candidate> {
    let mut rng = SmallRng::seed_from_u64(sample_seed(seed, sample));
    let shape = generator.generate(&mut rng);
    if !shape.is_rigid() {
        return None;
    }

    Some(Candidate {
        sample,
        shape: Arc::new(normalize(&shape)),
    })
}

fn sample_seed(seed: u64, sample: u64) -> u64 {
    // SplitMix64 turns adjacent sample numbers into independent-looking seeds.
    let mut value = seed.wrapping_add(sample.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn inspect_candidate(state: &mut SearchState, candidate: Candidate) -> io::Result<Option<Failure>> {
    if !state.rigid_shapes.insert(Arc::clone(&candidate.shape)) {
        return Ok(None);
    }

    for direction in candidate.shape.total_frame() {
        let (_, output_into_shape) = boundary(Sign::Output, direction, &candidate.shape);
        let (_, input_into_shape) = boundary(Sign::Input, direction, &candidate.shape);
        if output_into_shape.is_empty() || input_into_shape.is_empty() {
            return Err(io::Error::other(format!(
                "rigid sample {} has an empty total-frame direction {direction} boundary",
                candidate.sample
            )));
        }

        let output = BoundaryOccurrence {
            sample: candidate.sample,
            into_shape: output_into_shape,
        };
        let input = BoundaryOccurrence {
            sample: candidate.sample,
            into_shape: input_into_shape,
        };
        let output_key = (direction, Arc::new(normalize(&output.into_shape.dom)));
        let input_key = (direction, Arc::new(normalize(&input.into_shape.dom)));

        let mut tested_pastings = state.tested_pastings;
        if let Some(class) = state.boundary_classes.get(&output_key) {
            for second in &class.inputs {
                increment(&mut tested_pastings, "pasting counter")?;
                if let Some(failure) = test_pasting(tested_pastings, direction, &output, second)? {
                    state.tested_pastings = tested_pastings;
                    return Ok(Some(failure));
                }
            }
        }

        if let Some(class) = state.boundary_classes.get(&input_key) {
            for first in &class.outputs {
                increment(&mut tested_pastings, "pasting counter")?;
                if let Some(failure) = test_pasting(tested_pastings, direction, first, &input)? {
                    state.tested_pastings = tested_pastings;
                    return Ok(Some(failure));
                }
            }
        }

        if output_key == input_key {
            increment(&mut tested_pastings, "pasting counter")?;
            if let Some(failure) = test_pasting(tested_pastings, direction, &output, &input)? {
                state.tested_pastings = tested_pastings;
                return Ok(Some(failure));
            }
        }
        state.tested_pastings = tested_pastings;

        state
            .boundary_classes
            .entry(output_key)
            .or_default()
            .outputs
            .push(output);
        state
            .boundary_classes
            .entry(input_key)
            .or_default()
            .inputs
            .push(input);
    }

    Ok(None)
}

fn test_pasting(
    gluing: u64,
    direction: usize,
    first: &BoundaryOccurrence,
    second: &BoundaryOccurrence,
) -> io::Result<Option<Failure>> {
    let mut boundary_isomorphisms = isomorphisms(&first.into_shape.dom, &second.into_shape.dom);
    if boundary_isomorphisms.len() != 1 {
        return Err(io::Error::other(format!(
            "rigid boundaries in direction {direction} had {} isomorphisms instead of one",
            boundary_isomorphisms.len()
        )));
    }

    let boundary_isomorphism = boundary_isomorphisms.pop().unwrap();
    let boundary_into_second = Embedding::compose(&boundary_isomorphism, &second.into_shape);
    let pasted = pushout(&first.into_shape, &boundary_into_second);

    if pasted.tip.is_rigid() {
        Ok(None)
    } else {
        Ok(Some(Failure {
            gluing,
            direction,
            first: first.clone(),
            second: second.clone(),
            boundary_isomorphism,
            boundary_into_second,
            pushout: pasted,
        }))
    }
}

fn increment(counter: &mut u64, name: &str) -> io::Result<()> {
    *counter = counter
        .checked_add(1)
        .ok_or_else(|| io::Error::other(format!("{name} overflow")))?;
    Ok(())
}

fn report_progress(generated: u64, state: &SearchState, started: Instant) {
    let elapsed = started.elapsed();
    let rate = generated as f64 / elapsed.as_secs_f64();
    println!(
        "generated {generated} candidates ({rate:.0}/s); retained {} rigid candidates and {} distinct rigid shapes; indexed {} boundary classes; tested {} pastings ({elapsed:.1?})",
        state.rigid_candidates,
        state.rigid_shapes.len(),
        state.boundary_classes.len(),
        state.tested_pastings
    );
}

fn write_failure(
    options: &Options,
    generated: u64,
    state: &SearchState,
    failure: &Failure,
) -> io::Result<PathBuf> {
    let output_dir = unique_failure_directory(options.seed, failure.gluing)?;
    let first = &failure.first.into_shape.cod;
    let second = &failure.second.into_shape.cod;
    let report = FailureReport {
        seed: format!("{:#018x}", options.seed),
        dimension: options.dimension,
        cell_count: options.cell_count,
        worker_count: options.worker_count,
        generated_candidates: generated,
        rigid_candidates: state.rigid_candidates,
        distinct_rigid_shapes: state.rigid_shapes.len(),
        tested_pastings: state.tested_pastings,
        direction: failure.direction,
        first_sample: failure.first.sample,
        second_sample: failure.second.sample,
        first_and_second_are_equal: FramedPoset::equal(first, second),
        boundary_isomorphism_map: &failure.boundary_isomorphism.map,
        pushout_sizes: failure.pushout.tip.sizes(),
        pushout_connected: failure.pushout.tip.is_connected(),
    };

    fs::write(
        output_dir.join("report.json"),
        format!(
            "{}\n",
            serde_json::to_string_pretty(&report).map_err(io::Error::other)?
        ),
    )?;
    write_shape_artifacts(&output_dir, "first", first)?;
    write_shape_artifacts(&output_dir, "second", second)?;
    write_shape_artifacts(&output_dir, "pushout", &failure.pushout.tip)?;
    write_embedding_artifacts(
        &output_dir,
        "first_output_boundary",
        &failure.first.into_shape,
    )?;
    write_embedding_artifacts(
        &output_dir,
        "second_input_boundary",
        &failure.second.into_shape,
    )?;
    write_embedding_artifacts(
        &output_dir,
        "boundary_isomorphism",
        &failure.boundary_isomorphism,
    )?;
    write_embedding_artifacts(
        &output_dir,
        "boundary_into_second",
        &failure.boundary_into_second,
    )?;
    write_embedding_artifacts(&output_dir, "first_into_pushout", &failure.pushout.inl)?;
    write_embedding_artifacts(&output_dir, "second_into_pushout", &failure.pushout.inr)?;
    Ok(output_dir)
}

fn write_shape_artifacts(output_dir: &Path, name: &str, shape: &FramedPoset) -> io::Result<()> {
    fs::write(
        output_dir.join(format!("{name}.ofp.json")),
        format!(
            "{}\n",
            serde_json::to_string_pretty(shape).map_err(io::Error::other)?
        ),
    )?;
    fs::write(
        output_dir.join(format!("{name}_graded.dot")),
        to_dot(shape, Renderer::Ranked),
    )?;
    fs::write(
        output_dir.join(format!("{name}_compass_spring.dot")),
        to_dot(shape, Renderer::CompassSpring),
    )
}

fn write_embedding_artifacts(
    output_dir: &Path,
    name: &str,
    embedding: &Embedding,
) -> io::Result<()> {
    fs::write(
        output_dir.join(format!("{name}_graded.dot")),
        embedding_to_dot(embedding, Renderer::Ranked),
    )?;
    fs::write(
        output_dir.join(format!("{name}_compass_spring.dot")),
        embedding_to_dot(embedding, Renderer::CompassSpring),
    )
}

fn unique_failure_directory(seed: u64, gluing: u64) -> io::Result<PathBuf> {
    let root = failure_root();
    fs::create_dir_all(&root)?;

    for suffix in 0usize.. {
        let suffix = if suffix == 0 {
            String::new()
        } else {
            format!("_{suffix}")
        };
        let path = root.join(format!("seed_{seed:016x}_gluing_{gluing}{suffix}"));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    unreachable!()
}

fn failure_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FAILURE_ROOT)
}

fn arguments() -> io::Result<Options> {
    let mut dimension = DEFAULT_DIMENSION;
    let mut cell_count = DEFAULT_CELL_COUNT;
    let mut worker_count = thread::available_parallelism()?.get();
    let mut seed = None;
    let mut arguments = env::args().skip(1);

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--dimension" => dimension = parse_usize_argument(&mut arguments, "--dimension")?,
            "--cells" => cell_count = parse_usize_argument(&mut arguments, "--cells")?,
            "--threads" => worker_count = parse_usize_argument(&mut arguments, "--threads")?,
            "--seed" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| invalid_input("--seed requires a value"))?;
                seed = Some(parse_seed(&value)?);
            }
            "--help" | "-h" => {
                println!(
                    "usage: find_non_rigid_boundary_pasting [--dimension N] [--cells N] [--threads N] [--seed N|0xHEX]"
                );
                std::process::exit(0);
            }
            _ => return Err(invalid_input(format!("unknown argument {argument:?}"))),
        }
    }

    if dimension == 0 {
        return Err(invalid_input("dimension must be positive"));
    }
    if dimension >= usize::BITS as usize {
        return Err(invalid_input(format!(
            "dimension must be smaller than {}",
            usize::BITS
        )));
    }
    let minimum_cell_count = 1usize << dimension;
    if cell_count < minimum_cell_count {
        return Err(invalid_input(format!(
            "at least {minimum_cell_count} cells are required in dimension {dimension}"
        )));
    }
    if worker_count == 0 {
        return Err(invalid_input("thread count must be positive"));
    }

    Ok(Options {
        dimension,
        cell_count,
        worker_count,
        seed: seed.map_or_else(|| OsRng.try_next_u64().map_err(io::Error::other), Ok)?,
    })
}

fn parse_usize_argument(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> io::Result<usize> {
    let value = arguments
        .next()
        .ok_or_else(|| invalid_input(format!("{option} requires a value")))?;
    value
        .parse()
        .map_err(|error| invalid_input(format!("invalid value for {option}: {error}")))
}

fn parse_seed(value: &str) -> io::Result<u64> {
    let parsed = if let Some(hexadecimal) = value.strip_prefix("0x") {
        u64::from_str_radix(hexadecimal, 16)
    } else {
        value.parse()
    };
    parsed.map_err(|error| invalid_input(format!("invalid seed {value:?}: {error}")))
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
