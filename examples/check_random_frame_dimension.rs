use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ofposets::{
    BoundaryMode, CubularityMode, FramedPoset, RandomFramedPosetGenerator, Renderer, is_cubular,
    to_dot,
};
use rand::rngs::{OsRng, SmallRng};
use rand::{SeedableRng, TryRngCore};
use rayon::prelude::*;

const DEFAULT_SAMPLE_COUNT: u64 = 100_000;
const DEFAULT_CELL_COUNT: usize = 9;
const DEFAULT_FRAME_DIMENSION: usize = 3;
const REPORT_EVERY: u64 = 1_000_000;
const WORKER_COUNT: usize = 24;
const OUTPUT_DIR: &str = "visualizations/random_frame_dimension_counterexamples";

#[derive(Debug, Default)]
struct Statistics {
    generated: AtomicU64,
    strongly_cubular: AtomicU64,
    connected_and_strongly_cubular: AtomicU64,
    rigid_and_strongly_cubular: AtomicU64,
}

#[derive(Debug, Clone, Copy)]
struct StatisticsSnapshot {
    generated: u64,
    strongly_cubular: u64,
    connected_and_strongly_cubular: u64,
    rigid_and_strongly_cubular: u64,
}

struct Witness {
    sample: u64,
    shape: Arc<FramedPoset>,
}

fn main() -> io::Result<()> {
    let (sample_count, cell_count, frame_dimension, seed) = arguments()?;
    let generator = RandomFramedPosetGenerator::new_without_full_basis(frame_dimension, cell_count);
    let statistics = Statistics::default();
    let first_connected = Mutex::new(None);
    let first_rigid = Mutex::new(None);
    let started = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(WORKER_COUNT)
        .build()
        .map_err(io::Error::other)?;

    println!(
        "checking {sample_count} random {cell_count}-cell OFPs with frame directions {:?}, \
         dimension {}, and no full-frame cells on {WORKER_COUNT} threads (seed {seed:#018x})",
        (0..frame_dimension).collect::<Vec<_>>(),
        frame_dimension - 1,
    );

    pool.install(|| {
        (1..=sample_count).into_par_iter().for_each(|sample| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(sample));
            let shape = Arc::new(generator.generate(&mut rng));
            debug_assert_eq!(
                shape.active_directions(),
                (0..frame_dimension).collect::<Vec<_>>()
            );
            debug_assert_eq!(shape.dim(), frame_dimension as isize - 1);

            if is_cubular(BoundaryMode::Maximal, CubularityMode::Strong, &shape) {
                statistics.strongly_cubular.fetch_add(1, Ordering::Relaxed);

                if shape.is_connected() {
                    statistics
                        .connected_and_strongly_cubular
                        .fetch_add(1, Ordering::Relaxed);
                    retain_earliest(&first_connected, sample, &shape);

                    if shape.is_rigid() {
                        statistics
                            .rigid_and_strongly_cubular
                            .fetch_add(1, Ordering::Relaxed);
                        retain_earliest(&first_rigid, sample, &shape);
                    }
                }
            }

            let completed = statistics.generated.fetch_add(1, Ordering::Relaxed) + 1;
            report_progress(completed, sample_count, started);
        });
    });

    let statistics = statistics.snapshot();
    let first_connected = first_connected
        .into_inner()
        .expect("connected-witness mutex was poisoned");
    let first_rigid = first_rigid
        .into_inner()
        .expect("rigid-witness mutex was poisoned");

    if let Some(witness) = &first_connected {
        write_witness("connected", frame_dimension, witness.sample, &witness.shape)?;
    }
    if let Some(witness) = &first_rigid {
        write_witness("rigid", frame_dimension, witness.sample, &witness.shape)?;
    }

    println!("generated: {}", statistics.generated);
    println!("strongly cubular: {}", statistics.strongly_cubular);
    println!(
        "connected and strongly cubular: {}",
        statistics.connected_and_strongly_cubular
    );
    println!(
        "rigid and strongly cubular: {}",
        statistics.rigid_and_strongly_cubular
    );

    match &first_connected {
        Some(witness) => println!(
            "connectedness does not imply the frame/dimension thesis; first witness was sample \
             {}",
            witness.sample
        ),
        None => println!(
            "no connected strongly cubular counterexample to the frame/dimension thesis was found"
        ),
    }
    if first_connected.is_some() {
        match &first_rigid {
            Some(witness) => println!(
                "rigidity also does not imply the thesis; first rigid witness was sample {}",
                witness.sample
            ),
            None => println!("no rigid strongly cubular counterexample was found"),
        }
    }

    Ok(())
}

impl Statistics {
    fn snapshot(&self) -> StatisticsSnapshot {
        StatisticsSnapshot {
            generated: self.generated.load(Ordering::Relaxed),
            strongly_cubular: self.strongly_cubular.load(Ordering::Relaxed),
            connected_and_strongly_cubular: self
                .connected_and_strongly_cubular
                .load(Ordering::Relaxed),
            rigid_and_strongly_cubular: self.rigid_and_strongly_cubular.load(Ordering::Relaxed),
        }
    }
}

fn retain_earliest(slot: &Mutex<Option<Witness>>, sample: u64, shape: &Arc<FramedPoset>) {
    let mut witness = slot.lock().expect("witness mutex was poisoned");
    if witness
        .as_ref()
        .is_none_or(|current| sample < current.sample)
    {
        *witness = Some(Witness {
            sample,
            shape: Arc::clone(shape),
        });
    }
}

fn report_progress(completed: u64, sample_count: u64, started: Instant) {
    if completed.is_multiple_of(REPORT_EVERY) || completed == sample_count {
        println!(
            "checked {completed}/{sample_count} ({:.1?})",
            started.elapsed()
        );
    }
}

fn write_witness(
    kind: &str,
    frame_dimension: usize,
    sample: u64,
    shape: &FramedPoset,
) -> io::Result<PathBuf> {
    let output_dir = Path::new(OUTPUT_DIR);
    fs::create_dir_all(output_dir)?;
    let stem = format!("frame_{frame_dimension}_first_{kind}_sample_{sample}");

    fs::write(
        output_dir.join(format!("{stem}.ofp.json")),
        format!(
            "{}\n",
            serde_json::to_string_pretty(shape).map_err(io::Error::other)?
        ),
    )?;
    fs::write(
        output_dir.join(format!("{stem}_graded.dot")),
        to_dot(shape, Renderer::Ranked),
    )?;
    fs::write(
        output_dir.join(format!("{stem}_compass_spring.dot")),
        to_dot(shape, Renderer::CompassSpring),
    )?;

    Ok(output_dir.to_path_buf())
}

fn arguments() -> io::Result<(u64, usize, usize, u64)> {
    let mut arguments = env::args().skip(1);
    let sample_count = arguments
        .next()
        .map(|value| parse_u64("sample count", &value))
        .transpose()?
        .unwrap_or(DEFAULT_SAMPLE_COUNT);
    let cell_count = arguments
        .next()
        .map(|value| parse_usize("cell count", &value))
        .transpose()?
        .unwrap_or(DEFAULT_CELL_COUNT);
    let frame_dimension = arguments
        .next()
        .map(|value| parse_usize("frame dimension", &value))
        .transpose()?
        .unwrap_or(DEFAULT_FRAME_DIMENSION);
    let seed = arguments
        .next()
        .map(|value| parse_u64("seed", &value))
        .transpose()?
        .map_or_else(|| OsRng.try_next_u64().map_err(io::Error::other), Ok)?;

    if arguments.next().is_some() {
        return Err(invalid_input(
            "usage: check_random_frame_dimension [sample-count] [cell-count] \
             [frame-dimension] [seed]",
        ));
    }
    if sample_count == 0 {
        return Err(invalid_input("sample count must be positive"));
    }
    if !(2..usize::BITS as usize).contains(&frame_dimension) {
        return Err(invalid_input(
            "frame dimension must be at least 2 and smaller than the number of bits in usize",
        ));
    }
    let minimum_cell_count = (1usize << (frame_dimension - 1)) + 1;
    if cell_count < minimum_cell_count {
        return Err(invalid_input(format!(
            "cell count must be at least {minimum_cell_count} for frame dimension \
             {frame_dimension} and OFP dimension {}",
            frame_dimension - 1
        )));
    }

    Ok((sample_count, cell_count, frame_dimension, seed))
}

fn parse_u64(name: &str, value: &str) -> io::Result<u64> {
    let parsed = value
        .strip_prefix("0x")
        .map_or_else(|| value.parse(), |hex| u64::from_str_radix(hex, 16));
    parsed.map_err(|error| invalid_input(format!("invalid {name} {value:?}: {error}")))
}

fn parse_usize(name: &str, value: &str) -> io::Result<usize> {
    value
        .parse()
        .map_err(|error| invalid_input(format!("invalid {name} {value:?}: {error}")))
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
