use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ofposets::FramedPoset;
use ofposets::enumeration::{
    PolyvoxelCatalog, PolyvoxelEnumerationPhase, PolyvoxelEnumerationStage,
    PolyvoxelEnumerationTiming, PolyvoxelFactorization, enumerate_polyvoxels_profiled,
};
use serde::Serialize;

#[derive(Debug)]
struct Config {
    max_cells: usize,
    max_direction: usize,
    allowed_directions: Vec<usize>,
    length_bound: Option<usize>,
    output_file: PathBuf,
}

impl Config {
    fn from_args() -> io::Result<Option<Self>> {
        let mut args = env::args_os();
        let program = args
            .next()
            .unwrap_or_else(|| OsString::from("enumerate_polyvoxels"));
        let Some(max_cells) = args.next() else {
            return Err(invalid_arguments(&program));
        };
        if max_cells == "-h" || max_cells == "--help" {
            println!("{}", usage(&program));
            return Ok(None);
        }

        let max_cells = max_cells
            .to_str()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .ok_or_else(|| invalid_arguments(&program))?;
        let max_direction = args
            .next()
            .and_then(|value| value.to_str().and_then(|value| value.parse::<usize>().ok()))
            .ok_or_else(|| invalid_arguments(&program))?;
        let direction_count = max_direction
            .checked_add(1)
            .ok_or_else(|| invalid_arguments(&program))?;
        let allowed_directions = (0..direction_count).collect();
        let length_bound = args
            .next()
            .and_then(|value| value.to_str().and_then(|value| value.parse::<usize>().ok()))
            .map(|bound| (bound != 0).then_some(bound))
            .ok_or_else(|| invalid_arguments(&program))?;
        let output_file = args.next().map(PathBuf::from).unwrap_or_else(|| {
            let length_label = length_bound.map_or_else(
                || "unbounded_length".to_owned(),
                |bound| format!("length_below_{bound}"),
            );
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("visualizations")
                .join(format!(
                    "polyvoxels_up_to_{max_cells}_cells_directions_0_to_{max_direction}_{length_label}.jsonl"
                ))
        });
        if args.next().is_some() {
            return Err(invalid_arguments(&program));
        }

        Ok(Some(Self {
            max_cells,
            max_direction,
            allowed_directions,
            length_bound,
            output_file,
        }))
    }
}

fn usage(program: &OsString) -> String {
    format!(
        "usage: {} <max-cells> <max-direction> <length-bound> [output.jsonl]",
        Path::new(program)
            .file_name()
            .unwrap_or(program.as_os_str())
            .to_string_lossy(),
    )
}

fn invalid_arguments(program: &OsString) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "{}; <max-cells> must be positive, directions must be nonnegative, and a length bound of 0 means infinity",
            usage(program),
        ),
    )
}

#[derive(Serialize)]
struct CatalogRecord<'a> {
    version: usize,
    id: usize,
    cells: usize,
    active_directions: Vec<usize>,
    length: &'a [usize],
    layering_direction: Option<usize>,
    is_voxel: bool,
    factorizations: &'a [PolyvoxelFactorization],
    ofp: &'a FramedPoset,
}

#[derive(Debug, Default)]
struct TimingTotal {
    jobs: usize,
    wall_time: Duration,
    construction_work: Duration,
    canonicalisation_work: Duration,
    merge_time: Duration,
}

impl TimingTotal {
    fn record(&mut self, timing: PolyvoxelEnumerationTiming) {
        self.jobs += timing.jobs;
        self.wall_time += timing.wall_time;
        self.construction_work += timing.construction_work;
        self.canonicalisation_work += timing.canonicalisation_work;
        self.merge_time += timing.merge_time;
    }
}

fn main() -> io::Result<()> {
    let Some(config) = Config::from_args()? else {
        return Ok(());
    };

    let length_description = config.length_bound.map_or_else(
        || "without a directional-length bound".to_owned(),
        |bound| format!("with every directional length below {bound}"),
    );
    println!(
        "enumerating polyvoxels with at most {} cells, active directions from 0 through {}, and {}",
        config.max_cells, config.max_direction, length_description,
    );
    let started = Instant::now();
    let mut timings = BTreeMap::<PolyvoxelEnumerationStage, TimingTotal>::new();
    let catalog = enumerate_polyvoxels_profiled(
        config.max_cells,
        &config.allowed_directions,
        config.length_bound,
        |progress| {
            if progress.phase == PolyvoxelEnumerationPhase::Complete {
                println!(
                    "fixed point complete after {} rounds: {} representatives, {} factorizations ({:.1?})",
                    progress.round,
                    progress.representatives,
                    progress.factorizations,
                    started.elapsed(),
                );
            } else {
                println!(
                    "round {} {:?}: {}/{} jobs; {} representatives, {} factorizations ({:.1?})",
                    progress.round,
                    progress.phase,
                    progress.completed_jobs,
                    progress.total_jobs,
                    progress.representatives,
                    progress.factorizations,
                    started.elapsed(),
                );
            }
        },
        |timing| {
            print_timing(timing);
            timings.entry(timing.stage).or_default().record(timing);
        },
    );
    print_timing_summary(&timings);

    write_catalog(&config.output_file, &catalog)?;
    let total_factorizations: usize = catalog
        .entries()
        .iter()
        .map(|entry| entry.factorizations.len())
        .sum();
    println!(
        "wrote {} polyvoxels with {total_factorizations} immediate factorizations to {}",
        catalog.len(),
        config.output_file.display(),
    );

    Ok(())
}

fn print_timing(timing: PolyvoxelEnumerationTiming) {
    if timing.construction_work.is_zero()
        && timing.canonicalisation_work.is_zero()
        && timing.merge_time.is_zero()
    {
        println!(
            "timing round {} {:?}: {:.3?} wall, {} results",
            timing.round, timing.stage, timing.wall_time, timing.jobs,
        );
    } else {
        println!(
            "timing round {} {:?}: {:.3?} wall, {} jobs; worker sum: {:.3?} construction + {:.3?} canonicalisation; {:.3?} merge",
            timing.round,
            timing.stage,
            timing.wall_time,
            timing.jobs,
            timing.construction_work,
            timing.canonicalisation_work,
            timing.merge_time,
        );
    }
}

fn print_timing_summary(timings: &BTreeMap<PolyvoxelEnumerationStage, TimingTotal>) {
    let total_wall: Duration = timings.values().map(|timing| timing.wall_time).sum();
    println!("enumeration timing summary:");
    for (stage, timing) in timings {
        let wall_percent = if total_wall.is_zero() {
            0.0
        } else {
            100.0 * timing.wall_time.as_secs_f64() / total_wall.as_secs_f64()
        };
        println!(
            "  {stage:?}: {:.3?} wall ({wall_percent:.1}%), {} jobs/results; worker sum: {:.3?} construction + {:.3?} canonicalisation; {:.3?} merge",
            timing.wall_time,
            timing.jobs,
            timing.construction_work,
            timing.canonicalisation_work,
            timing.merge_time,
        );
    }
}

fn write_catalog(output_file: &Path, catalog: &PolyvoxelCatalog) -> io::Result<()> {
    if let Some(parent) = output_file.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let temporary_file = output_file.with_extension("jsonl.tmp");
    {
        let mut jsonl = BufWriter::new(File::create(&temporary_file)?);
        for (id, entry) in catalog.entries().iter().enumerate() {
            serde_json::to_writer(
                &mut jsonl,
                &CatalogRecord {
                    version: 1,
                    id,
                    cells: entry.shape.sizes().iter().sum(),
                    active_directions: entry.shape.active_directions(),
                    length: entry.shape.length(),
                    layering_direction: entry.shape.layering_direction(),
                    is_voxel: entry.is_voxel,
                    factorizations: &entry.factorizations,
                    ofp: &entry.shape,
                },
            )
            .map_err(io::Error::other)?;
            writeln!(jsonl)?;
        }
        jsonl.flush()?;
    }
    fs::rename(temporary_file, output_file)
}
