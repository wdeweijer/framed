use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ofposets::FramedPoset;
use ofposets::enumeration::{
    PasteCandidateCounts, PolyvoxelCatalog, PolyvoxelEnumerationPhase, PolyvoxelEnumerationStage,
    PolyvoxelEnumerationTiming, PolyvoxelFactorization, enumerate_polyvoxels_profiled,
};
use serde::Serialize;

const DEFAULT_MAX_CELLS: usize = 27;
const DEFAULT_MAX_DIRECTION: usize = 2;
const PROGRESS_PRINT_INTERVAL: Duration = Duration::from_secs(5);
const TIMING_PRINT_THRESHOLD: Duration = Duration::from_secs(1);

#[derive(Debug)]
struct Config {
    max_cells: usize,
    max_direction: usize,
    allowed_directions: Vec<usize>,
    length_bound: Option<usize>,
    thread_count: Option<usize>,
    output_file: PathBuf,
}

impl Config {
    fn from_args() -> io::Result<Option<Self>> {
        let mut args = env::args_os();
        let program = args
            .next()
            .unwrap_or_else(|| OsString::from("enumerate_polyvoxels"));
        Self::parse(program, args)
    }

    fn parse(
        program: OsString,
        args: impl IntoIterator<Item = OsString>,
    ) -> io::Result<Option<Self>> {
        let mut max_cells = DEFAULT_MAX_CELLS;
        let mut max_direction = DEFAULT_MAX_DIRECTION;
        let mut length_bound = None;
        let mut thread_count = None;
        let mut output_file = None;
        let mut args = args.into_iter();

        while let Some(argument) = args.next() {
            let Some(option) = argument.to_str() else {
                return Err(argument_error(&program, "option names must be valid UTF-8"));
            };
            match option {
                "-h" | "--help" => {
                    println!("{}", usage(&program));
                    return Ok(None);
                }
                "--max-cells" => {
                    max_cells = parse_usize_option(&mut args, &program, option)?;
                }
                "--max-direction" => {
                    max_direction = parse_usize_option(&mut args, &program, option)?;
                }
                "--length-bound" => {
                    let bound = parse_usize_option(&mut args, &program, option)?;
                    length_bound = (bound != 0).then_some(bound);
                }
                "--threads" => {
                    let count = parse_usize_option(&mut args, &program, option)?;
                    thread_count = (count != 0).then_some(count);
                }
                "--output" => {
                    output_file = Some(PathBuf::from(next_option_value(
                        &mut args, &program, option,
                    )?));
                }
                _ => {
                    return Err(argument_error(
                        &program,
                        &format!("unknown option: {option}"),
                    ));
                }
            }
        }

        if max_cells == 0 {
            return Err(argument_error(&program, "--max-cells must be positive"));
        }
        let direction_count = max_direction
            .checked_add(1)
            .ok_or_else(|| argument_error(&program, "--max-direction is too large"))?;
        let allowed_directions = (0..direction_count).collect();
        let output_file = output_file.unwrap_or_else(|| {
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

        Ok(Some(Self {
            max_cells,
            max_direction,
            allowed_directions,
            length_bound,
            thread_count,
            output_file,
        }))
    }
}

fn usage(program: &OsString) -> String {
    format!(
        "Usage: {} [OPTIONS]\n\
         \n\
         Options:\n\
           --max-cells <N>       Maximum number of cells [default: {DEFAULT_MAX_CELLS}]\n\
           --max-direction <N>   Allow directions 0 through N [default: {DEFAULT_MAX_DIRECTION}]\n\
           --length-bound <N>    Exclusive directional-length bound; 0 is unbounded [default: 0]\n\
           --threads <N>         Rayon worker threads; 0 uses all available threads [default: 0]\n\
           --output <PATH>       Output JSONL file [default: derived from the bounds]\n\
           -h, --help            Print help",
        Path::new(program)
            .file_name()
            .unwrap_or(program.as_os_str())
            .to_string_lossy(),
    )
}

fn next_option_value(
    args: &mut impl Iterator<Item = OsString>,
    program: &OsString,
    option: &str,
) -> io::Result<OsString> {
    args.next()
        .ok_or_else(|| argument_error(program, &format!("missing value for {option}")))
}

fn parse_usize_option(
    args: &mut impl Iterator<Item = OsString>,
    program: &OsString,
    option: &str,
) -> io::Result<usize> {
    let value = next_option_value(args, program, option)?;
    value
        .to_str()
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| argument_error(program, &format!("{option} must be a nonnegative integer")))
}

fn argument_error(program: &OsString, message: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("{message}\n\n{}", usage(program)),
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
    results: usize,
    wall_time: Duration,
    construction_work: Duration,
    canonicalisation_work: Duration,
    merge_time: Duration,
    paste_candidates: PasteCandidateCounts,
}

impl TimingTotal {
    fn record(&mut self, timing: PolyvoxelEnumerationTiming) {
        self.jobs += timing.jobs;
        self.results += timing.results;
        self.wall_time += timing.wall_time;
        self.construction_work += timing.construction_work;
        self.canonicalisation_work += timing.canonicalisation_work;
        self.merge_time += timing.merge_time;
        if let Some(counts) = timing.paste_candidates {
            self.paste_candidates.merge(counts);
        }
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
    let mut pool_builder = rayon::ThreadPoolBuilder::new();
    if let Some(thread_count) = config.thread_count {
        pool_builder = pool_builder.num_threads(thread_count);
    }
    let pool = pool_builder.build().map_err(io::Error::other)?;
    println!(
        "enumerating polyvoxels with at most {} cells, active directions from 0 through {}, {}, using {} threads",
        config.max_cells,
        config.max_direction,
        length_description,
        pool.current_num_threads(),
    );
    let started = Instant::now();
    let last_status_print = Mutex::new(started);
    let mut timings = BTreeMap::<PolyvoxelEnumerationStage, TimingTotal>::new();
    let catalog = pool.install(|| {
        enumerate_polyvoxels_profiled(
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
                } else if mark_if_interval_elapsed(
                    &last_status_print,
                    PROGRESS_PRINT_INTERVAL,
                ) {
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
                if timing.wall_time >= TIMING_PRINT_THRESHOLD {
                    print_timing(timing);
                    mark_printed_now(&last_status_print);
                }
                timings.entry(timing.stage).or_default().record(timing);
            },
        )
    });
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

fn mark_if_interval_elapsed(last_print: &Mutex<Instant>, interval: Duration) -> bool {
    let mut last_print = last_print.lock().expect("status-print mutex was poisoned");
    if last_print.elapsed() < interval {
        return false;
    }
    *last_print = Instant::now();
    true
}

fn mark_printed_now(last_print: &Mutex<Instant>) {
    *last_print.lock().expect("status-print mutex was poisoned") = Instant::now();
}

fn print_timing(timing: PolyvoxelEnumerationTiming) {
    let counts = format_counts(timing.jobs, timing.results);
    if timing.construction_work.is_zero()
        && timing.canonicalisation_work.is_zero()
        && timing.merge_time.is_zero()
    {
        println!(
            "timing round {} {:?}: {:.3?} wall, {counts}",
            timing.round, timing.stage, timing.wall_time,
        );
    } else {
        println!(
            "timing round {} {:?}: {:.3?} wall, {counts}; worker sum: {:.3?} construction + {:.3?} canonicalisation; {:.3?} merge",
            timing.round,
            timing.stage,
            timing.wall_time,
            timing.construction_work,
            timing.canonicalisation_work,
            timing.merge_time,
        );
    }
    if let Some(counts) = timing.paste_candidates {
        print_paste_candidate_counts("  ", counts);
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
        let counts = format_counts(timing.jobs, timing.results);
        println!(
            "  {stage:?}: {:.3?} wall ({wall_percent:.1}%), {counts}; worker sum: {:.3?} construction + {:.3?} canonicalisation; {:.3?} merge",
            timing.wall_time,
            timing.construction_work,
            timing.canonicalisation_work,
            timing.merge_time,
        );
        if timing.paste_candidates.examined != 0 {
            print_paste_candidate_counts("    ", timing.paste_candidates);
        }
    }
}

fn format_counts(jobs: usize, results: usize) -> String {
    if jobs == results {
        format!("{jobs} inputs/results")
    } else {
        format!("{jobs} inputs -> {results} results")
    }
}

fn print_paste_candidate_counts(indent: &str, counts: PasteCandidateCounts) {
    println!(
        "{indent}index scan: {} lookups, {} nonempty buckets, {} indexed pairs; {} pruned by cell bound, {} examined",
        counts.boundary_index_lookups,
        counts.nonempty_boundary_buckets,
        counts.indexed_pairs,
        counts.pruned_by_cell_bound,
        counts.examined,
    );
    println!(
        "{indent}examined outcomes: {} length-bound + {} boundary-form rejections; {} accepted",
        counts.rejected_by_length_bound, counts.rejected_by_boundary_normal_form, counts.accepted,
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(arguments: &[&str]) -> io::Result<Option<Config>> {
        Config::parse(
            OsString::from("enumerate_polyvoxels"),
            arguments.iter().map(OsString::from),
        )
    }

    #[test]
    fn omitted_options_use_the_historical_defaults() {
        let config = parse(&[]).unwrap().unwrap();

        assert_eq!(config.max_cells, 27);
        assert_eq!(config.max_direction, 2);
        assert_eq!(config.allowed_directions, vec![0, 1, 2]);
        assert_eq!(config.length_bound, None);
        assert_eq!(config.thread_count, None);
        assert!(config.output_file.ends_with(
            "visualizations/polyvoxels_up_to_27_cells_directions_0_to_2_unbounded_length.jsonl"
        ));
    }

    #[test]
    fn named_options_override_the_defaults_in_any_order() {
        let config = parse(&[
            "--threads",
            "6",
            "--output",
            "catalog.jsonl",
            "--length-bound",
            "4",
            "--max-direction",
            "3",
            "--max-cells",
            "50",
        ])
        .unwrap()
        .unwrap();

        assert_eq!(config.max_cells, 50);
        assert_eq!(config.max_direction, 3);
        assert_eq!(config.allowed_directions, vec![0, 1, 2, 3]);
        assert_eq!(config.length_bound, Some(4));
        assert_eq!(config.thread_count, Some(6));
        assert_eq!(config.output_file, Path::new("catalog.jsonl"));
    }

    #[test]
    fn zero_selects_unbounded_length_and_automatic_thread_count() {
        let config = parse(&["--length-bound", "0", "--threads", "0"])
            .unwrap()
            .unwrap();

        assert_eq!(config.length_bound, None);
        assert_eq!(config.thread_count, None);
    }

    #[test]
    fn invalid_options_are_rejected() {
        assert!(parse(&["--max-cells", "0"]).is_err());
        assert!(parse(&["--threads"]).is_err());
        assert!(parse(&["--unknown", "1"]).is_err());
    }
}
