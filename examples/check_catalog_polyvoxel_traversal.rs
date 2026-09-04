use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs::File;
use std::hint::black_box;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ofposets::{FramedPoset, normalize, randomly_permute, traversal_normalisation_of_shape};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use serde::Deserialize;

const DEFAULT_INPUT: &str =
    "visualizations/polyvoxels_up_to_55_cells_directions_0_to_3_length_below_4.jsonl";
const DEFAULT_PERMUTATIONS: usize = 1;
const SEED: u64 = 0x7a4e_25a1_2026_0902;
const BUFFER_CAPACITY: usize = 8 * 1024 * 1024;
const LOAD_REPORT_INTERVAL: usize = 100_000;

#[derive(Debug)]
struct Config {
    input_file: PathBuf,
    permutations: usize,
    thread_count: Option<usize>,
    limit: Option<usize>,
}

impl Config {
    fn from_args() -> io::Result<Option<Self>> {
        let mut args = env::args_os();
        let program = args
            .next()
            .unwrap_or_else(|| OsString::from("check_catalog_polyvoxel_traversal"));
        Self::parse(program, args)
    }

    fn parse(
        program: OsString,
        args: impl IntoIterator<Item = OsString>,
    ) -> io::Result<Option<Self>> {
        let mut input_file = Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_INPUT);
        let mut permutations = DEFAULT_PERMUTATIONS;
        let mut thread_count = None;
        let mut limit = None;
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
                "--input" => {
                    input_file = PathBuf::from(next_option_value(&mut args, &program, option)?);
                }
                "--permutations" => {
                    permutations = parse_usize_option(&mut args, &program, option)?;
                }
                "--threads" => {
                    let count = parse_usize_option(&mut args, &program, option)?;
                    thread_count = (count != 0).then_some(count);
                }
                "--limit" => {
                    let count = parse_usize_option(&mut args, &program, option)?;
                    limit = (count != 0).then_some(count);
                }
                _ => {
                    return Err(argument_error(
                        &program,
                        &format!("unknown option: {option}"),
                    ));
                }
            }
        }

        if permutations == 0 {
            return Err(argument_error(&program, "--permutations must be positive"));
        }

        Ok(Some(Self {
            input_file,
            permutations,
            thread_count,
            limit,
        }))
    }
}

fn usage(program: &OsString) -> String {
    format!(
        "Usage: {} [OPTIONS]\n\
         \n\
         Options:\n\
           --input <PATH>        Catalogue JSONL [default: {DEFAULT_INPUT}]\n\
           --permutations <N>    Random permutations per polyvoxel [default: {DEFAULT_PERMUTATIONS}]\n\
           --threads <N>         Rayon worker threads; 0 uses all available threads [default: 0]\n\
           --limit <N>           Check only the first N records; 0 checks all [default: 0]\n\
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCatalogRecord {
    version: usize,
    id: usize,
    cells: usize,
    active_directions: Vec<usize>,
    #[serde(rename = "length")]
    _length: Vec<usize>,
    #[serde(rename = "layering_direction")]
    _layering_direction: Option<usize>,
    #[serde(rename = "is_voxel")]
    _is_voxel: bool,
    #[serde(rename = "factorizations")]
    _factorizations: serde::de::IgnoredAny,
    ofp: FramedPoset,
}

struct CatalogRecord {
    id: usize,
    ofp: Arc<FramedPoset>,
}

#[derive(Debug, Default, Clone, Copy)]
struct NormalisationTimings {
    samples: usize,
    traversal: Duration,
    graph: Duration,
    traversal_faster: usize,
}

impl NormalisationTimings {
    fn record(&mut self, traversal: Duration, graph: Duration) {
        self.samples += 1;
        self.traversal += traversal;
        self.graph += graph;
        self.traversal_faster += usize::from(traversal < graph);
    }

    fn merge(&mut self, other: Self) {
        self.samples += other.samples;
        self.traversal += other.traversal;
        self.graph += other.graph;
        self.traversal_faster += other.traversal_faster;
    }

    fn graph_over_traversal(&self) -> f64 {
        self.graph.as_secs_f64() / self.traversal.as_secs_f64()
    }
}

struct EntryTimings {
    cells: usize,
    timings: NormalisationTimings,
}

fn main() -> io::Result<()> {
    let Some(config) = Config::from_args()? else {
        return Ok(());
    };
    if cfg!(debug_assertions) {
        eprintln!("timing results include debug-only validation; use --release for comparison");
    }

    let loading_started = Instant::now();
    let records = load_records(&config.input_file, config.limit)?;
    println!(
        "loaded {} catalogue records from {} in {:.1?}",
        records.len(),
        config.input_file.display(),
        loading_started.elapsed(),
    );

    let total_checks = records
        .len()
        .checked_mul(config.permutations)
        .ok_or_else(|| invalid_data("number of traversal checks exceeds usize"))?;
    let mut pool_builder = rayon::ThreadPoolBuilder::new();
    if let Some(thread_count) = config.thread_count {
        pool_builder = pool_builder.num_threads(thread_count);
    }
    let pool = pool_builder.build().map_err(io::Error::other)?;
    println!(
        "checking {total_checks} permutations of {} catalogue OFPs using {} threads",
        records.len(),
        pool.current_num_threads(),
    );

    let checking_started = Instant::now();
    let completed = AtomicUsize::new(0);
    let progress_step = records.len().div_ceil(20).max(1);
    let results = pool.install(|| {
        records
            .par_iter()
            .map(|record| {
                let result = check_ofp(record.id, &record.ofp, config.permutations)?;
                let completed = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if completed == records.len() || completed.is_multiple_of(progress_step) {
                    println!(
                        "checked {}/{} catalogue OFPs ({}/{} permutations, {:.1?})",
                        completed,
                        records.len(),
                        completed * config.permutations,
                        total_checks,
                        checking_started.elapsed(),
                    );
                }
                Ok(result)
            })
            .collect::<io::Result<Vec<_>>>()
    })?;

    let mut timings = NormalisationTimings::default();
    let mut timings_by_cells = BTreeMap::<usize, NormalisationTimings>::new();
    for result in results {
        timings.merge(result.timings);
        timings_by_cells
            .entry(result.cells)
            .or_default()
            .merge(result.timings);
    }

    println!(
        "all {total_checks} random permutations produced the same traversal form (base seed {SEED:#018x}, {:.1?})",
        checking_started.elapsed(),
    );
    print_timing_statistics(&timings, &timings_by_cells);
    Ok(())
}

fn load_records(path: &Path, limit: Option<usize>) -> io::Result<Vec<CatalogRecord>> {
    let mut reader = BufReader::with_capacity(BUFFER_CAPACITY, File::open(path)?);
    let mut records = Vec::new();
    let mut line = String::new();
    let mut line_number = 0usize;

    loop {
        if limit.is_some_and(|limit| records.len() >= limit) {
            break;
        }
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        line_number += 1;
        let raw: RawCatalogRecord = serde_json::from_str(&line)
            .map_err(|error| invalid_data(format!("{}:{line_number}: {error}", path.display())))?;
        if raw.version != 1 {
            return Err(invalid_data(format!(
                "{}:{line_number}: unsupported catalogue version {}",
                path.display(),
                raw.version,
            )));
        }
        if raw.id != records.len() {
            return Err(invalid_data(format!(
                "{}:{line_number}: expected record id {}, found {}",
                path.display(),
                records.len(),
                raw.id,
            )));
        }
        let actual_cells = raw.ofp.sizes().iter().sum::<usize>();
        if raw.cells != actual_cells {
            return Err(invalid_data(format!(
                "{}:{line_number}: record {} declares {} cells but its OFP has {actual_cells}",
                path.display(),
                raw.id,
                raw.cells,
            )));
        }
        let actual_directions = raw.ofp.active_directions();
        if raw.active_directions != actual_directions {
            return Err(invalid_data(format!(
                "{}:{line_number}: record {} has incorrect active directions",
                path.display(),
                raw.id,
            )));
        }
        records.push(CatalogRecord {
            id: raw.id,
            ofp: Arc::new(raw.ofp),
        });
        if records.len().is_multiple_of(LOAD_REPORT_INTERVAL) {
            println!("loaded {} records", records.len());
        }
    }

    if records.is_empty() {
        return Err(invalid_data(format!(
            "{} contains no catalogue records",
            path.display(),
        )));
    }
    Ok(records)
}
fn check_ofp(
    index: usize,
    shape: &Arc<FramedPoset>,
    permutations: usize,
) -> io::Result<EntryTimings> {
    let (expected, _) = traversal_normalisation_of_shape(shape).map_err(|error| {
        invalid_data(format!(
            "baseline traversal failed for catalogue entry {index}: {error}; OFP: {}",
            serde_json::to_string(shape.as_ref()).unwrap(),
        ))
    })?;
    let cells = shape.sizes().iter().sum::<usize>();
    let rng_seed = seed_for_entry(index);
    let mut rng = SmallRng::seed_from_u64(rng_seed);
    let mut expected_graph_normal = None;
    let mut timings = NormalisationTimings::default();

    for permutation_index in 0..permutations {
        let (permuted, _) = randomly_permute(shape, &mut rng);
        let sample = index * permutations + permutation_index;
        let ((traversal_result, traversal_elapsed), (graph_normal, graph_elapsed)) =
            if sample.is_multiple_of(2) {
                let traversal = measure(|| traversal_normalisation_of_shape(black_box(&permuted)));
                let graph = measure(|| normalize(black_box(permuted.as_ref())));
                (traversal, graph)
            } else {
                let graph = measure(|| normalize(black_box(permuted.as_ref())));
                let traversal = measure(|| traversal_normalisation_of_shape(black_box(&permuted)));
                (traversal, graph)
            };

        let (actual, _) = traversal_result.map_err(|error| {
            invalid_data(format!(
                "traversal failed for catalogue entry {index}, permutation {permutation_index}, seed {rng_seed:#018x}: {error}; permuted OFP: {}",
                serde_json::to_string(permuted.as_ref()).unwrap(),
            ))
        })?;
        if !FramedPoset::equal(&expected, &actual) {
            return Err(invalid_data(format!(
                "traversal is not invariant for catalogue entry {index}, permutation {permutation_index}, seed {rng_seed:#018x}; original OFP: {}; permuted OFP: {}; expected traversal form: {}; actual traversal form: {}",
                serde_json::to_string(shape.as_ref()).unwrap(),
                serde_json::to_string(permuted.as_ref()).unwrap(),
                serde_json::to_string(expected.as_ref()).unwrap(),
                serde_json::to_string(actual.as_ref()).unwrap(),
            )));
        }

        if let Some(expected) = &expected_graph_normal {
            if !FramedPoset::equal(expected, &graph_normal) {
                return Err(invalid_data(format!(
                    "graph normalisation is not invariant for catalogue entry {index}, permutation {permutation_index}, seed {rng_seed:#018x}",
                )));
            }
        } else {
            expected_graph_normal = Some(graph_normal);
        }
        timings.record(traversal_elapsed, graph_elapsed);
    }

    Ok(EntryTimings { cells, timings })
}

fn seed_for_entry(index: usize) -> u64 {
    let mut value = SEED.wrapping_add((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn measure<T>(operation: impl FnOnce() -> T) -> (T, Duration) {
    let started = Instant::now();
    let result = black_box(operation());
    (result, started.elapsed())
}

fn print_timing_statistics(
    total: &NormalisationTimings,
    by_cells: &BTreeMap<usize, NormalisationTimings>,
) {
    println!("normalisation timings:");
    println!(
        "  total: traversal {:.3?}, graph {:.3?}, graph/traversal {:.3}x, traversal faster in {}/{} samples ({:.1}%)",
        total.traversal,
        total.graph,
        total.graph_over_traversal(),
        total.traversal_faster,
        total.samples,
        100.0 * total.traversal_faster as f64 / total.samples as f64,
    );
    println!(
        "cells\tsamples\ttraversal_mean_us\tgraph_mean_us\tgraph_over_traversal\ttraversal_faster_percent"
    );
    for (&cells, timing) in by_cells {
        println!(
            "{cells}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.1}",
            timing.samples,
            timing.traversal.as_secs_f64() * 1_000_000.0 / timing.samples as f64,
            timing.graph.as_secs_f64() * 1_000_000.0 / timing.samples as f64,
            timing.graph_over_traversal(),
            100.0 * timing.traversal_faster as f64 / timing.samples as f64,
        );
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arrow() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![1]]],
        ))
    }

    #[test]
    fn named_options_have_safe_large_catalogue_defaults() {
        let config = Config::parse(
            OsString::from("check_catalog_polyvoxel_traversal"),
            std::iter::empty(),
        )
        .unwrap()
        .unwrap();

        assert!(config.input_file.ends_with(DEFAULT_INPUT));
        assert_eq!(config.permutations, 1);
        assert_eq!(config.thread_count, None);
        assert_eq!(config.limit, None);
    }

    #[test]
    fn traversal_check_uses_an_ofp_directly() {
        let result = check_ofp(0, &arrow(), 2).unwrap();

        assert_eq!(result.cells, 3);
        assert_eq!(result.timings.samples, 2);
    }
}
