use std::collections::BTreeMap;
use std::hint::black_box;
use std::io;
use std::time::{Duration, Instant};

use ofposets::enumeration::{PolyvoxelEnumerationPhase, enumerate_polyvoxels_with_progress};
use ofposets::{FramedPoset, Polyvoxel, normalize, randomly_permute, traversal_normalisation};
use rand::SeedableRng;
use rand::rngs::SmallRng;

const MAX_CELLS: usize = 27;
const ALLOWED_DIRECTIONS: &[usize] = &[0, 1, 2];
const PERMUTATIONS_PER_POLYVOXEL: usize = 100;
const SEED: u64 = 0x7a4e_25a1_2026_0902;

#[derive(Debug, Default)]
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

    fn graph_over_traversal(&self) -> f64 {
        self.graph.as_secs_f64() / self.traversal.as_secs_f64()
    }
}

fn main() -> io::Result<()> {
    if cfg!(debug_assertions) {
        eprintln!("timing results include debug-only validation; use --release for comparison");
    }
    println!(
        "enumerating polyvoxels with at most {MAX_CELLS} cells and total frame contained in {ALLOWED_DIRECTIONS:?}"
    );
    let enumeration_started = Instant::now();
    let catalog = enumerate_polyvoxels_with_progress(MAX_CELLS, ALLOWED_DIRECTIONS, |progress| {
        if progress.phase == PolyvoxelEnumerationPhase::Complete {
            println!(
                "enumeration complete: {} representatives in {:.1?}",
                progress.representatives,
                enumeration_started.elapsed(),
            );
        } else if progress.completed_jobs == progress.total_jobs {
            println!(
                "round {} {:?} complete: {} representatives ({:.1?})",
                progress.round,
                progress.phase,
                progress.representatives,
                enumeration_started.elapsed(),
            );
        }
    });

    let total_checks = catalog
        .len()
        .checked_mul(PERMUTATIONS_PER_POLYVOXEL)
        .expect("number of traversal checks exceeds usize");
    let progress_step = catalog.len().div_ceil(20).max(1);
    let checking_started = Instant::now();
    let mut rng = SmallRng::seed_from_u64(SEED);
    let mut checked = 0usize;
    let mut timings = NormalisationTimings::default();
    let mut timings_by_cells = BTreeMap::<usize, NormalisationTimings>::new();

    for (polyvoxel_index, entry) in catalog.entries().iter().enumerate() {
        let (expected, _) = traversal_normalisation(&entry.shape).map_err(|error| {
            io::Error::other(format!(
                "baseline traversal failed for catalogue entry {polyvoxel_index}: {error}; OFP: {}",
                serde_json::to_string(entry.shape.as_ref()).unwrap(),
            ))
        })?;
        let mut expected_graph_normal = None;
        let cells = entry.shape.sizes().iter().sum::<usize>();

        for permutation_index in 0..PERMUTATIONS_PER_POLYVOXEL {
            let (permuted_shape, into_original) =
                randomly_permute(entry.shape.as_framed_poset(), &mut rng);
            let permuted =
                Polyvoxel::from_isomorphism(permuted_shape, &into_original, &entry.shape);
            let ((traversal_result, traversal_elapsed), (graph_normal, graph_elapsed)) =
                if checked.is_multiple_of(2) {
                    let traversal = measure(|| traversal_normalisation(black_box(&permuted)));
                    let graph = measure(|| normalize(black_box(permuted.as_ref())));
                    (traversal, graph)
                } else {
                    let graph = measure(|| normalize(black_box(permuted.as_ref())));
                    let traversal = measure(|| traversal_normalisation(black_box(&permuted)));
                    (traversal, graph)
                };

            let (actual, _) = traversal_result.map_err(|error| {
                io::Error::other(format!(
                    "traversal failed for catalogue entry {polyvoxel_index}, permutation {permutation_index}, seed {SEED:#018x}: {error}; permuted OFP: {}",
                    serde_json::to_string(permuted.as_ref()).unwrap(),
                ))
            })?;

            if !FramedPoset::equal(&expected, &actual) {
                return Err(io::Error::other(format!(
                    "traversal is not invariant for catalogue entry {polyvoxel_index}, permutation {permutation_index}, seed {SEED:#018x}; original OFP: {}; permuted OFP: {}; expected traversal form: {}; actual traversal form: {}",
                    serde_json::to_string(entry.shape.as_ref()).unwrap(),
                    serde_json::to_string(permuted.as_ref()).unwrap(),
                    serde_json::to_string(expected.as_ref()).unwrap(),
                    serde_json::to_string(actual.as_ref()).unwrap(),
                )));
            }

            if let Some(expected) = &expected_graph_normal {
                if !FramedPoset::equal(expected, &graph_normal) {
                    return Err(io::Error::other(format!(
                        "graph normalisation is not invariant for catalogue entry {polyvoxel_index}, permutation {permutation_index}, seed {SEED:#018x}; permuted OFP: {}; expected graph form: {}; actual graph form: {}",
                        serde_json::to_string(permuted.as_ref()).unwrap(),
                        serde_json::to_string(expected).unwrap(),
                        serde_json::to_string(&graph_normal).unwrap(),
                    )));
                }
            } else {
                expected_graph_normal = Some(graph_normal);
            }

            timings.record(traversal_elapsed, graph_elapsed);
            timings_by_cells
                .entry(cells)
                .or_default()
                .record(traversal_elapsed, graph_elapsed);
            checked += 1;
        }

        let completed = polyvoxel_index + 1;
        if completed == catalog.len() || completed.is_multiple_of(progress_step) {
            println!(
                "checked {checked}/{total_checks} permutations across {completed}/{} polyvoxels ({:.1?})",
                catalog.len(),
                checking_started.elapsed(),
            );
            println!(
                "  traversal {:.2?}, graph {:.2?}, graph/traversal {:.2}x",
                timings.traversal,
                timings.graph,
                timings.graph_over_traversal(),
            );
        }
    }

    println!(
        "all {checked} random permutations produced the same traversal form (seed {SEED:#018x}, {:.1?})",
        checking_started.elapsed(),
    );
    print_timing_statistics(&timings, &timings_by_cells);
    Ok(())
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
