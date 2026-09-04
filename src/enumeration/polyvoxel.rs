//! Small exhaustive enumerations of bounded polyvoxels.
//!
//! This keeps the inductive construction simple, while indexing canonical
//! boundary forms so that cylinder and paste candidates need not be found by
//! scanning every pair. Factorizations form a packed graph: they refer to
//! operand polyvoxels, whose own factorizations represent all recursive
//! choices.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::embedding::Embedding;
use crate::intset::{self, IntSet};
use crate::isomorphism::normalisation as graph_normalisation;
use crate::polyvoxel::{Polyvoxel, cylinder, paste, point, shift};
use crate::poset::{FramedPoset, Sign, boundary};
use crate::volumetric::is_volumetric;

#[cfg(test)]
use crate::isomorphism::{isomorphic, isomorphisms, normalize};
/// One immediate construction of a polyvoxel in a bounded catalogue.
///
/// Operand values are indices into the containing [`PolyvoxelCatalog`].
/// Recursive factorization trees are represented implicitly by following
/// those operands' own factorizations.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "constructor", rename_all = "snake_case")]
pub enum PolyvoxelFactorization {
    Point,
    Shift {
        source: usize,
    },
    Cylinder {
        input: usize,
        output: usize,
    },
    Paste {
        direction: usize,
        left: usize,
        right: usize,
        boundary_isomorphism: Vec<Vec<usize>>,
    },
}

impl PolyvoxelFactorization {
    fn remap(self, old_to_new: &[usize]) -> Self {
        match self {
            Self::Point => Self::Point,
            Self::Shift { source } => Self::Shift {
                source: old_to_new[source],
            },
            Self::Cylinder { input, output } => Self::Cylinder {
                input: old_to_new[input],
                output: old_to_new[output],
            },
            Self::Paste {
                direction,
                left,
                right,
                boundary_isomorphism,
            } => Self::Paste {
                direction,
                left: old_to_new[left],
                right: old_to_new[right],
                boundary_isomorphism,
            },
        }
    }
}

/// One isomorphism class in a bounded polyvoxel catalogue.
#[derive(Debug, Clone)]
pub struct PolyvoxelEntry {
    pub shape: Polyvoxel,
    pub is_voxel: bool,
    pub factorizations: Vec<PolyvoxelFactorization>,
}

/// Polyvoxels in canonical order, together with every immediate construction
/// found within the same cell and direction bounds.
#[derive(Debug, Clone)]
pub struct PolyvoxelCatalog {
    entries: Vec<PolyvoxelEntry>,
}

impl PolyvoxelCatalog {
    pub fn entries(&self) -> &[PolyvoxelEntry] {
        &self.entries
    }

    pub fn entry(&self, index: usize) -> &PolyvoxelEntry {
        &self.entries[index]
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// One phase of a bounded polyvoxel fixed-point round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyvoxelEnumerationPhase {
    Shift,
    Cylinder,
    Paste,
    Complete,
}

/// A milestone emitted during bounded polyvoxel enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolyvoxelEnumerationProgress {
    pub round: usize,
    pub phase: PolyvoxelEnumerationPhase,
    pub completed_jobs: usize,
    pub total_jobs: usize,
    pub representatives: usize,
    pub factorizations: usize,
}

/// A separately timed stage of polyvoxel enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PolyvoxelEnumerationStage {
    BoundaryIndexing,
    Shift,
    CylinderMatching,
    Cylinder,
    PasteCandidateScanning,
    PasteCandidateSorting,
    PasteCandidateDeduplication,
    PasteProcessedFiltering,
    Paste,
    BoundaryCaching,
}

/// Counts collected while scanning indexed partners for paste candidates.
///
/// Rejections are classified by the first failed test, in the order listed
/// here. `accepted` counts candidates before duplicate and previously
/// processed jobs are removed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PasteCandidateCounts {
    pub boundary_index_lookups: usize,
    pub nonempty_boundary_buckets: usize,
    pub indexed_pairs: usize,
    pub pruned_by_cell_bound: usize,
    pub examined: usize,
    pub rejected_by_length_bound: usize,
    pub rejected_by_boundary_normal_form: usize,
    pub accepted: usize,
}

impl PasteCandidateCounts {
    pub fn merge(&mut self, other: Self) {
        self.boundary_index_lookups += other.boundary_index_lookups;
        self.nonempty_boundary_buckets += other.nonempty_boundary_buckets;
        self.indexed_pairs += other.indexed_pairs;
        self.pruned_by_cell_bound += other.pruned_by_cell_bound;
        self.examined += other.examined;
        self.rejected_by_length_bound += other.rejected_by_length_bound;
        self.rejected_by_boundary_normal_form += other.rejected_by_boundary_normal_form;
        self.accepted += other.accepted;
    }
}

/// Basic performance measurements for one enumeration stage.
///
/// `construction_work` and `canonicalisation_work` are sums over Rayon worker
/// tasks and can therefore exceed `wall_time`. `merge_time` is sequential.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolyvoxelEnumerationTiming {
    pub round: usize,
    pub stage: PolyvoxelEnumerationStage,
    /// Number of inputs examined by this stage.
    pub jobs: usize,
    /// Number of outputs retained by this stage.
    pub results: usize,
    pub wall_time: Duration,
    pub construction_work: Duration,
    pub canonicalisation_work: Duration,
    pub merge_time: Duration,
    pub paste_candidates: Option<PasteCandidateCounts>,
}

/// Enumerate polyvoxels up to isomorphism within finite cell and direction
/// bounds.
///
/// `allowed_directions` must be sorted and deduplicated. The enumeration is
/// closed under point, shift, elementary cylinder, and every directional
/// pasting along every boundary isomorphism that remains inside the bounds.
/// Repeated immediate constructions are retained as distinct factorizations.
pub fn enumerate_polyvoxels(max_cells: usize, allowed_directions: &[usize]) -> PolyvoxelCatalog {
    enumerate_polyvoxels_with_length_bound(max_cells, allowed_directions, None)
}

/// Enumerate bounded polyvoxels whose directional lengths are strictly below
/// `length_bound`. Passing `None` imposes no length bound.
pub fn enumerate_polyvoxels_with_length_bound(
    max_cells: usize,
    allowed_directions: &[usize],
    length_bound: Option<usize>,
) -> PolyvoxelCatalog {
    enumerate_polyvoxels_with_length_bound_and_progress(
        max_cells,
        allowed_directions,
        length_bound,
        |_| {},
    )
}

/// Enumerate bounded polyvoxels while reporting phase milestones.
///
/// At most about ten milestones are emitted per phase and fixed-point round,
/// in addition to phase starts and the final completion event.
pub fn enumerate_polyvoxels_with_progress(
    max_cells: usize,
    allowed_directions: &[usize],
    report: impl FnMut(PolyvoxelEnumerationProgress),
) -> PolyvoxelCatalog {
    enumerate_polyvoxels_with_length_bound_and_progress(max_cells, allowed_directions, None, report)
}

/// Enumerate length-bounded polyvoxels while reporting phase milestones.
///
/// Every entry of a retained polyvoxel's length vector is strictly below
/// `length_bound`. Passing `None` imposes no length bound.
pub fn enumerate_polyvoxels_with_length_bound_and_progress(
    max_cells: usize,
    allowed_directions: &[usize],
    length_bound: Option<usize>,
    report: impl FnMut(PolyvoxelEnumerationProgress),
) -> PolyvoxelCatalog {
    enumerate_polyvoxels_profiled(max_cells, allowed_directions, length_bound, report, |_| {})
}

/// Enumerate polyvoxels while reporting both progress and timing metrics.
pub fn enumerate_polyvoxels_profiled(
    max_cells: usize,
    allowed_directions: &[usize],
    length_bound: Option<usize>,
    mut report: impl FnMut(PolyvoxelEnumerationProgress),
    mut report_timing: impl FnMut(PolyvoxelEnumerationTiming),
) -> PolyvoxelCatalog {
    assert!(
        intset::is_sorted_unique(allowed_directions),
        "allowed directions must be sorted and deduplicated",
    );

    let mut builder = CatalogBuilder::new(max_cells, allowed_directions.to_vec(), length_bound);
    let initial = prepare_candidate(
        Candidate {
            shape: point(),
            is_voxel: true,
            factorization: PolyvoxelFactorization::Point,
        },
        max_cells,
        allowed_directions,
        length_bound,
    )
    .expect("the point must fit every polyvoxel catalogue");
    let initial = builder.record(initial);
    debug_assert!(initial.new_entry);
    debug_assert!(initial.new_voxel);
    let mut entry_frontier = vec![initial.index];
    let mut voxel_frontier = vec![initial.index];
    let mut boundary_index = BoundaryIndex::default();
    let mut processed_jobs = ProcessedJobs::default();
    let cache_started = Instant::now();
    let cached_boundaries = builder.uncached_entry_count();
    if cached_boundaries > 0 {
        builder.populate_boundary_caches();
    }
    report_timing(stage_timing(
        0,
        PolyvoxelEnumerationStage::BoundaryCaching,
        cached_boundaries,
        cache_started.elapsed(),
    ));

    loop {
        let round = builder.round;
        let indexing_started = Instant::now();
        let entries = builder.snapshot();
        let indexed_entries =
            boundary_index.add_entries(&entries, &entry_frontier, allowed_directions);
        let indexed_voxels =
            boundary_index.add_voxels(&entries, &voxel_frontier, allowed_directions);
        report_timing(stage_timing(
            round,
            PolyvoxelEnumerationStage::BoundaryIndexing,
            indexed_entries + indexed_voxels,
            indexing_started.elapsed(),
        ));

        let shift_jobs: Vec<_> = voxel_frontier
            .iter()
            .copied()
            .filter(|source| processed_jobs.shifts.insert(*source))
            .collect();

        let shift = process_jobs(
            &mut builder,
            &shift_jobs,
            round,
            PolyvoxelEnumerationPhase::Shift,
            &mut report,
            |&source| Candidate {
                shape: shift(&entries[source].shape),
                is_voxel: true,
                factorization: PolyvoxelFactorization::Shift { source },
            },
        );
        report_timing(shift.timing(round, PolyvoxelEnumerationStage::Shift, shift_jobs.len()));

        let matching_started = Instant::now();
        let cylinder_jobs = compatible_cylinder_jobs(
            &entries,
            &boundary_index,
            &entry_frontier,
            &voxel_frontier,
            allowed_directions,
            length_bound,
            &mut processed_jobs.cylinders,
        );
        report_timing(stage_timing(
            round,
            PolyvoxelEnumerationStage::CylinderMatching,
            cylinder_jobs.len(),
            matching_started.elapsed(),
        ));
        let cylinder = process_jobs(
            &mut builder,
            &cylinder_jobs,
            round,
            PolyvoxelEnumerationPhase::Cylinder,
            &mut report,
            |job| Candidate {
                shape: cylinder(&entries[job.input].shape, &entries[job.output].shape),
                is_voxel: true,
                factorization: PolyvoxelFactorization::Cylinder {
                    input: job.input,
                    output: job.output,
                },
            },
        );
        report_timing(cylinder.timing(
            round,
            PolyvoxelEnumerationStage::Cylinder,
            cylinder_jobs.len(),
        ));

        let paste_matching = compatible_paste_jobs(
            &entries,
            &boundary_index,
            &entry_frontier,
            max_cells,
            length_bound,
            &mut processed_jobs.pastes,
        );
        for timing in paste_matching.timings(round) {
            report_timing(timing);
        }
        let paste_jobs = paste_matching.jobs;
        let paste = process_jobs(
            &mut builder,
            &paste_jobs,
            round,
            PolyvoxelEnumerationPhase::Paste,
            &mut report,
            |job| {
                let left_boundary = entries[job.left].boundary(job.direction).output();
                let right_boundary = entries[job.right].boundary(job.direction).input();
                let isomorphism = left_boundary
                    .isomorphism_to(right_boundary)
                    .expect("indexed boundary normal forms must agree");
                let shape = paste(
                    &entries[job.left].shape,
                    &entries[job.right].shape,
                    job.direction,
                )
                .1;
                Candidate {
                    shape,
                    is_voxel: false,
                    factorization: PolyvoxelFactorization::Paste {
                        direction: job.direction,
                        left: job.left,
                        right: job.right,
                        boundary_isomorphism: isomorphism.map,
                    },
                }
            },
        );
        report_timing(paste.timing(round, PolyvoxelEnumerationStage::Paste, paste_jobs.len()));

        let mut next_entries = BTreeSet::new();
        let mut next_voxels = BTreeSet::new();
        for processing in [&shift, &cylinder, &paste] {
            next_entries.extend(processing.new_entries.iter().copied());
            next_voxels.extend(processing.new_voxels.iter().copied());
        }

        let cache_started = Instant::now();
        let cached_boundaries = builder.uncached_entry_count();
        if cached_boundaries > 0 {
            builder.populate_boundary_caches();
        }
        report_timing(stage_timing(
            round,
            PolyvoxelEnumerationStage::BoundaryCaching,
            cached_boundaries,
            cache_started.elapsed(),
        ));

        if next_entries.is_empty() && next_voxels.is_empty() {
            report(PolyvoxelEnumerationProgress {
                round,
                phase: PolyvoxelEnumerationPhase::Complete,
                completed_jobs: 0,
                total_jobs: 0,
                representatives: builder.entries.len(),
                factorizations: builder.factorization_count(),
            });
            return builder.finish();
        }
        entry_frontier = next_entries.into_iter().collect();
        voxel_frontier = next_voxels.into_iter().collect();
        builder.round += 1;
    }
}

fn report_milestone(
    builder: &CatalogBuilder,
    report: &mut impl FnMut(PolyvoxelEnumerationProgress),
    round: usize,
    phase: PolyvoxelEnumerationPhase,
    completed_jobs: usize,
    total_jobs: usize,
) {
    let step = total_jobs.div_ceil(10).max(1);
    if completed_jobs == 0 || completed_jobs == total_jobs || completed_jobs.is_multiple_of(step) {
        report(PolyvoxelEnumerationProgress {
            round,
            phase,
            completed_jobs,
            total_jobs,
            representatives: builder.entries.len(),
            factorizations: builder.factorization_count(),
        });
    }
}

fn stage_timing(
    round: usize,
    stage: PolyvoxelEnumerationStage,
    jobs: usize,
    wall_time: Duration,
) -> PolyvoxelEnumerationTiming {
    stage_timing_with_results(round, stage, jobs, jobs, wall_time)
}

fn stage_timing_with_results(
    round: usize,
    stage: PolyvoxelEnumerationStage,
    jobs: usize,
    results: usize,
    wall_time: Duration,
) -> PolyvoxelEnumerationTiming {
    PolyvoxelEnumerationTiming {
        round,
        stage,
        jobs,
        results,
        wall_time,
        construction_work: Duration::ZERO,
        canonicalisation_work: Duration::ZERO,
        merge_time: Duration::ZERO,
        paste_candidates: None,
    }
}

fn process_jobs<J, F>(
    builder: &mut CatalogBuilder,
    jobs: &[J],
    round: usize,
    phase: PolyvoxelEnumerationPhase,
    report: &mut impl FnMut(PolyvoxelEnumerationProgress),
    construct: F,
) -> JobProcessing
where
    J: Sync,
    F: Fn(&J) -> Candidate + Sync,
{
    let wall_started = Instant::now();
    report_milestone(builder, report, round, phase, 0, jobs.len());
    let chunk_size = jobs.len().div_ceil(10).max(1);
    let max_cells = builder.max_cells;
    let allowed_directions = builder.allowed_directions.clone();
    let length_bound = builder.length_bound;
    let mut new_entries = BTreeSet::new();
    let mut new_voxels = BTreeSet::new();
    let mut construction_work = Duration::ZERO;
    let mut canonicalisation_work = Duration::ZERO;
    let mut merge_time = Duration::ZERO;

    for (chunk_index, chunk) in jobs.chunks(chunk_size).enumerate() {
        let candidates: Vec<_> = chunk
            .par_iter()
            .map(|job| {
                let construction_started = Instant::now();
                let candidate = construct(job);
                let construction_time = construction_started.elapsed();
                let canonicalisation_started = Instant::now();
                let candidate =
                    prepare_candidate(candidate, max_cells, &allowed_directions, length_bound);
                let canonicalisation_time = canonicalisation_started.elapsed();
                (candidate, construction_time, canonicalisation_time)
            })
            .collect();
        let merge_started = Instant::now();
        for (candidate, construction_time, canonicalisation_time) in candidates {
            construction_work += construction_time;
            canonicalisation_work += canonicalisation_time;
            if let Some(candidate) = candidate {
                let outcome = builder.record(candidate);
                if outcome.new_entry {
                    new_entries.insert(outcome.index);
                }
                if outcome.new_voxel {
                    new_voxels.insert(outcome.index);
                }
            }
        }
        merge_time += merge_started.elapsed();

        let completed = ((chunk_index + 1) * chunk_size).min(jobs.len());
        report_milestone(builder, report, round, phase, completed, jobs.len());
    }

    JobProcessing {
        new_entries,
        new_voxels,
        wall_time: wall_started.elapsed(),
        construction_work,
        canonicalisation_work,
        merge_time,
    }
}

struct JobProcessing {
    new_entries: BTreeSet<usize>,
    new_voxels: BTreeSet<usize>,
    wall_time: Duration,
    construction_work: Duration,
    canonicalisation_work: Duration,
    merge_time: Duration,
}

impl JobProcessing {
    fn timing(
        &self,
        round: usize,
        stage: PolyvoxelEnumerationStage,
        jobs: usize,
    ) -> PolyvoxelEnumerationTiming {
        PolyvoxelEnumerationTiming {
            round,
            stage,
            jobs,
            results: jobs,
            wall_time: self.wall_time,
            construction_work: self.construction_work,
            canonicalisation_work: self.canonicalisation_work,
            merge_time: self.merge_time,
            paste_candidates: None,
        }
    }
}

#[derive(Default)]
struct ProcessedJobs {
    shifts: HashSet<usize>,
    cylinders: HashSet<CylinderJob>,
    pastes: HashSet<PasteJob>,
}

type CylinderBoundaryKey = (u64, u64);
type SignedBoundaryKey = (usize, u64, usize);

#[derive(Default)]
struct CellIndexedEntries {
    by_cells: BTreeMap<usize, Vec<usize>>,
    len: usize,
}

impl CellIndexedEntries {
    fn push(&mut self, cells: usize, index: usize) {
        self.by_cells.entry(cells).or_default().push(index);
        self.len += 1;
    }

    fn up_to(&self, max_cells: usize) -> impl Iterator<Item = usize> + '_ {
        self.by_cells
            .range(..=max_cells)
            .flat_map(|(_, entries)| entries.iter().copied())
    }
}

#[derive(Default)]
struct BoundaryIndex {
    indexed_entries: HashSet<usize>,
    indexed_voxels: HashSet<usize>,
    cylinder_outputs: HashMap<CylinderBoundaryKey, Vec<usize>>,
    cylinder_inputs: HashMap<CylinderBoundaryKey, Vec<usize>>,
    paste_inputs: HashMap<SignedBoundaryKey, CellIndexedEntries>,
    paste_outputs: HashMap<SignedBoundaryKey, CellIndexedEntries>,
}

impl BoundaryIndex {
    fn add_entries(
        &mut self,
        entries: &[SnapshotEntry],
        new_entries: &[usize],
        allowed_directions: &[usize],
    ) -> usize {
        let cylinder_enabled = allowed_directions.binary_search(&0).is_ok();
        let mut added = 0;
        for &index in new_entries {
            if !self.indexed_entries.insert(index) {
                continue;
            }
            added += 1;
            let entry = &entries[index];
            if cylinder_enabled {
                self.cylinder_outputs
                    .entry(cylinder_boundary_key(entry))
                    .or_default()
                    .push(index);
            }
            for &direction in &entry.frame {
                let boundary = entry.boundary(direction);
                self.paste_inputs
                    .entry((direction, boundary.input().hash, boundary.input().cells))
                    .or_default()
                    .push(entry.cells, index);
                self.paste_outputs
                    .entry((direction, boundary.output().hash, boundary.output().cells))
                    .or_default()
                    .push(entry.cells, index);
            }
        }
        added
    }

    fn add_voxels(
        &mut self,
        entries: &[SnapshotEntry],
        new_voxels: &[usize],
        allowed_directions: &[usize],
    ) -> usize {
        let cylinder_enabled = allowed_directions.binary_search(&0).is_ok();
        let mut added = 0;
        for &index in new_voxels {
            if !self.indexed_voxels.insert(index) {
                continue;
            }
            debug_assert!(entries[index].is_voxel);
            added += 1;
            if cylinder_enabled {
                self.cylinder_inputs
                    .entry(cylinder_boundary_key(&entries[index]))
                    .or_default()
                    .push(index);
            }
        }
        added
    }
}

fn cylinder_boundary_key(entry: &SnapshotEntry) -> CylinderBoundaryKey {
    let boundary = entry.boundary(0);
    (boundary.input().hash, boundary.output().hash)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CylinderJob {
    input: usize,
    output: usize,
}

fn compatible_cylinder_jobs(
    entries: &[SnapshotEntry],
    boundary_index: &BoundaryIndex,
    new_entries: &[usize],
    new_voxels: &[usize],
    allowed_directions: &[usize],
    length_bound: Option<usize>,
    processed: &mut HashSet<CylinderJob>,
) -> Vec<CylinderJob> {
    if allowed_directions.binary_search(&0).is_err() || length_bound.is_some_and(|bound| 1 >= bound)
    {
        return vec![];
    }

    let mut jobs = Vec::new();
    for &input in new_voxels {
        if let Some(outputs) = boundary_index
            .cylinder_outputs
            .get(&cylinder_boundary_key(&entries[input]))
        {
            for &output in outputs {
                push_cylinder_job_if_compatible(entries, input, output, &mut jobs);
            }
        }
    }
    for &output in new_entries {
        if let Some(inputs) = boundary_index
            .cylinder_inputs
            .get(&cylinder_boundary_key(&entries[output]))
        {
            for &input in inputs {
                push_cylinder_job_if_compatible(entries, input, output, &mut jobs);
            }
        }
    }

    jobs.sort_unstable();
    jobs.dedup();
    jobs.retain(|job| processed.insert(*job));
    jobs
}

fn push_cylinder_job_if_compatible(
    entries: &[SnapshotEntry],
    input: usize,
    output: usize,
    jobs: &mut Vec<CylinderJob>,
) {
    let input_entry = &entries[input];
    let output_entry = &entries[output];
    let input_boundary = input_entry.boundary(0);
    let output_boundary = output_entry.boundary(0);
    if cylinder_frames_compatible(input_entry, output_entry)
        && input_boundary
            .input()
            .same_normal_form(output_boundary.input())
        && input_boundary
            .output()
            .same_normal_form(output_boundary.output())
    {
        jobs.push(CylinderJob { input, output });
    }
}

fn cylinder_frames_compatible(input: &SnapshotEntry, output: &SnapshotEntry) -> bool {
    let input_without_zero: IntSet = input
        .frame
        .iter()
        .copied()
        .filter(|&direction| direction != 0)
        .collect();
    intset::is_subset(&input_without_zero, &output.frame)
        && intset::is_subset(&output.frame, &input.frame)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PasteJob {
    direction: usize,
    left: usize,
    right: usize,
}

#[derive(Default)]
struct PasteCandidateBatch {
    jobs: Vec<PasteJob>,
    counts: PasteCandidateCounts,
}

impl PasteCandidateBatch {
    fn merge(mut self, mut other: Self) -> Self {
        self.jobs.append(&mut other.jobs);
        self.counts.merge(other.counts);
        self
    }
}

struct PasteJobSelection {
    jobs: Vec<PasteJob>,
    counts: PasteCandidateCounts,
    scanning_time: Duration,
    sorting_time: Duration,
    deduplication_time: Duration,
    jobs_before_deduplication: usize,
    processed_filtering_time: Duration,
    jobs_before_processed_filtering: usize,
}

impl PasteJobSelection {
    fn timings(&self, round: usize) -> [PolyvoxelEnumerationTiming; 4] {
        let mut scanning = stage_timing_with_results(
            round,
            PolyvoxelEnumerationStage::PasteCandidateScanning,
            self.counts.examined,
            self.counts.accepted,
            self.scanning_time,
        );
        scanning.paste_candidates = Some(self.counts);
        [
            scanning,
            stage_timing(
                round,
                PolyvoxelEnumerationStage::PasteCandidateSorting,
                self.jobs_before_deduplication,
                self.sorting_time,
            ),
            stage_timing_with_results(
                round,
                PolyvoxelEnumerationStage::PasteCandidateDeduplication,
                self.jobs_before_deduplication,
                self.jobs_before_processed_filtering,
                self.deduplication_time,
            ),
            stage_timing_with_results(
                round,
                PolyvoxelEnumerationStage::PasteProcessedFiltering,
                self.jobs_before_processed_filtering,
                self.jobs.len(),
                self.processed_filtering_time,
            ),
        ]
    }
}

fn compatible_paste_jobs(
    entries: &[SnapshotEntry],
    boundary_index: &BoundaryIndex,
    new_entries: &[usize],
    max_cells: usize,
    length_bound: Option<usize>,
    processed: &mut HashSet<PasteJob>,
) -> PasteJobSelection {
    let scanning_started = Instant::now();
    let batch = new_entries
        .par_iter()
        .map(|&new_entry| {
            let mut batch = PasteCandidateBatch::default();
            push_paste_jobs_with_new_left(
                entries,
                boundary_index,
                new_entry,
                max_cells,
                length_bound,
                &mut batch,
            );
            push_paste_jobs_with_new_right(
                entries,
                boundary_index,
                new_entry,
                max_cells,
                length_bound,
                &mut batch,
            );
            batch
        })
        .reduce(PasteCandidateBatch::default, PasteCandidateBatch::merge);
    let scanning_time = scanning_started.elapsed();

    let PasteCandidateBatch { mut jobs, counts } = batch;
    debug_assert_eq!(
        counts.indexed_pairs,
        counts.pruned_by_cell_bound + counts.examined,
    );
    debug_assert_eq!(
        counts.examined,
        counts.rejected_by_length_bound + counts.rejected_by_boundary_normal_form + counts.accepted,
    );
    debug_assert_eq!(counts.accepted, jobs.len());

    let sorting_started = Instant::now();
    jobs.par_sort_unstable_by_key(|job| (job.left, job.right, job.direction));
    let sorting_time = sorting_started.elapsed();

    let jobs_before_deduplication = jobs.len();
    let deduplication_started = Instant::now();
    jobs.dedup();
    let deduplication_time = deduplication_started.elapsed();

    let jobs_before_processed_filtering = jobs.len();
    let processed_filtering_started = Instant::now();
    jobs.retain(|job| processed.insert(*job));
    let processed_filtering_time = processed_filtering_started.elapsed();

    PasteJobSelection {
        jobs,
        counts,
        scanning_time,
        sorting_time,
        deduplication_time,
        jobs_before_deduplication,
        processed_filtering_time,
        jobs_before_processed_filtering,
    }
}

fn push_paste_jobs_with_new_left(
    entries: &[SnapshotEntry],
    boundary_index: &BoundaryIndex,
    left: usize,
    max_cells: usize,
    length_bound: Option<usize>,
    batch: &mut PasteCandidateBatch,
) {
    let left_entry = &entries[left];
    for &direction in &left_entry.frame {
        let left_boundary = left_entry.boundary(direction).output();
        batch.counts.boundary_index_lookups += 1;
        if let Some(rights) =
            boundary_index
                .paste_inputs
                .get(&(direction, left_boundary.hash, left_boundary.cells))
        {
            batch.counts.nonempty_boundary_buckets += 1;
            batch.counts.indexed_pairs += rights.len;
            let max_right_cells = max_cells
                .saturating_add(left_boundary.cells)
                .saturating_sub(left_entry.cells);
            let examined_before = batch.counts.examined;
            for right in rights.up_to(max_right_cells) {
                push_paste_job_if_compatible(
                    entries,
                    direction,
                    left,
                    right,
                    max_cells,
                    length_bound,
                    batch,
                );
            }
            batch.counts.pruned_by_cell_bound +=
                rights.len - (batch.counts.examined - examined_before);
        }
    }
}

fn push_paste_jobs_with_new_right(
    entries: &[SnapshotEntry],
    boundary_index: &BoundaryIndex,
    right: usize,
    max_cells: usize,
    length_bound: Option<usize>,
    batch: &mut PasteCandidateBatch,
) {
    let right_entry = &entries[right];
    for &direction in &right_entry.frame {
        let right_boundary = right_entry.boundary(direction).input();
        batch.counts.boundary_index_lookups += 1;
        if let Some(lefts) = boundary_index.paste_outputs.get(&(
            direction,
            right_boundary.hash,
            right_boundary.cells,
        )) {
            batch.counts.nonempty_boundary_buckets += 1;
            batch.counts.indexed_pairs += lefts.len;
            let max_left_cells = max_cells
                .saturating_add(right_boundary.cells)
                .saturating_sub(right_entry.cells);
            let examined_before = batch.counts.examined;
            for left in lefts.up_to(max_left_cells) {
                push_paste_job_if_compatible(
                    entries,
                    direction,
                    left,
                    right,
                    max_cells,
                    length_bound,
                    batch,
                );
            }
            batch.counts.pruned_by_cell_bound +=
                lefts.len - (batch.counts.examined - examined_before);
        }
    }
}

fn push_paste_job_if_compatible(
    entries: &[SnapshotEntry],
    direction: usize,
    left: usize,
    right: usize,
    max_cells: usize,
    length_bound: Option<usize>,
    batch: &mut PasteCandidateBatch,
) {
    batch.counts.examined += 1;
    let left_entry = &entries[left];
    let right_entry = &entries[right];
    let left_boundary = left_entry.boundary(direction).output();
    let right_boundary = right_entry.boundary(direction).input();
    let result_cells = left_entry
        .cells
        .saturating_add(right_entry.cells)
        .saturating_sub(left_boundary.cells);
    debug_assert!(result_cells <= max_cells);

    let result_length = left_entry
        .shape
        .length_at(direction)
        .saturating_add(right_entry.shape.length_at(direction));
    if length_bound.is_some_and(|bound| result_length >= bound) {
        batch.counts.rejected_by_length_bound += 1;
        return;
    }

    if !left_boundary.same_normal_form(right_boundary) {
        batch.counts.rejected_by_boundary_normal_form += 1;
        return;
    }

    batch.counts.accepted += 1;
    batch.jobs.push(PasteJob {
        direction,
        left,
        right,
    });
}

struct Candidate {
    shape: Polyvoxel,
    is_voxel: bool,
    factorization: PolyvoxelFactorization,
}

struct PreparedCandidate {
    shape: Polyvoxel,
    canonical_shape: Arc<FramedPoset>,
    canonical_into_shape: Embedding,
    is_voxel: bool,
    factorization: PolyvoxelFactorization,
}

fn prepare_candidate(
    candidate: Candidate,
    max_cells: usize,
    allowed_directions: &[usize],
    length_bound: Option<usize>,
) -> Option<PreparedCandidate> {
    if cell_count(&candidate.shape) > max_cells
        || !intset::is_subset(&candidate.shape.active_directions(), allowed_directions)
        || length_bound.is_some_and(|bound| {
            candidate
                .shape
                .length()
                .iter()
                .any(|&length| length >= bound)
        })
    {
        return None;
    }

    let (canonical_shape, canonical_into_shape) =
        normalise_for_enumeration(candidate.shape.as_framed_poset());
    Some(PreparedCandidate {
        shape: candidate.shape,
        canonical_shape,
        canonical_into_shape,
        is_voxel: candidate.is_voxel,
        factorization: candidate.factorization,
    })
}

/// The single switch point for the canonicalisation used by enumeration.
fn normalise_for_enumeration(shape: &Arc<FramedPoset>) -> (Arc<FramedPoset>, Embedding) {
    graph_normalisation(shape)
}

#[derive(Clone)]
struct BoundaryNormalForm {
    hash: u64,
    cells: usize,
    normal: Arc<FramedPoset>,
    normal_into_boundary: Embedding,
}

impl BoundaryNormalForm {
    fn new(shape: Arc<FramedPoset>) -> Self {
        let (normal, normal_into_boundary) = normalise_for_enumeration(&shape);
        Self::from_normalisation(normal, normal_into_boundary)
    }

    fn from_normalisation(normal: Arc<FramedPoset>, normal_into_boundary: Embedding) -> Self {
        debug_assert!(FramedPoset::equal(&normal, &normal_into_boundary.dom));
        let cells = cell_count(&normal);
        let hash = structural_hash(&normal);
        Self {
            hash,
            cells,
            normal,
            normal_into_boundary,
        }
    }

    fn same_normal_form(&self, other: &Self) -> bool {
        self.hash == other.hash && FramedPoset::equal(&self.normal, &other.normal)
    }

    fn isomorphism_to(&self, other: &Self) -> Option<Embedding> {
        self.same_normal_form(other).then(|| {
            Embedding::compose(
                &self.normal_into_boundary.inverse_isomorphism(),
                &other.normal_into_boundary,
            )
        })
    }
}

#[derive(Clone)]
struct DirectionalBoundaryCache {
    direction: usize,
    input: BoundaryNormalForm,
    output: BoundaryNormalForm,
}

impl DirectionalBoundaryCache {
    fn input(&self) -> &BoundaryNormalForm {
        &self.input
    }

    fn output(&self) -> &BoundaryNormalForm {
        &self.output
    }
}

#[derive(Clone)]
struct SnapshotEntry {
    shape: Polyvoxel,
    is_voxel: bool,
    cells: usize,
    frame: IntSet,
    boundaries: Arc<Vec<DirectionalBoundaryCache>>,
}

impl SnapshotEntry {
    fn boundary(&self, direction: usize) -> &DirectionalBoundaryCache {
        let index = self
            .boundaries
            .binary_search_by_key(&direction, |boundary| boundary.direction)
            .expect("every required direction must have a cached boundary");
        &self.boundaries[index]
    }
}

fn structural_hash(shape: &FramedPoset) -> u64 {
    let mut hasher = DefaultHasher::new();
    shape.hash(&mut hasher);
    hasher.finish()
}

fn cell_count(shape: &FramedPoset) -> usize {
    shape.sizes().iter().sum()
}

struct WorkingEntry {
    shape: Polyvoxel,
    canonical_shape: Arc<FramedPoset>,
    canonical_into_shape: Option<Embedding>,
    cells: usize,
    frame: IntSet,
    boundaries: Option<Arc<Vec<DirectionalBoundaryCache>>>,
    is_voxel: bool,
    factorizations: BTreeSet<PolyvoxelFactorization>,
}

struct RecordOutcome {
    index: usize,
    new_entry: bool,
    new_voxel: bool,
}

struct CatalogBuilder {
    max_cells: usize,
    allowed_directions: IntSet,
    length_bound: Option<usize>,
    round: usize,
    entries: Vec<WorkingEntry>,
    indices: HashMap<Arc<FramedPoset>, usize>,
}

impl CatalogBuilder {
    fn new(max_cells: usize, allowed_directions: IntSet, length_bound: Option<usize>) -> Self {
        Self {
            max_cells,
            allowed_directions,
            length_bound,
            round: 1,
            entries: Vec::new(),
            indices: HashMap::new(),
        }
    }

    fn factorization_count(&self) -> usize {
        self.entries
            .iter()
            .map(|entry| entry.factorizations.len())
            .sum()
    }

    fn uncached_entry_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.boundaries.is_none())
            .count()
    }

    fn record(&mut self, candidate: PreparedCandidate) -> RecordOutcome {
        let PreparedCandidate {
            shape,
            canonical_shape,
            canonical_into_shape,
            is_voxel,
            factorization,
        } = candidate;
        if let Some(&index) = self.indices.get(&canonical_shape) {
            let entry = &mut self.entries[index];
            let became_voxel = is_voxel && !entry.is_voxel;
            entry.is_voxel |= is_voxel;
            entry.factorizations.insert(factorization);
            return RecordOutcome {
                index,
                new_entry: false,
                new_voxel: became_voxel,
            };
        }

        debug_assert!(shape.well_formed());
        debug_assert!(is_volumetric(shape.as_framed_poset()));
        let cells = cell_count(&shape);
        let frame = shape.active_directions();
        let index = self.entries.len();
        self.indices.insert(Arc::clone(&canonical_shape), index);
        self.entries.push(WorkingEntry {
            shape,
            canonical_shape,
            canonical_into_shape: Some(canonical_into_shape),
            cells,
            frame,
            boundaries: None,
            is_voxel,
            factorizations: BTreeSet::from([factorization]),
        });
        RecordOutcome {
            index,
            new_entry: true,
            new_voxel: is_voxel,
        }
    }

    fn populate_boundary_caches(&mut self) {
        let cylinder_enabled = self.allowed_directions.binary_search(&0).is_ok();
        self.entries
            .par_iter_mut()
            .filter(|entry| entry.boundaries.is_none())
            .for_each(|entry| {
                let mut directions = entry.frame.clone();
                if cylinder_enabled {
                    intset::insert(&mut directions, 0);
                }
                let canonical_into_shape = entry
                    .canonical_into_shape
                    .take()
                    .expect("an uncached entry must retain its normalization embedding");
                entry.boundaries = Some(Arc::new(
                    directions
                        .iter()
                        .copied()
                        .map(|direction| {
                            if entry.frame.binary_search(&direction).is_err() {
                                debug_assert_eq!(direction, 0);
                                let boundary = BoundaryNormalForm::from_normalisation(
                                    Arc::clone(&entry.canonical_shape),
                                    canonical_into_shape.clone(),
                                );
                                return DirectionalBoundaryCache {
                                    direction,
                                    input: boundary.clone(),
                                    output: boundary,
                                };
                            }

                            let input =
                                boundary(Sign::Input, direction, entry.shape.as_framed_poset()).0;
                            let output =
                                boundary(Sign::Output, direction, entry.shape.as_framed_poset()).0;
                            DirectionalBoundaryCache {
                                direction,
                                input: BoundaryNormalForm::new(input),
                                output: BoundaryNormalForm::new(output),
                            }
                        })
                        .collect(),
                ));
            });
    }

    fn snapshot(&self) -> Vec<SnapshotEntry> {
        self.entries
            .iter()
            .map(|entry| SnapshotEntry {
                shape: entry.shape.clone(),
                is_voxel: entry.is_voxel,
                cells: entry.cells,
                frame: entry.frame.clone(),
                boundaries: Arc::clone(
                    entry
                        .boundaries
                        .as_ref()
                        .expect("boundary caches must be populated before a round"),
                ),
            })
            .collect()
    }

    fn finish(self) -> PolyvoxelCatalog {
        let serialized: Vec<_> = self
            .entries
            .iter()
            .map(|entry| {
                serde_json::to_string(entry.canonical_shape.as_ref())
                    .expect("serializing a framed poset to a string cannot fail")
            })
            .collect();
        let mut order: Vec<_> = (0..self.entries.len()).collect();
        order.sort_unstable_by(|&left, &right| {
            self.entries[left]
                .cells
                .cmp(&self.entries[right].cells)
                .then_with(|| serialized[left].cmp(&serialized[right]))
        });

        let mut old_to_new = vec![0; order.len()];
        for (new, &old) in order.iter().enumerate() {
            old_to_new[old] = new;
        }

        let entries = order
            .into_iter()
            .map(|old| {
                let entry = &self.entries[old];
                let factorizations = entry
                    .factorizations
                    .iter()
                    .cloned()
                    .map(|factorization| factorization.remap(&old_to_new))
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                PolyvoxelEntry {
                    shape: entry.shape.clone(),
                    is_voxel: entry.is_voxel,
                    factorizations,
                }
            })
            .collect();

        PolyvoxelCatalog { entries }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poset::{polyvoxel_layering_direction, polyvoxel_length};

    #[test]
    fn directions_zero_through_two_give_three_tight_arrows() {
        let catalog = enumerate_polyvoxels(3, &[0, 1, 2]);

        assert_eq!(catalog.len(), 4);
        assert_eq!(
            catalog
                .entries()
                .iter()
                .map(|entry| entry.shape.active_directions())
                .collect::<Vec<_>>(),
            vec![vec![], vec![0], vec![1], vec![2]],
        );
        assert!(catalog.entries().iter().all(|entry| entry.is_voxel));
    }

    #[test]
    fn every_recorded_factorization_reconstructs_its_result() {
        let catalog = enumerate_polyvoxels(5, &[0, 1, 2]);

        for (result, entry) in catalog.entries().iter().enumerate() {
            assert_eq!(
                entry.shape.length(),
                polyvoxel_length(entry.shape.as_framed_poset())
            );
            assert_eq!(
                entry.shape.layering_direction(),
                polyvoxel_layering_direction(entry.shape.as_framed_poset())
            );
            assert!(!entry.factorizations.is_empty());
            for factorization in &entry.factorizations {
                let reconstructed = reconstruct(&catalog, factorization);
                assert_eq!(
                    normalize(reconstructed.as_ref()),
                    normalize(entry.shape.as_ref()),
                    "factorization {factorization:?} did not reconstruct entry {result}",
                );
            }
        }
    }

    #[test]
    fn progress_reports_each_phase_and_the_final_fixed_point() {
        let mut progress = Vec::new();
        let catalog = enumerate_polyvoxels_with_progress(3, &[0, 1, 2], |event| {
            progress.push(event);
        });

        for phase in [
            PolyvoxelEnumerationPhase::Shift,
            PolyvoxelEnumerationPhase::Cylinder,
            PolyvoxelEnumerationPhase::Paste,
        ] {
            assert!(progress.iter().any(|event| event.phase == phase));
        }

        let complete = progress.last().unwrap();
        assert_eq!(complete.phase, PolyvoxelEnumerationPhase::Complete);
        assert_eq!(complete.representatives, catalog.len());
        assert_eq!(
            complete.factorizations,
            catalog
                .entries()
                .iter()
                .map(|entry| entry.factorizations.len())
                .sum::<usize>(),
        );
    }

    #[test]
    fn profiling_reports_every_enumeration_stage() {
        let mut timings = Vec::new();
        enumerate_polyvoxels_profiled(
            3,
            &[0, 1, 2],
            Some(4),
            |_| {},
            |timing| timings.push(timing),
        );

        for stage in [
            PolyvoxelEnumerationStage::BoundaryIndexing,
            PolyvoxelEnumerationStage::Shift,
            PolyvoxelEnumerationStage::CylinderMatching,
            PolyvoxelEnumerationStage::Cylinder,
            PolyvoxelEnumerationStage::PasteCandidateScanning,
            PolyvoxelEnumerationStage::PasteCandidateSorting,
            PolyvoxelEnumerationStage::PasteCandidateDeduplication,
            PolyvoxelEnumerationStage::PasteProcessedFiltering,
            PolyvoxelEnumerationStage::Paste,
            PolyvoxelEnumerationStage::BoundaryCaching,
        ] {
            assert!(timings.iter().any(|timing| timing.stage == stage));
        }
        assert!(timings.iter().any(|timing| {
            timing.round == 0 && timing.stage == PolyvoxelEnumerationStage::BoundaryCaching
        }));
        for timing in timings
            .iter()
            .filter(|timing| timing.stage == PolyvoxelEnumerationStage::PasteCandidateScanning)
        {
            let counts = timing
                .paste_candidates
                .expect("paste candidate scans must report rejection counts");
            assert_eq!(timing.jobs, counts.examined);
            assert_eq!(timing.results, counts.accepted);
            assert!(counts.nonempty_boundary_buckets <= counts.boundary_index_lookups);
            assert_eq!(
                counts.indexed_pairs,
                counts.pruned_by_cell_bound + counts.examined,
            );
            assert_eq!(
                counts.examined,
                counts.rejected_by_length_bound
                    + counts.rejected_by_boundary_normal_form
                    + counts.accepted,
            );
        }
    }

    #[test]
    fn constructor_jobs_are_processed_only_once() {
        let mut timings = Vec::new();
        let catalog = enumerate_polyvoxels_profiled(
            5,
            &[0, 1, 2],
            Some(4),
            |_| {},
            |timing| timings.push(timing),
        );

        let shift_jobs = timings
            .iter()
            .filter(|timing| timing.stage == PolyvoxelEnumerationStage::Shift)
            .map(|timing| timing.jobs)
            .sum::<usize>();
        assert_eq!(
            shift_jobs,
            catalog
                .entries()
                .iter()
                .filter(|entry| entry.is_voxel)
                .count(),
        );

        let paste_jobs = timings
            .iter()
            .filter(|timing| timing.stage == PolyvoxelEnumerationStage::Paste)
            .map(|timing| timing.jobs)
            .sum::<usize>();
        let paste_factorizations = catalog
            .entries()
            .iter()
            .flat_map(|entry| &entry.factorizations)
            .filter(|factorization| matches!(factorization, PolyvoxelFactorization::Paste { .. }))
            .count();
        assert_eq!(paste_jobs, paste_factorizations);
    }

    #[test]
    fn length_bound_is_exclusive() {
        let bounded = enumerate_polyvoxels_with_length_bound(9, &[0], Some(4));
        assert!(
            bounded
                .entries()
                .iter()
                .all(|entry| { entry.shape.length().iter().all(|&length| length < 4) })
        );
        assert!(
            bounded
                .entries()
                .iter()
                .any(|entry| entry.shape.length_at(0) == 3)
        );

        let unbounded = enumerate_polyvoxels_with_length_bound(9, &[0], None);
        assert!(
            unbounded
                .entries()
                .iter()
                .any(|entry| entry.shape.length_at(0) == 4)
        );
    }

    #[test]
    fn boundary_indices_find_exactly_the_pairwise_compatible_jobs() {
        let max_cells = 5;
        let directions = [0, 1, 2];
        let catalog = enumerate_polyvoxels(max_cells, &directions);
        let entries: Vec<_> = catalog
            .entries()
            .iter()
            .map(|entry| SnapshotEntry {
                shape: entry.shape.clone(),
                is_voxel: entry.is_voxel,
                cells: cell_count(&entry.shape),
                frame: entry.shape.active_directions(),
                boundaries: Arc::new(
                    directions
                        .iter()
                        .copied()
                        .map(|direction| DirectionalBoundaryCache {
                            direction,
                            input: BoundaryNormalForm::new(
                                boundary(Sign::Input, direction, entry.shape.as_framed_poset()).0,
                            ),
                            output: BoundaryNormalForm::new(
                                boundary(Sign::Output, direction, entry.shape.as_framed_poset()).0,
                            ),
                        })
                        .collect(),
                ),
            })
            .collect();
        let all_entries: Vec<_> = (0..entries.len()).collect();
        let voxels: Vec<_> = entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| entry.is_voxel.then_some(index))
            .collect();
        let mut boundary_index = BoundaryIndex::default();
        boundary_index.add_entries(&entries, &all_entries, &directions);
        boundary_index.add_voxels(&entries, &voxels, &directions);

        let indexed_cylinders = compatible_cylinder_jobs(
            &entries,
            &boundary_index,
            &all_entries,
            &voxels,
            &directions,
            None,
            &mut HashSet::new(),
        );
        let mut pairwise_cylinders = Vec::new();
        for &input in &voxels {
            for output in 0..entries.len() {
                if cylinder_frames_compatible(&entries[input], &entries[output])
                    && [Sign::Input, Sign::Output].into_iter().all(|sign| {
                        let input_boundary =
                            boundary(sign, 0, entries[input].shape.as_framed_poset()).0;
                        let output_boundary =
                            boundary(sign, 0, entries[output].shape.as_framed_poset()).0;
                        isomorphic(&input_boundary, &output_boundary)
                    })
                {
                    pairwise_cylinders.push(CylinderJob { input, output });
                }
            }
        }
        assert_eq!(indexed_cylinders, pairwise_cylinders);

        let indexed_pastes = compatible_paste_jobs(
            &entries,
            &boundary_index,
            &all_entries,
            max_cells,
            None,
            &mut HashSet::new(),
        );
        let mut pairwise_pastes = Vec::new();
        for (left, left_entry) in entries.iter().enumerate() {
            for (right, right_entry) in entries.iter().enumerate() {
                for &direction in &left_entry.frame {
                    if right_entry.frame.binary_search(&direction).is_err() {
                        continue;
                    }
                    let left_boundary =
                        boundary(Sign::Output, direction, left_entry.shape.as_framed_poset()).0;
                    let right_boundary =
                        boundary(Sign::Input, direction, right_entry.shape.as_framed_poset()).0;
                    let result_cells = left_entry
                        .cells
                        .saturating_add(right_entry.cells)
                        .saturating_sub(cell_count(&left_boundary));
                    if result_cells <= max_cells && isomorphic(&left_boundary, &right_boundary) {
                        pairwise_pastes.push(PasteJob {
                            direction,
                            left,
                            right,
                        });
                    }
                }
            }
        }
        pairwise_pastes.sort_unstable_by_key(|job| (job.left, job.right, job.direction));
        assert_eq!(indexed_pastes.jobs, pairwise_pastes);

        let mut incremental_index = BoundaryIndex::default();
        let mut processed_cylinders = HashSet::new();
        let mut processed_pastes = HashSet::new();
        let mut incremental_cylinders = Vec::new();
        let mut incremental_pastes = Vec::new();
        for batch in all_entries.chunks(2) {
            let new_voxels: Vec<_> = batch
                .iter()
                .copied()
                .filter(|&index| entries[index].is_voxel)
                .collect();
            incremental_index.add_entries(&entries, batch, &directions);
            incremental_index.add_voxels(&entries, &new_voxels, &directions);
            incremental_cylinders.extend(compatible_cylinder_jobs(
                &entries,
                &incremental_index,
                batch,
                &new_voxels,
                &directions,
                None,
                &mut processed_cylinders,
            ));
            incremental_pastes.extend(
                compatible_paste_jobs(
                    &entries,
                    &incremental_index,
                    batch,
                    max_cells,
                    None,
                    &mut processed_pastes,
                )
                .jobs,
            );
        }
        incremental_cylinders.sort_unstable();
        incremental_pastes.sort_unstable_by_key(|job| (job.left, job.right, job.direction));
        assert_eq!(incremental_cylinders, pairwise_cylinders);
        assert_eq!(incremental_pastes, pairwise_pastes);
        assert_eq!(processed_cylinders.len(), pairwise_cylinders.len());
        assert_eq!(processed_pastes.len(), pairwise_pastes.len());

        let mut promotion_index = BoundaryIndex::default();
        promotion_index.add_entries(&entries, &all_entries, &directions);
        let mut promotion_cylinders = Vec::new();
        let mut processed_promotions = HashSet::new();
        for batch in voxels.chunks(2) {
            promotion_index.add_voxels(&entries, batch, &directions);
            promotion_cylinders.extend(compatible_cylinder_jobs(
                &entries,
                &promotion_index,
                &[],
                batch,
                &directions,
                None,
                &mut processed_promotions,
            ));
        }
        promotion_cylinders.sort_unstable();
        assert_eq!(promotion_cylinders, pairwise_cylinders);
    }

    fn reconstruct(
        catalog: &PolyvoxelCatalog,
        factorization: &PolyvoxelFactorization,
    ) -> Polyvoxel {
        match factorization {
            PolyvoxelFactorization::Point => point(),
            PolyvoxelFactorization::Shift { source } => shift(&catalog.entry(*source).shape),
            PolyvoxelFactorization::Cylinder { input, output } => {
                cylinder(&catalog.entry(*input).shape, &catalog.entry(*output).shape)
            }
            PolyvoxelFactorization::Paste {
                direction,
                left,
                right,
                boundary_isomorphism,
            } => {
                let left = &catalog.entry(*left).shape;
                let right = &catalog.entry(*right).shape;
                let (left_boundary, _) = boundary(Sign::Output, *direction, left.as_framed_poset());
                let (right_boundary, _) =
                    boundary(Sign::Input, *direction, right.as_framed_poset());
                isomorphisms(&left_boundary, &right_boundary)
                    .into_iter()
                    .find(|isomorphism| &isomorphism.map == boundary_isomorphism)
                    .expect("recorded boundary isomorphism must still exist");
                paste(left, right, *direction).1
            }
        }
    }
}
