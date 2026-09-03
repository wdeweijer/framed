//! Small exhaustive enumerations of bounded polyvoxels.
//!
//! This keeps the inductive construction simple, while indexing canonical
//! boundary forms so that cylinder and paste candidates need not be found by
//! scanning every pair. Factorizations form a packed graph: they refer to
//! operand polyvoxels, whose own factorizations represent all recursive
//! choices.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, HashMap};
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
    Shift,
    CylinderMatching,
    Cylinder,
    PasteMatching,
    Paste,
    BoundaryCaching,
}

/// Basic performance measurements for one enumeration stage.
///
/// `construction_work` and `canonicalisation_work` are sums over Rayon worker
/// tasks and can therefore exceed `wall_time`. `merge_time` is sequential.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolyvoxelEnumerationTiming {
    pub round: usize,
    pub stage: PolyvoxelEnumerationStage,
    pub jobs: usize,
    pub wall_time: Duration,
    pub construction_work: Duration,
    pub canonicalisation_work: Duration,
    pub merge_time: Duration,
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
    builder.record(initial);
    let cache_started = Instant::now();
    let cached_boundaries = builder.uncached_entry_count();
    builder.populate_boundary_caches();
    report_timing(stage_timing(
        0,
        PolyvoxelEnumerationStage::BoundaryCaching,
        cached_boundaries,
        cache_started.elapsed(),
    ));

    loop {
        let entries = builder.snapshot();
        let voxels: Vec<_> = entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| entry.is_voxel.then_some(index))
            .collect();
        let mut changed = false;
        let round = builder.round;

        let shift = process_jobs(
            &mut builder,
            &voxels,
            round,
            PolyvoxelEnumerationPhase::Shift,
            &mut report,
            |&source| Candidate {
                shape: shift(&entries[source].shape),
                is_voxel: true,
                factorization: PolyvoxelFactorization::Shift { source },
            },
        );
        changed |= shift.changed;
        report_timing(shift.timing(round, PolyvoxelEnumerationStage::Shift, voxels.len()));

        let matching_started = Instant::now();
        let cylinder_jobs =
            compatible_cylinder_jobs(&entries, &voxels, allowed_directions, length_bound);
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
        changed |= cylinder.changed;
        report_timing(cylinder.timing(
            round,
            PolyvoxelEnumerationStage::Cylinder,
            cylinder_jobs.len(),
        ));

        let matching_started = Instant::now();
        let paste_jobs = compatible_paste_jobs(&entries, max_cells, length_bound);
        report_timing(stage_timing(
            round,
            PolyvoxelEnumerationStage::PasteMatching,
            paste_jobs.len(),
            matching_started.elapsed(),
        ));
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
        changed |= paste.changed;
        report_timing(paste.timing(round, PolyvoxelEnumerationStage::Paste, paste_jobs.len()));

        let cache_started = Instant::now();
        let cached_boundaries = builder.uncached_entry_count();
        builder.populate_boundary_caches();
        report_timing(stage_timing(
            round,
            PolyvoxelEnumerationStage::BoundaryCaching,
            cached_boundaries,
            cache_started.elapsed(),
        ));

        if !changed {
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
    PolyvoxelEnumerationTiming {
        round,
        stage,
        jobs,
        wall_time,
        construction_work: Duration::ZERO,
        canonicalisation_work: Duration::ZERO,
        merge_time: Duration::ZERO,
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
    let mut changed = false;
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
                changed |= builder.record(candidate);
            }
        }
        merge_time += merge_started.elapsed();

        let completed = ((chunk_index + 1) * chunk_size).min(jobs.len());
        report_milestone(builder, report, round, phase, completed, jobs.len());
    }

    JobProcessing {
        changed,
        wall_time: wall_started.elapsed(),
        construction_work,
        canonicalisation_work,
        merge_time,
    }
}

struct JobProcessing {
    changed: bool,
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
            wall_time: self.wall_time,
            construction_work: self.construction_work,
            canonicalisation_work: self.canonicalisation_work,
            merge_time: self.merge_time,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CylinderJob {
    input: usize,
    output: usize,
}

fn compatible_cylinder_jobs(
    entries: &[SnapshotEntry],
    voxels: &[usize],
    allowed_directions: &[usize],
    length_bound: Option<usize>,
) -> Vec<CylinderJob> {
    if allowed_directions.binary_search(&0).is_err() || length_bound.is_some_and(|bound| 1 >= bound)
    {
        return vec![];
    }

    let mut outputs_by_hash = HashMap::<(u64, u64), Vec<usize>>::new();
    for (output, entry) in entries.iter().enumerate() {
        let boundary = entry.boundary(0);
        outputs_by_hash
            .entry((boundary.input().hash, boundary.output().hash))
            .or_default()
            .push(output);
    }

    let mut jobs = Vec::new();
    for &input in voxels {
        let input_entry = &entries[input];
        let boundary = input_entry.boundary(0);
        let key = (boundary.input().hash, boundary.output().hash);
        let Some(outputs) = outputs_by_hash.get(&key) else {
            continue;
        };

        for &output in outputs {
            let output_entry = &entries[output];
            let output_boundary = output_entry.boundary(0);
            if cylinder_frames_compatible(input_entry, output_entry)
                && boundary.input().same_normal_form(output_boundary.input())
                && boundary.output().same_normal_form(output_boundary.output())
            {
                jobs.push(CylinderJob { input, output });
            }
        }
    }

    jobs
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PasteJob {
    direction: usize,
    left: usize,
    right: usize,
}

fn compatible_paste_jobs(
    entries: &[SnapshotEntry],
    max_cells: usize,
    length_bound: Option<usize>,
) -> Vec<PasteJob> {
    let mut inputs_by_hash = HashMap::<(usize, u64), Vec<usize>>::new();
    for (right, entry) in entries.iter().enumerate() {
        for &direction in &entry.frame {
            inputs_by_hash
                .entry((direction, entry.boundary(direction).input().hash))
                .or_default()
                .push(right);
        }
    }

    let mut jobs = Vec::new();
    for (left, left_entry) in entries.iter().enumerate() {
        for &direction in &left_entry.frame {
            let left_boundary = left_entry.boundary(direction).output();
            let Some(rights) = inputs_by_hash.get(&(direction, left_boundary.hash)) else {
                continue;
            };

            for &right in rights {
                let right_boundary = entries[right].boundary(direction).input();
                let result_length = left_entry
                    .shape
                    .length_at(direction)
                    .saturating_add(entries[right].shape.length_at(direction));
                let result_cells = left_entry
                    .cells
                    .saturating_add(entries[right].cells)
                    .saturating_sub(left_boundary.cells);
                if result_cells <= max_cells
                    && length_bound.is_none_or(|bound| result_length < bound)
                    && left_boundary.same_normal_form(right_boundary)
                {
                    jobs.push(PasteJob {
                        direction,
                        left,
                        right,
                    });
                }
            }
        }
    }
    jobs.sort_unstable_by_key(|job| (job.left, job.right, job.direction));
    jobs
}

struct Candidate {
    shape: Polyvoxel,
    is_voxel: bool,
    factorization: PolyvoxelFactorization,
}

struct PreparedCandidate {
    shape: Polyvoxel,
    canonical_shape: Arc<FramedPoset>,
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

    let canonical_shape = normalise_for_enumeration(candidate.shape.as_framed_poset()).0;
    Some(PreparedCandidate {
        shape: candidate.shape,
        canonical_shape,
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
        let cells = cell_count(&shape);
        let (normal, normal_into_boundary) = normalise_for_enumeration(&shape);
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
    boundaries: Vec<DirectionalBoundaryCache>,
}

impl SnapshotEntry {
    fn boundary(&self, direction: usize) -> &DirectionalBoundaryCache {
        let index = self
            .boundaries
            .binary_search_by_key(&direction, |boundary| boundary.direction)
            .expect("every allowed direction must have a cached boundary");
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
    boundaries: Option<Vec<DirectionalBoundaryCache>>,
    is_voxel: bool,
    factorizations: BTreeSet<PolyvoxelFactorization>,
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

    fn record(&mut self, candidate: PreparedCandidate) -> bool {
        let PreparedCandidate {
            shape,
            canonical_shape,
            is_voxel,
            factorization,
        } = candidate;
        if let Some(&index) = self.indices.get(&canonical_shape) {
            let entry = &mut self.entries[index];
            let became_voxel = is_voxel && !entry.is_voxel;
            entry.is_voxel |= is_voxel;
            return became_voxel || entry.factorizations.insert(factorization);
        }

        debug_assert!(shape.well_formed());
        debug_assert!(is_volumetric(shape.as_framed_poset()));
        let index = self.entries.len();
        self.indices.insert(Arc::clone(&canonical_shape), index);
        self.entries.push(WorkingEntry {
            shape,
            canonical_shape,
            boundaries: None,
            is_voxel,
            factorizations: BTreeSet::from([factorization]),
        });
        true
    }

    fn populate_boundary_caches(&mut self) {
        let directions = self.allowed_directions.clone();
        self.entries
            .par_iter_mut()
            .filter(|entry| entry.boundaries.is_none())
            .for_each(|entry| {
                entry.boundaries = Some(
                    directions
                        .iter()
                        .copied()
                        .map(|direction| {
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
                );
            });
    }

    fn snapshot(&self) -> Vec<SnapshotEntry> {
        self.entries
            .iter()
            .map(|entry| SnapshotEntry {
                shape: entry.shape.clone(),
                is_voxel: entry.is_voxel,
                cells: cell_count(&entry.shape),
                frame: entry.shape.active_directions(),
                boundaries: entry
                    .boundaries
                    .as_ref()
                    .expect("boundary caches must be populated before a round")
                    .clone(),
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
            cell_count(&self.entries[left].shape)
                .cmp(&cell_count(&self.entries[right].shape))
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
            PolyvoxelEnumerationStage::Shift,
            PolyvoxelEnumerationStage::CylinderMatching,
            PolyvoxelEnumerationStage::Cylinder,
            PolyvoxelEnumerationStage::PasteMatching,
            PolyvoxelEnumerationStage::Paste,
            PolyvoxelEnumerationStage::BoundaryCaching,
        ] {
            assert!(timings.iter().any(|timing| timing.stage == stage));
        }
        assert!(timings.iter().any(|timing| {
            timing.round == 0 && timing.stage == PolyvoxelEnumerationStage::BoundaryCaching
        }));
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
                boundaries: directions
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
            })
            .collect();
        let voxels: Vec<_> = entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| entry.is_voxel.then_some(index))
            .collect();

        let indexed_cylinders = compatible_cylinder_jobs(&entries, &voxels, &directions, None);
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

        let indexed_pastes = compatible_paste_jobs(&entries, max_cells, None);
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
        assert_eq!(indexed_pastes, pairwise_pastes);
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
