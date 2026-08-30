//! Small exhaustive enumerations of bounded polyvoxels.
//!
//! This deliberately favors a direct implementation of the inductive
//! definition over the more elaborate indexing used by alifib's globular
//! enumerator. Factorizations form a packed graph: they refer to operand
//! polyvoxels, whose own factorizations represent all recursive choices.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::box_construction::elementary_cylinder;
use crate::embedding::Embedding;
use crate::intset::{self, IntSet};
use crate::isomorphism::{isomorphic, isomorphisms, normalize};
use crate::poset::{FramedPoset, Sign, boundary, shift};
use crate::pushout::pushout;
use crate::volumetric::is_volumetric;

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
    pub shape: Arc<FramedPoset>,
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

/// Enumerate polyvoxels up to isomorphism within finite cell and direction
/// bounds.
///
/// `allowed_directions` must be sorted and deduplicated. The enumeration is
/// closed under point, shift, elementary cylinder, and every directional
/// pasting along every boundary isomorphism that remains inside the bounds.
/// Repeated immediate constructions are retained as distinct factorizations.
pub fn enumerate_polyvoxels(max_cells: usize, allowed_directions: &[usize]) -> PolyvoxelCatalog {
    enumerate_polyvoxels_with_progress(max_cells, allowed_directions, |_| {})
}

/// Enumerate bounded polyvoxels while reporting phase milestones.
///
/// At most about ten milestones are emitted per phase and fixed-point round,
/// in addition to phase starts and the final completion event.
pub fn enumerate_polyvoxels_with_progress(
    max_cells: usize,
    allowed_directions: &[usize],
    mut report: impl FnMut(PolyvoxelEnumerationProgress),
) -> PolyvoxelCatalog {
    assert!(
        intset::is_sorted_unique(allowed_directions),
        "allowed directions must be sorted and deduplicated",
    );

    let mut builder = CatalogBuilder::new(max_cells, allowed_directions.to_vec());
    builder.record(
        Arc::new(FramedPoset::point()),
        true,
        PolyvoxelFactorization::Point,
    );

    loop {
        let shapes: Vec<_> = builder
            .entries
            .iter()
            .map(|entry| Arc::clone(&entry.shape))
            .collect();
        let voxels: Vec<_> = builder
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| entry.is_voxel.then_some(index))
            .collect();
        let mut changed = false;
        let round = builder.round;

        report_milestone(
            &builder,
            &mut report,
            round,
            PolyvoxelEnumerationPhase::Shift,
            0,
            voxels.len(),
        );
        for (job, &source) in voxels.iter().enumerate() {
            changed |= builder.record(
                Arc::new(shift(&shapes[source])),
                true,
                PolyvoxelFactorization::Shift { source },
            );
            report_milestone(
                &builder,
                &mut report,
                round,
                PolyvoxelEnumerationPhase::Shift,
                job + 1,
                voxels.len(),
            );
        }

        let cylinder_jobs = voxels.len().saturating_mul(shapes.len());
        let mut cylinder_job = 0;
        report_milestone(
            &builder,
            &mut report,
            round,
            PolyvoxelEnumerationPhase::Cylinder,
            cylinder_job,
            cylinder_jobs,
        );
        for &input in &voxels {
            for output in 0..shapes.len() {
                if cylinder_is_defined(&shapes[input], &shapes[output]) {
                    let cylinder = elementary_cylinder(&shapes[input], &shapes[output]);
                    changed |= builder.record(
                        cylinder,
                        true,
                        PolyvoxelFactorization::Cylinder { input, output },
                    );
                }

                cylinder_job += 1;
                report_milestone(
                    &builder,
                    &mut report,
                    round,
                    PolyvoxelEnumerationPhase::Cylinder,
                    cylinder_job,
                    cylinder_jobs,
                );
            }
        }

        let paste_jobs = shapes.len().saturating_mul(shapes.len());
        let mut paste_job = 0;
        report_milestone(
            &builder,
            &mut report,
            round,
            PolyvoxelEnumerationPhase::Paste,
            paste_job,
            paste_jobs,
        );
        for left in 0..shapes.len() {
            for right in 0..shapes.len() {
                let left_frame = shapes[left].active_directions();
                let right_frame = shapes[right].active_directions();
                let common_directions = left_frame
                    .iter()
                    .copied()
                    .filter(|direction| right_frame.binary_search(direction).is_ok());

                for direction in common_directions {
                    let (left_boundary, into_left) =
                        boundary(Sign::Output, direction, &shapes[left]);
                    let (right_boundary, into_right) =
                        boundary(Sign::Input, direction, &shapes[right]);

                    let result_cells = cell_count(&shapes[left])
                        .saturating_add(cell_count(&shapes[right]))
                        .saturating_sub(cell_count(&left_boundary));
                    if result_cells > max_cells {
                        continue;
                    }

                    let boundary_isomorphisms = isomorphisms(&left_boundary, &right_boundary);
                    debug_assert!(
                        boundary_isomorphisms.len() <= 1,
                        "polyvoxel boundaries should have at most one isomorphism",
                    );
                    for isomorphism in boundary_isomorphisms {
                        let into_right = Embedding::compose(&isomorphism, &into_right);
                        let pasted = pushout(&into_left, &into_right);
                        changed |= builder.record(
                            pasted.tip,
                            false,
                            PolyvoxelFactorization::Paste {
                                direction,
                                left,
                                right,
                                boundary_isomorphism: isomorphism.map,
                            },
                        );
                    }
                }

                paste_job += 1;
                report_milestone(
                    &builder,
                    &mut report,
                    round,
                    PolyvoxelEnumerationPhase::Paste,
                    paste_job,
                    paste_jobs,
                );
            }
        }

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

fn cylinder_is_defined(input: &Arc<FramedPoset>, output: &Arc<FramedPoset>) -> bool {
    let input_frame = input.active_directions();
    let output_frame = output.active_directions();
    let input_without_zero: IntSet = input_frame
        .iter()
        .copied()
        .filter(|&direction| direction != 0)
        .collect();

    intset::is_subset(&input_without_zero, &output_frame)
        && intset::is_subset(&output_frame, &input_frame)
        && [Sign::Input, Sign::Output].into_iter().all(|sign| {
            let (input_boundary, _) = boundary(sign, 0, input);
            let (output_boundary, _) = boundary(sign, 0, output);
            isomorphic(&input_boundary, &output_boundary)
        })
}

fn cell_count(shape: &FramedPoset) -> usize {
    shape.sizes().iter().sum()
}

struct WorkingEntry {
    shape: Arc<FramedPoset>,
    is_voxel: bool,
    factorizations: BTreeSet<PolyvoxelFactorization>,
}

struct CatalogBuilder {
    max_cells: usize,
    allowed_directions: IntSet,
    round: usize,
    entries: Vec<WorkingEntry>,
    indices: HashMap<Arc<FramedPoset>, usize>,
}

impl CatalogBuilder {
    fn new(max_cells: usize, allowed_directions: IntSet) -> Self {
        Self {
            max_cells,
            allowed_directions,
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

    fn record(
        &mut self,
        shape: Arc<FramedPoset>,
        is_voxel: bool,
        factorization: PolyvoxelFactorization,
    ) -> bool {
        if cell_count(&shape) > self.max_cells
            || !intset::is_subset(&shape.active_directions(), &self.allowed_directions)
        {
            return false;
        }

        let shape = Arc::new(normalize(&shape));
        if let Some(&index) = self.indices.get(&shape) {
            let entry = &mut self.entries[index];
            let became_voxel = is_voxel && !entry.is_voxel;
            entry.is_voxel |= is_voxel;
            return became_voxel || entry.factorizations.insert(factorization);
        }

        debug_assert!(shape.well_formed());
        debug_assert!(is_volumetric(&shape));
        let index = self.entries.len();
        self.indices.insert(Arc::clone(&shape), index);
        self.entries.push(WorkingEntry {
            shape,
            is_voxel,
            factorizations: BTreeSet::from([factorization]),
        });
        true
    }

    fn finish(self) -> PolyvoxelCatalog {
        let serialized: Vec<_> = self
            .entries
            .iter()
            .map(|entry| {
                serde_json::to_string(entry.shape.as_ref())
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
                    shape: Arc::clone(&entry.shape),
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
            assert!(!entry.factorizations.is_empty());
            for factorization in &entry.factorizations {
                let reconstructed = reconstruct(&catalog, factorization);
                assert_eq!(
                    normalize(&reconstructed),
                    *entry.shape,
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

    fn reconstruct(
        catalog: &PolyvoxelCatalog,
        factorization: &PolyvoxelFactorization,
    ) -> Arc<FramedPoset> {
        match factorization {
            PolyvoxelFactorization::Point => Arc::new(FramedPoset::point()),
            PolyvoxelFactorization::Shift { source } => {
                Arc::new(shift(&catalog.entry(*source).shape))
            }
            PolyvoxelFactorization::Cylinder { input, output } => {
                elementary_cylinder(&catalog.entry(*input).shape, &catalog.entry(*output).shape)
            }
            PolyvoxelFactorization::Paste {
                direction,
                left,
                right,
                boundary_isomorphism,
            } => {
                let left = &catalog.entry(*left).shape;
                let right = &catalog.entry(*right).shape;
                let (left_boundary, into_left) = boundary(Sign::Output, *direction, left);
                let (right_boundary, into_right) = boundary(Sign::Input, *direction, right);
                let isomorphism = isomorphisms(&left_boundary, &right_boundary)
                    .into_iter()
                    .find(|isomorphism| &isomorphism.map == boundary_isomorphism)
                    .expect("recorded boundary isomorphism must still exist");
                let into_right = Embedding::compose(&isomorphism, &into_right);
                pushout(&into_left, &into_right).tip
            }
        }
    }
}
