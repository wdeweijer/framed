//! Intrinsic input-first traversals of polyvoxels.
//!
//! This is an experimental adaptation of the traversal of directed molecules.
//! A polyvoxel has one input and one output boundary for every direction in
//! its frame, so those boundaries are visited in increasing direction order.
//! If the polyvoxel is not a voxel, its full-frame maximal cells are then
//! reached in the greatest frame direction from the earliest already visited
//! input face. No existing cell index is used to break a tie: a failure of
//! existence or uniqueness is reported as a [`TraversalError`].

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::embedding::Embedding;
use crate::intset;
use crate::polyvoxel::Polyvoxel;
use crate::poset::{Element, FramedPoset, FramedPosetSubset, Sign, boundary, closure};

/// A failed invariant of the proposed polyvoxel traversal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraversalError {
    /// Taking an active directional boundary failed to reduce the shape.
    BoundaryDidNotDecrease {
        sign: Sign,
        direction: usize,
        cells: usize,
    },
    /// Taking the closure of a maximal cell failed to reduce a non-voxel.
    CellClosureDidNotDecrease { cell: Element, cells: usize },
    /// More than one unvisited maximal cell continues from the same face.
    AmbiguousContinuation {
        direction: usize,
        face: Element,
        cells: Vec<Element>,
    },
    /// Some maximal cells cannot be reached by the directional walk.
    MissingContinuation {
        direction: usize,
        cells: Vec<Element>,
    },
    /// The traversal rules terminated without visiting every cell.
    Incomplete { cells: Vec<Element> },
}

impl fmt::Display for TraversalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BoundaryDidNotDecrease {
                sign,
                direction,
                cells,
            } => write!(
                f,
                "the {sign:?} boundary in direction {direction} did not reduce a {cells}-cell shape"
            ),
            Self::CellClosureDidNotDecrease { cell, cells } => write!(
                f,
                "the closure of cell ({}, {}) did not reduce a {cells}-cell non-voxel",
                cell.dim, cell.pos
            ),
            Self::AmbiguousContinuation {
                direction,
                face,
                cells,
            } => write!(
                f,
                "face ({}, {}) has {} unvisited input cofaces in traversal direction {direction}",
                face.dim,
                face.pos,
                cells.len()
            ),
            Self::MissingContinuation { direction, cells } => write!(
                f,
                "could not reach {} full-frame maximal cells in traversal direction {direction}",
                cells.len()
            ),
            Self::Incomplete { cells } => {
                write!(f, "the traversal left {} cells unvisited", cells.len())
            }
        }
    }
}

impl Error for TraversalError {}

/// Traverse a polyvoxel in intrinsic input-first order.
///
/// The returned elements are indices in the original polyvoxel. Every cell
/// occurs exactly once. Directions and cells within recursively visited
/// boundaries are ordered intrinsically; old cell indices are never used as a
/// tie-breaker.
pub fn traversal_order(polyvoxel: &Polyvoxel) -> Result<Vec<Element>, TraversalError> {
    traverse_shape(polyvoxel.as_framed_poset())
}

/// Relabel a polyvoxel by its intrinsic traversal.
///
/// Cells at each basis cardinality retain their relative traversal order. The
/// returned embedding is an isomorphism from the relabelled shape to the
/// original shape. The relabelled OFP is deliberately not marked as the
/// graph-isomorphism normal form used by [`crate::isomorphism::normalize`].
pub fn traversal_normalisation(
    polyvoxel: &Polyvoxel,
) -> Result<(Arc<FramedPoset>, Embedding), TraversalError> {
    let shape = polyvoxel.as_framed_poset();
    traversal_normalisation_of_shape(shape)
}

/// Relabel an OFP by the traversal used for polyvoxels.
///
/// This is the unchecked shape-level counterpart of
/// [`traversal_normalisation`]. It is useful when polyvoxelhood is known from
/// external provenance, such as an enumeration catalogue, but has not been
/// reconstructed as a [`Polyvoxel`].
pub fn traversal_normalisation_of_shape(
    shape: &Arc<FramedPoset>,
) -> Result<(Arc<FramedPoset>, Embedding), TraversalError> {
    let order = traverse_shape(shape)?;
    let sizes = shape.sizes();

    let mut new_to_old: Vec<Vec<usize>> =
        sizes.iter().map(|&size| Vec::with_capacity(size)).collect();
    for cell in order {
        new_to_old[cell.dim].push(cell.pos);
    }

    let mut old_to_new: Vec<Vec<usize>> = sizes.iter().map(|&size| vec![0; size]).collect();
    for (dim, level) in new_to_old.iter().enumerate() {
        for (new_pos, &old_pos) in level.iter().enumerate() {
            old_to_new[dim][old_pos] = new_pos;
        }
    }

    let mut basis = Vec::with_capacity(sizes.len());
    let mut faces_in = Vec::with_capacity(sizes.len());
    let mut faces_out = Vec::with_capacity(sizes.len());
    for (dim, level) in new_to_old.iter().enumerate() {
        basis.push(
            level
                .iter()
                .map(|&old_pos| shape.basis_of(dim, old_pos).clone())
                .collect(),
        );
        if dim == 0 {
            faces_in.push(vec![vec![]; level.len()]);
            faces_out.push(vec![vec![]; level.len()]);
            continue;
        }
        faces_in.push(
            level
                .iter()
                .map(|&old_pos| {
                    intset::collect_sorted(
                        shape
                            .faces_of(Sign::Input, dim, old_pos)
                            .iter()
                            .map(|&face| old_to_new[dim - 1][face]),
                    )
                })
                .collect(),
        );
        faces_out.push(
            level
                .iter()
                .map(|&old_pos| {
                    intset::collect_sorted(
                        shape
                            .faces_of(Sign::Output, dim, old_pos)
                            .iter()
                            .map(|&face| old_to_new[dim - 1][face]),
                    )
                })
                .collect(),
        );
    }

    let normal = Arc::new(FramedPoset::from_faces(basis, faces_in, faces_out));
    let into_original = Embedding::from_map(Arc::clone(&normal), Arc::clone(shape), new_to_old);
    debug_assert!(into_original.is_isomorphism());
    Ok((normal, into_original))
}

fn traverse_shape(shape: &Arc<FramedPoset>) -> Result<Vec<Element>, TraversalError> {
    let total_cells = cell_count(shape);
    if total_cells == 0 {
        return Ok(vec![]);
    }

    let frame = shape.active_directions();
    let mut traversal = PartialTraversal::new(shape);

    for &direction in &frame {
        traversal.append_boundary(Sign::Input, direction, total_cells)?;
    }

    if let Some(greatest) = shape.greatest_element() {
        traversal.append(greatest);
    } else if let Some(&direction) = frame.last() {
        traversal.walk_maximal_cells(&frame, direction, total_cells)?;
    }

    for &direction in &frame {
        traversal.append_boundary(Sign::Output, direction, total_cells)?;
    }

    let unvisited = traversal.unvisited_cells();
    if !unvisited.is_empty() {
        return Err(TraversalError::Incomplete { cells: unvisited });
    }

    Ok(traversal.order)
}

struct PartialTraversal<'a> {
    shape: &'a Arc<FramedPoset>,
    order: Vec<Element>,
    seen: Vec<Vec<bool>>,
}

impl<'a> PartialTraversal<'a> {
    fn new(shape: &'a Arc<FramedPoset>) -> Self {
        Self {
            shape,
            order: Vec::with_capacity(cell_count(shape)),
            seen: shape
                .sizes()
                .into_iter()
                .map(|size| vec![false; size])
                .collect(),
        }
    }

    fn append(&mut self, cell: Element) {
        if !self.seen[cell.dim][cell.pos] {
            self.seen[cell.dim][cell.pos] = true;
            self.order.push(cell);
        }
    }

    fn append_subtraversal(&mut self, order: Vec<Element>, embedding: &Embedding) {
        for cell in order {
            self.append(embedding.apply(cell));
        }
    }

    fn append_boundary(
        &mut self,
        sign: Sign,
        direction: usize,
        parent_cells: usize,
    ) -> Result<(), TraversalError> {
        let (boundary_shape, into_parent) = boundary(sign, direction, self.shape);
        if cell_count(&boundary_shape) >= parent_cells {
            return Err(TraversalError::BoundaryDidNotDecrease {
                sign,
                direction,
                cells: parent_cells,
            });
        }
        let order = traverse_shape(&boundary_shape)?;
        self.append_subtraversal(order, &into_parent);
        Ok(())
    }

    fn walk_maximal_cells(
        &mut self,
        frame: &[usize],
        direction: usize,
        parent_cells: usize,
    ) -> Result<(), TraversalError> {
        let full_dim = frame.len();
        let full_maxima: Vec<Element> = self
            .shape
            .sizes()
            .get(full_dim)
            .map(|&size| {
                (0..size)
                    .filter(|&pos| self.shape.basis_of(full_dim, pos) == frame)
                    .map(|pos| Element { dim: full_dim, pos })
                    .collect()
            })
            .unwrap_or_default();

        while full_maxima
            .iter()
            .any(|cell| !self.seen[cell.dim][cell.pos])
        {
            let mut continuation = None;

            for &face in &self.order {
                let candidates = self.continuations(face, &full_maxima, direction);
                if candidates.len() > 1 {
                    return Err(TraversalError::AmbiguousContinuation {
                        direction,
                        face,
                        cells: candidates,
                    });
                }
                if let Some(&cell) = candidates.first() {
                    continuation = Some(cell);
                    break;
                }
            }

            let Some(cell) = continuation else {
                let cells = full_maxima
                    .iter()
                    .copied()
                    .filter(|cell| !self.seen[cell.dim][cell.pos])
                    .collect();
                return Err(TraversalError::MissingContinuation { direction, cells });
            };

            let mut keep: Vec<Vec<bool>> = self
                .shape
                .sizes()
                .into_iter()
                .map(|size| vec![false; size])
                .collect();
            keep[cell.dim][cell.pos] = true;
            let subset = FramedPosetSubset::make(Arc::clone(self.shape), keep);
            let (cell_shape, into_parent) = closure(&subset);
            if cell_count(&cell_shape) >= parent_cells {
                return Err(TraversalError::CellClosureDidNotDecrease {
                    cell,
                    cells: parent_cells,
                });
            }
            let order = traverse_shape(&cell_shape)?;
            self.append_subtraversal(order, &into_parent);
        }

        Ok(())
    }

    fn continuations(
        &self,
        face: Element,
        full_maxima: &[Element],
        direction: usize,
    ) -> Vec<Element> {
        if face.dim + 1 >= self.seen.len() {
            return vec![];
        }

        self.shape
            .cofaces_of(Sign::Input, face.dim, face.pos)
            .iter()
            .copied()
            .map(|pos| Element {
                dim: face.dim + 1,
                pos,
            })
            .filter(|cell| {
                full_maxima.contains(cell)
                    && !self.seen[cell.dim][cell.pos]
                    && intset::cover_direction(
                        self.shape.basis_of_element(face),
                        self.shape.basis_of_element(*cell),
                    ) == Ok(direction)
            })
            .collect()
    }

    fn unvisited_cells(&self) -> Vec<Element> {
        self.seen
            .iter()
            .enumerate()
            .flat_map(|(dim, level)| {
                level
                    .iter()
                    .enumerate()
                    .filter_map(move |(pos, &seen)| (!seen).then_some(Element { dim, pos }))
            })
            .collect()
    }
}

fn cell_count(shape: &FramedPoset) -> usize {
    shape.sizes().into_iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polyvoxel::{cylinder, paste, point};
    use crate::random::randomly_permute;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn arrow() -> Polyvoxel {
        let point = point();
        cylinder(&point, &point)
    }

    fn square() -> Polyvoxel {
        let arrow = arrow();
        cylinder(&arrow, &arrow)
    }

    fn cube() -> Polyvoxel {
        let square = square();
        cylinder(&square, &square)
    }

    fn double_degen_example() -> Polyvoxel {
        let p = point();
        let e = cylinder(&p, &p);
        let t = cylinder(&e, &p);
        let prism = cylinder(&t, &t);
        let t2 = crate::polyvoxel::shift(&t);
        let u1 = cylinder(&prism, &t2);

        let square = cylinder(&e, &e);
        let e2 = crate::polyvoxel::shift(&e);
        let u2 = cylinder(&square, &e2);
        let (_, u) = paste(&u1, &u2, 2);

        let v1 = cylinder(&t2, &t2);
        let v2 = cylinder(&e2, &e2);
        let (_, v) = paste(&v1, &v2, 2);

        paste(&u, &v, 0).1
    }

    fn non_uniform_layering_direction_example() -> Polyvoxel {
        let p = point();
        let e = cylinder(&p, &p);
        let t = cylinder(&e, &p);
        let prism = cylinder(&t, &t);

        let double_prism = paste(&prism, &prism, 2).1;
        let u = paste(&double_prism, &t, 1).1;
        paste(&u, &double_prism, 0).1
    }

    fn assert_traversal_normalises(polyvoxel: &Polyvoxel) -> Arc<FramedPoset> {
        let (normal, into_original) = traversal_normalisation(polyvoxel).unwrap();
        assert!(!normal.is_normal());
        assert!(into_original.is_isomorphism());
        assert!(FramedPoset::equal(&normal, &into_original.dom));
        assert!(FramedPoset::equal(
            polyvoxel.as_framed_poset(),
            &into_original.cod,
        ));
        normal
    }

    #[test]
    fn traverses_a_point_and_an_arrow_input_first() {
        let point = point();
        assert_eq!(
            traversal_order(&point).unwrap(),
            vec![Element { dim: 0, pos: 0 }]
        );

        let arrow = arrow();
        let order = traversal_order(&arrow).unwrap();
        assert_eq!(order.len(), 3);
        let input = order[0];
        let edge = order[1];
        let output = order[2];
        assert_eq!(input.dim, 0);
        assert_eq!(edge.dim, 1);
        assert_eq!(output.dim, 0);
        assert_eq!(
            arrow.faces_of(Sign::Input, edge.dim, edge.pos),
            &[input.pos]
        );
        assert_eq!(
            arrow.faces_of(Sign::Output, edge.dim, edge.pos),
            &[output.pos]
        );

        assert_traversal_normalises(&arrow);
    }

    #[test]
    fn square_uses_increasing_direction_boundaries() {
        let square = square();
        let normal = assert_traversal_normalises(&square);

        assert_eq!(normal.sizes(), vec![4, 4, 1]);
        assert_eq!(
            (0..4)
                .map(|pos| normal.basis_of(1, pos).clone())
                .collect::<Vec<_>>(),
            vec![vec![1], vec![0], vec![1], vec![0]],
        );
        assert_eq!(normal.faces_of(Sign::Input, 1, 0), &[0]);
        assert_eq!(normal.faces_of(Sign::Output, 1, 0), &[1]);
        assert_eq!(normal.faces_of(Sign::Input, 1, 1), &[0]);
        assert_eq!(normal.faces_of(Sign::Output, 1, 1), &[2]);
        assert_eq!(normal.faces_of(Sign::Input, 1, 2), &[2]);
        assert_eq!(normal.faces_of(Sign::Output, 1, 2), &[3]);
        assert_eq!(normal.faces_of(Sign::Input, 1, 3), &[1]);
        assert_eq!(normal.faces_of(Sign::Output, 1, 3), &[3]);
        assert_eq!(normal.faces_of(Sign::Input, 2, 0), &[0, 1]);
        assert_eq!(normal.faces_of(Sign::Output, 2, 0), &[2, 3]);
    }

    #[test]
    fn traverses_basic_pasted_polyvoxels() {
        let arrow = arrow();
        let square = square();
        let (_, path) = paste(&arrow, &arrow, 0);
        let (_, horizontal_rectangle) = paste(&square, &square, 0);
        let (_, vertical_rectangle) = paste(&square, &square, 1);

        for shape in [&path, &horizontal_rectangle, &vertical_rectangle] {
            assert_traversal_normalises(shape);
        }
    }

    #[test]
    fn traverses_a_cube_and_pastings_in_each_direction() {
        let cube = cube();
        assert_eq!(cube.sizes(), vec![8, 12, 6, 1]);
        assert_traversal_normalises(&cube);

        for direction in 0..3 {
            let (_, pair) = paste(&cube, &cube, direction);
            assert_traversal_normalises(&pair);
        }
    }

    #[test]
    fn traversal_is_independent_of_path_parenthesisation() {
        let arrow = arrow();
        let (_, left_pair) = paste(&arrow, &arrow, 0);
        let (_, left_associated) = paste(&left_pair, &arrow, 0);
        let (_, right_pair) = paste(&arrow, &arrow, 0);
        let (_, right_associated) = paste(&arrow, &right_pair, 0);

        let left_normal = assert_traversal_normalises(&left_associated);
        let right_normal = assert_traversal_normalises(&right_associated);
        assert!(FramedPoset::equal(&left_normal, &right_normal));
    }

    #[test]
    fn traversal_normalisation_is_invariant_under_random_cell_orders() {
        let square = square();
        let (_, rectangle) = paste(&square, &square, 0);
        let cube = cube();
        let mut rng = SmallRng::seed_from_u64(0x7a_a4_3e_25_a1);

        for polyvoxel in [&square, &rectangle, &cube] {
            assert_random_permutations_have_same_traversal(polyvoxel, &mut rng);
        }
    }

    #[test]
    fn traversal_handles_the_nonstandard_example_polyvoxels() {
        let double_degen = double_degen_example();
        let non_uniform = non_uniform_layering_direction_example();
        let mut rng = SmallRng::seed_from_u64(0x5ec1_a1ca_5e50_5e70);

        for polyvoxel in [&double_degen, &non_uniform] {
            assert_random_permutations_have_same_traversal(polyvoxel, &mut rng);
        }
    }

    fn assert_random_permutations_have_same_traversal(polyvoxel: &Polyvoxel, rng: &mut SmallRng) {
        let (expected, _) = traversal_normalisation(polyvoxel).unwrap();
        for _ in 0..32 {
            let (permuted_shape, into_original) =
                randomly_permute(polyvoxel.as_framed_poset(), rng);
            let permuted = Polyvoxel::from_isomorphism(permuted_shape, &into_original, polyvoxel);

            let (actual, _) = traversal_normalisation(&permuted).unwrap();
            assert!(FramedPoset::equal(&expected, &actual));
        }
    }
}
