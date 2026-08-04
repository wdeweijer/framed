//! Orthogonal products of oriented framed posets and embeddings.

use std::sync::Arc;

use crate::embedding::{Embedding, NO_PREIMAGE};
use crate::intset::{self, IntSet};
use crate::poset::{FramedPoset, Sign};

/// Form the orthogonal tensor product of two framed posets.
///
/// Its cells are pairs whose bases are disjoint. The basis of a retained pair
/// is the union of the two bases, and a signed cover changes one coordinate
/// along a cover of the same sign in that factor.
pub fn orthogonal_product(left: &FramedPoset, right: &FramedPoset) -> FramedPoset {
    orthogonal_product_data(left, right).shape
}

/// Form the orthogonal product of two embeddings.
///
/// It maps every retained product cell `(x, y)` to `(left(x), right(y))`.
pub fn orthogonal_product_embedding(left: &Embedding, right: &Embedding) -> Embedding {
    let source = orthogonal_product_data(&left.dom, &right.dom);
    let target = orthogonal_product_data(&left.cod, &right.cod);
    let left_cod_offsets = cell_offsets(&left.cod.sizes());
    let right_cod_offsets = cell_offsets(&right.cod.sizes());
    let map = source
        .cells
        .iter()
        .map(|level| {
            level
                .iter()
                .map(|cell| {
                    let left_cod_pos = left.map[cell.left_dim][cell.left_pos];
                    let right_cod_pos = right.map[cell.right_dim][cell.right_pos];
                    let left_cod_id = left_cod_offsets[cell.left_dim] + left_cod_pos;
                    let right_cod_id = right_cod_offsets[cell.right_dim] + right_cod_pos;
                    target.pair_position(left_cod_id, right_cod_id)
                })
                .collect()
        })
        .collect();

    product_embedding(source.shape, target.shape, map)
}

/// The commutativity isomorphism `left * right -> right * left`.
///
/// It maps every retained product cell `(x, y)` to `(y, x)`.
pub fn orthogonal_product_commutator(left: &FramedPoset, right: &FramedPoset) -> Embedding {
    let forward = orthogonal_product_data(left, right);
    let backward = orthogonal_product_data(right, left);
    let map = forward
        .cells
        .iter()
        .map(|level| {
            level
                .iter()
                .map(|cell| backward.pair_position(cell.right_id, cell.left_id))
                .collect()
        })
        .collect();

    product_isomorphism(forward.shape, backward.shape, map)
}

/// The associativity isomorphism `(left * middle) * right -> left * (middle * right)`.
///
/// It maps every retained product cell `((x, y), z)` to `(x, (y, z))`.
pub fn orthogonal_product_associator(
    left: &FramedPoset,
    middle: &FramedPoset,
    right: &FramedPoset,
) -> Embedding {
    let left_middle = orthogonal_product_data(left, middle);
    let middle_right = orthogonal_product_data(middle, right);
    let source = orthogonal_product_data(&left_middle.shape, right);
    let target = orthogonal_product_data(left, &middle_right.shape);
    let map = source
        .cells
        .iter()
        .map(|level| {
            level
                .iter()
                .map(|cell| {
                    let left_middle_cell = left_middle.cells[cell.left_dim][cell.left_pos];
                    let middle_right_dim = left_middle_cell.right_dim + cell.right_dim;
                    let middle_right_pos =
                        middle_right.pair_position(left_middle_cell.right_id, cell.right_id);
                    let middle_right_id = middle_right.cell_id(middle_right_dim, middle_right_pos);
                    target.pair_position(left_middle_cell.left_id, middle_right_id)
                })
                .collect()
        })
        .collect();

    product_isomorphism(source.shape, target.shape, map)
}

#[derive(Clone, Copy)]
struct OrthogonalProductCell {
    left_dim: usize,
    left_pos: usize,
    left_id: usize,
    right_dim: usize,
    right_pos: usize,
    right_id: usize,
}

struct OrthogonalProductData {
    shape: FramedPoset,
    cells: Vec<Vec<OrthogonalProductCell>>,
    pair_positions: Vec<Option<usize>>,
    right_count: usize,
    offsets: Vec<usize>,
}

impl OrthogonalProductData {
    fn pair_position(&self, left_id: usize, right_id: usize) -> usize {
        self.pair_positions[left_id * self.right_count + right_id]
            .expect("the corresponding orthogonal product cell must exist")
    }

    fn cell_id(&self, dim: usize, pos: usize) -> usize {
        self.offsets[dim] + pos
    }
}

fn cell_offsets(sizes: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(sizes.len() + 1);
    offsets.push(0);
    for &size in sizes {
        offsets.push(offsets.last().copied().unwrap() + size);
    }
    offsets
}

fn orthogonal_product_data(left: &FramedPoset, right: &FramedPoset) -> OrthogonalProductData {
    let left_sizes = left.sizes();
    let right_sizes = right.sizes();
    let left_count = left_sizes.iter().sum::<usize>();
    let right_count = right_sizes.iter().sum::<usize>();
    if left_count == 0 || right_count == 0 {
        return OrthogonalProductData {
            shape: FramedPoset::empty(),
            cells: Vec::new(),
            pair_positions: Vec::new(),
            right_count,
            offsets: vec![0],
        };
    }

    let left_offsets = cell_offsets(&left_sizes);
    let right_offsets = cell_offsets(&right_sizes);
    let pair_count = left_count
        .checked_mul(right_count)
        .expect("orthogonal product has too many potential cell pairs");
    let level_count = left_sizes.len() + right_sizes.len() - 1;
    let mut pair_positions = vec![None; pair_count];
    let mut basis = vec![Vec::new(); level_count];
    let mut product_cells = vec![Vec::new(); level_count];

    for left_dim in 0..left_sizes.len() {
        for left_pos in 0..left_sizes[left_dim] {
            let left_id = left_offsets[left_dim] + left_pos;
            for right_dim in 0..right_sizes.len() {
                for right_pos in 0..right_sizes[right_dim] {
                    if !intset::is_disjoint(
                        left.basis_of(left_dim, left_pos),
                        right.basis_of(right_dim, right_pos),
                    ) {
                        continue;
                    }

                    let right_id = right_offsets[right_dim] + right_pos;
                    let dim = left_dim + right_dim;
                    let pos = basis[dim].len();
                    pair_positions[left_id * right_count + right_id] = Some(pos);
                    basis[dim].push(intset::union(
                        left.basis_of(left_dim, left_pos),
                        right.basis_of(right_dim, right_pos),
                    ));
                    product_cells[dim].push(OrthogonalProductCell {
                        left_dim,
                        left_pos,
                        left_id,
                        right_dim,
                        right_pos,
                        right_id,
                    });
                }
            }
        }
    }

    while basis.last().is_some_and(Vec::is_empty) {
        basis.pop();
        product_cells.pop();
    }

    let mut faces_in: Vec<Vec<IntSet>> = basis
        .iter()
        .map(|level| vec![vec![]; level.len()])
        .collect();
    let mut faces_out = faces_in.clone();

    for dim in 1..basis.len() {
        for (pos, cell) in product_cells[dim].iter().enumerate() {
            for sign in [Sign::Input, Sign::Output] {
                let mut faces = Vec::new();

                if cell.left_dim > 0 {
                    faces.extend(
                        left.faces_of(sign, cell.left_dim, cell.left_pos)
                            .iter()
                            .map(|&face| {
                                let left_face_id = left_offsets[cell.left_dim - 1] + face;
                                pair_positions[left_face_id * right_count + cell.right_id]
                                    .expect("a face of a disjoint pair must remain disjoint")
                            }),
                    );
                }
                if cell.right_dim > 0 {
                    faces.extend(
                        right
                            .faces_of(sign, cell.right_dim, cell.right_pos)
                            .iter()
                            .map(|&face| {
                                let right_face_id = right_offsets[cell.right_dim - 1] + face;
                                pair_positions[cell.left_id * right_count + right_face_id]
                                    .expect("a face of a disjoint pair must remain disjoint")
                            }),
                    );
                }

                let faces = intset::collect_sorted(faces.into_iter());
                match sign {
                    Sign::Input => faces_in[dim][pos] = faces,
                    Sign::Output => faces_out[dim][pos] = faces,
                }
            }
        }
    }

    let product = FramedPoset::from_faces(basis, faces_in, faces_out);
    debug_assert!(product.well_formed());
    let offsets = cell_offsets(&product.sizes());
    OrthogonalProductData {
        shape: product,
        cells: product_cells,
        pair_positions,
        right_count,
        offsets,
    }
}

fn product_isomorphism(dom: FramedPoset, cod: FramedPoset, map: Vec<Vec<usize>>) -> Embedding {
    let embedding = product_embedding(dom, cod, map);
    debug_assert!(embedding.is_isomorphism());
    embedding
}

fn product_embedding(dom: FramedPoset, cod: FramedPoset, map: Vec<Vec<usize>>) -> Embedding {
    debug_assert_eq!(map.iter().map(Vec::len).collect::<Vec<_>>(), dom.sizes());
    let mut inv: Vec<Vec<usize>> = cod
        .sizes()
        .into_iter()
        .map(|size| vec![NO_PREIMAGE; size])
        .collect();

    for (dim, level) in map.iter().enumerate() {
        for (dom_pos, &cod_pos) in level.iter().enumerate() {
            debug_assert_eq!(inv[dim][cod_pos], NO_PREIMAGE);
            inv[dim][cod_pos] = dom_pos;
        }
    }

    Embedding::make(Arc::new(dom), Arc::new(cod), map, inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poset::shift;

    fn tight_arrow() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![1]]],
        ))
    }

    fn square() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![
                vec![vec![], vec![], vec![], vec![]],
                vec![vec![0], vec![0], vec![1], vec![1]],
                vec![vec![0, 1]],
            ],
            vec![
                vec![vec![], vec![], vec![], vec![]],
                vec![vec![0], vec![2], vec![0], vec![1]],
                vec![vec![0, 2]],
            ],
            vec![
                vec![vec![], vec![], vec![], vec![]],
                vec![vec![1], vec![3], vec![2], vec![3]],
                vec![vec![1, 3]],
            ],
        ))
    }

    #[test]
    fn point_is_the_unit_and_empty_is_absorbing() {
        let empty = FramedPoset::empty();
        let point = FramedPoset::point();
        let arrow = tight_arrow();

        assert!(FramedPoset::equal(
            &orthogonal_product(&point, &arrow),
            &arrow,
        ));
        assert!(FramedPoset::equal(
            &orthogonal_product(&arrow, &point),
            &arrow,
        ));
        assert!(orthogonal_product(&empty, &arrow).sizes().is_empty());
        assert!(orthogonal_product(&arrow, &empty).sizes().is_empty());
    }

    #[test]
    fn orthogonal_arrows_form_the_oriented_square() {
        let horizontal = tight_arrow();
        let vertical = Arc::new(shift(&horizontal));
        let product = orthogonal_product(&horizontal, &vertical);

        assert_eq!(product.sizes(), vec![4, 4, 1]);
        assert_eq!(product.active_directions(), vec![0, 1]);
        assert!(crate::isomorphism::isomorphic(&product, &square()));
    }

    #[test]
    fn product_discards_pairs_with_overlapping_bases() {
        let arrow = tight_arrow();
        let product = orthogonal_product(&arrow, &arrow);

        assert_eq!(product.sizes(), vec![4, 4]);
        assert_eq!(product.active_directions(), vec![0]);
        assert_eq!(product.sizes().iter().sum::<usize>(), 8);
        assert!(product.well_formed());
    }

    #[test]
    fn commutator_is_an_isomorphism() {
        let arrow_0 = tight_arrow();
        let arrow_1 = Arc::new(shift(&arrow_0));

        for (left, right) in [
            (arrow_0.as_ref(), arrow_1.as_ref()),
            (arrow_0.as_ref(), arrow_0.as_ref()),
        ] {
            let commutator = orthogonal_product_commutator(left, right);

            assert!(FramedPoset::equal(
                &commutator.dom,
                &orthogonal_product(left, right),
            ));
            assert!(FramedPoset::equal(
                &commutator.cod,
                &orthogonal_product(right, left),
            ));
            assert!(commutator.is_isomorphism());
        }
    }

    #[test]
    fn product_of_isomorphisms_is_an_isomorphism() {
        let arrow_0 = tight_arrow();
        let arrow_1 = Arc::new(shift(&arrow_0));
        let commutator = orthogonal_product_commutator(&arrow_0, &arrow_1);
        let point_identity = Embedding::id(Arc::new(FramedPoset::point()));
        let product = orthogonal_product_embedding(&commutator, &point_identity);

        assert!(FramedPoset::equal(
            &product.dom,
            &orthogonal_product(&commutator.dom, &point_identity.dom),
        ));
        assert!(FramedPoset::equal(
            &product.cod,
            &orthogonal_product(&commutator.cod, &point_identity.cod),
        ));
        assert!(product.is_isomorphism());
    }

    #[test]
    fn associator_is_an_isomorphism() {
        let point = FramedPoset::point();
        let arrow_0 = tight_arrow();
        let arrow_1 = Arc::new(shift(&arrow_0));
        let arrow_2 = Arc::new(shift(&arrow_1));
        let cases = [
            (&point, arrow_0.as_ref(), arrow_1.as_ref()),
            (arrow_0.as_ref(), arrow_1.as_ref(), arrow_2.as_ref()),
            (arrow_0.as_ref(), arrow_0.as_ref(), arrow_1.as_ref()),
        ];

        for (left, middle, right) in cases {
            let associator = orthogonal_product_associator(left, middle, right);
            let left_middle = orthogonal_product(left, middle);
            let middle_right = orthogonal_product(middle, right);

            assert!(FramedPoset::equal(
                &associator.dom,
                &orthogonal_product(&left_middle, right),
            ));
            assert!(FramedPoset::equal(
                &associator.cod,
                &orthogonal_product(left, &middle_right),
            ));
            assert!(associator.is_isomorphism());
        }
    }
}
