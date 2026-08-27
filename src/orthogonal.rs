//! Orthogonal products of oriented framed posets and embeddings.

use std::collections::HashMap;
use std::sync::Arc;

use crate::embedding::Embedding;
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
    let map = source
        .cells
        .iter()
        .map(|level| {
            level
                .iter()
                .map(|cell| {
                    let left_image = Cell {
                        dimension: cell.left.dimension,
                        position: left.map[cell.left.dimension][cell.left.position],
                    };
                    let right_image = Cell {
                        dimension: cell.right.dimension,
                        position: right.map[cell.right.dimension][cell.right.position],
                    };
                    target.product_cell(left_image, right_image).position
                })
                .collect()
        })
        .collect();

    Embedding::from_map(Arc::new(source.shape), Arc::new(target.shape), map)
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
                .map(|cell| backward.product_cell(cell.right, cell.left).position)
                .collect()
        })
        .collect();

    Embedding::from_map(Arc::new(forward.shape), Arc::new(backward.shape), map)
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
                    let left_middle_cell =
                        left_middle.cells[cell.left.dimension][cell.left.position];
                    let middle_right_cell =
                        middle_right.product_cell(left_middle_cell.right, cell.right);
                    target
                        .product_cell(left_middle_cell.left, middle_right_cell)
                        .position
                })
                .collect()
        })
        .collect();

    Embedding::from_map(Arc::new(source.shape), Arc::new(target.shape), map)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Cell {
    dimension: usize,
    position: usize,
}

#[derive(Debug, Clone, Copy)]
struct OrthogonalProductCell {
    left: Cell,
    right: Cell,
}

struct OrthogonalProductData {
    shape: FramedPoset,
    cells: Vec<Vec<OrthogonalProductCell>>,
    product_cells: HashMap<(Cell, Cell), Cell>,
}

impl OrthogonalProductData {
    fn product_cell(&self, left: Cell, right: Cell) -> Cell {
        self.product_cells
            .get(&(left, right))
            .copied()
            .expect("the corresponding orthogonal product cell must exist")
    }
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
            product_cells: HashMap::new(),
        };
    }

    let level_count = left_sizes.len() + right_sizes.len() - 1;
    let mut cell_pairs = HashMap::new();
    let mut basis = vec![Vec::new(); level_count];
    let mut product_cells = vec![Vec::new(); level_count];

    for (left_dim, &left_size) in left_sizes.iter().enumerate() {
        for left_pos in 0..left_size {
            let left_cell = Cell {
                dimension: left_dim,
                position: left_pos,
            };
            for (right_dim, &right_size) in right_sizes.iter().enumerate() {
                for right_pos in 0..right_size {
                    if !intset::is_disjoint(
                        left.basis_of(left_dim, left_pos),
                        right.basis_of(right_dim, right_pos),
                    ) {
                        continue;
                    }

                    let right_cell = Cell {
                        dimension: right_dim,
                        position: right_pos,
                    };
                    let dim = left_dim + right_dim;
                    let pos = basis[dim].len();
                    let previous = cell_pairs.insert(
                        (left_cell, right_cell),
                        Cell {
                            dimension: dim,
                            position: pos,
                        },
                    );
                    debug_assert!(previous.is_none());
                    basis[dim].push(intset::union(
                        left.basis_of(left_dim, left_pos),
                        right.basis_of(right_dim, right_pos),
                    ));
                    product_cells[dim].push(OrthogonalProductCell {
                        left: left_cell,
                        right: right_cell,
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

                if cell.left.dimension > 0 {
                    faces.extend(
                        left.faces_of(sign, cell.left.dimension, cell.left.position)
                            .iter()
                            .map(|&face| {
                                cell_pairs
                                    .get(&(
                                        Cell {
                                            dimension: cell.left.dimension - 1,
                                            position: face,
                                        },
                                        cell.right,
                                    ))
                                    .expect("a face of a disjoint pair must remain disjoint")
                                    .position
                            }),
                    );
                }
                if cell.right.dimension > 0 {
                    faces.extend(
                        right
                            .faces_of(sign, cell.right.dimension, cell.right.position)
                            .iter()
                            .map(|&face| {
                                cell_pairs
                                    .get(&(
                                        cell.left,
                                        Cell {
                                            dimension: cell.right.dimension - 1,
                                            position: face,
                                        },
                                    ))
                                    .expect("a face of a disjoint pair must remain disjoint")
                                    .position
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
    OrthogonalProductData {
        shape: product,
        cells: product_cells,
        product_cells: cell_pairs,
    }
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
