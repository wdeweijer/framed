//! Volumetric oriented framed posets.
//!
//! Volumetricity includes strong cubularity, so the order of distinct
//! boundary operations does not affect a volumetric candidate. We consistently
//! apply them in ascending frame order; the standalone predicates use the same
//! deterministic convention even for non-cubular inputs. Following Remark
//! 1.71, we use the exact active frame, since adding absent directions would
//! only insert identity boundary operations.

use std::sync::Arc;

use crate::cubularity::{CubularityMode, is_cubular};
use crate::isomorphism::isomorphic;
use crate::orthogonal::orthogonal_product;
use crate::poset::{FramedPoset, Sign, boundary, iterated_boundary};

/// Check every convolution equation from Definition 1.69.
pub fn satisfies_convolution_equations(shape: &Arc<FramedPoset>) -> bool {
    let frame = shape.active_directions();

    frame.iter().enumerate().all(|(index, &direction)| {
        [Sign::Input, Sign::Output].into_iter().all(|sign| {
            let (direct_boundary, _) = boundary(sign, direction, shape);
            let left = boundary_block(sign, &frame[..=index], shape);
            let right = boundary_block(sign, &frame[index..], shape);
            let convolution = orthogonal_product(&left, &right);

            isomorphic(&direct_boundary, &convolution)
        })
    })
}

/// Check every left sign-equation from Definition 1.69.
pub fn satisfies_left_sign_equations(shape: &Arc<FramedPoset>) -> bool {
    let frame = shape.active_directions();

    index_pairs(frame.len()).all(|(left, right)| {
        [Sign::Input, Sign::Output].into_iter().all(|sign| {
            let mut lhs_word = boundary_word(sign.opposite(), &frame[..=left]);
            lhs_word.extend(boundary_word(sign, &frame[left + 1..=right]));
            let (lhs, _) = iterated_boundary(&lhs_word, shape);

            let rhs_word = boundary_word(sign, &frame[..=right]);
            let (rhs, _) = iterated_boundary(&rhs_word, shape);

            isomorphic(&lhs, &rhs)
        })
    })
}

/// Check every right sign-equation from Definition 1.69.
pub fn satisfies_right_sign_equations(shape: &Arc<FramedPoset>) -> bool {
    let frame = shape.active_directions();

    index_pairs(frame.len()).all(|(left, right)| {
        [Sign::Input, Sign::Output].into_iter().all(|sign| {
            let mut lhs_word = boundary_word(sign, &frame[left..right]);
            lhs_word.extend(boundary_word(sign.opposite(), &frame[right..]));
            let (lhs, _) = iterated_boundary(&lhs_word, shape);

            let rhs_word = boundary_word(sign, &frame[left..]);
            let (rhs, _) = iterated_boundary(&rhs_word, shape);

            isomorphic(&lhs, &rhs)
        })
    })
}

/// Check whether an oriented framed poset is volumetric.
///
/// Rigidity is checked last because enumerating automorphisms recursively over
/// all boundaries can be substantially more expensive than the other checks.
pub fn is_volumetric(shape: &Arc<FramedPoset>) -> bool {
    is_cubular(CubularityMode::Strong, shape)
        && satisfies_convolution_equations(shape)
        && satisfies_left_sign_equations(shape)
        && satisfies_right_sign_equations(shape)
        && shape.is_rigid()
}

/// Ascending operation order for the compact notation delta_{k_1 ... k_n}.
fn boundary_word(sign: Sign, directions: &[usize]) -> Vec<(Sign, usize)> {
    directions
        .iter()
        .map(|&direction| (sign, direction))
        .collect()
}

fn boundary_block(sign: Sign, directions: &[usize], shape: &Arc<FramedPoset>) -> Arc<FramedPoset> {
    iterated_boundary(&boundary_word(sign, directions), shape).0
}

fn index_pairs(length: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..length).flat_map(move |left| (left + 1..length).map(move |right| (left, right)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arrow(direction: usize) -> FramedPoset {
        FramedPoset::from_faces(
            vec![vec![vec![], vec![]], vec![vec![direction]]],
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![1]]],
        )
    }

    fn cube(dimension: usize) -> Arc<FramedPoset> {
        let mut result = FramedPoset::point();
        for direction in 0..dimension {
            result = orthogonal_product(&result, &arrow(direction));
        }
        Arc::new(result)
    }

    fn arrow_with_two_input_vertices() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![vec![vec![], vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![], vec![]], vec![vec![0, 1]]],
            vec![vec![vec![], vec![], vec![]], vec![vec![2]]],
        ))
    }

    fn sign_equation_counterexample() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![vec![vec![]], vec![vec![0], vec![1]], vec![vec![0, 1]]],
            vec![vec![vec![]], vec![vec![], vec![]], vec![vec![0, 1]]],
            vec![vec![vec![]], vec![vec![0], vec![0]], vec![vec![]]],
        ))
    }

    #[test]
    fn standard_cubes_satisfy_all_three_equation_families() {
        for dimension in 0..=3 {
            let shape = cube(dimension);
            assert!(satisfies_convolution_equations(&shape));
            assert!(satisfies_left_sign_equations(&shape));
            assert!(satisfies_right_sign_equations(&shape));
        }
    }

    #[test]
    fn convolution_detects_a_non_point_extreme_boundary() {
        assert!(!satisfies_convolution_equations(
            &arrow_with_two_input_vertices()
        ));
    }

    #[test]
    fn left_sign_equations_detect_a_failure() {
        assert!(!satisfies_left_sign_equations(
            &sign_equation_counterexample()
        ));
    }

    #[test]
    fn right_sign_equations_detect_a_failure() {
        assert!(!satisfies_right_sign_equations(
            &sign_equation_counterexample()
        ));
    }

    #[test]
    fn standard_cubes_are_volumetric() {
        for dimension in 0..=3 {
            assert!(is_volumetric(&cube(dimension)));
        }
    }

    #[test]
    fn volumetricity_rejects_a_disconnected_zero_dimensional_shape() {
        let two_points = Arc::new(FramedPoset::from_faces(
            vec![vec![vec![], vec![]]],
            vec![vec![vec![], vec![]]],
            vec![vec![vec![], vec![]]],
        ));

        assert!(is_cubular(CubularityMode::Strong, &two_points));
        assert!(satisfies_convolution_equations(&two_points));
        assert!(satisfies_left_sign_equations(&two_points));
        assert!(satisfies_right_sign_equations(&two_points));
        assert!(!is_volumetric(&two_points));
    }
}
