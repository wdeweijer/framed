//! Random generation of oriented framed posets.

use rand::Rng;

use crate::intset::IntSet;
use crate::poset::FramedPoset;

/// Generate an oriented framed poset with exactly `cell_count` cells.
///
/// Cell bases are restricted to `∅`, `{0}`, `{1}`, and `{0, 1}`. Every
/// result has at least one `{0, 1}` cell. Every oriented framed poset with
/// those bases, at least one `{0, 1}` cell, and the requested number of cells
/// has positive probability, up to the ordering of cells within each level.
///
/// # Panics
///
/// Panics if `cell_count` is less than four, the minimum number of cells needed
/// for a well-formed poset containing a `{0, 1}` cell.
pub fn random_framed_poset<R: Rng + ?Sized>(cell_count: usize, rng: &mut R) -> FramedPoset {
    assert!(
        cell_count >= 4,
        "at least four cells are required to generate a poset with a {{0, 1}} cell"
    );
    let profiles = feasible_profiles(cell_count);
    let profile = profiles[rng.random_range(0..profiles.len())];
    generate_with_profile(profile, rng)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CellProfile {
    vertices: usize,
    edges_0: usize,
    edges_1: usize,
    faces: usize,
}

impl CellProfile {
    fn total(self) -> usize {
        self.vertices + self.edges_0 + self.edges_1 + self.faces
    }
}

fn feasible_profiles(cell_count: usize) -> Vec<CellProfile> {
    let mut profiles = Vec::new();

    for vertices in 0..=cell_count {
        let after_vertices = cell_count - vertices;
        for edges_0 in 0..=after_vertices {
            let after_edges_0 = after_vertices - edges_0;
            for edges_1 in 0..=after_edges_0 {
                let faces = after_edges_0 - edges_1;
                let edges_have_faces = edges_0 + edges_1 == 0 || vertices > 0;
                let faces_have_basis_faces = faces == 0 || (edges_0 > 0 && edges_1 > 0);

                if faces > 0 && edges_have_faces && faces_have_basis_faces {
                    profiles.push(CellProfile {
                        vertices,
                        edges_0,
                        edges_1,
                        faces,
                    });
                }
            }
        }
    }

    debug_assert!(!profiles.is_empty());
    profiles
}

fn generate_with_profile<R: Rng + ?Sized>(profile: CellProfile, rng: &mut R) -> FramedPoset {
    debug_assert!(profile_is_feasible(profile));

    if profile.total() == 0 {
        return FramedPoset::empty();
    }

    let edge_count = profile.edges_0 + profile.edges_1;
    let levels = if profile.faces > 0 {
        3
    } else if edge_count > 0 {
        2
    } else {
        1
    };

    let mut basis = Vec::with_capacity(levels);
    basis.push(vec![vec![]; profile.vertices]);
    if levels > 1 {
        let mut edge_basis = vec![vec![0]; profile.edges_0];
        edge_basis.extend(vec![vec![1]; profile.edges_1]);
        basis.push(edge_basis);
    }
    if levels > 2 {
        basis.push(vec![vec![0, 1]; profile.faces]);
    }

    let mut faces_in: Vec<Vec<IntSet>> = basis
        .iter()
        .map(|level| vec![vec![]; level.len()])
        .collect();
    let mut faces_out = faces_in.clone();

    for edge in 0..edge_count {
        let (input, output) = random_nonempty_signed_subset(profile.vertices, 0, rng);
        faces_in[1][edge] = input;
        faces_out[1][edge] = output;
    }

    for face in 0..profile.faces {
        let (mut input, mut output) = random_nonempty_signed_subset(profile.edges_0, 0, rng);
        let (input_1, output_1) =
            random_nonempty_signed_subset(profile.edges_1, profile.edges_0, rng);
        input.extend(input_1);
        output.extend(output_1);
        faces_in[2][face] = input;
        faces_out[2][face] = output;
    }

    let poset = FramedPoset::from_faces(basis, faces_in, faces_out);
    debug_assert!(poset.well_formed());
    poset
}

fn profile_is_feasible(profile: CellProfile) -> bool {
    profile.faces > 0 && profile.vertices > 0 && profile.edges_0 > 0 && profile.edges_1 > 0
}

fn random_nonempty_signed_subset<R: Rng + ?Sized>(
    size: usize,
    offset: usize,
    rng: &mut R,
) -> (IntSet, IntSet) {
    debug_assert!(size > 0);

    loop {
        let mut input = Vec::new();
        let mut output = Vec::new();

        for element in 0..size {
            match rng.random_range(0..3) {
                0 => {}
                1 => input.push(offset + element),
                2 => output.push(offset + element),
                _ => unreachable!(),
            }
        }

        if !input.is_empty() || !output.is_empty() {
            return (input, output);
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    #[test]
    fn generates_the_requested_number_of_cells() {
        let mut rng = SmallRng::seed_from_u64(0x0f_50_5e_75);

        for cell_count in 4..=20 {
            for _ in 0..64 {
                let poset = random_framed_poset(cell_count, &mut rng);
                assert_eq!(poset.sizes().iter().sum::<usize>(), cell_count);
                assert!(poset.sizes().get(2).is_some_and(|&size| size > 0));
                assert_eq!(poset.basis_of(2, 0), &vec![0, 1]);
                assert!(poset.well_formed());
            }
        }
    }

    #[test]
    #[should_panic(expected = "at least four cells are required")]
    fn rejects_cell_count_too_small_for_a_two_directional_cell() {
        let mut rng = SmallRng::seed_from_u64(0);
        random_framed_poset(3, &mut rng);
    }

    #[test]
    fn every_feasible_profile_generates_a_well_formed_poset() {
        let mut rng = SmallRng::seed_from_u64(0x0ba5_15c1_05ed);

        for cell_count in 4..=20 {
            for profile in feasible_profiles(cell_count) {
                assert!(profile.faces > 0);
                let poset = generate_with_profile(profile, &mut rng);
                assert_eq!(poset.sizes().iter().sum::<usize>(), cell_count);
                assert!(poset.well_formed());
            }
        }
    }

    #[test]
    fn generation_is_reproducible_from_the_rng_seed() {
        let mut first_rng = SmallRng::seed_from_u64(42);
        let mut second_rng = SmallRng::seed_from_u64(42);

        for cell_count in 4..=20 {
            let first = random_framed_poset(cell_count, &mut first_rng);
            let second = random_framed_poset(cell_count, &mut second_rng);
            assert!(FramedPoset::equal(&first, &second));
        }
    }
}
