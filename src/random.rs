//! Random generation of oriented framed posets.

use std::ops::Range;

use rand::Rng;
use rand::seq::index;

use crate::intset::IntSet;
use crate::poset::FramedPoset;

/// Reusable generator for finite oriented framed posets.
///
/// A generator with `dimension = d` uses precisely the directions
/// `0, ..., d - 1`. Every generated poset has exactly `cell_count` cells and
/// at least one cell for each of the `2^d` possible bases. Consequently it
/// has at least one cell with full basis `{0, ..., d - 1}`.
///
/// The Boolean lattice of bases and its cover relations are computed once by
/// [`Self::new`]. A generator is immutable and can be shared between threads.
#[derive(Debug)]
pub struct RandomFramedPosetGenerator {
    dimension: usize,
    cell_count: usize,
    basis_by_mask: Vec<IntSet>,
    face_masks_by_mask: Vec<Vec<usize>>,
}

impl RandomFramedPosetGenerator {
    /// Prepare a generator of `dimension`-dimensional, `cell_count`-cell OFPs.
    ///
    /// Every OFP with these active directions, at least one full-dimensional
    /// cell, and the requested number of cells has positive probability, up
    /// to the ordering of cells within each level.
    ///
    /// # Panics
    ///
    /// Panics if `dimension` cannot be represented by the internal bit masks,
    /// or if fewer than `2^dimension` cells are requested.
    pub fn new(dimension: usize, cell_count: usize) -> Self {
        assert!(
            dimension < usize::BITS as usize,
            "dimension must be smaller than the number of bits in usize"
        );

        let basis_count = 1usize << dimension;
        assert!(
            cell_count >= basis_count,
            "at least {basis_count} cells are required to generate a \
             {dimension}-dimensional poset"
        );

        let basis_by_mask: Vec<_> = (0..basis_count)
            .map(|mask| basis_from_mask(mask, dimension))
            .collect();
        let face_masks_by_mask = (0..basis_count)
            .map(|mask| {
                (0..dimension)
                    .filter(|&direction| mask & (1 << direction) != 0)
                    .map(|direction| mask & !(1 << direction))
                    .collect()
            })
            .collect();

        Self {
            dimension,
            cell_count,
            basis_by_mask,
            face_masks_by_mask,
        }
    }

    /// Number of active directions in every generated OFP.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of cells in every generated OFP.
    pub fn cell_count(&self) -> usize {
        self.cell_count
    }

    /// Generate one random well-formed OFP.
    pub fn generate<R: Rng + ?Sized>(&self, rng: &mut R) -> FramedPoset {
        let profile = self.random_profile(rng);
        self.generate_with_profile(&profile, rng)
    }

    /// Sample a profile uniformly from the positive compositions of the cell
    /// count, using the standard stars-and-bars bijection.
    fn random_profile<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<usize> {
        let basis_count = self.basis_by_mask.len();
        let mut cuts = index::sample(rng, self.cell_count - 1, basis_count - 1).into_vec();
        for cut in &mut cuts {
            *cut += 1;
        }
        cuts.sort_unstable();

        let mut profile = Vec::with_capacity(basis_count);
        let mut previous = 0;
        for cut in cuts.into_iter().chain(std::iter::once(self.cell_count)) {
            profile.push(cut - previous);
            previous = cut;
        }
        profile
    }

    fn generate_with_profile<R: Rng + ?Sized>(
        &self,
        profile: &[usize],
        rng: &mut R,
    ) -> FramedPoset {
        debug_assert_eq!(profile.len(), self.basis_by_mask.len());
        debug_assert!(profile.iter().all(|&count| count > 0));
        debug_assert_eq!(profile.iter().sum::<usize>(), self.cell_count);

        let mut basis = vec![Vec::new(); self.dimension + 1];
        let mut ranges: Vec<Range<usize>> = vec![0..0; self.basis_by_mask.len()];

        for (mask, cell_basis) in self.basis_by_mask.iter().enumerate() {
            let level = cell_basis.len();
            let start = basis[level].len();
            basis[level].extend((0..profile[mask]).map(|_| cell_basis.clone()));
            ranges[mask] = start..basis[level].len();
        }

        let mut faces_in: Vec<Vec<IntSet>> = basis
            .iter()
            .map(|level| vec![vec![]; level.len()])
            .collect();
        let mut faces_out = faces_in.clone();

        for mask in 1..self.basis_by_mask.len() {
            let level = self.basis_by_mask[mask].len();
            for pos in ranges[mask].clone() {
                for &face_mask in &self.face_masks_by_mask[mask] {
                    let face_range = ranges[face_mask].clone();
                    let (input, output) =
                        random_nonempty_signed_subset(face_range.len(), face_range.start, rng);
                    faces_in[level][pos].extend(input);
                    faces_out[level][pos].extend(output);
                }
                faces_in[level][pos].sort_unstable();
                faces_out[level][pos].sort_unstable();
            }
        }

        let poset = FramedPoset::from_faces(basis, faces_in, faces_out);
        debug_assert_eq!(poset.sizes().iter().sum::<usize>(), self.cell_count);
        debug_assert!(poset.well_formed());
        poset
    }
}

fn basis_from_mask(mask: usize, dimension: usize) -> IntSet {
    (0..dimension)
        .filter(|&direction| mask & (1 << direction) != 0)
        .collect()
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
    fn generates_well_formed_posets_in_arbitrary_small_dimensions() {
        let mut rng = SmallRng::seed_from_u64(0x0f_50_5e_75);

        for dimension in 0..=4 {
            let minimum = 1usize << dimension;
            for cell_count in minimum..=minimum + 4 {
                let generator = RandomFramedPosetGenerator::new(dimension, cell_count);

                for _ in 0..32 {
                    let poset = generator.generate(&mut rng);
                    assert_eq!(poset.sizes().iter().sum::<usize>(), cell_count);
                    assert_eq!(poset.dim(), dimension as isize);
                    assert_eq!(
                        poset.active_directions(),
                        (0..dimension).collect::<Vec<_>>()
                    );
                    assert!(all_bases_occur(&poset, dimension));
                    assert!(poset.well_formed());
                }
            }
        }
    }

    #[test]
    fn minimum_profile_has_one_cell_per_basis() {
        let mut rng = SmallRng::seed_from_u64(0x08);

        for dimension in 0..=4 {
            let basis_count = 1usize << dimension;
            let generator = RandomFramedPosetGenerator::new(dimension, basis_count);
            let poset = generator.generate(&mut rng);
            let mut expected_sizes = vec![0; dimension + 1];
            for mask in 0usize..basis_count {
                expected_sizes[mask.count_ones() as usize] += 1;
            }

            assert_eq!(poset.sizes(), expected_sizes);
            assert!(all_bases_occur(&poset, dimension));
            assert!(poset.well_formed());
        }
    }

    #[test]
    #[should_panic(expected = "at least 8 cells are required")]
    fn rejects_too_few_cells() {
        RandomFramedPosetGenerator::new(3, 7);
    }

    #[test]
    fn every_small_feasible_profile_generates_a_well_formed_poset() {
        let mut rng = SmallRng::seed_from_u64(0x03_ba_51_51);

        for dimension in 0..=3 {
            let basis_count = 1usize << dimension;
            for cell_count in basis_count..=basis_count + 3 {
                let generator = RandomFramedPosetGenerator::new(dimension, cell_count);
                for profile in feasible_profiles_for_test(basis_count, cell_count) {
                    let poset = generator.generate_with_profile(&profile, &mut rng);
                    assert_eq!(poset.sizes().iter().sum::<usize>(), cell_count);
                    assert!(all_bases_occur(&poset, dimension));
                    assert!(poset.well_formed());
                }
            }
        }
    }

    #[test]
    fn generation_is_reproducible_from_the_rng_seed() {
        let first_generator = RandomFramedPosetGenerator::new(4, 20);
        let second_generator = RandomFramedPosetGenerator::new(4, 20);
        let mut first_rng = SmallRng::seed_from_u64(42);
        let mut second_rng = SmallRng::seed_from_u64(42);

        for _ in 0..64 {
            let first = first_generator.generate(&mut first_rng);
            let second = second_generator.generate(&mut second_rng);
            assert!(FramedPoset::equal(&first, &second));
        }
    }

    #[test]
    fn generator_reports_its_configuration() {
        let generator = RandomFramedPosetGenerator::new(3, 12);

        assert_eq!(generator.dimension(), 3);
        assert_eq!(generator.cell_count(), 12);
    }

    fn all_bases_occur(poset: &FramedPoset, dimension: usize) -> bool {
        (0usize..1usize << dimension).all(|mask| {
            let expected = basis_from_mask(mask, dimension);
            let level = expected.len();
            (0..poset.sizes()[level]).any(|pos| poset.basis_of(level, pos) == &expected)
        })
    }

    fn feasible_profiles_for_test(basis_count: usize, cell_count: usize) -> Vec<Vec<usize>> {
        let mut profiles = Vec::new();
        let mut profile = vec![0; basis_count];
        collect_profiles(cell_count, 0, &mut profile, &mut profiles);
        profiles
    }

    fn collect_profiles(
        remaining: usize,
        basis: usize,
        profile: &mut [usize],
        profiles: &mut Vec<Vec<usize>>,
    ) {
        let remaining_bases = profile.len() - basis;
        if remaining_bases == 1 {
            if remaining > 0 {
                profile[basis] = remaining;
                profiles.push(profile.to_vec());
            }
            return;
        }

        let maximum = remaining.saturating_sub(remaining_bases - 1);
        for count in 1..=maximum {
            profile[basis] = count;
            collect_profiles(remaining - count, basis + 1, profile, profiles);
        }
    }
}
