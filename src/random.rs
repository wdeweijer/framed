//! Random generation of oriented framed posets.

use std::ops::Range;

use rand::Rng;
use rand::seq::index;

use crate::intset::IntSet;
use crate::poset::FramedPoset;

/// Reusable generator for finite oriented framed posets.
///
/// A generator with `dimension = d` uses precisely the directions
/// `0, ..., d - 1` and produces posets with exactly `cell_count` cells. The
/// constructor determines whether the full basis is present.
///
/// The Boolean lattice of bases and its cover relations are computed once by
/// [`Self::new`]. A generator is immutable and can be shared between threads.
#[derive(Debug)]
pub struct RandomFramedPosetGenerator {
    dimension: usize,
    cell_count: usize,
    basis_by_mask: Vec<IntSet>,
    face_masks_by_mask: Vec<Vec<usize>>,
    profile_mode: ProfileMode,
}

#[derive(Debug, Clone, Copy)]
enum ProfileMode {
    AllBases,
    ProperBases,
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
        let basis_count = basis_count(dimension);
        assert!(
            cell_count >= basis_count,
            "at least {basis_count} cells are required to generate a \
             {dimension}-dimensional poset"
        );

        Self::with_basis_count(dimension, cell_count, basis_count, ProfileMode::AllBases)
    }

    /// Prepare a generator whose frame has `frame_dimension` directions but
    /// whose cells all have proper subsets of the full frame as their bases.
    ///
    /// Every generated OFP has all `frame_dimension` directions active and
    /// has dimension exactly `frame_dimension - 1`. Its realized bases form a
    /// random downward-closed family. Thus every OFP with these dimensions and
    /// the requested number of cells has positive probability, up to the
    /// ordering of cells within each level.
    ///
    /// # Panics
    ///
    /// Panics if `frame_dimension` is less than two or cannot be represented
    /// by the internal bit masks, or if there are too few cells to contain one
    /// codimension-one basis, all of its subbases, and the remaining direction.
    pub fn new_without_full_basis(frame_dimension: usize, cell_count: usize) -> Self {
        assert!(frame_dimension >= 2, "frame dimension must be at least 2");
        let full_basis_count = basis_count(frame_dimension);
        let minimum_cell_count = (1usize << (frame_dimension - 1)) + 1;
        assert!(
            cell_count >= minimum_cell_count,
            "at least {minimum_cell_count} cells are required to generate a poset with frame \
             dimension {frame_dimension}, dimension {}, and no full-basis cell",
            frame_dimension - 1
        );

        Self::with_basis_count(
            frame_dimension,
            cell_count,
            full_basis_count - 1,
            ProfileMode::ProperBases,
        )
    }

    fn with_basis_count(
        dimension: usize,
        cell_count: usize,
        basis_count: usize,
        profile_mode: ProfileMode,
    ) -> Self {
        debug_assert!(
            dimension < usize::BITS as usize,
            "dimension must be smaller than the number of bits in usize"
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
            profile_mode,
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
    /// count, or sample a downward-closed proper-basis profile.
    fn random_profile<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<usize> {
        match self.profile_mode {
            ProfileMode::AllBases => self.random_all_bases_profile(rng),
            ProfileMode::ProperBases => self.random_proper_bases_profile(rng),
        }
    }

    /// Sample uniformly from positive compositions using stars and bars.
    fn random_all_bases_profile<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<usize> {
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

    fn random_proper_bases_profile<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<usize> {
        let mut profile = vec![0; self.basis_by_mask.len()];
        let full_mask = (1usize << self.dimension) - 1;
        let omitted_direction = rng.random_range(0..self.dimension);
        let primary_coatom = full_mask & !(1 << omitted_direction);

        // The primary coatom and its subbases ensure the requested dimension;
        // all singleton bases ensure that every frame direction is active.
        for (mask, count) in profile.iter_mut().enumerate() {
            if mask & !primary_coatom == 0 || mask.count_ones() == 1 {
                *count = 1;
            }
        }

        let mut selected_count = profile.iter().sum::<usize>();
        for mask in 1..profile.len() {
            if profile[mask] == 0
                && selected_count < self.cell_count
                && self.face_masks_by_mask[mask]
                    .iter()
                    .all(|&face_mask| profile[face_mask] != 0)
                && rng.random_bool(0.5)
            {
                profile[mask] = 1;
                selected_count += 1;
            }
        }

        let selected_masks: Vec<_> = profile
            .iter()
            .enumerate()
            .filter_map(|(mask, &count)| (count != 0).then_some(mask))
            .collect();
        for _ in selected_count..self.cell_count {
            let mask = selected_masks[rng.random_range(0..selected_masks.len())];
            profile[mask] += 1;
        }

        profile
    }

    fn generate_with_profile<R: Rng + ?Sized>(
        &self,
        profile: &[usize],
        rng: &mut R,
    ) -> FramedPoset {
        debug_assert_eq!(profile.len(), self.basis_by_mask.len());
        debug_assert_eq!(profile.iter().sum::<usize>(), self.cell_count);
        debug_assert!(profile.iter().enumerate().all(|(mask, &count)| {
            count == 0
                || self.face_masks_by_mask[mask]
                    .iter()
                    .all(|&face_mask| profile[face_mask] != 0)
        }));

        let level_count = self
            .basis_by_mask
            .iter()
            .map(Vec::len)
            .max()
            .map_or(0, |maximum| maximum + 1);
        let mut basis = vec![Vec::new(); level_count];
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

fn basis_count(dimension: usize) -> usize {
    assert!(
        dimension < usize::BITS as usize,
        "dimension must be smaller than the number of bits in usize"
    );
    1usize << dimension
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
    fn generates_well_formed_posets_without_full_basis() {
        let mut rng = SmallRng::seed_from_u64(0x0f_50_5e_75_00);

        for frame_dimension in 2..=5 {
            let minimum = (1usize << (frame_dimension - 1)) + 1;
            for cell_count in minimum..=minimum + 4 {
                let generator =
                    RandomFramedPosetGenerator::new_without_full_basis(frame_dimension, cell_count);

                for _ in 0..32 {
                    let poset = generator.generate(&mut rng);
                    assert_eq!(poset.sizes().iter().sum::<usize>(), cell_count);
                    assert_eq!(poset.dim(), frame_dimension as isize - 1);
                    assert_eq!(
                        poset.active_directions(),
                        (0..frame_dimension).collect::<Vec<_>>()
                    );
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
    #[should_panic(expected = "at least 9 cells are required")]
    fn rejects_too_few_cells_without_full_basis() {
        RandomFramedPosetGenerator::new_without_full_basis(4, 8);
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
