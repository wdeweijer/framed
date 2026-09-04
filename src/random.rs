//! Random generation of oriented framed posets.

use std::ops::Range;
use std::sync::Arc;

use rand::Rng;
use rand::seq::SliceRandom;
use rand::seq::index;

use crate::embedding::Embedding;
use crate::intset::{self, IntSet};
use crate::poset::{FramedPoset, Sign};

/// Randomly reorder the cells at every level of a framed poset.
///
/// The returned embedding is the relabelling isomorphism from the permuted OFP
/// to `shape`; its forward map records the old position of every new cell.
pub fn randomly_permute<R: Rng + ?Sized>(
    shape: &Arc<FramedPoset>,
    rng: &mut R,
) -> (Arc<FramedPoset>, Embedding) {
    let sizes = shape.sizes();
    let new_to_old: Vec<Vec<usize>> = sizes
        .iter()
        .map(|&size| {
            let mut level: Vec<_> = (0..size).collect();
            level.shuffle(rng);
            level
        })
        .collect();

    let mut old_to_new: Vec<Vec<usize>> = sizes.iter().map(|&size| vec![0; size]).collect();
    for (dim, level) in new_to_old.iter().enumerate() {
        for (new_pos, &old_pos) in level.iter().enumerate() {
            old_to_new[dim][old_pos] = new_pos;
        }
    }

    let mut frames = Vec::with_capacity(sizes.len());
    let mut faces_in = Vec::with_capacity(sizes.len());
    let mut faces_out = Vec::with_capacity(sizes.len());
    for (dim, level) in new_to_old.iter().enumerate() {
        frames.push(
            level
                .iter()
                .map(|&old_pos| shape.frame_of(dim, old_pos).clone())
                .collect(),
        );
        if dim == 0 {
            faces_in.push(vec![vec![]; level.len()]);
            faces_out.push(vec![vec![]; level.len()]);
            continue;
        }

        let remap_faces = |sign| {
            level
                .iter()
                .map(|&old_pos| {
                    intset::collect_sorted(
                        shape
                            .faces_of(sign, dim, old_pos)
                            .iter()
                            .map(|&old_face| old_to_new[dim - 1][old_face]),
                    )
                })
                .collect()
        };
        faces_in.push(remap_faces(Sign::Input));
        faces_out.push(remap_faces(Sign::Output));
    }

    let permuted = Arc::new(FramedPoset::from_faces(frames, faces_in, faces_out));
    let isomorphism = Embedding::from_map(Arc::clone(&permuted), Arc::clone(shape), new_to_old);
    debug_assert!(isomorphism.is_isomorphism());
    (permuted, isomorphism)
}

/// Reusable generator for finite oriented framed posets.
///
/// A generator with total-frame size `d` uses precisely the directions
/// `0, ..., d - 1` and produces posets with exactly `cell_count` cells. Its
/// constructor determines whether the full frame is present.
///
/// The Boolean lattice of frames and its cover relations are computed once by
/// [`Self::new`]. A generator is immutable and can be shared between threads.
#[derive(Debug)]
pub struct RandomFramedPosetGenerator {
    total_frame_size: usize,
    cell_count: usize,
    frame_by_mask: Vec<IntSet>,
    face_masks_by_mask: Vec<Vec<usize>>,
    profile_mode: ProfileMode,
}

#[derive(Debug, Clone, Copy)]
enum ProfileMode {
    AllFrames,
    ProperFrames,
}

impl RandomFramedPosetGenerator {
    /// Prepare a generator of `dimension`-dimensional, `cell_count`-cell OFPs.
    ///
    /// Every OFP with this total frame, at least one full-dimensional
    /// cell, and the requested number of cells has positive probability, up
    /// to the ordering of cells within each level.
    ///
    /// # Panics
    ///
    /// Panics if `dimension` cannot be represented by the internal bit masks,
    /// or if fewer than `2^dimension` cells are requested.
    pub fn new(dimension: usize, cell_count: usize) -> Self {
        let frame_count = frame_count(dimension);
        assert!(
            cell_count >= frame_count,
            "at least {frame_count} cells are required to generate a \
             {dimension}-dimensional poset"
        );

        Self::with_frame_count(dimension, cell_count, frame_count, ProfileMode::AllFrames)
    }

    /// Prepare a generator whose total frame has `total_frame_size` directions
    /// but whose cells all have proper subsets of the total frame as their
    /// frames.
    ///
    /// Every generated OFP has total frame `0, ..., total_frame_size - 1` and
    /// has dimension exactly `total_frame_size - 1`. Its realized frames form a
    /// random downward-closed family. Thus every OFP with these dimensions and
    /// the requested number of cells has positive probability, up to the
    /// ordering of cells within each level.
    ///
    /// # Panics
    ///
    /// Panics if `total_frame_size` is less than two or cannot be represented
    /// by the internal bit masks, or if there are too few cells to contain one
    /// codimension-one frame, all of its subframes, and the remaining direction.
    pub fn new_without_full_frame(total_frame_size: usize, cell_count: usize) -> Self {
        assert!(total_frame_size >= 2, "total frame size must be at least 2");
        let full_frame_count = frame_count(total_frame_size);
        let minimum_cell_count = (1usize << (total_frame_size - 1)) + 1;
        assert!(
            cell_count >= minimum_cell_count,
            "at least {minimum_cell_count} cells are required to generate a poset with total \
             frame size {total_frame_size}, dimension {}, and no full-frame cell",
            total_frame_size - 1
        );

        Self::with_frame_count(
            total_frame_size,
            cell_count,
            full_frame_count - 1,
            ProfileMode::ProperFrames,
        )
    }

    fn with_frame_count(
        total_frame_size: usize,
        cell_count: usize,
        frame_count: usize,
        profile_mode: ProfileMode,
    ) -> Self {
        debug_assert!(
            total_frame_size < usize::BITS as usize,
            "total frame size must be smaller than the number of bits in usize"
        );

        let frame_by_mask: Vec<_> = (0..frame_count)
            .map(|mask| frame_from_mask(mask, total_frame_size))
            .collect();
        let face_masks_by_mask = (0..frame_count)
            .map(|mask| {
                (0..total_frame_size)
                    .filter(|&direction| mask & (1 << direction) != 0)
                    .map(|direction| mask & !(1 << direction))
                    .collect()
            })
            .collect();

        Self {
            total_frame_size,
            cell_count,
            frame_by_mask,
            face_masks_by_mask,
            profile_mode,
        }
    }

    /// Cardinality of the total frame of every generated OFP.
    pub fn total_frame_size(&self) -> usize {
        self.total_frame_size
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
    /// count, or sample a downward-closed proper-frame profile.
    fn random_profile<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<usize> {
        match self.profile_mode {
            ProfileMode::AllFrames => self.random_all_frames_profile(rng),
            ProfileMode::ProperFrames => self.random_proper_frames_profile(rng),
        }
    }

    /// Sample uniformly from positive compositions using stars and bars.
    fn random_all_frames_profile<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<usize> {
        let frame_count = self.frame_by_mask.len();
        let mut cuts = index::sample(rng, self.cell_count - 1, frame_count - 1).into_vec();
        for cut in &mut cuts {
            *cut += 1;
        }
        cuts.sort_unstable();

        let mut profile = Vec::with_capacity(frame_count);
        let mut previous = 0;
        for cut in cuts.into_iter().chain(std::iter::once(self.cell_count)) {
            profile.push(cut - previous);
            previous = cut;
        }
        profile
    }

    fn random_proper_frames_profile<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<usize> {
        let mut profile = vec![0; self.frame_by_mask.len()];
        let full_mask = (1usize << self.total_frame_size) - 1;
        let omitted_direction = rng.random_range(0..self.total_frame_size);
        let primary_coatom = full_mask & !(1 << omitted_direction);

        // The primary coatom and its subframes ensure the requested dimension;
        // all singleton frames ensure the requested total frame.
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
        debug_assert_eq!(profile.len(), self.frame_by_mask.len());
        debug_assert_eq!(profile.iter().sum::<usize>(), self.cell_count);
        debug_assert!(profile.iter().enumerate().all(|(mask, &count)| {
            count == 0
                || self.face_masks_by_mask[mask]
                    .iter()
                    .all(|&face_mask| profile[face_mask] != 0)
        }));

        let level_count = self
            .frame_by_mask
            .iter()
            .map(Vec::len)
            .max()
            .map_or(0, |maximum| maximum + 1);
        let mut frames = vec![Vec::new(); level_count];
        let mut ranges: Vec<Range<usize>> = vec![0..0; self.frame_by_mask.len()];

        for (mask, cell_frame) in self.frame_by_mask.iter().enumerate() {
            let level = cell_frame.len();
            let start = frames[level].len();
            frames[level].extend((0..profile[mask]).map(|_| cell_frame.clone()));
            ranges[mask] = start..frames[level].len();
        }

        let mut faces_in: Vec<Vec<IntSet>> = frames
            .iter()
            .map(|level| vec![vec![]; level.len()])
            .collect();
        let mut faces_out = faces_in.clone();

        for mask in 1..self.frame_by_mask.len() {
            let level = self.frame_by_mask[mask].len();
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

        let poset = FramedPoset::from_faces(frames, faces_in, faces_out);
        debug_assert_eq!(poset.sizes().iter().sum::<usize>(), self.cell_count);
        debug_assert!(poset.well_formed());
        poset
    }
}

fn frame_count(total_frame_size: usize) -> usize {
    assert!(
        total_frame_size < usize::BITS as usize,
        "total frame size must be smaller than the number of bits in usize"
    );
    1usize << total_frame_size
}

fn frame_from_mask(mask: usize, total_frame_size: usize) -> IntSet {
    (0..total_frame_size)
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
    fn random_permutation_returns_an_explicit_isomorphism() {
        let arrow = Arc::new(FramedPoset::from_faces(
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![1]]],
        ));
        let mut rng = SmallRng::seed_from_u64(0xce_11_0a_de);

        let (permuted, into_arrow) = randomly_permute(&arrow, &mut rng);

        assert!(into_arrow.is_isomorphism());
        assert!(Arc::ptr_eq(&permuted, &into_arrow.dom));
        assert!(Arc::ptr_eq(&arrow, &into_arrow.cod));
    }

    #[test]
    fn random_permutation_is_the_identity_when_no_reordering_is_possible() {
        let point = Arc::new(FramedPoset::point());
        let mut rng = SmallRng::seed_from_u64(0x0009_01a7);

        let (permuted, into_point) = randomly_permute(&point, &mut rng);

        assert!(FramedPoset::equal(&point, &permuted));
        assert_eq!(into_point.map, vec![vec![0]]);
        assert!(into_point.is_isomorphism());
    }

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
                    assert_eq!(poset.total_frame(), (0..dimension).collect::<Vec<_>>());
                    assert!(all_frames_occur(&poset, dimension));
                    assert!(poset.well_formed());
                }
            }
        }
    }

    #[test]
    fn generates_well_formed_posets_without_full_frame() {
        let mut rng = SmallRng::seed_from_u64(0x0f_50_5e_75_00);

        for total_frame_size in 2..=5 {
            let minimum = (1usize << (total_frame_size - 1)) + 1;
            for cell_count in minimum..=minimum + 4 {
                let generator = RandomFramedPosetGenerator::new_without_full_frame(
                    total_frame_size,
                    cell_count,
                );

                for _ in 0..32 {
                    let poset = generator.generate(&mut rng);
                    assert_eq!(poset.sizes().iter().sum::<usize>(), cell_count);
                    assert_eq!(poset.dim(), total_frame_size as isize - 1);
                    assert_eq!(
                        poset.total_frame(),
                        (0..total_frame_size).collect::<Vec<_>>()
                    );
                    assert!(poset.well_formed());
                }
            }
        }
    }

    #[test]
    fn minimum_profile_has_one_cell_per_frame() {
        let mut rng = SmallRng::seed_from_u64(0x08);

        for dimension in 0..=4 {
            let frame_count = 1usize << dimension;
            let generator = RandomFramedPosetGenerator::new(dimension, frame_count);
            let poset = generator.generate(&mut rng);
            let mut expected_sizes = vec![0; dimension + 1];
            for mask in 0usize..frame_count {
                expected_sizes[mask.count_ones() as usize] += 1;
            }

            assert_eq!(poset.sizes(), expected_sizes);
            assert!(all_frames_occur(&poset, dimension));
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
    fn rejects_too_few_cells_without_full_frame() {
        RandomFramedPosetGenerator::new_without_full_frame(4, 8);
    }

    #[test]
    fn every_small_feasible_profile_generates_a_well_formed_poset() {
        let mut rng = SmallRng::seed_from_u64(0x03_ba_51_51);

        for dimension in 0..=3 {
            let frame_count = 1usize << dimension;
            for cell_count in frame_count..=frame_count + 3 {
                let generator = RandomFramedPosetGenerator::new(dimension, cell_count);
                for profile in feasible_profiles_for_test(frame_count, cell_count) {
                    let poset = generator.generate_with_profile(&profile, &mut rng);
                    assert_eq!(poset.sizes().iter().sum::<usize>(), cell_count);
                    assert!(all_frames_occur(&poset, dimension));
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

        assert_eq!(generator.total_frame_size(), 3);
        assert_eq!(generator.cell_count(), 12);
    }

    fn all_frames_occur(poset: &FramedPoset, dimension: usize) -> bool {
        (0usize..1usize << dimension).all(|mask| {
            let expected = frame_from_mask(mask, dimension);
            let level = expected.len();
            (0..poset.sizes()[level]).any(|pos| poset.frame_of(level, pos) == &expected)
        })
    }

    fn feasible_profiles_for_test(frame_count: usize, cell_count: usize) -> Vec<Vec<usize>> {
        let mut profiles = Vec::new();
        let mut profile = vec![0; frame_count];
        collect_profiles(cell_count, 0, &mut profile, &mut profiles);
        profiles
    }

    fn collect_profiles(
        remaining: usize,
        frame_index: usize,
        profile: &mut [usize],
        profiles: &mut Vec<Vec<usize>>,
    ) {
        let remaining_frames = profile.len() - frame_index;
        if remaining_frames == 1 {
            if remaining > 0 {
                profile[frame_index] = remaining;
                profiles.push(profile.to_vec());
            }
            return;
        }

        let maximum = remaining.saturating_sub(remaining_frames - 1);
        for count in 1..=maximum {
            profile[frame_index] = count;
            collect_profiles(remaining - count, frame_index + 1, profile, profiles);
        }
    }
}
