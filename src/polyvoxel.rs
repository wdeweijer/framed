//! Polyvoxels and their inductive constructors.
//!
//! [`point`] is the base value; [`shift`], [`cylinder`], and [`paste`] are the
//! three inductive constructions.

use std::ops::Deref;
use std::sync::Arc;

use crate::box_construction::elementary_cylinder;
use crate::embedding::Embedding;
use crate::poset::{FramedPoset, shift as shift_poset};
use crate::pushout::{Pushout, paste_along_boundary};

/// An oriented framed poset known to be a polyvoxel.
///
/// The wrapped shape is immutable, and the field is private so that public code
/// can obtain a `Polyvoxel` only from [`point`], [`shift`], [`cylinder`],
/// [`paste`], certified transport along [`Polyvoxel::from_isomorphism`], or by
/// cloning an existing value. These operations form the trusted kernel for
/// polyvoxelhood.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Polyvoxel {
    shape: Arc<FramedPoset>,
    length: Vec<usize>,
    layering_direction: Option<usize>,
}

impl Polyvoxel {
    /// Access the shared underlying oriented framed poset.
    pub fn as_framed_poset(&self) -> &Arc<FramedPoset> {
        &self.shape
    }

    /// The rank, equal to one more than the greatest active direction.
    ///
    /// The point has rank zero.
    pub fn rank(&self) -> usize {
        self.length.len()
    }

    /// The finite vector representing the length function below the rank.
    ///
    /// An entry is zero exactly when its index is not an active direction.
    pub fn length(&self) -> &[usize] {
        &self.length
    }

    /// The length in one direction, with directions at or above the rank
    /// implicitly having length zero.
    pub fn length_at(&self, direction: usize) -> usize {
        self.length.get(direction).copied().unwrap_or(0)
    }

    /// The least direction of length greater than one.
    ///
    /// `None` represents infinity and is equivalent to this polyvoxel being a
    /// voxel.
    pub fn layering_direction(&self) -> Option<usize> {
        self.layering_direction
    }

    /// Whether this polyvoxel is a voxel.
    pub fn is_voxel(&self) -> bool {
        self.layering_direction.is_none()
    }

    /// Regard an isomorphic OFP as a polyvoxel.
    ///
    /// `isomorphism` must have `shape` as its domain and `known_polyvoxel` as
    /// its codomain. Length and layering direction are intrinsic, so they are
    /// transported unchanged from `known_polyvoxel`.
    pub fn from_isomorphism(
        shape: Arc<FramedPoset>,
        isomorphism: &Embedding,
        known_polyvoxel: &Self,
    ) -> Self {
        assert!(
            FramedPoset::equal(&shape, &isomorphism.dom),
            "the supplied OFP must be the domain of the isomorphism",
        );
        assert!(
            FramedPoset::equal(known_polyvoxel.as_framed_poset(), &isomorphism.cod),
            "the known polyvoxel must be the codomain of the isomorphism",
        );
        assert!(
            isomorphism.is_isomorphism(),
            "the supplied embedding must be an isomorphism",
        );

        Self {
            shape,
            length: known_polyvoxel.length.clone(),
            layering_direction: known_polyvoxel.layering_direction,
        }
    }
}

impl Deref for Polyvoxel {
    type Target = FramedPoset;

    fn deref(&self) -> &Self::Target {
        &self.shape
    }
}

impl AsRef<FramedPoset> for Polyvoxel {
    fn as_ref(&self) -> &FramedPoset {
        self
    }
}

/// Construct the point polyvoxel.
pub fn point() -> Polyvoxel {
    Polyvoxel {
        shape: Arc::new(FramedPoset::point()),
        length: vec![],
        layering_direction: None,
    }
}

/// Shift every direction of a polyvoxel by one.
pub fn shift(shape: &Polyvoxel) -> Polyvoxel {
    let length = if shape.length.is_empty() {
        vec![]
    } else {
        std::iter::once(0)
            .chain(shape.length.iter().copied())
            .collect()
    };
    let layering_direction = shape.layering_direction.map(|direction| {
        direction
            .checked_add(1)
            .expect("cannot shift direction usize::MAX")
    });
    Polyvoxel {
        shape: Arc::new(shift_poset(shape)),
        length,
        layering_direction,
    }
}

/// Construct an elementary cylinder with a voxel as its input.
///
/// This checks that `input` has a greatest element, which for a polyvoxel is
/// equivalent to being a voxel.
pub fn cylinder(input: &Polyvoxel, output: &Polyvoxel) -> Polyvoxel {
    assert!(
        input.is_voxel(),
        "the elementary-cylinder input must be a voxel",
    );
    let shape = elementary_cylinder(input.as_framed_poset(), output.as_framed_poset());
    Polyvoxel {
        shape,
        length: std::iter::once(1)
            .chain(input.length.iter().copied())
            .collect(),
        layering_direction: None,
    }
}

/// Paste the output boundary of `left` to the input boundary of `right`.
///
/// The returned pushout contains the canonical embeddings of the operands. Its
/// tip and the returned polyvoxel share the same underlying framed poset.
/// These OFP embeddings are already morphisms of polyvoxels, so they need no
/// separate wrapper type.
pub fn paste(left: &Polyvoxel, right: &Polyvoxel, direction: usize) -> (Pushout, Polyvoxel) {
    let left_length = left.length_at(direction);
    let right_length = right.length_at(direction);
    assert!(
        left_length > 0 && right_length > 0,
        "pasting direction {direction} must be active in both polyvoxels",
    );

    let pushout = paste_along_boundary(left.as_framed_poset(), right.as_framed_poset(), direction);
    let mut length = left.length.clone();
    length[direction] = left_length
        .checked_add(right_length)
        .expect("polyvoxel length exceeds usize::MAX");
    let layering_direction = Some(
        left.layering_direction
            .map_or(direction, |layering| direction.min(layering)),
    );
    let polyvoxel = Polyvoxel {
        shape: Arc::clone(&pushout.tip),
        length,
        layering_direction,
    };
    (pushout, polyvoxel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poset::{polyvoxel_layering_direction, polyvoxel_length};
    use crate::random::randomly_permute;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn assert_cached_metadata_matches_shape(polyvoxel: &Polyvoxel) {
        assert_eq!(
            polyvoxel.length(),
            polyvoxel_length(polyvoxel.as_framed_poset())
        );
        assert_eq!(
            polyvoxel.layering_direction(),
            polyvoxel_layering_direction(polyvoxel.as_framed_poset())
        );

        let frame = polyvoxel.active_directions();
        let expected_rank = frame.last().map_or(0, |direction| direction + 1);
        assert_eq!(polyvoxel.rank(), expected_rank);
        for direction in 0..polyvoxel.rank() {
            assert_eq!(
                polyvoxel.length_at(direction) == 0,
                frame.binary_search(&direction).is_err()
            );
        }
        assert_eq!(polyvoxel.length_at(polyvoxel.rank() + 3), 0);
        assert_eq!(polyvoxel.is_voxel(), polyvoxel.greatest_element().is_some());
    }

    #[test]
    fn constructors_return_shared_polyvoxel_shapes() {
        let point = point();
        let arrow = cylinder(&point, &point);
        let shifted_arrow = shift(&arrow);
        let (path_pushout, path) = paste(&arrow, &arrow, 0);

        assert_eq!(point.sizes(), vec![1]);
        assert_eq!(arrow.sizes(), vec![2, 1]);
        assert_eq!(shifted_arrow.active_directions(), vec![1]);
        assert_eq!(path.sizes(), vec![3, 2]);
        assert!(path.greatest_element().is_none());
        assert!(Arc::ptr_eq(path.as_framed_poset(), &path_pushout.tip));
        assert!(Arc::ptr_eq(&path_pushout.tip, &path_pushout.inl.cod));
        assert!(Arc::ptr_eq(&path_pushout.tip, &path_pushout.inr.cod));
    }

    #[test]
    fn constructor_metadata_matches_the_underlying_poset_definition() {
        let point = point();
        let arrow = cylinder(&point, &point);
        let shifted_arrow = shift(&arrow);
        let square = cylinder(&arrow, &arrow);
        let gapped_square = cylinder(&shifted_arrow, &shifted_arrow);
        let (_, horizontal_path) = paste(&square, &square, 0);
        let (_, vertical_path) = paste(&square, &square, 1);
        let (_, shifted_path) = paste(&shifted_arrow, &shifted_arrow, 1);

        assert!(point.length().is_empty());
        assert_eq!(arrow.length(), &[1]);
        assert_eq!(shifted_arrow.length(), &[0, 1]);
        assert_eq!(square.length(), &[1, 1]);
        assert_eq!(gapped_square.length(), &[1, 0, 1]);
        assert_eq!(horizontal_path.length(), &[2, 1]);
        assert_eq!(horizontal_path.layering_direction(), Some(0));
        assert_eq!(vertical_path.length(), &[1, 2]);
        assert_eq!(vertical_path.layering_direction(), Some(1));
        assert_eq!(shifted_path.length(), &[0, 2]);
        assert_eq!(shifted_path.layering_direction(), Some(1));

        for polyvoxel in [
            &point,
            &arrow,
            &shifted_arrow,
            &square,
            &gapped_square,
            &horizontal_path,
            &vertical_path,
            &shifted_path,
        ] {
            assert_cached_metadata_matches_shape(polyvoxel);
        }
    }

    #[test]
    #[should_panic(expected = "the elementary-cylinder input must be a voxel")]
    fn cylinder_rejects_a_polyvoxel_without_a_greatest_element() {
        let point = point();
        let arrow = cylinder(&point, &point);
        let (_, path) = paste(&arrow, &arrow, 0);

        let _ = cylinder(&path, &path);
    }

    #[test]
    #[should_panic(expected = "pasting direction 0 must be active in both polyvoxels")]
    fn paste_rejects_an_inactive_direction() {
        let point = point();

        let _ = paste(&point, &point, 0);
    }

    #[test]
    fn isomorphism_transports_polyvoxel_structure_and_metadata() {
        let square = square_for_test();
        let (_, rectangle) = paste(&square, &square, 0);
        let mut rng = SmallRng::seed_from_u64(0x1500_40f0_5e70_0001);
        let (permuted_shape, into_rectangle) =
            randomly_permute(rectangle.as_framed_poset(), &mut rng);
        let expected_shape = Arc::clone(&permuted_shape);

        let permuted = Polyvoxel::from_isomorphism(permuted_shape, &into_rectangle, &rectangle);

        assert!(Arc::ptr_eq(permuted.as_framed_poset(), &expected_shape));
        assert_eq!(permuted.length(), rectangle.length());
        assert_eq!(
            permuted.layering_direction(),
            rectangle.layering_direction()
        );
    }

    #[test]
    #[should_panic(expected = "the supplied embedding must be an isomorphism")]
    fn isomorphism_transport_rejects_a_non_isomorphism() {
        let known_polyvoxel = point();
        let shape = Arc::new(FramedPoset::empty());
        let embedding = Embedding::empty(Arc::clone(known_polyvoxel.as_framed_poset()));

        let _ = Polyvoxel::from_isomorphism(shape, &embedding, &known_polyvoxel);
    }

    fn square_for_test() -> Polyvoxel {
        let point = point();
        let arrow = cylinder(&point, &point);
        cylinder(&arrow, &arrow)
    }
}
