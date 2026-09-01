//! Polyvoxels and their inductive constructors.
//!
//! [`point`] is the base value; [`shift`], [`cylinder`], and [`paste`] are the
//! three inductive constructions.

use std::ops::Deref;
use std::sync::Arc;

use crate::box_construction::elementary_cylinder;
use crate::poset::{FramedPoset, shift as shift_poset};
use crate::pushout::{Pushout, paste_along_boundary};

/// An oriented framed poset known to be a polyvoxel.
///
/// The wrapped shape is immutable, and the field is private so that public code
/// can obtain a `Polyvoxel` only from [`point`], [`shift`], [`cylinder`],
/// [`paste`], or by cloning an existing value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Polyvoxel {
    shape: Arc<FramedPoset>,
}

impl Polyvoxel {
    /// Access the shared underlying oriented framed poset.
    pub fn as_framed_poset(&self) -> &Arc<FramedPoset> {
        &self.shape
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
    }
}

/// Shift every direction of a polyvoxel by one.
pub fn shift(shape: &Polyvoxel) -> Polyvoxel {
    Polyvoxel {
        shape: Arc::new(shift_poset(shape)),
    }
}

/// Construct an elementary cylinder with a voxel as its input.
///
/// This checks that `input` has a greatest element, which for a polyvoxel is
/// equivalent to being a voxel.
pub fn cylinder(input: &Polyvoxel, output: &Polyvoxel) -> Polyvoxel {
    assert!(
        input.greatest_element().is_some(),
        "the elementary-cylinder input must be a voxel",
    );
    Polyvoxel {
        shape: elementary_cylinder(input.as_framed_poset(), output.as_framed_poset()),
    }
}

/// Paste the output boundary of `left` to the input boundary of `right`.
///
/// The returned pushout contains the canonical embeddings of the operands. Its
/// tip and the returned polyvoxel share the same underlying framed poset.
/// These OFP embeddings are already morphisms of polyvoxels, so they need no
/// separate wrapper type.
pub fn paste(left: &Polyvoxel, right: &Polyvoxel, direction: usize) -> (Pushout, Polyvoxel) {
    let pushout = paste_along_boundary(left.as_framed_poset(), right.as_framed_poset(), direction);
    let polyvoxel = Polyvoxel {
        shape: Arc::clone(&pushout.tip),
    };
    (pushout, polyvoxel)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    #[should_panic(expected = "the elementary-cylinder input must be a voxel")]
    fn cylinder_rejects_a_polyvoxel_without_a_greatest_element() {
        let point = point();
        let arrow = cylinder(&point, &point);
        let (_, path) = paste(&arrow, &arrow, 0);

        let _ = cylinder(&path, &path);
    }
}
