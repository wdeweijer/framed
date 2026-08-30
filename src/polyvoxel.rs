//! Convenient constructors for polyvoxels.
//!
//! Except for [`point`], these functions assume that every input shape is
//! already known to be a polyvoxel.

use std::sync::Arc;

use crate::box_construction::elementary_cylinder;
use crate::poset::{FramedPoset, shift as shift_poset};
use crate::pushout::{Pushout, paste_along_boundary};

/// The point, as a shared shape suitable for further polyvoxel construction.
pub fn point() -> Arc<FramedPoset> {
    Arc::new(FramedPoset::point())
}

/// Shift every direction of a polyvoxel by one and return the result as a
/// shared shape.
///
/// This assumes that `shape` is a polyvoxel.
pub fn shift(shape: &Arc<FramedPoset>) -> Arc<FramedPoset> {
    Arc::new(shift_poset(shape))
}

/// Construct an elementary cylinder with a voxel as its input.
///
/// This assumes that both `input` and `output` are polyvoxels.
///
/// This checks that `input` has a greatest element. For a shape already known
/// to be a polyvoxel, this is equivalent to being a voxel.
pub fn cylinder(input: &Arc<FramedPoset>, output: &Arc<FramedPoset>) -> Arc<FramedPoset> {
    assert!(
        input.greatest_element().is_some(),
        "the elementary-cylinder input must be a voxel",
    );
    elementary_cylinder(input, output)
}

/// Paste the output boundary of `left` to the input boundary of `right`.
///
/// This assumes that both `left` and `right` are polyvoxels.
///
/// The returned pushout retains both canonical embeddings into the result.
pub fn paste(left: &Arc<FramedPoset>, right: &Arc<FramedPoset>, direction: usize) -> Pushout {
    paste_along_boundary(left, right, direction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_return_shared_polyvoxel_shapes() {
        let point = point();
        let arrow = cylinder(&point, &point);
        let shifted_arrow = shift(&arrow);
        let path = paste(&arrow, &arrow, 0);

        assert_eq!(point.sizes(), vec![1]);
        assert_eq!(arrow.sizes(), vec![2, 1]);
        assert_eq!(shifted_arrow.active_directions(), vec![1]);
        assert_eq!(path.tip.sizes(), vec![3, 2]);
        assert!(path.tip.greatest_element().is_none());
    }

    #[test]
    #[should_panic(expected = "the elementary-cylinder input must be a voxel")]
    fn cylinder_rejects_a_polyvoxel_without_a_greatest_element() {
        let point = point();
        let arrow = cylinder(&point, &point);
        let path = paste(&arrow, &arrow, 0).tip;

        let _ = cylinder(&path, &path);
    }
}
