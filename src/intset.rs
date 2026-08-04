//! Small sorted integer sets.

use std::error::Error;
use std::fmt;

/// A sorted, deduplicated vector of `usize`s.
pub type IntSet = Vec<usize>;

/// The second set contains no direction absent from the first set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoverDirectionError;

impl fmt::Display for CoverDirectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the second set contains no direction absent from the first"
        )
    }
}

impl Error for CoverDirectionError {}

/// Return true exactly when `xs` is sorted and contains no duplicates.
pub fn is_sorted_unique(xs: &[usize]) -> bool {
    xs.windows(2).all(|w| w[0] < w[1])
}

/// Insert `x`, preserving sorted deduplicated order.
pub fn insert(xs: &mut IntSet, x: usize) {
    match xs.binary_search(&x) {
        Ok(_) => {}
        Err(i) => xs.insert(i, x),
    }
}

/// Collect an iterator into a sorted, deduplicated set.
pub fn collect_sorted(iter: impl Iterator<Item = usize>) -> IntSet {
    let mut xs: Vec<usize> = iter.collect();
    xs.sort_unstable();
    xs.dedup();
    xs
}

/// Union of two sorted sets.
pub fn union(a: &IntSet, b: &IntSet) -> IntSet {
    use std::cmp::Ordering::*;

    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Less => {
                out.push(a[i]);
                i += 1;
            }
            Greater => {
                out.push(b[j]);
                j += 1;
            }
            Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// Return true when two sorted sets have no element in common.
pub fn is_disjoint(a: &[usize], b: &[usize]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return false,
        }
    }
    true
}

/// Return true if every element of `small` lies in `big`.
pub fn is_subset(small: &[usize], big: &[usize]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < small.len() && j < big.len() {
        if small[i] == big[j] {
            i += 1;
            j += 1;
        } else if small[i] > big[j] {
            j += 1;
        } else {
            return false;
        }
    }
    i == small.len()
}

/// Return the first element of `second` that is not in `first`.
///
/// Both sets must be sorted and deduplicated.
pub fn cover_direction(first: &[usize], second: &[usize]) -> Result<usize, CoverDirectionError> {
    let mut i = 0;

    for &candidate in second {
        while i < first.len() && first[i] < candidate {
            i += 1;
        }
        if i == first.len() || first[i] != candidate {
            return Ok(candidate);
        }
        i += 1;
    }

    Err(CoverDirectionError)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_sorted_deduplicates() {
        assert_eq!(collect_sorted([3, 1, 2, 1].into_iter()), vec![1, 2, 3]);
    }

    #[test]
    fn union_merges_sorted_sets() {
        assert_eq!(union(&vec![0, 2], &vec![1, 2, 4]), vec![0, 1, 2, 4]);
    }

    #[test]
    fn disjointness_uses_sorted_set_intersection() {
        assert!(is_disjoint(&[0, 2], &[1, 3]));
        assert!(is_disjoint(&[], &[0]));
        assert!(!is_disjoint(&[0, 2, 4], &[1, 2, 3]));
    }

    #[test]
    fn cover_direction_finds_the_first_new_element() {
        assert_eq!(cover_direction(&[0, 2, 4], &[0, 1, 2, 3, 4]), Ok(1));
        assert_eq!(cover_direction(&[1, 2], &[0, 1, 2]), Ok(0));
        assert_eq!(cover_direction(&[0, 1], &[0, 1, 5]), Ok(5));
    }

    #[test]
    fn cover_direction_errors_when_there_is_no_new_element() {
        assert_eq!(cover_direction(&[0, 1], &[0, 1]), Err(CoverDirectionError));
        assert_eq!(
            cover_direction(&[0, 1, 2], &[0, 2]),
            Err(CoverDirectionError)
        );
        assert_eq!(cover_direction(&[], &[]), Err(CoverDirectionError));
    }
}
