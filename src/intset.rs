//! Small sorted integer sets.

/// A sorted, deduplicated vector of `usize`s.
pub type IntSet = Vec<usize>;

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
}
