//! Oriented framed posets.

use std::sync::Arc;

use crate::embedding::{Embedding, NO_PREIMAGE};
use crate::intset::{self, IntSet};

/// Input/output polarity for oriented cover relations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sign {
    Input,
    Output,
}

impl Sign {
    fn opposite(self) -> Self {
        match self {
            Self::Input => Self::Output,
            Self::Output => Self::Input,
        }
    }
}

/// A finite oriented framed poset.
///
/// Cells are grouped by the cardinality of their basis.  For a cell `p` at
/// level `d`, `basis[d][p]` is a finite set of directions of cardinality `d`.
#[derive(Debug, Clone)]
pub struct FramedPoset {
    pub(crate) dim: isize,
    pub(crate) basis: Vec<Vec<IntSet>>,
    pub(crate) faces_in: Vec<Vec<IntSet>>,
    pub(crate) faces_out: Vec<Vec<IntSet>>,
    pub(crate) cofaces_in: Vec<Vec<IntSet>>,
    pub(crate) cofaces_out: Vec<Vec<IntSet>>,
}

impl FramedPoset {
    /// Construct a framed poset from basis and adjacency tables.
    pub fn make(
        basis: Vec<Vec<IntSet>>,
        faces_in: Vec<Vec<IntSet>>,
        faces_out: Vec<Vec<IntSet>>,
        cofaces_in: Vec<Vec<IntSet>>,
        cofaces_out: Vec<Vec<IntSet>>,
    ) -> Self {
        let dim = basis.len() as isize - 1;
        let poset = Self {
            dim,
            basis,
            faces_in,
            faces_out,
            cofaces_in,
            cofaces_out,
        };
        debug_assert!(poset.well_formed());
        poset
    }

    /// Construct a framed poset from basis and signed face tables.
    ///
    /// The signed coface tables are generated as the reverse adjacency of the
    /// supplied signed face tables.
    pub fn from_faces(
        basis: Vec<Vec<IntSet>>,
        faces_in: Vec<Vec<IntSet>>,
        faces_out: Vec<Vec<IntSet>>,
    ) -> Self {
        let sizes: Vec<usize> = basis.iter().map(Vec::len).collect();
        let mut cofaces_in: Vec<Vec<IntSet>> = sizes.iter().map(|&n| vec![vec![]; n]).collect();
        let mut cofaces_out: Vec<Vec<IntSet>> = sizes.iter().map(|&n| vec![vec![]; n]).collect();

        for dim in 1..basis.len() {
            for pos in 0..basis[dim].len() {
                for &face in &faces_in[dim][pos] {
                    intset::insert(&mut cofaces_in[dim - 1][face], pos);
                }
                for &face in &faces_out[dim][pos] {
                    intset::insert(&mut cofaces_out[dim - 1][face], pos);
                }
            }
        }

        FramedPoset::make(basis, faces_in, faces_out, cofaces_in, cofaces_out)
    }

    /// Empty framed poset.
    pub fn empty() -> Self {
        FramedPoset::from_faces(vec![], vec![], vec![])
    }

    /// The point: one 0-cell with empty basis.
    pub fn point() -> Self {
        FramedPoset::from_faces(vec![vec![vec![]]], vec![vec![vec![]]], vec![vec![vec![]]])
    }

    /// Highest basis cardinality present, or `-1` for empty.
    pub fn dim(&self) -> isize {
        self.dim
    }

    /// Number of cells at each basis cardinality.
    pub fn sizes(&self) -> Vec<usize> {
        self.basis.iter().map(Vec::len).collect()
    }

    /// Basis of a cell.
    pub fn basis_of(&self, dim: usize, pos: usize) -> &IntSet {
        &self.basis[dim][pos]
    }

    /// Signed faces of a cell.
    pub fn faces_of(&self, sign: Sign, dim: usize, pos: usize) -> &IntSet {
        match sign {
            Sign::Input => &self.faces_in[dim][pos],
            Sign::Output => &self.faces_out[dim][pos],
        }
    }

    /// Signed cofaces of a cell.
    pub fn cofaces_of(&self, sign: Sign, dim: usize, pos: usize) -> &IntSet {
        match sign {
            Sign::Input => &self.cofaces_in[dim][pos],
            Sign::Output => &self.cofaces_out[dim][pos],
        }
    }

    /// Structural equality of basis and signed face tables.
    pub fn equal(a: &Self, b: &Self) -> bool {
        a.basis == b.basis && a.faces_in == b.faces_in && a.faces_out == b.faces_out
    }

    /// True if the cell is orthogonal to `direction`.
    pub fn is_orthogonal_to(&self, dim: usize, pos: usize, direction: usize) -> bool {
        self.basis[dim][pos].binary_search(&direction).is_err()
    }

    /// Cells at `dim` with no cofaces of either sign.
    pub fn maximal(&self, dim: usize) -> IntSet {
        if self.dim < 0 || dim > self.dim as usize {
            return vec![];
        }
        (0..self.basis[dim].len())
            .filter(|&pos| {
                self.cofaces_in[dim][pos].is_empty() && self.cofaces_out[dim][pos].is_empty()
            })
            .collect()
    }

    /// Render this poset's Hasse diagram as Graphviz DOT.
    pub fn to_dot(&self) -> String {
        crate::dot::to_dot(self)
    }

    /// Render this poset as DOT with node coordinates fixed by the compass
    /// spring layout.
    pub fn to_compass_spring_dot(&self) -> String {
        crate::dot::to_compass_spring_dot(self)
    }

    fn faces_all(&self, dim: usize, pos: usize) -> IntSet {
        intset::union(&self.faces_in[dim][pos], &self.faces_out[dim][pos])
    }

    fn well_formed(&self) -> bool {
        let levels = self.basis.len();

        if self.dim != levels as isize - 1 {
            return false;
        }

        if self.faces_in.len() != levels
            || self.faces_out.len() != levels
            || self.cofaces_in.len() != levels
            || self.cofaces_out.len() != levels
        {
            return false;
        }

        for dim in 0..levels {
            let n = self.basis[dim].len();
            if self.faces_in[dim].len() != n
                || self.faces_out[dim].len() != n
                || self.cofaces_in[dim].len() != n
                || self.cofaces_out[dim].len() != n
            {
                return false;
            }

            for basis in &self.basis[dim] {
                if basis.len() != dim || !intset::is_sorted_unique(basis) {
                    return false;
                }
            }
        }

        for dim in 0..levels {
            for pos in 0..self.basis[dim].len() {
                if dim == 0
                    && (!self.faces_in[dim][pos].is_empty() || !self.faces_out[dim][pos].is_empty())
                {
                    return false;
                }

                if dim > 0 {
                    for &face in &self.faces_in[dim][pos] {
                        if !self.valid_face(dim, pos, face)
                            || !self.cofaces_in[dim - 1][face].contains(&pos)
                        {
                            return false;
                        }
                    }
                    for &face in &self.faces_out[dim][pos] {
                        if !self.valid_face(dim, pos, face)
                            || !self.cofaces_out[dim - 1][face].contains(&pos)
                        {
                            return false;
                        }
                    }
                }

                if dim + 1 < levels {
                    for &coface in &self.cofaces_in[dim][pos] {
                        if coface >= self.basis[dim + 1].len()
                            || !self.faces_in[dim + 1][coface].contains(&pos)
                        {
                            return false;
                        }
                    }
                    for &coface in &self.cofaces_out[dim][pos] {
                        if coface >= self.basis[dim + 1].len()
                            || !self.faces_out[dim + 1][coface].contains(&pos)
                        {
                            return false;
                        }
                    }
                } else if !self.cofaces_in[dim][pos].is_empty()
                    || !self.cofaces_out[dim][pos].is_empty()
                {
                    return false;
                }
            }
        }

        true
    }

    fn valid_face(&self, dim: usize, pos: usize, face: usize) -> bool {
        face < self.basis[dim - 1].len()
            && intset::is_subset(&self.basis[dim - 1][face], &self.basis[dim][pos])
    }
}

/// Compute the directional boundary of a framed poset.
///
/// The boundary is the closure of the cells orthogonal to `direction` that have
/// no opposite-signed coface in the whole shape.
pub fn boundary(
    sign: Sign,
    direction: usize,
    shape: &Arc<FramedPoset>,
) -> (Arc<FramedPoset>, Embedding) {
    if shape.dim < 0 {
        return (
            Arc::new(FramedPoset::empty()),
            Embedding::empty(Arc::clone(shape)),
        );
    }

    let levels = shape.basis.len();
    let mut keep: Vec<Vec<bool>> = shape
        .basis
        .iter()
        .map(|level| vec![false; level.len()])
        .collect();

    let opposite = sign.opposite();
    for dim in 0..levels {
        for pos in 0..shape.basis[dim].len() {
            if shape.is_orthogonal_to(dim, pos, direction)
                && shape.cofaces_of(opposite, dim, pos).is_empty()
            {
                keep[dim][pos] = true;
            }
        }
    }

    for dim in (1..levels).rev() {
        for pos in 0..shape.basis[dim].len() {
            if !keep[dim][pos] {
                continue;
            }
            for face in shape.faces_all(dim, pos) {
                keep[dim - 1][face] = true;
            }
        }
    }

    restrict(shape, &keep)
}

fn restrict(shape: &Arc<FramedPoset>, keep: &[Vec<bool>]) -> (Arc<FramedPoset>, Embedding) {
    let sizes = shape.sizes();
    let top_dim = keep
        .iter()
        .enumerate()
        .rev()
        .find(|(_, row)| row.iter().any(|&x| x))
        .map(|(dim, _)| dim as isize)
        .unwrap_or(-1);

    if top_dim < 0 {
        return (
            Arc::new(FramedPoset::empty()),
            Embedding::empty(Arc::clone(shape)),
        );
    }

    let levels = top_dim as usize + 1;
    let mut map: Vec<Vec<usize>> = Vec::with_capacity(levels);
    let mut inv: Vec<Vec<usize>> = sizes.iter().map(|&n| vec![NO_PREIMAGE; n]).collect();

    for dim in 0..levels {
        let mut row = Vec::new();
        for (old, &is_kept) in keep[dim].iter().enumerate() {
            if is_kept {
                inv[dim][old] = row.len();
                row.push(old);
            }
        }
        map.push(row);
    }

    let remap_faces = |dim: usize, old: usize, table: &[Vec<IntSet>]| -> IntSet {
        if dim == 0 {
            vec![]
        } else {
            intset::collect_sorted(table[dim][old].iter().map(|&x| inv[dim - 1][x]))
        }
    };

    let remap_cofaces = |dim: usize, old: usize, table: &[Vec<IntSet>]| -> IntSet {
        if dim + 1 >= levels {
            vec![]
        } else {
            intset::collect_sorted(table[dim][old].iter().filter_map(|&x| {
                let y = inv[dim + 1][x];
                (y != NO_PREIMAGE).then_some(y)
            }))
        }
    };

    let mut basis = Vec::with_capacity(levels);
    let mut faces_in = Vec::with_capacity(levels);
    let mut faces_out = Vec::with_capacity(levels);
    let mut cofaces_in = Vec::with_capacity(levels);
    let mut cofaces_out = Vec::with_capacity(levels);

    for dim in 0..levels {
        basis.push(
            map[dim]
                .iter()
                .map(|&old| shape.basis[dim][old].clone())
                .collect(),
        );
        faces_in.push(
            map[dim]
                .iter()
                .map(|&old| remap_faces(dim, old, &shape.faces_in))
                .collect(),
        );
        faces_out.push(
            map[dim]
                .iter()
                .map(|&old| remap_faces(dim, old, &shape.faces_out))
                .collect(),
        );
        cofaces_in.push(
            map[dim]
                .iter()
                .map(|&old| remap_cofaces(dim, old, &shape.cofaces_in))
                .collect(),
        );
        cofaces_out.push(
            map[dim]
                .iter()
                .map(|&old| remap_cofaces(dim, old, &shape.cofaces_out))
                .collect(),
        );
    }

    let sub = Arc::new(FramedPoset::make(
        basis,
        faces_in,
        faces_out,
        cofaces_in,
        cofaces_out,
    ));
    let emb = Embedding::make(Arc::clone(&sub), Arc::clone(shape), map, inv);
    (sub, emb)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tight_arrow() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![1]]],
        ))
    }

    fn square() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![
                vec![vec![], vec![], vec![], vec![]],
                vec![vec![0], vec![0], vec![1], vec![1]],
                vec![vec![0, 1]],
            ],
            vec![
                vec![vec![], vec![], vec![], vec![]],
                vec![vec![0], vec![2], vec![0], vec![1]],
                vec![vec![0, 2]],
            ],
            vec![
                vec![vec![], vec![], vec![], vec![]],
                vec![vec![1], vec![3], vec![2], vec![3]],
                vec![vec![1, 3]],
            ],
        ))
    }

    #[test]
    fn empty_and_point() {
        let empty = FramedPoset::empty();
        assert_eq!(empty.dim(), -1);
        assert_eq!(empty.sizes(), Vec::<usize>::new());

        let point = FramedPoset::point();
        assert_eq!(point.dim(), 0);
        assert_eq!(point.sizes(), vec![1]);
        assert_eq!(point.basis_of(0, 0), &Vec::<usize>::new());
    }

    #[test]
    fn from_faces_generates_cofaces() {
        let arrow = tight_arrow();

        assert_eq!(arrow.cofaces_of(Sign::Input, 0, 0), &vec![0]);
        assert_eq!(arrow.cofaces_of(Sign::Input, 0, 1), &Vec::<usize>::new());
        assert_eq!(arrow.cofaces_of(Sign::Output, 0, 0), &Vec::<usize>::new());
        assert_eq!(arrow.cofaces_of(Sign::Output, 0, 1), &vec![0]);
    }

    #[test]
    fn tight_arrow_boundaries() {
        let arrow = tight_arrow();

        let (input, input_emb) = boundary(Sign::Input, 0, &arrow);
        assert_eq!(input.sizes(), vec![1]);
        assert_eq!(input_emb.map, vec![vec![0]]);

        let (output, output_emb) = boundary(Sign::Output, 0, &arrow);
        assert_eq!(output.sizes(), vec![1]);
        assert_eq!(output_emb.map, vec![vec![1]]);
    }

    #[test]
    fn two_direction_boundaries_differ() {
        let square = square();

        let (left, left_emb) = boundary(Sign::Input, 0, &square);
        assert_eq!(left.sizes(), vec![2, 1]);
        assert_eq!(left_emb.map, vec![vec![0, 2], vec![2]]);

        let (bottom, bottom_emb) = boundary(Sign::Input, 1, &square);
        assert_eq!(bottom.sizes(), vec![2, 1]);
        assert_eq!(bottom_emb.map, vec![vec![0, 1], vec![0]]);
    }

    #[test]
    fn boundary_closes_downward() {
        let square = square();

        let (left, _) = boundary(Sign::Input, 0, &square);
        assert_eq!(left.basis_of(1, 0), &vec![1]);
        assert_eq!(left.faces_of(Sign::Input, 1, 0), &vec![0]);
        assert_eq!(left.faces_of(Sign::Output, 1, 0), &vec![1]);
    }
}
