//! Oriented framed posets.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::embedding::{Embedding, NO_PREIMAGE};
use crate::intset::{self, IntSet};

const SERIALIZATION_VERSION: usize = 1;

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
    pub(crate) normal: bool,
    pub(crate) dim: isize,
    pub(crate) basis: Vec<Vec<IntSet>>,
    pub(crate) faces_in: Vec<Vec<IntSet>>,
    pub(crate) faces_out: Vec<Vec<IntSet>>,
    pub(crate) cofaces_in: Vec<Vec<IntSet>>,
    pub(crate) cofaces_out: Vec<Vec<IntSet>>,
}

impl PartialEq for FramedPoset {
    fn eq(&self, other: &Self) -> bool {
        Self::equal(self, other)
    }
}

impl Eq for FramedPoset {}

impl Hash for FramedPoset {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.basis.hash(state);
        self.faces_in.hash(state);
        self.faces_out.hash(state);
    }
}

#[derive(Serialize)]
struct FramedPosetRef<'a> {
    version: usize,
    basis: &'a [Vec<IntSet>],
    faces_in: &'a [Vec<IntSet>],
    faces_out: &'a [Vec<IntSet>],
}

#[derive(Deserialize)]
struct FramedPosetData {
    version: usize,
    basis: Vec<Vec<IntSet>>,
    faces_in: Vec<Vec<IntSet>>,
    faces_out: Vec<Vec<IntSet>>,
}

impl Serialize for FramedPoset {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        FramedPosetRef {
            version: SERIALIZATION_VERSION,
            basis: &self.basis,
            faces_in: &self.faces_in,
            faces_out: &self.faces_out,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FramedPoset {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data = FramedPosetData::deserialize(deserializer)?;
        if data.version != SERIALIZATION_VERSION {
            return Err(D::Error::custom(format!(
                "unsupported framed-poset serialization version {}",
                data.version
            )));
        }

        let levels = data.basis.len();
        if data.faces_in.len() != levels || data.faces_out.len() != levels {
            return Err(D::Error::custom(
                "basis and signed face tables must have the same number of levels",
            ));
        }

        let sizes: Vec<usize> = data.basis.iter().map(Vec::len).collect();
        for dim in 0..levels {
            if data.faces_in[dim].len() != sizes[dim] || data.faces_out[dim].len() != sizes[dim] {
                return Err(D::Error::custom(format!(
                    "signed face rows at level {dim} must match the number of cells"
                )));
            }

            if dim > 0
                && data.faces_in[dim]
                    .iter()
                    .chain(&data.faces_out[dim])
                    .flatten()
                    .any(|&face| face >= sizes[dim - 1])
            {
                return Err(D::Error::custom(format!(
                    "face index at level {dim} is out of bounds"
                )));
            }
        }

        let mut cofaces_in: Vec<Vec<IntSet>> = sizes.iter().map(|&n| vec![vec![]; n]).collect();
        let mut cofaces_out: Vec<Vec<IntSet>> = sizes.iter().map(|&n| vec![vec![]; n]).collect();

        for dim in 1..levels {
            for pos in 0..sizes[dim] {
                for &face in &data.faces_in[dim][pos] {
                    intset::insert(&mut cofaces_in[dim - 1][face], pos);
                }
                for &face in &data.faces_out[dim][pos] {
                    intset::insert(&mut cofaces_out[dim - 1][face], pos);
                }
            }
        }

        let poset = Self {
            normal: false,
            dim: levels as isize - 1,
            basis: data.basis,
            faces_in: data.faces_in,
            faces_out: data.faces_out,
            cofaces_in,
            cofaces_out,
        };
        if !poset.well_formed() {
            return Err(D::Error::custom(
                "serialized framed poset is not well-formed",
            ));
        }

        Ok(poset)
    }
}

/// A subset of cells of a framed poset.
///
/// The `keep` table is indexed by basis cardinality and then cell position,
/// matching [`FramedPoset`]'s internal tables.
#[derive(Debug, Clone)]
pub struct FramedPosetSubset {
    pub shape: Arc<FramedPoset>,
    pub keep: Vec<Vec<bool>>,
}

impl FramedPosetSubset {
    /// Construct a subset from a shape and a keep table.
    pub fn make(shape: Arc<FramedPoset>, keep: Vec<Vec<bool>>) -> Self {
        let subset = Self { shape, keep };
        debug_assert!(subset.well_formed());
        subset
    }

    /// The image subset of an embedding in its codomain.
    pub fn from_embedding(embedding: &Embedding) -> Self {
        let keep = embedding
            .inv
            .iter()
            .map(|row| row.iter().map(|&x| x != NO_PREIMAGE).collect())
            .collect();
        Self::make(Arc::clone(&embedding.cod), keep)
    }

    /// True when the subset contains a given cell.
    pub fn contains(&self, dim: usize, pos: usize) -> bool {
        self.keep
            .get(dim)
            .and_then(|row| row.get(pos))
            .copied()
            .unwrap_or(false)
    }

    /// True when every cell in this subset also belongs to `other`.
    ///
    /// The ambient framed posets are compared structurally rather than by
    /// pointer identity.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.well_formed()
            && other.well_formed()
            && FramedPoset::equal(&self.shape, &other.shape)
            && self
                .keep
                .iter()
                .zip(&other.keep)
                .all(|(left, right)| left.iter().zip(right).all(|(&x, &y)| !x || y))
    }

    fn well_formed(&self) -> bool {
        let sizes = self.shape.sizes();
        self.keep.len() == sizes.len()
            && self
                .keep
                .iter()
                .zip(sizes)
                .all(|(row, size)| row.len() == size)
    }
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
            normal: false,
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

    /// Whether this framed poset is known to be in canonical normal form.
    pub fn is_normal(&self) -> bool {
        self.normal
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
    ///
    /// The derived [`Self::is_normal`] marker is not part of the mathematical
    /// framed poset and is therefore ignored.
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
    pub fn to_dot(&self, renderer: crate::dot::Renderer) -> String {
        crate::dot::to_dot(self, renderer)
    }

    fn faces_all(&self, dim: usize, pos: usize) -> IntSet {
        intset::union(&self.faces_in[dim][pos], &self.faces_out[dim][pos])
    }

    pub(crate) fn well_formed(&self) -> bool {
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
                if !intset::is_sorted_unique(&self.faces_in[dim][pos])
                    || !intset::is_sorted_unique(&self.faces_out[dim][pos])
                    || !intset::is_sorted_unique(&self.cofaces_in[dim][pos])
                    || !intset::is_sorted_unique(&self.cofaces_out[dim][pos])
                {
                    return false;
                }

                if self.faces_in[dim][pos]
                    .iter()
                    .any(|face| self.faces_out[dim][pos].contains(face))
                {
                    return false;
                }

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

                    // Closedness of the basis map is equivalent, in this
                    // finite ranked representation, to realizing every
                    // immediate face of this cell's basis.
                    if self.basis[dim][pos].iter().any(|direction| {
                        !self.faces_in[dim][pos]
                            .iter()
                            .chain(&self.faces_out[dim][pos])
                            .any(|&face| {
                                self.basis[dim - 1][face].binary_search(direction).is_err()
                            })
                    }) {
                        return false;
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
    let result = if shape.dim < 0 {
        (
            Arc::new(FramedPoset::empty()),
            Embedding::empty(Arc::clone(shape)),
        )
    } else {
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

        let subset = FramedPosetSubset::make(Arc::clone(shape), keep);
        closure(&subset)
    };

    debug_assert!(result.1.is_closed());
    result
}

/// Compute the smallest closed embedding whose image contains `subset`.
pub fn closure(subset: &FramedPosetSubset) -> (Arc<FramedPoset>, Embedding) {
    debug_assert!(subset.well_formed());

    let shape = &subset.shape;
    let mut keep = subset.keep.clone();

    for dim in (1..keep.len()).rev() {
        for pos in 0..keep[dim].len() {
            if !keep[dim][pos] {
                continue;
            }
            for face in shape.faces_all(dim, pos) {
                keep[dim - 1][face] = true;
            }
        }
    }

    let closed_subset = FramedPosetSubset::make(Arc::clone(shape), keep);
    let result = embedding_from_closed_subset(&closed_subset);
    debug_assert!(result.1.is_closed());
    result
}

fn embedding_from_closed_subset(subset: &FramedPosetSubset) -> (Arc<FramedPoset>, Embedding) {
    let shape = &subset.shape;
    let keep = &subset.keep;
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
            Embedding::empty(Arc::clone(&subset.shape)),
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
    let emb = Embedding::make(Arc::clone(&sub), Arc::clone(&subset.shape), map, inv);
    (sub, emb)
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use super::*;

    fn structural_hash(shape: &FramedPoset) -> u64 {
        let mut hasher = DefaultHasher::new();
        shape.hash(&mut hasher);
        hasher.finish()
    }

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
        assert!(!empty.is_normal());

        let point = FramedPoset::point();
        assert_eq!(point.dim(), 0);
        assert_eq!(point.sizes(), vec![1]);
        assert_eq!(point.basis_of(0, 0), &Vec::<usize>::new());
        assert!(!point.is_normal());
    }

    #[test]
    fn equality_and_hash_ignore_the_normal_form_marker() {
        let shape = tight_arrow();
        let mut marked_normal = shape.as_ref().clone();
        marked_normal.normal = true;

        assert_eq!(shape.as_ref(), &marked_normal);
        assert_eq!(structural_hash(&shape), structural_hash(&marked_normal));
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
    fn serialization_round_trips_defining_data_and_regenerates_cofaces() {
        let original = square();

        let json = serde_json::to_string_pretty(original.as_ref()).unwrap();
        let restored: FramedPoset = serde_json::from_str(&json).unwrap();

        assert!(FramedPoset::equal(&original, &restored));
        assert_eq!(
            original.cofaces_of(Sign::Input, 1, 0),
            restored.cofaces_of(Sign::Input, 1, 0)
        );
        assert!(json.contains("\"version\": 1"));
        assert!(!json.contains("cofaces"));
        assert!(!json.contains("normal"));
        assert!(!restored.is_normal());
    }

    #[test]
    fn serialization_round_trips_empty_poset() {
        let json = serde_json::to_string(&FramedPoset::empty()).unwrap();
        let restored: FramedPoset = serde_json::from_str(&json).unwrap();

        assert!(FramedPoset::equal(&FramedPoset::empty(), &restored));
    }

    #[test]
    fn deserialization_rejects_invalid_face_index() {
        let json = serde_json::json!({
            "version": 1,
            "basis": [[[]], [[0]]],
            "faces_in": [[[]], [[1]]],
            "faces_out": [[[]], [[]]],
        });

        assert!(serde_json::from_value::<FramedPoset>(json).is_err());
    }

    #[test]
    fn deserialization_rejects_poset_that_is_not_well_formed() {
        let json = serde_json::json!({
            "version": 1,
            "basis": [[[]], [[0]]],
            "faces_in": [[[]], [[0]]],
            "faces_out": [[[]], [[0]]],
        });

        assert!(serde_json::from_value::<FramedPoset>(json).is_err());
    }

    #[test]
    fn subset_from_embedding_marks_image_cells() {
        let arrow = tight_arrow();
        let (_, input_embedding) = boundary(Sign::Input, 0, &arrow);
        let subset = FramedPosetSubset::from_embedding(&input_embedding);

        assert!(subset.contains(0, 0));
        assert!(!subset.contains(0, 1));
    }

    #[test]
    fn subset_predicate_compares_cells_in_a_common_ambient_poset() {
        let arrow = tight_arrow();
        let endpoint =
            FramedPosetSubset::make(Arc::clone(&arrow), vec![vec![true, false], vec![false]]);
        let whole = FramedPosetSubset::make(Arc::clone(&arrow), vec![vec![true, true], vec![true]]);
        let point = FramedPosetSubset::make(Arc::new(FramedPoset::point()), vec![vec![true]]);

        assert!(endpoint.is_subset_of(&endpoint));
        assert!(endpoint.is_subset_of(&whole));
        assert!(!whole.is_subset_of(&endpoint));
        assert!(!endpoint.is_subset_of(&point));
    }

    #[test]
    fn closure_adds_missing_faces_and_returns_closed_embedding() {
        let arrow = tight_arrow();
        let subset =
            FramedPosetSubset::make(Arc::clone(&arrow), vec![vec![false, false], vec![true]]);

        let (closed, embedding) = closure(&subset);

        assert_eq!(closed.sizes(), vec![2, 1]);
        assert!(embedding.is_closed());
        assert!(Embedding::equal(&embedding, &Embedding::id(arrow)));
    }

    #[test]
    fn closure_of_closed_embedding_subset_is_equal_to_original() {
        let square = square();
        let (_, original) = boundary(Sign::Input, 0, &square);
        let subset = FramedPosetSubset::from_embedding(&original);

        let (_, closed) = closure(&subset);

        assert!(original.is_closed());
        assert!(closed.is_closed());
        assert!(Embedding::equal(&closed, &original));
    }

    #[test]
    fn closure_of_empty_subset_is_empty_embedding() {
        let arrow = tight_arrow();
        let subset =
            FramedPosetSubset::make(Arc::clone(&arrow), vec![vec![false, false], vec![false]]);

        let (closed, embedding) = closure(&subset);

        assert_eq!(closed.sizes(), Vec::<usize>::new());
        assert_eq!(embedding.map, Vec::<Vec<usize>>::new());
        assert!(embedding.is_closed());
    }

    #[test]
    fn well_formed_rejects_signed_face_overlap() {
        let poset = FramedPoset {
            normal: false,
            dim: 1,
            basis: vec![vec![vec![]], vec![vec![0]]],
            faces_in: vec![vec![vec![]], vec![vec![0]]],
            faces_out: vec![vec![vec![]], vec![vec![0]]],
            cofaces_in: vec![vec![vec![0]], vec![vec![]]],
            cofaces_out: vec![vec![vec![0]], vec![vec![]]],
        };

        assert!(!poset.well_formed());
    }

    #[test]
    fn well_formed_rejects_positive_cell_without_faces() {
        let poset = FramedPoset {
            normal: false,
            dim: 1,
            basis: vec![vec![vec![]], vec![vec![0]]],
            faces_in: vec![vec![vec![]], vec![vec![]]],
            faces_out: vec![vec![vec![]], vec![vec![]]],
            cofaces_in: vec![vec![vec![]], vec![vec![]]],
            cofaces_out: vec![vec![vec![]], vec![vec![]]],
        };

        assert!(!poset.well_formed());
    }

    #[test]
    fn well_formed_rejects_non_closed_basis_map() {
        let poset = FramedPoset {
            normal: false,
            dim: 2,
            basis: vec![vec![vec![]], vec![vec![0], vec![1]], vec![vec![0, 1]]],
            faces_in: vec![vec![vec![]], vec![vec![0], vec![0]], vec![vec![0]]],
            faces_out: vec![vec![vec![]], vec![vec![], vec![]], vec![vec![]]],
            cofaces_in: vec![vec![vec![0, 1]], vec![vec![0], vec![]], vec![vec![]]],
            cofaces_out: vec![vec![vec![]], vec![vec![], vec![]], vec![vec![]]],
        };

        assert!(!poset.well_formed());
    }

    #[test]
    fn well_formed_accepts_one_sided_faces_over_every_basis_face() {
        let poset = FramedPoset {
            normal: false,
            dim: 2,
            basis: vec![vec![vec![]], vec![vec![0], vec![1]], vec![vec![0, 1]]],
            faces_in: vec![vec![vec![]], vec![vec![0], vec![0]], vec![vec![0, 1]]],
            faces_out: vec![vec![vec![]], vec![vec![], vec![]], vec![vec![]]],
            cofaces_in: vec![vec![vec![0, 1]], vec![vec![0], vec![0]], vec![vec![]]],
            cofaces_out: vec![vec![vec![]], vec![vec![], vec![]], vec![vec![]]],
        };

        assert!(poset.well_formed());
    }

    #[test]
    fn well_formed_rejects_duplicate_adjacency() {
        let poset = FramedPoset {
            normal: false,
            dim: 1,
            basis: vec![vec![vec![]], vec![vec![0]]],
            faces_in: vec![vec![vec![]], vec![vec![0, 0]]],
            faces_out: vec![vec![vec![]], vec![vec![]]],
            cofaces_in: vec![vec![vec![0]], vec![vec![]]],
            cofaces_out: vec![vec![vec![]], vec![vec![]]],
        };

        assert!(!poset.well_formed());
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
