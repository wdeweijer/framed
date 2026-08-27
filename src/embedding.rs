//! Embeddings of framed posets.

use std::collections::HashSet;
use std::sync::Arc;

use crate::intset;
use crate::poset::{FramedPoset, FramedPosetSubset, Sign, closure};

/// Sentinel stored in inverse maps when a codomain cell has no preimage.
pub const NO_PREIMAGE: usize = usize::MAX;

/// An injective basis-preserving map of framed posets.
///
/// The map is indexed by basis cardinality:
/// - `map[d][i]` is the codomain index of domain cell `i` at level `d`.
/// - `inv[d][j]` is the domain preimage of codomain cell `j`, or
///   [`NO_PREIMAGE`] when no preimage exists.
#[derive(Debug, Clone)]
pub struct Embedding {
    pub dom: Arc<FramedPoset>,
    pub cod: Arc<FramedPoset>,
    pub map: Vec<Vec<usize>>,
    pub inv: Vec<Vec<usize>>,
}

/// Intersection of two closed embeddings into a common codomain.
#[derive(Debug, Clone)]
pub struct EmbeddingIntersection {
    /// The closed intersection embedding into the original common codomain.
    pub into_codomain: Embedding,
    /// The embedding from the intersection into the left domain.
    pub into_left: Embedding,
    /// The embedding from the intersection into the right domain.
    pub into_right: Embedding,
}

/// Union of two closed embeddings into a common codomain.
#[derive(Debug, Clone)]
pub struct EmbeddingUnion {
    /// The closed union embedding into the original common codomain.
    pub into_codomain: Embedding,
    /// The embedding from the left domain into the union.
    pub left_into_union: Embedding,
    /// The embedding from the right domain into the union.
    pub right_into_union: Embedding,
}

impl Embedding {
    /// Construct an embedding from its forward map.
    ///
    /// The partial inverse is derived from `map`. As with
    /// [`FramedPoset::from_faces`], validity of the supplied data is checked in
    /// debug builds by the lower-level constructor.
    pub fn from_map(dom: Arc<FramedPoset>, cod: Arc<FramedPoset>, map: Vec<Vec<usize>>) -> Self {
        let mut inv: Vec<Vec<usize>> = cod
            .sizes()
            .into_iter()
            .map(|size| vec![NO_PREIMAGE; size])
            .collect();

        for (dim, level) in map.iter().enumerate() {
            for (dom_pos, &cod_pos) in level.iter().enumerate() {
                debug_assert_eq!(
                    inv[dim][cod_pos], NO_PREIMAGE,
                    "an embedding map must be injective",
                );
                inv[dim][cod_pos] = dom_pos;
            }
        }

        Self::make(dom, cod, map, inv)
    }

    /// Construct an embedding from precomputed tables.
    pub fn make(
        dom: Arc<FramedPoset>,
        cod: Arc<FramedPoset>,
        map: Vec<Vec<usize>>,
        inv: Vec<Vec<usize>>,
    ) -> Self {
        let emb = Self { dom, cod, map, inv };
        debug_assert!(emb.well_formed());
        emb
    }

    /// Identity embedding.
    pub fn id(x: Arc<FramedPoset>) -> Self {
        let sizes = x.sizes();
        let map: Vec<Vec<usize>> = sizes.iter().map(|&n| (0..n).collect()).collect();
        Self::from_map(Arc::clone(&x), x, map)
    }

    /// The unique embedding from the empty poset into `cod`.
    pub fn empty(cod: Arc<FramedPoset>) -> Self {
        Self::from_map(Arc::new(FramedPoset::empty()), cod, vec![])
    }

    /// True when the domain has no cells.
    pub fn is_empty(&self) -> bool {
        self.dom.sizes().into_iter().all(|size| size == 0)
    }

    /// True when every codomain cell has a preimage.
    pub fn is_surjective(&self) -> bool {
        self.inv
            .iter()
            .flatten()
            .all(|&preimage| preimage != NO_PREIMAGE)
    }

    /// True when this embedding is an isomorphism of framed posets.
    pub fn is_isomorphism(&self) -> bool {
        self.is_surjective() && self.is_closed()
    }

    /// Invert an embedding known to be an isomorphism.
    pub fn inverse_isomorphism(&self) -> Self {
        debug_assert!(self.is_isomorphism());
        Self::make(
            Arc::clone(&self.cod),
            Arc::clone(&self.dom),
            self.inv.clone(),
            self.map.clone(),
        )
    }

    /// Compose `f: A -> B` with `g: B -> C`, returning `g f: A -> C`.
    ///
    /// The middle framed posets are compared structurally, not by pointer
    /// identity, so separately allocated but equal middle objects can compose.
    pub fn compose(f: &Self, g: &Self) -> Self {
        assert!(
            FramedPoset::equal(&f.cod, &g.dom),
            "embeddings are not composable"
        );

        let map = f
            .map
            .iter()
            .enumerate()
            .map(|(dim, row)| row.iter().map(|&pos| g.map[dim][pos]).collect())
            .collect();

        let inv = g
            .inv
            .iter()
            .enumerate()
            .map(|(dim, row)| {
                row.iter()
                    .map(|&pos| {
                        if pos == NO_PREIMAGE {
                            NO_PREIMAGE
                        } else {
                            f.inv[dim][pos]
                        }
                    })
                    .collect()
            })
            .collect();

        Self::make(Arc::clone(&f.dom), Arc::clone(&g.cod), map, inv)
    }

    /// Intersection of two closed embeddings into a common codomain.
    pub fn intersection(a: &Self, b: &Self) -> EmbeddingIntersection {
        let into_codomain = closed_image_combination(a, b, |x, y| x && y);
        let into_left = common_subobject_to_argument(&into_codomain, a);
        let into_right = common_subobject_to_argument(&into_codomain, b);

        debug_assert!(into_codomain.is_closed());
        debug_assert!(into_left.is_closed());
        debug_assert!(into_right.is_closed());

        EmbeddingIntersection {
            into_codomain,
            into_left,
            into_right,
        }
    }

    /// Union of two closed embeddings into a common codomain.
    pub fn union(a: &Self, b: &Self) -> EmbeddingUnion {
        let into_codomain = closed_image_combination(a, b, |x, y| x || y);
        let left_into_union = argument_to_common_subobject(a, &into_codomain);
        let right_into_union = argument_to_common_subobject(b, &into_codomain);

        debug_assert!(into_codomain.is_closed());
        debug_assert!(left_into_union.is_closed());
        debug_assert!(right_into_union.is_closed());

        EmbeddingUnion {
            into_codomain,
            left_into_union,
            right_into_union,
        }
    }

    /// Equality as subobjects of a common codomain.
    ///
    /// The codomains are compared structurally, but the domains are not.
    /// Instead, two embeddings are equal when they have the same image cells
    /// and the same signed image cover relations in that codomain.
    pub fn equal(a: &Self, b: &Self) -> bool {
        FramedPoset::equal(&a.cod, &b.cod)
            && image_cells(a) == image_cells(b)
            && image_edges(a) == image_edges(b)
    }

    /// True when the image is an induced downward sub-poset of the codomain.
    ///
    /// For every domain cell, its signed faces must be exactly the preimages of
    /// the corresponding signed faces of its image in the codomain.  This is
    /// stronger than asking whether the image cells form a closed subset:
    /// incidences between image cells must also be present in the domain.
    pub fn is_closed(&self) -> bool {
        debug_assert!(self.well_formed());

        for dim in 1..self.map.len() {
            for (dom_pos, &cod_pos) in self.map[dim].iter().enumerate() {
                for sign in [Sign::Input, Sign::Output] {
                    let mapped_faces = intset::collect_sorted(
                        self.dom
                            .faces_of(sign, dim, dom_pos)
                            .iter()
                            .map(|&face| self.map[dim - 1][face]),
                    );
                    if mapped_faces != *self.cod.faces_of(sign, dim, cod_pos) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Render this embedding as Graphviz DOT.
    pub fn to_dot(&self, renderer: crate::dot::Renderer) -> String {
        crate::dot::embedding_to_dot(self, renderer)
    }

    pub(crate) fn well_formed(&self) -> bool {
        let dom_sizes = self.dom.sizes();
        let cod_sizes = self.cod.sizes();

        if self.map.len() != dom_sizes.len() || self.inv.len() != cod_sizes.len() {
            return false;
        }
        if !self
            .map
            .iter()
            .zip(&dom_sizes)
            .all(|(row, &n)| row.len() == n)
        {
            return false;
        }
        if !self
            .inv
            .iter()
            .zip(&cod_sizes)
            .all(|(row, &n)| row.len() == n)
        {
            return false;
        }

        for (dim, row) in self.map.iter().enumerate() {
            let mut seen = vec![false; cod_sizes[dim]];
            for (dom_pos, &cod_pos) in row.iter().enumerate() {
                if cod_pos >= cod_sizes[dim] || seen[cod_pos] {
                    return false;
                }
                seen[cod_pos] = true;
                if self.inv[dim][cod_pos] != dom_pos {
                    return false;
                }
                if self.dom.basis_of(dim, dom_pos) != self.cod.basis_of(dim, cod_pos) {
                    return false;
                }
            }
        }

        for (dim, row) in self.inv.iter().enumerate() {
            for (cod_pos, &dom_pos) in row.iter().enumerate() {
                if dom_pos == NO_PREIMAGE {
                    continue;
                }
                if dom_pos >= dom_sizes[dim] || self.map[dim][dom_pos] != cod_pos {
                    return false;
                }
            }
        }

        true
    }
}

fn closed_image_combination(
    a: &Embedding,
    b: &Embedding,
    combine: impl Fn(bool, bool) -> bool,
) -> Embedding {
    assert!(
        FramedPoset::equal(&a.cod, &b.cod),
        "embeddings must have equal codomains"
    );
    debug_assert!(a.is_closed(), "left embedding must be closed");
    debug_assert!(b.is_closed(), "right embedding must be closed");

    let keep = a
        .cod
        .sizes()
        .iter()
        .enumerate()
        .map(|(dim, &n)| {
            (0..n)
                .map(|pos| {
                    combine(
                        a.inv[dim][pos] != NO_PREIMAGE,
                        b.inv[dim][pos] != NO_PREIMAGE,
                    )
                })
                .collect()
        })
        .collect();
    let subset = FramedPosetSubset::make(Arc::clone(&a.cod), keep);
    let (_, embedding) = closure(&subset);
    debug_assert!(embedding.is_closed());
    embedding
}

fn common_subobject_to_argument(subobject: &Embedding, argument: &Embedding) -> Embedding {
    let map: Vec<Vec<usize>> = subobject
        .map
        .iter()
        .enumerate()
        .map(|(dim, row)| {
            row.iter()
                .map(|&cod_pos| {
                    let argument_pos = argument.inv[dim][cod_pos];
                    assert!(argument_pos != NO_PREIMAGE);
                    argument_pos
                })
                .collect()
        })
        .collect();

    Embedding::from_map(Arc::clone(&subobject.dom), Arc::clone(&argument.dom), map)
}

fn argument_to_common_subobject(argument: &Embedding, subobject: &Embedding) -> Embedding {
    let map: Vec<Vec<usize>> = argument
        .map
        .iter()
        .enumerate()
        .map(|(dim, row)| {
            row.iter()
                .map(|&cod_pos| {
                    let subobject_pos = subobject.inv[dim][cod_pos];
                    assert!(subobject_pos != NO_PREIMAGE);
                    subobject_pos
                })
                .collect()
        })
        .collect();

    Embedding::from_map(Arc::clone(&argument.dom), Arc::clone(&subobject.dom), map)
}

fn image_cells(embedding: &Embedding) -> HashSet<(usize, usize)> {
    embedding
        .map
        .iter()
        .enumerate()
        .flat_map(|(dim, row)| row.iter().copied().map(move |pos| (dim, pos)))
        .collect()
}

fn image_edges(embedding: &Embedding) -> HashSet<(Sign, usize, usize, usize, usize)> {
    let mut edges = HashSet::new();

    for dim in 1..embedding.dom.sizes().len() {
        for pos in 0..embedding.dom.sizes()[dim] {
            for &face in embedding.dom.faces_of(Sign::Input, dim, pos) {
                edges.insert((
                    Sign::Input,
                    dim - 1,
                    embedding.map[dim - 1][face],
                    dim,
                    embedding.map[dim][pos],
                ));
            }
            for &face in embedding.dom.faces_of(Sign::Output, dim, pos) {
                edges.insert((
                    Sign::Output,
                    dim - 1,
                    embedding.map[dim - 1][face],
                    dim,
                    embedding.map[dim][pos],
                ));
            }
        }
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poset::boundary;

    fn point() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::point())
    }

    fn arrow() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![1]]],
        ))
    }

    fn reversed_arrow() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![1]]],
            vec![vec![vec![], vec![]], vec![vec![0]]],
        ))
    }

    fn input_half_arrow() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![]]],
        ))
    }

    fn open_edge() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![vec![vec![]], vec![vec![0]]],
            vec![vec![vec![]], vec![vec![0]]],
            vec![vec![vec![]], vec![vec![]]],
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

    fn endpoint_embedding(endpoint: usize, cod: Arc<FramedPoset>) -> Embedding {
        let dom = point();
        let map = vec![vec![endpoint]];
        Embedding::from_map(dom, cod, map)
    }

    fn bottom_edge_embedding(dom: Arc<FramedPoset>, cod: Arc<FramedPoset>) -> Embedding {
        let map = vec![vec![0, 1], vec![0]];
        Embedding::from_map(dom, cod, map)
    }

    #[test]
    fn from_map_derives_the_partial_inverse() {
        let embedding = Embedding::from_map(point(), arrow(), vec![vec![1]]);

        assert_eq!(embedding.map, vec![vec![1]]);
        assert_eq!(embedding.inv, vec![vec![NO_PREIMAGE, 0], vec![NO_PREIMAGE]],);
    }

    #[test]
    fn equal_compares_images_not_domain_presentations() {
        let cod = arrow();
        let id = Embedding::id(Arc::clone(&cod));
        let reversed = reversed_arrow();
        let reversed_map = vec![vec![1, 0], vec![0]];
        let reversed_inv = reversed_map.clone();
        let reversed_embedding = Embedding::make(reversed, cod, reversed_map, reversed_inv);

        assert!(!FramedPoset::equal(&id.dom, &reversed_embedding.dom));
        assert!(Embedding::equal(&id, &reversed_embedding));
    }

    #[test]
    fn is_empty_checks_the_domain() {
        let cod = arrow();

        assert!(Embedding::empty(Arc::clone(&cod)).is_empty());
        assert!(Embedding::id(Arc::new(FramedPoset::empty())).is_empty());
        assert!(!endpoint_embedding(0, Arc::clone(&cod)).is_empty());
        assert!(!Embedding::id(cod).is_empty());
    }

    #[test]
    fn is_surjective_checks_every_inverse_entry() {
        let cod = arrow();

        assert!(Embedding::id(Arc::clone(&cod)).is_surjective());
        assert!(!endpoint_embedding(0, cod).is_surjective());
    }

    #[test]
    fn is_isomorphism_requires_closedness_and_surjectivity() {
        let cod = arrow();
        let identity = Embedding::id(Arc::clone(&cod));
        let proper_closed_subobject = endpoint_embedding(0, Arc::clone(&cod));
        let bijective_with_missing_incidence = Embedding::make(
            input_half_arrow(),
            cod,
            vec![vec![0, 1], vec![0]],
            vec![vec![0, 1], vec![0]],
        );

        assert!(identity.is_isomorphism());
        assert!(proper_closed_subobject.is_closed());
        assert!(!proper_closed_subobject.is_surjective());
        assert!(!proper_closed_subobject.is_isomorphism());
        assert!(bijective_with_missing_incidence.is_surjective());
        assert!(!bijective_with_missing_incidence.is_closed());
        assert!(!bijective_with_missing_incidence.is_isomorphism());
    }

    #[test]
    fn inverse_isomorphism_swaps_maps_and_undoes_the_original() {
        let dom = arrow();
        let cod = reversed_arrow();
        let isomorphism = Embedding::make(
            Arc::clone(&dom),
            Arc::clone(&cod),
            vec![vec![1, 0], vec![0]],
            vec![vec![1, 0], vec![0]],
        );

        let inverse = isomorphism.inverse_isomorphism();

        assert!(FramedPoset::equal(&inverse.dom, &cod));
        assert!(FramedPoset::equal(&inverse.cod, &dom));
        assert_eq!(inverse.map, isomorphism.inv);
        assert_eq!(inverse.inv, isomorphism.map);
        assert_eq!(
            Embedding::compose(&isomorphism, &inverse).map,
            Embedding::id(dom).map
        );
        assert_eq!(
            Embedding::compose(&inverse, &isomorphism).map,
            Embedding::id(cod).map
        );
    }

    #[test]
    fn equal_sees_signed_image_edges() {
        let cod = arrow();
        let id = Embedding::id(Arc::clone(&cod));
        let same_cells = Embedding::make(
            input_half_arrow(),
            cod,
            vec![vec![0, 1], vec![0]],
            vec![vec![0, 1], vec![0]],
        );

        assert_eq!(image_cells(&id), image_cells(&same_cells));
        assert!(!Embedding::equal(&id, &same_cells));
    }

    #[test]
    fn equal_rejects_different_codomains() {
        let into_arrow = endpoint_embedding(0, arrow());
        let into_square = endpoint_embedding(0, square());

        assert!(!Embedding::equal(&into_arrow, &into_square));
    }

    #[test]
    fn is_closed_accepts_downward_closed_image() {
        assert!(Embedding::id(arrow()).is_closed());
        assert!(endpoint_embedding(0, arrow()).is_closed());
    }

    #[test]
    fn is_closed_rejects_image_missing_faces() {
        let cod = arrow();
        let embedding = Embedding::make(
            open_edge(),
            cod,
            vec![vec![0], vec![0]],
            vec![vec![0, NO_PREIMAGE], vec![0]],
        );

        assert!(!embedding.is_closed());
    }

    #[test]
    fn is_closed_rejects_bijective_image_with_missing_incidence() {
        let cod = arrow();
        let embedding = Embedding::make(
            input_half_arrow(),
            cod,
            vec![vec![0, 1], vec![0]],
            vec![vec![0, 1], vec![0]],
        );

        assert_eq!(embedding.map, Embedding::id(Arc::clone(&embedding.cod)).map);
        assert!(!embedding.is_closed());
    }

    #[test]
    fn intersection_returns_maps_to_both_arguments() {
        let cod = square();
        let (_, left) = boundary(Sign::Input, 0, &cod);
        let (_, bottom) = boundary(Sign::Input, 1, &cod);

        let intersection = Embedding::intersection(&left, &bottom);

        assert_eq!(intersection.into_codomain.map, vec![vec![0]]);
        assert_eq!(intersection.into_left.map, vec![vec![0]]);
        assert_eq!(intersection.into_right.map, vec![vec![0]]);
        assert!(intersection.into_codomain.is_closed());
        assert!(intersection.into_left.is_closed());
        assert!(intersection.into_right.is_closed());
        assert!(Embedding::equal(
            &Embedding::compose(&intersection.into_left, &left),
            &intersection.into_codomain,
        ));
        assert!(Embedding::equal(
            &Embedding::compose(&intersection.into_right, &bottom),
            &intersection.into_codomain,
        ));
    }

    #[test]
    fn union_returns_maps_from_both_arguments() {
        let cod = square();
        let (_, left) = boundary(Sign::Input, 0, &cod);
        let (_, bottom) = boundary(Sign::Input, 1, &cod);

        let union = Embedding::union(&left, &bottom);

        assert_eq!(union.into_codomain.map, vec![vec![0, 1, 2], vec![0, 2]]);
        assert_eq!(union.left_into_union.map, vec![vec![0, 2], vec![1]]);
        assert_eq!(union.right_into_union.map, vec![vec![0, 1], vec![0]]);
        assert!(union.into_codomain.is_closed());
        assert!(union.left_into_union.is_closed());
        assert!(union.right_into_union.is_closed());
        assert!(Embedding::equal(
            &Embedding::compose(&union.left_into_union, &union.into_codomain),
            &left,
        ));
        assert!(Embedding::equal(
            &Embedding::compose(&union.right_into_union, &union.into_codomain),
            &bottom,
        ));
    }

    #[test]
    fn compose_uses_structural_middle_equality() {
        let endpoint = endpoint_embedding(1, arrow());
        let bottom_edge = bottom_edge_embedding(arrow(), square());

        let composite = Embedding::compose(&endpoint, &bottom_edge);

        assert_eq!(composite.map, vec![vec![1]]);
        assert_eq!(
            composite.inv,
            vec![
                vec![NO_PREIMAGE, 0, NO_PREIMAGE, NO_PREIMAGE],
                vec![NO_PREIMAGE, NO_PREIMAGE, NO_PREIMAGE, NO_PREIMAGE],
                vec![NO_PREIMAGE],
            ]
        );
    }

    #[test]
    #[should_panic(expected = "embeddings are not composable")]
    fn compose_rejects_unequal_middle_posets() {
        let endpoint = endpoint_embedding(0, arrow());
        let id_point = Embedding::id(point());

        let _ = Embedding::compose(&endpoint, &id_point);
    }
}
