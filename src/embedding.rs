//! Embeddings of framed posets.

use std::collections::HashSet;
use std::sync::Arc;

use crate::intset;
use crate::poset::{FramedPoset, Sign};

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

impl Embedding {
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
        let inv = map.clone();
        Self {
            dom: Arc::clone(&x),
            cod: x,
            map,
            inv,
        }
    }

    /// The unique embedding from the empty poset into `cod`.
    pub fn empty(cod: Arc<FramedPoset>) -> Self {
        let inv: Vec<Vec<usize>> = cod.sizes().iter().map(|&n| vec![NO_PREIMAGE; n]).collect();
        Self {
            dom: Arc::new(FramedPoset::empty()),
            cod,
            map: vec![],
            inv,
        }
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
        if !self.well_formed() {
            return false;
        }

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

    fn well_formed(&self) -> bool {
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
        let mut inv: Vec<Vec<usize>> = cod.sizes().iter().map(|&n| vec![NO_PREIMAGE; n]).collect();
        inv[0][endpoint] = 0;
        Embedding::make(dom, cod, map, inv)
    }

    fn bottom_edge_embedding(dom: Arc<FramedPoset>, cod: Arc<FramedPoset>) -> Embedding {
        let map = vec![vec![0, 1], vec![0]];
        let inv = vec![
            vec![0, 1, NO_PREIMAGE, NO_PREIMAGE],
            vec![0, NO_PREIMAGE, NO_PREIMAGE, NO_PREIMAGE],
            vec![NO_PREIMAGE],
        ];
        Embedding::make(dom, cod, map, inv)
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
