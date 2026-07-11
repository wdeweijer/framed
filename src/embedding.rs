//! Embeddings of framed posets.

use std::sync::Arc;

use crate::poset::FramedPoset;

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

    /// Render this embedding as Graphviz DOT.
    pub fn to_dot(&self) -> String {
        crate::dot::embedding_to_dot(self)
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
