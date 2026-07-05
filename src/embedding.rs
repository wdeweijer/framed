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
