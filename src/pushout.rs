//! Pushouts of framed-poset embeddings.

use std::sync::Arc;

use crate::embedding::{Embedding, NO_PREIMAGE};
use crate::intset::{self, IntSet};
use crate::isomorphism::isomorphisms;
use crate::poset::{BoundaryMode, FramedPoset, Sign, boundary};

/// Binary pushout result.
#[derive(Debug, Clone)]
pub struct Pushout {
    pub tip: Arc<FramedPoset>,
    pub inl: Embedding,
    pub inr: Embedding,
}

/// One extension attached to a base along a shared domain.
#[derive(Debug, Clone, Copy)]
pub struct Span<'a> {
    pub into_base: &'a Embedding,
    pub into_ext: &'a Embedding,
}

/// Multi-pushout result.
#[derive(Debug, Clone)]
pub struct MultiPushout {
    pub tip: Arc<FramedPoset>,
    pub inl: Embedding,
    pub inrs: Vec<Embedding>,
}

/// Paste two framed posets along a uniquely isomorphic directional boundary.
///
/// The output `direction` boundary of `first` is identified with the input
/// `direction` boundary of `second`. The left and right pushout injections
/// therefore have `first` and `second` as their respective domains.
///
/// # Panics
///
/// Panics unless there is exactly one signed, basis-preserving isomorphism
/// from the output boundary of `first` to the input boundary of `second`.
pub fn paste_along_boundary(
    first: &Arc<FramedPoset>,
    second: &Arc<FramedPoset>,
    direction: usize,
) -> Pushout {
    let (output_boundary, output_into_first) =
        boundary(BoundaryMode::Plain, Sign::Output, direction, first);
    let (input_boundary, input_into_second) =
        boundary(BoundaryMode::Plain, Sign::Input, direction, second);

    let mut boundary_isomorphisms = isomorphisms(&output_boundary, &input_boundary);
    assert_eq!(
        boundary_isomorphisms.len(),
        1,
        "direction {direction} boundaries must have exactly one isomorphism"
    );

    let boundary_isomorphism = boundary_isomorphisms.pop().unwrap();
    let output_into_second = Embedding::compose(&boundary_isomorphism, &input_into_second);
    debug_assert!(Arc::ptr_eq(&output_into_first.dom, &output_into_second.dom));

    pushout(&output_into_first, &output_into_second)
}

/// Compute a binary pushout of embeddings.
pub fn pushout(f: &Embedding, g: &Embedding) -> Pushout {
    let size_sum = |x: &FramedPoset| x.sizes().iter().sum::<usize>();
    let (base_emb, ext_emb, swapped) = if size_sum(&f.cod) >= size_sum(&g.cod) {
        (f, g, false)
    } else {
        (g, f, true)
    };

    let mp = multi_pushout(
        &base_emb.cod,
        &[Span {
            into_base: base_emb,
            into_ext: ext_emb,
        }],
    );
    let inr = mp
        .inrs
        .into_iter()
        .next()
        .expect("one span has one right injection");

    if swapped {
        Pushout {
            tip: mp.tip,
            inl: inr,
            inr: mp.inl,
        }
    } else {
        Pushout {
            tip: mp.tip,
            inl: mp.inl,
            inr,
        }
    }
}

/// Compute the colimit of a base with any number of extensions.
pub fn multi_pushout(base: &Arc<FramedPoset>, spans: &[Span<'_>]) -> MultiPushout {
    let base_sizes = base.sizes();
    let tip_dim = spans
        .iter()
        .map(|span| span.into_ext.cod.dim())
        .fold(base.dim(), isize::max);
    let levels = if tip_dim < 0 { 0 } else { tip_dim as usize + 1 };

    let base_level_sizes: Vec<usize> = (0..levels)
        .map(|dim| base_sizes.get(dim).copied().unwrap_or(0))
        .collect();

    let mut extra_counts = vec![0usize; levels];
    for span in spans {
        let ext = &span.into_ext.cod;
        let ext_sizes = ext.sizes();
        for dim in 0..ext_sizes.len().min(levels) {
            for pos in 0..ext_sizes[dim] {
                if preimage_at(&span.into_ext.inv, dim, pos) == NO_PREIMAGE {
                    extra_counts[dim] += 1;
                }
            }
        }
    }

    let total_sizes: Vec<usize> = (0..levels)
        .map(|dim| base_level_sizes[dim] + extra_counts[dim])
        .collect();

    let mut tip_basis = alloc_rows(&base.basis, &total_sizes);
    let mut tip_faces_in = alloc_rows(&base.faces_in, &total_sizes);
    let mut tip_faces_out = alloc_rows(&base.faces_out, &total_sizes);
    let mut tip_cofaces_in = alloc_rows(&base.cofaces_in, &total_sizes);
    let mut tip_cofaces_out = alloc_rows(&base.cofaces_out, &total_sizes);

    let mut counters = base_level_sizes.clone();
    let mut inr_data = Vec::with_capacity(spans.len());

    for span in spans {
        let ext = &span.into_ext.cod;
        let ext_sizes = ext.sizes();
        let ext_levels = ext_sizes.len();

        let mut inr_map: Vec<Vec<usize>> = ext_sizes.iter().map(|&n| vec![0; n]).collect();
        let mut inr_inv: Vec<Vec<usize>> =
            total_sizes.iter().map(|&n| vec![NO_PREIMAGE; n]).collect();

        for dim in 0..ext_levels.min(levels) {
            for pos in 0..ext_sizes[dim] {
                let preimage = preimage_at(&span.into_ext.inv, dim, pos);
                if preimage != NO_PREIMAGE {
                    let target = span.into_base.map[dim][preimage];
                    debug_assert_eq!(base.basis_of(dim, target), ext.basis_of(dim, pos));
                    inr_map[dim][pos] = target;
                    inr_inv[dim][target] = pos;
                    continue;
                }

                let idx = counters[dim];
                counters[dim] += 1;
                inr_map[dim][pos] = idx;
                inr_inv[dim][idx] = pos;
                tip_basis[dim][idx] = ext.basis_of(dim, pos).clone();

                if dim > 0 {
                    let fi = intset::collect_sorted(
                        ext.faces_of(crate::poset::Sign::Input, dim, pos)
                            .iter()
                            .map(|&q| inr_map[dim - 1][q]),
                    );
                    let fo = intset::collect_sorted(
                        ext.faces_of(crate::poset::Sign::Output, dim, pos)
                            .iter()
                            .map(|&q| inr_map[dim - 1][q]),
                    );

                    for &face in &fi {
                        intset::insert(&mut tip_cofaces_in[dim - 1][face], idx);
                    }
                    for &face in &fo {
                        intset::insert(&mut tip_cofaces_out[dim - 1][face], idx);
                    }

                    tip_faces_in[dim][idx] = fi;
                    tip_faces_out[dim][idx] = fo;
                }
            }
        }

        inr_data.push((inr_map, inr_inv));
    }

    let tip = Arc::new(FramedPoset::make(
        tip_basis,
        tip_faces_in,
        tip_faces_out,
        tip_cofaces_in,
        tip_cofaces_out,
    ));

    let inl_map: Vec<Vec<usize>> = base_sizes.iter().map(|&n| (0..n).collect()).collect();
    let inl_inv: Vec<Vec<usize>> = (0..levels)
        .map(|dim| {
            let mut row = vec![NO_PREIMAGE; total_sizes[dim]];
            for (pos, value) in row.iter_mut().enumerate().take(base_level_sizes[dim]) {
                *value = pos;
            }
            row
        })
        .collect();
    let inl = Embedding::make(Arc::clone(base), Arc::clone(&tip), inl_map, inl_inv);

    let inrs = spans
        .iter()
        .zip(inr_data)
        .map(|(span, (map, inv))| {
            Embedding::make(Arc::clone(&span.into_ext.cod), Arc::clone(&tip), map, inv)
        })
        .collect();

    MultiPushout { tip, inl, inrs }
}

fn preimage_at(inv: &[Vec<usize>], dim: usize, pos: usize) -> usize {
    inv.get(dim)
        .and_then(|row| row.get(pos))
        .copied()
        .unwrap_or(NO_PREIMAGE)
}

fn alloc_rows(base: &[Vec<IntSet>], total_sizes: &[usize]) -> Vec<Vec<IntSet>> {
    (0..total_sizes.len())
        .map(|dim| {
            let mut row = vec![vec![]; total_sizes[dim]];
            if let Some(base_row) = base.get(dim) {
                for (pos, set) in base_row.iter().enumerate() {
                    row[pos] = set.clone();
                }
            }
            row
        })
        .collect()
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

    fn endpoint_embedding(endpoint: usize, cod: Arc<FramedPoset>) -> Embedding {
        let dom = point();
        let map = vec![vec![endpoint]];
        let mut inv = vec![vec![NO_PREIMAGE; 2], vec![NO_PREIMAGE; 1]];
        inv[0][endpoint] = 0;
        Embedding::make(dom, cod, map, inv)
    }

    #[test]
    fn pushout_pastes_two_arrows() {
        let left = arrow();
        let right = arrow();
        let out_left = endpoint_embedding(1, Arc::clone(&left));
        let in_right = endpoint_embedding(0, Arc::clone(&right));

        let po = pushout(&out_left, &in_right);

        assert_eq!(po.tip.sizes(), vec![3, 2]);
        assert_eq!(po.inl.map, vec![vec![0, 1], vec![0]]);
        assert_eq!(po.inr.map, vec![vec![1, 2], vec![1]]);
        assert_eq!(po.tip.basis_of(1, 0), &vec![0]);
        assert_eq!(po.tip.basis_of(1, 1), &vec![0]);
        assert_eq!(po.tip.faces_of(Sign::Input, 1, 1), &vec![1]);
        assert_eq!(po.tip.faces_of(Sign::Output, 1, 1), &vec![2]);
    }

    #[test]
    fn pastes_two_arrows_along_their_unique_directional_boundaries() {
        let first = arrow();
        let second = arrow();

        let pasted = paste_along_boundary(&first, &second, 0);

        assert_eq!(pasted.tip.sizes(), vec![3, 2]);
        assert_eq!(pasted.inl.dom.sizes(), vec![2, 1]);
        assert_eq!(pasted.inr.dom.sizes(), vec![2, 1]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn pushout_detects_glued_basis_mismatch_in_debug() {
        let base = Arc::new(FramedPoset::from_faces(
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![1]]],
        ));
        let ext = Arc::new(FramedPoset::from_faces(
            vec![vec![vec![], vec![]], vec![vec![1]]],
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![1]]],
        ));
        let dom = Arc::new(FramedPoset::from_faces(
            vec![vec![vec![]], vec![vec![0]]],
            vec![vec![vec![]], vec![vec![0]]],
            vec![vec![vec![]], vec![vec![]]],
        ));

        let into_base = Embedding {
            dom: Arc::clone(&dom),
            cod: Arc::clone(&base),
            map: vec![vec![0], vec![0]],
            inv: vec![vec![0, NO_PREIMAGE], vec![0]],
        };
        let into_ext = Embedding {
            dom,
            cod: Arc::clone(&ext),
            map: vec![vec![0], vec![0]],
            inv: vec![vec![0, NO_PREIMAGE], vec![0]],
        };

        let _ = multi_pushout(
            &base,
            &[Span {
                into_base: &into_base,
                into_ext: &into_ext,
            }],
        );
    }
}
