//! Isomorphisms of oriented framed posets.

use std::sync::Arc;

use rust_igraph::{Graph, get_isomorphisms_vf2};

use crate::embedding::{Embedding, NO_PREIMAGE};
use crate::intset::IntSet;
use crate::poset::{FramedPoset, Sign};

struct ColoredHasseGraph {
    graph: Graph,
    vertex_colors: Vec<u32>,
    edge_colors: Vec<u32>,
    cells: Vec<(usize, usize)>,
}

/// Enumerate all isomorphisms from `dom` to `cod`.
///
/// The result can be extremely large: a framed poset consisting of `n`
/// indistinguishable isolated points has `n!` automorphisms.
pub fn isomorphisms(dom: &Arc<FramedPoset>, cod: &Arc<FramedPoset>) -> Vec<Embedding> {
    let palette = basis_palette(dom, cod);
    let dom_graph = ColoredHasseGraph::new(dom, &palette);
    let cod_graph = ColoredHasseGraph::new(cod, &palette);

    let inverse_mappings = get_isomorphisms_vf2(
        &dom_graph.graph,
        &cod_graph.graph,
        Some(&dom_graph.vertex_colors),
        Some(&cod_graph.vertex_colors),
        Some(&dom_graph.edge_colors),
        Some(&cod_graph.edge_colors),
    )
    .expect("well-formed framed posets must produce valid VF2 input graphs");

    inverse_mappings
        .into_iter()
        .map(|inverse_mapping| {
            mapping_to_embedding(dom, cod, &dom_graph, &cod_graph, &inverse_mapping)
        })
        .collect()
}

impl ColoredHasseGraph {
    fn new(shape: &FramedPoset, palette: &[IntSet]) -> Self {
        let sizes = shape.sizes();
        let cells: Vec<(usize, usize)> = sizes
            .iter()
            .enumerate()
            .flat_map(|(dim, &size)| (0..size).map(move |pos| (dim, pos)))
            .collect();
        let vertex_count =
            u32::try_from(cells.len()).expect("framed poset has too many cells for rust-igraph");
        let mut graph = Graph::new(vertex_count, true)
            .expect("vertex count accepted by rust-igraph must form a graph");
        let vertex_colors = cells
            .iter()
            .map(|&(dim, pos)| {
                let color = palette
                    .binary_search(shape.basis_of(dim, pos))
                    .expect("every cell basis must occur in the shared palette");
                u32::try_from(color).expect("too many distinct bases for rust-igraph colors")
            })
            .collect();

        let mut offsets = Vec::with_capacity(sizes.len() + 1);
        offsets.push(0usize);
        for &size in &sizes {
            offsets.push(offsets.last().copied().unwrap() + size);
        }

        let mut edge_colors = Vec::new();
        for dim in 1..sizes.len() {
            for pos in 0..sizes[dim] {
                for sign in [Sign::Input, Sign::Output] {
                    for &face in shape.faces_of(sign, dim, pos) {
                        let source = u32::try_from(offsets[dim - 1] + face)
                            .expect("cell index must fit rust-igraph");
                        let target = u32::try_from(offsets[dim] + pos)
                            .expect("cell index must fit rust-igraph");
                        graph
                            .add_edge(source, target)
                            .expect("framed-poset cover must form a valid graph edge");
                        edge_colors.push(sign_color(sign));
                    }
                }
            }
        }

        Self {
            graph,
            vertex_colors,
            edge_colors,
            cells,
        }
    }
}

fn basis_palette(a: &FramedPoset, b: &FramedPoset) -> Vec<IntSet> {
    let mut palette = Vec::new();
    for shape in [a, b] {
        for (dim, size) in shape.sizes().into_iter().enumerate() {
            for pos in 0..size {
                palette.push(shape.basis_of(dim, pos).clone());
            }
        }
    }
    palette.sort_unstable();
    palette.dedup();
    palette
}

fn sign_color(sign: Sign) -> u32 {
    match sign {
        Sign::Input => 0,
        Sign::Output => 1,
    }
}

fn mapping_to_embedding(
    dom: &Arc<FramedPoset>,
    cod: &Arc<FramedPoset>,
    dom_graph: &ColoredHasseGraph,
    cod_graph: &ColoredHasseGraph,
    inverse_mapping: &[u32],
) -> Embedding {
    assert_eq!(inverse_mapping.len(), cod_graph.cells.len());

    let mut map: Vec<Vec<usize>> = dom
        .sizes()
        .into_iter()
        .map(|size| vec![NO_PREIMAGE; size])
        .collect();
    let mut inv: Vec<Vec<usize>> = cod
        .sizes()
        .into_iter()
        .map(|size| vec![NO_PREIMAGE; size])
        .collect();

    for (cod_vertex, &dom_vertex) in inverse_mapping.iter().enumerate() {
        let (dom_dim, dom_pos) = dom_graph.cells[dom_vertex as usize];
        let (cod_dim, cod_pos) = cod_graph.cells[cod_vertex];
        debug_assert_eq!(dom_dim, cod_dim);
        map[dom_dim][dom_pos] = cod_pos;
        inv[cod_dim][cod_pos] = dom_pos;
    }

    let embedding = Embedding::make(Arc::clone(dom), Arc::clone(cod), map, inv);
    debug_assert!(embedding.is_isomorphism());
    embedding
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points(count: usize) -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![vec![vec![]; count]],
            vec![vec![vec![]; count]],
            vec![vec![vec![]; count]],
        ))
    }

    fn half_arrow(sign: Sign, direction: usize) -> Arc<FramedPoset> {
        let (faces_in, faces_out) = match sign {
            Sign::Input => (
                vec![vec![vec![]], vec![vec![0]]],
                vec![vec![vec![]], vec![vec![]]],
            ),
            Sign::Output => (
                vec![vec![vec![]], vec![vec![]]],
                vec![vec![vec![]], vec![vec![0]]],
            ),
        };
        Arc::new(FramedPoset::from_faces(
            vec![vec![vec![]], vec![vec![direction]]],
            faces_in,
            faces_out,
        ))
    }

    fn arrow() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![1]]],
        ))
    }

    fn arrow_with_reordered_vertices() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![1]]],
            vec![vec![vec![], vec![]], vec![vec![0]]],
        ))
    }

    #[test]
    fn empty_posets_have_one_isomorphism() {
        let a = Arc::new(FramedPoset::empty());
        let b = Arc::new(FramedPoset::empty());
        let isomorphisms = isomorphisms(&a, &b);

        assert_eq!(isomorphisms.len(), 1);
        assert!(isomorphisms[0].is_isomorphism());
    }

    #[test]
    fn enumerates_every_permutation_of_indistinguishable_points() {
        let shape = points(3);
        let isomorphisms = isomorphisms(&shape, &shape);
        let mut maps: Vec<Vec<usize>> = isomorphisms
            .iter()
            .map(|embedding| embedding.map[0].clone())
            .collect();
        maps.sort_unstable();
        maps.dedup();

        assert_eq!(isomorphisms.len(), 6);
        assert_eq!(maps.len(), 6);
        assert!(isomorphisms.iter().all(Embedding::is_isomorphism));
    }

    #[test]
    fn recovers_isomorphism_between_different_cell_orders() {
        let source = arrow();
        let target = arrow_with_reordered_vertices();
        let isomorphisms = isomorphisms(&source, &target);

        assert_eq!(isomorphisms.len(), 1);
        assert_eq!(isomorphisms[0].map, vec![vec![1, 0], vec![0]]);
        assert_eq!(isomorphisms[0].inv, vec![vec![1, 0], vec![0]]);
        assert!(isomorphisms[0].is_isomorphism());
    }

    #[test]
    fn edge_orientation_is_preserved() {
        assert!(isomorphisms(&half_arrow(Sign::Input, 0), &half_arrow(Sign::Output, 0)).is_empty());
    }

    #[test]
    fn basis_is_preserved() {
        assert!(isomorphisms(&half_arrow(Sign::Input, 0), &half_arrow(Sign::Input, 1)).is_empty());
    }
}
