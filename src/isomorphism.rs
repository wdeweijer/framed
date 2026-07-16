//! Isomorphisms of oriented framed posets.

use std::sync::Arc;

use rust_igraph::{Graph, canonical_permutation, get_isomorphisms_vf2, isomorphic_vf2};

use crate::embedding::{Embedding, NO_PREIMAGE};
use crate::intset::{self, IntSet};
use crate::poset::{FramedPoset, Sign};

struct ColoredHasseGraph {
    graph: Graph,
    vertex_colors: Vec<u32>,
    edge_colors: Vec<u32>,
    cells: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum CanonicalVertexLabel {
    Cell(IntSet),
    InputRelation,
    OutputRelation,
}

struct CanonicalGraph {
    graph: Graph,
    colors: Vec<u32>,
    cell_count: usize,
}

/// Return the canonical normal form of a framed poset.
pub fn normalize(shape: &FramedPoset) -> FramedPoset {
    if shape.is_normal() {
        return shape.clone();
    }

    let canonical = CanonicalGraph::new(shape);
    let permutation = canonical.permutation();
    let sizes = shape.sizes();
    debug_assert_eq!(canonical.cell_count, sizes.iter().copied().sum::<usize>());

    let mut offsets = Vec::with_capacity(sizes.len() + 1);
    offsets.push(0usize);
    for &size in &sizes {
        offsets.push(offsets.last().copied().unwrap() + size);
    }

    let mut new_to_old = Vec::with_capacity(sizes.len());
    let mut old_to_new = Vec::with_capacity(sizes.len());
    for (dim, &size) in sizes.iter().enumerate() {
        let mut order: Vec<usize> = (0..size).collect();
        order.sort_unstable_by_key(|&pos| permutation[offsets[dim] + pos]);

        let mut inverse = vec![0usize; size];
        for (new_pos, &old_pos) in order.iter().enumerate() {
            inverse[old_pos] = new_pos;
        }
        new_to_old.push(order);
        old_to_new.push(inverse);
    }

    let mut basis = Vec::with_capacity(sizes.len());
    let mut faces_in = Vec::with_capacity(sizes.len());
    let mut faces_out = Vec::with_capacity(sizes.len());
    for dim in 0..sizes.len() {
        let mut basis_level = Vec::with_capacity(sizes[dim]);
        let mut faces_in_level = Vec::with_capacity(sizes[dim]);
        let mut faces_out_level = Vec::with_capacity(sizes[dim]);

        for &old_pos in &new_to_old[dim] {
            basis_level.push(shape.basis_of(dim, old_pos).clone());
            if dim == 0 {
                faces_in_level.push(vec![]);
                faces_out_level.push(vec![]);
            } else {
                faces_in_level.push(intset::collect_sorted(
                    shape
                        .faces_of(Sign::Input, dim, old_pos)
                        .iter()
                        .map(|&face| old_to_new[dim - 1][face]),
                ));
                faces_out_level.push(intset::collect_sorted(
                    shape
                        .faces_of(Sign::Output, dim, old_pos)
                        .iter()
                        .map(|&face| old_to_new[dim - 1][face]),
                ));
            }
        }

        basis.push(basis_level);
        faces_in.push(faces_in_level);
        faces_out.push(faces_out_level);
    }

    let mut normalized = FramedPoset::from_faces(basis, faces_in, faces_out);
    normalized.normal = true;
    debug_assert!(vf2_isomorphic(shape, &normalized));
    normalized
}

/// Test whether two framed posets are isomorphic.
pub fn isomorphic(a: &FramedPoset, b: &FramedPoset) -> bool {
    match (a.is_normal(), b.is_normal()) {
        (true, true) => FramedPoset::equal(a, b),
        (true, false) => FramedPoset::equal(a, &normalize(b)),
        (false, true) => FramedPoset::equal(&normalize(a), b),
        (false, false) => FramedPoset::equal(&normalize(a), &normalize(b)),
    }
}

fn vf2_isomorphic(a: &FramedPoset, b: &FramedPoset) -> bool {
    let palette = basis_palette(a, b);
    let a_graph = ColoredHasseGraph::new(a, &palette);
    let b_graph = ColoredHasseGraph::new(b, &palette);

    isomorphic_vf2(
        &a_graph.graph,
        &b_graph.graph,
        Some(&a_graph.vertex_colors),
        Some(&b_graph.vertex_colors),
        Some(&a_graph.edge_colors),
        Some(&b_graph.edge_colors),
    )
    .expect("well-formed framed posets must produce valid VF2 input graphs")
    .iso
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

impl CanonicalGraph {
    fn new(shape: &FramedPoset) -> Self {
        let sizes = shape.sizes();
        let mut offsets = Vec::with_capacity(sizes.len() + 1);
        offsets.push(0usize);
        for &size in &sizes {
            offsets.push(offsets.last().copied().unwrap() + size);
        }

        let mut labels: Vec<CanonicalVertexLabel> = sizes
            .iter()
            .enumerate()
            .flat_map(|(dim, &size)| {
                (0..size)
                    .map(move |pos| CanonicalVertexLabel::Cell(shape.basis_of(dim, pos).clone()))
            })
            .collect();
        let cell_count = labels.len();
        let mut relations = Vec::new();

        for dim in 1..sizes.len() {
            for pos in 0..sizes[dim] {
                for sign in [Sign::Input, Sign::Output] {
                    for &face in shape.faces_of(sign, dim, pos) {
                        relations.push((
                            offsets[dim - 1] + face,
                            offsets[dim] + pos,
                            canonical_relation_label(sign),
                        ));
                    }
                }
            }
        }

        let vertex_count = u32::try_from(cell_count + relations.len())
            .expect("framed poset has too many cells and covers for rust-igraph");
        let mut graph = Graph::new(vertex_count, true)
            .expect("vertex count accepted by rust-igraph must form a graph");

        for (relation_pos, (source, target, label)) in relations.into_iter().enumerate() {
            let relation = cell_count + relation_pos;
            graph
                .add_edge(
                    u32::try_from(source).expect("cell index must fit rust-igraph"),
                    u32::try_from(relation).expect("relation index must fit rust-igraph"),
                )
                .expect("framed-poset cover must form a valid graph edge");
            graph
                .add_edge(
                    u32::try_from(relation).expect("relation index must fit rust-igraph"),
                    u32::try_from(target).expect("cell index must fit rust-igraph"),
                )
                .expect("framed-poset cover must form a valid graph edge");
            labels.push(label);
        }

        let mut palette = labels.clone();
        palette.sort_unstable();
        palette.dedup();
        let colors = labels
            .iter()
            .map(|label| {
                let color = palette
                    .binary_search(label)
                    .expect("every canonical vertex label must occur in its palette");
                u32::try_from(color).expect("too many canonical vertex labels for rust-igraph")
            })
            .collect();

        Self {
            graph,
            colors,
            cell_count,
        }
    }

    fn permutation(&self) -> Vec<u32> {
        canonical_permutation(&self.graph, Some(&self.colors))
            .expect("well-formed framed posets must produce valid canonical-labeling graphs")
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

fn canonical_relation_label(sign: Sign) -> CanonicalVertexLabel {
    match sign {
        Sign::Input => CanonicalVertexLabel::InputRelation,
        Sign::Output => CanonicalVertexLabel::OutputRelation,
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
    use std::collections::HashMap;

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

        assert!(isomorphic(&a, &b));
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

        assert!(isomorphic(&source, &target));
        assert_eq!(isomorphisms.len(), 1);
        assert_eq!(isomorphisms[0].map, vec![vec![1, 0], vec![0]]);
        assert_eq!(isomorphisms[0].inv, vec![vec![1, 0], vec![0]]);
        assert!(isomorphisms[0].is_isomorphism());
    }

    #[test]
    fn edge_orientation_is_preserved() {
        let input = half_arrow(Sign::Input, 0);
        let output = half_arrow(Sign::Output, 0);

        assert!(!isomorphic(&input, &output));
        assert!(isomorphisms(&input, &output).is_empty());
    }

    #[test]
    fn basis_is_preserved() {
        let direction_0 = half_arrow(Sign::Input, 0);
        let direction_1 = half_arrow(Sign::Input, 1);

        assert!(!isomorphic(&direction_0, &direction_1));
        assert!(isomorphisms(&direction_0, &direction_1).is_empty());
    }

    #[test]
    fn normalization_is_canonical_and_idempotent() {
        let source = arrow();
        let reordered = arrow_with_reordered_vertices();

        let normal_source = normalize(&source);
        let normal_reordered = normalize(&reordered);
        let normal_again = normalize(&normal_source);

        assert!(!source.is_normal());
        assert!(!reordered.is_normal());
        assert!(normal_source.is_normal());
        assert!(normal_reordered.is_normal());
        assert!(normal_again.is_normal());
        assert!(FramedPoset::equal(&normal_source, &normal_reordered));
        assert!(FramedPoset::equal(&normal_source, &normal_again));
        assert!(vf2_isomorphic(&source, &normal_source));
        assert!(vf2_isomorphic(&reordered, &normal_reordered));
    }

    #[test]
    fn normalized_posets_deduplicate_as_hash_map_keys() {
        let mut counts = HashMap::new();

        *counts.entry(normalize(&arrow())).or_insert(0) += 1;
        *counts
            .entry(normalize(&arrow_with_reordered_vertices()))
            .or_insert(0) += 1;

        assert_eq!(counts.len(), 1);
        assert_eq!(counts.into_values().next(), Some(2));
    }

    #[test]
    fn normalized_equality_agrees_with_vf2() {
        let cases = [
            arrow(),
            arrow_with_reordered_vertices(),
            half_arrow(Sign::Input, 0),
            half_arrow(Sign::Output, 0),
            half_arrow(Sign::Input, 1),
            points(3),
        ];

        for a in &cases {
            for b in &cases {
                assert_eq!(isomorphic(a, b), vf2_isomorphic(a, b));
            }
        }
    }
}
