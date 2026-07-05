//! DOT export for framed-poset Hasse diagrams and embeddings.

use std::collections::HashSet;
use std::fmt::Write;

use crate::embedding::Embedding;
use crate::poset::{FramedPoset, Sign};

/// Render a framed poset as a Graphviz DOT directed graph.
///
/// Edges point from each face to the cell that covers it. Input edges are
/// orange and labelled `-`; output edges are blue and labelled `+`.
pub fn to_dot(shape: &FramedPoset) -> String {
    let mut out = String::new();

    writeln!(&mut out, "digraph ofposet {{").unwrap();
    write_header(&mut out);
    write_nodes(&mut out, shape, &HashSet::new(), false);
    write_ranks(&mut out, shape);
    write_edges(&mut out, shape, &HashSet::new(), false);
    writeln!(&mut out, "}}").unwrap();

    out
}

/// Render an embedding as a Graphviz DOT directed graph.
///
/// The codomain is drawn in full. Nodes in the image are highlighted, and an
/// edge is highlighted exactly when it is the image of an edge in the domain.
/// In particular, a codomain edge whose endpoints are both image nodes is not
/// highlighted unless the corresponding domain edge exists.
pub fn embedding_to_dot(embedding: &Embedding) -> String {
    let mut out = String::new();
    let image_nodes = image_nodes(embedding);
    let image_edges = image_edges(embedding);
    let codomain = embedding.cod.as_ref();

    writeln!(&mut out, "digraph ofposet_embedding {{").unwrap();
    write_header(&mut out);
    write_nodes(&mut out, codomain, &image_nodes, true);
    write_ranks(&mut out, codomain);
    write_edges(&mut out, codomain, &image_edges, true);
    writeln!(&mut out, "}}").unwrap();

    out
}

fn write_header(out: &mut String) {
    writeln!(out, "  graph [rankdir=BT];").unwrap();
    writeln!(
        out,
        "  node [shape=box, style=rounded, fontname=\"sans-serif\"];"
    )
    .unwrap();
    writeln!(out, "  edge [fontname=\"sans-serif\"];").unwrap();
}

fn write_nodes(out: &mut String, shape: &FramedPoset, image: &HashSet<Cell>, mark_image: bool) {
    for dim in 0..shape.sizes().len() {
        for pos in 0..shape.sizes()[dim] {
            let in_image = mark_image && image.contains(&Cell { dim, pos });
            write_node(out, shape, dim, pos, mark_image, in_image);
        }
    }
}

fn write_node(
    out: &mut String,
    shape: &FramedPoset,
    dim: usize,
    pos: usize,
    mark_image: bool,
    in_image: bool,
) {
    if mark_image && in_image {
        writeln!(
            out,
            "  {} [label=\"{}\", style=\"rounded,filled\", color=\"#0891b2\", fillcolor=\"#ecfeff\", penwidth=3];",
            node_id(dim, pos),
            escape_label(&node_label(shape, dim, pos)),
        )
        .unwrap();
    } else if mark_image {
        writeln!(
            out,
            "  {} [label=\"{}\", color=\"#a1a1aa\", fontcolor=\"#71717a\"];",
            node_id(dim, pos),
            escape_label(&node_label(shape, dim, pos)),
        )
        .unwrap();
    } else {
        writeln!(
            out,
            "  {} [label=\"{}\"];",
            node_id(dim, pos),
            escape_label(&node_label(shape, dim, pos)),
        )
        .unwrap();
    }
}

fn write_ranks(out: &mut String, shape: &FramedPoset) {
    for dim in 0..shape.sizes().len() {
        writeln!(out, "  {{ rank=same;").unwrap();
        for pos in 0..shape.sizes()[dim] {
            writeln!(out, "    {};", node_id(dim, pos)).unwrap();
        }
        writeln!(out, "  }}").unwrap();
    }
}

fn write_edges(out: &mut String, shape: &FramedPoset, image: &HashSet<Edge>, mark_image: bool) {
    for dim in 1..shape.sizes().len() {
        for pos in 0..shape.sizes()[dim] {
            for &face in shape.faces_of(Sign::Input, dim, pos) {
                let edge = Edge {
                    sign: Sign::Input,
                    face_dim: dim - 1,
                    face_pos: face,
                    cell_dim: dim,
                    cell_pos: pos,
                };
                write_edge(out, edge, mark_image, image.contains(&edge));
            }
            for &face in shape.faces_of(Sign::Output, dim, pos) {
                let edge = Edge {
                    sign: Sign::Output,
                    face_dim: dim - 1,
                    face_pos: face,
                    cell_dim: dim,
                    cell_pos: pos,
                };
                write_edge(out, edge, mark_image, image.contains(&edge));
            }
        }
    }
}

fn write_edge(out: &mut String, edge: Edge, mark_image: bool, in_image: bool) {
    let (label, sign_color) = sign_style(edge.sign);
    let color = if !mark_image || in_image {
        sign_color
    } else {
        "#d4d4d8"
    };
    let fontcolor = if !mark_image || in_image {
        sign_color
    } else {
        "#71717a"
    };
    let penwidth = if mark_image && in_image { 3 } else { 1 };
    let style = if mark_image && in_image {
        "bold"
    } else {
        "solid"
    };

    writeln!(
        out,
        "  {} -> {} [label=\"{}\", color=\"{}\", fontcolor=\"{}\", penwidth={}, style=\"{}\"];",
        node_id(edge.face_dim, edge.face_pos),
        node_id(edge.cell_dim, edge.cell_pos),
        label,
        color,
        fontcolor,
        penwidth,
        style,
    )
    .unwrap();
}

fn sign_style(sign: Sign) -> (&'static str, &'static str) {
    match sign {
        Sign::Input => ("-", "#c2410c"),
        Sign::Output => ("+", "#2563eb"),
    }
}

fn image_nodes(embedding: &Embedding) -> HashSet<Cell> {
    embedding
        .map
        .iter()
        .enumerate()
        .flat_map(|(dim, row)| row.iter().copied().map(move |pos| Cell { dim, pos }))
        .collect()
}

fn image_edges(embedding: &Embedding) -> HashSet<Edge> {
    let mut edges = HashSet::new();

    for dim in 1..embedding.dom.sizes().len() {
        for pos in 0..embedding.dom.sizes()[dim] {
            for &face in embedding.dom.faces_of(Sign::Input, dim, pos) {
                edges.insert(Edge {
                    sign: Sign::Input,
                    face_dim: dim - 1,
                    face_pos: embedding.map[dim - 1][face],
                    cell_dim: dim,
                    cell_pos: embedding.map[dim][pos],
                });
            }
            for &face in embedding.dom.faces_of(Sign::Output, dim, pos) {
                edges.insert(Edge {
                    sign: Sign::Output,
                    face_dim: dim - 1,
                    face_pos: embedding.map[dim - 1][face],
                    cell_dim: dim,
                    cell_pos: embedding.map[dim][pos],
                });
            }
        }
    }

    edges
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Cell {
    dim: usize,
    pos: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Edge {
    sign: Sign,
    face_dim: usize,
    face_pos: usize,
    cell_dim: usize,
    cell_pos: usize,
}

fn node_id(dim: usize, pos: usize) -> String {
    format!("c{}_{}", dim, pos)
}

fn node_label(shape: &FramedPoset, dim: usize, pos: usize) -> String {
    format!(
        "({}, {})\n{}",
        dim,
        pos,
        basis_label(shape.basis_of(dim, pos))
    )
}

fn basis_label(basis: &[usize]) -> String {
    if basis.is_empty() {
        return "{}".to_owned();
    }

    let body = basis
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{}}}", body)
}

fn escape_label(label: &str) -> String {
    label
        .chars()
        .flat_map(|ch| match ch {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '\n' => "\\n".chars().collect::<Vec<_>>(),
            _ => vec![ch],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::poset::boundary;

    use super::*;

    fn square() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::make(
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
            vec![
                vec![vec![0, 2], vec![3], vec![1], vec![]],
                vec![vec![0], vec![], vec![0], vec![]],
                vec![vec![]],
            ],
            vec![
                vec![vec![], vec![0], vec![2], vec![1, 3]],
                vec![vec![], vec![0], vec![], vec![0]],
                vec![vec![]],
            ],
        ))
    }

    #[test]
    fn renders_embedding_image_edges_distinct_from_endpoint_image() {
        let square = square();
        let (_, embedding) = boundary(Sign::Input, 0, &square);
        let dot = embedding_to_dot(&embedding);

        assert!(dot.contains("digraph ofposet_embedding"));
        assert!(dot.contains("c0_0 [label=\"(0, 0)\\n{}\", style=\"rounded,filled\""));
        assert!(dot.contains("c0_2 [label=\"(0, 2)\\n{}\", style=\"rounded,filled\""));
        assert!(dot.contains("c1_2 [label=\"(1, 2)\\n{1}\", style=\"rounded,filled\""));
        assert!(dot.contains("c0_0 -> c1_2 [label=\"-\", color=\"#c2410c\""));
        assert!(dot.contains("c0_2 -> c1_2 [label=\"+\", color=\"#2563eb\""));
        assert!(dot.contains("c0_0 -> c1_0 [label=\"-\", color=\"#d4d4d8\""));
    }
}
