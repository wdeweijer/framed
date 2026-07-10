//! DOT export for framed-poset Hasse diagrams and embeddings.

use std::collections::HashSet;
use std::fmt::Write;

use crate::compass_spring_nd::{
    AxisPort, Edge as SpringEdge, Graph as SpringGraph, SimParams, simulate_projected_2d,
};
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

/// Render a framed poset as DOT with node coordinates fixed by the
/// compass-directed spring simulation.
///
/// The generated DOT contains `pos="x,y!"` and `pin=true` attributes. Render it
/// with Graphviz's position-respecting engines, for example:
///
/// ```text
/// neato -n2 -Tsvg input.dot -o output.svg
/// ```
pub fn to_compass_spring_dot(shape: &FramedPoset) -> String {
    to_compass_spring_dot_with_params(shape, &SimParams::default())
}

/// Like [`to_compass_spring_dot`], but using caller-provided simulation
/// parameters.
pub fn to_compass_spring_dot_with_params(shape: &FramedPoset, params: &SimParams) -> String {
    let mut out = String::new();
    let positions = compass_spring_positions(shape, params);

    writeln!(&mut out, "digraph ofposet_compass_spring {{").unwrap();
    write_positioned_header(&mut out);
    write_positioned_nodes(&mut out, shape, &positions);
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

fn write_positioned_header(out: &mut String) {
    writeln!(
        out,
        "  graph [layout=neato, outputorder=edgesfirst, overlap=false, splines=true];"
    )
    .unwrap();
    writeln!(
        out,
        "  node [shape=box, style=rounded, fontname=\"sans-serif\", pin=true];"
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

fn write_positioned_nodes(out: &mut String, shape: &FramedPoset, positions: &[[f64; 2]]) {
    let mut index = 0;
    for dim in 0..shape.sizes().len() {
        for pos in 0..shape.sizes()[dim] {
            let [x, y] = positions[index];
            writeln!(
                out,
                "  {} [label=\"{}\", pos=\"{:.6},{:.6}!\"];",
                node_id(dim, pos),
                escape_label(&node_label(shape, dim, pos)),
                x,
                y,
            )
            .unwrap();
            index += 1;
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

fn compass_spring_positions(shape: &FramedPoset, params: &SimParams) -> Vec<[f64; 2]> {
    let graph = compass_spring_graph(shape);
    orient_for_screen(simulate_projected_2d(&graph, params))
}

fn orient_for_screen(positions: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    positions.into_iter().map(|[x, y]| [x, -y]).collect()
}

fn compass_spring_graph(shape: &FramedPoset) -> SpringGraph {
    let sizes = shape.sizes();
    let mut node_of_cell: Vec<Vec<usize>> = sizes.iter().map(|&n| vec![0; n]).collect();
    let mut node_count = 0;
    let mut max_axis = None::<usize>;

    for dim in 0..sizes.len() {
        for pos in 0..sizes[dim] {
            node_of_cell[dim][pos] = node_count;
            node_count += 1;
            for &axis in shape.basis_of(dim, pos) {
                max_axis = Some(max_axis.map_or(axis, |current| current.max(axis)));
            }
        }
    }

    let mut edges = Vec::new();
    for dim in 1..sizes.len() {
        for pos in 0..sizes[dim] {
            for &face in shape.faces_of(Sign::Input, dim, pos) {
                edges.push(compass_spring_edge(
                    shape,
                    &node_of_cell,
                    Sign::Input,
                    dim - 1,
                    face,
                    dim,
                    pos,
                ));
            }
            for &face in shape.faces_of(Sign::Output, dim, pos) {
                edges.push(compass_spring_edge(
                    shape,
                    &node_of_cell,
                    Sign::Output,
                    dim - 1,
                    face,
                    dim,
                    pos,
                ));
            }
        }
    }

    SpringGraph {
        dim: max_axis.map_or(1, |axis| axis + 1),
        node_count,
        edges,
    }
}

fn compass_spring_edge(
    shape: &FramedPoset,
    node_of_cell: &[Vec<usize>],
    sign: Sign,
    face_dim: usize,
    face_pos: usize,
    cell_dim: usize,
    cell_pos: usize,
) -> SpringEdge {
    let axis = added_axis(
        shape.basis_of(face_dim, face_pos),
        shape.basis_of(cell_dim, cell_pos),
    );
    let positive = sign == Sign::Input;

    SpringEdge {
        tail: node_of_cell[face_dim][face_pos],
        head: node_of_cell[cell_dim][cell_pos],
        tail_port: Some(AxisPort::new(axis, positive)),
        head_port: Some(AxisPort::new(axis, !positive)),
    }
}

fn added_axis(face_basis: &[usize], cell_basis: &[usize]) -> usize {
    cell_basis
        .iter()
        .copied()
        .find(|axis| face_basis.binary_search(axis).is_err())
        .expect("a cover relation must add one basis direction")
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

    #[test]
    fn compass_spring_dot_pins_nodes() {
        let square = square();
        let dot = to_compass_spring_dot(&square);

        assert!(dot.contains("digraph ofposet_compass_spring"));
        assert!(dot.contains("layout=neato"));
        assert!(dot.contains("pos=\""));
        assert!(!dot.contains("rank=same"));
    }

    #[test]
    fn compass_spring_orients_square_input_top_left() {
        let square = square();
        let positions = compass_spring_positions(&square, &SimParams::default());

        let input = positions[0];
        let output_0 = positions[1];
        let output_1 = positions[2];

        assert!(output_0[0] > input[0], "{{0}} direction should point right");
        assert!(
            (output_0[1] - input[1]).abs() < 1.0,
            "{{0}} direction should stay horizontal"
        );
        assert!(
            (output_1[0] - input[0]).abs() < 1.0,
            "{{1}} direction should stay vertical"
        );
        assert!(
            output_1[1] < input[1],
            "{{1}} direction should point down in DOT coordinates"
        );
    }
}
