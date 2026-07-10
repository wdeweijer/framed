use std::collections::HashMap;

/// A handful of arbitrary-length vector operations. Vectors are plain
/// `Vec<f64>`; every function just assumes its arguments have the same
/// length rather than assuming any particular dimension.
mod vector {
    pub fn zero(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }

    pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    pub fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x - y).collect()
    }

    pub fn scale(a: &[f64], s: f64) -> Vec<f64> {
        a.iter().map(|x| x * s).collect()
    }

    pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    pub fn length_sq(a: &[f64]) -> f64 {
        dot(a, a)
    }

    pub fn length(a: &[f64]) -> f64 {
        length_sq(a).sqrt()
    }

    pub fn add_in_place(a: &mut [f64], b: &[f64]) {
        for (x, y) in a.iter_mut().zip(b) {
            *x += y;
        }
    }

    pub fn sub_in_place(a: &mut [f64], b: &[f64]) {
        for (x, y) in a.iter_mut().zip(b) {
            *x -= y;
        }
    }
}

/// A compass direction generalized to n dimensions: positive or negative
/// along a single basis axis. `axis(0)`/`axis(1)` correspond to what would
/// be `e`/`n` in the 2D version; there's no bound on how high `axis` can go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxisPort {
    pub axis: usize,
    pub positive: bool,
}

impl AxisPort {
    pub fn new(axis: usize, positive: bool) -> Self {
        AxisPort { axis, positive }
    }

    fn unit_vector(self, dim: usize) -> Vec<f64> {
        let mut v = vector::zero(dim);
        v[self.axis] = if self.positive { 1.0 } else { -1.0 };
        v
    }
}

/// A directed edge with an optional compass constraint at each end, same
/// semantics as the 2D/3D versions: `tail_port` constrains the direction
/// from tail to head, `head_port` constrains head to tail. An edge with
/// neither set falls back to a plain distance-only spring.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub tail: usize,
    pub head: usize,
    pub tail_port: Option<AxisPort>,
    pub head_port: Option<AxisPort>,
}

/// A graph to lay out in `dim` dimensions: `node_count` nodes, indexed
/// `0..node_count`, and a list of edges between them.
#[derive(Clone, Debug)]
pub struct Graph {
    pub dim: usize,
    pub node_count: usize,
    pub edges: Vec<Edge>,
}

/// Tunable simulation parameters, same roles as the 2D/3D versions.
#[derive(Clone, Copy, Debug)]
pub struct SimParams {
    pub edge_length: f64,
    pub spring_k: f64,
    pub repulsion: f64,
    pub damping: f64,
    pub center_k: f64,
    pub steps: usize,
}

impl Default for SimParams {
    fn default() -> Self {
        SimParams {
            edge_length: 100.0,
            spring_k: 0.06,
            repulsion: 6000.0,
            damping: 0.82,
            center_k: 0.004,
            steps: 1000,
        }
    }
}

/// Initial layout: nodes placed evenly spaced along a straight line (the
/// first axis), `spacing` apart, centered on the origin. Every other axis
/// starts at zero. This replaces a random initial layout - the directional
/// springs don't need a randomized start to converge to the target shape,
/// since (unlike a plain distance-only spring layout) they aren't
/// rotationally symmetric to begin with.
fn line_start(node_count: usize, dim: usize, spacing: f64) -> Vec<Vec<f64>> {
    let offset = spacing * (node_count as f64 - 1.0) / 2.0;
    (0..node_count)
        .map(|i| {
            let mut p = vector::zero(dim);
            p[0] = i as f64 * spacing - offset;
            p
        })
        .collect()
}

/// Runs the compass-directed spring simulation in `graph.dim` dimensions
/// and returns the final position of every node as an n-vector, indexed
/// the same way as the input graph.
///
/// Structurally identical to the 2D/3D versions - repulsion, the
/// directional spring, and the centering force are each one call into
/// `vector`, regardless of how many dimensions `graph.dim` is.
pub fn simulate(graph: &Graph, params: &SimParams) -> Vec<Vec<f64>> {
    let n = graph.node_count;
    let dim = graph.dim;

    let mut pos: Vec<Vec<f64>> = line_start(n, dim, params.edge_length * 0.5);
    let mut vel: Vec<Vec<f64>> = (0..n).map(|_| vector::zero(dim)).collect();

    for _ in 0..params.steps {
        let mut force: Vec<Vec<f64>> = (0..n).map(|_| vector::zero(dim)).collect();

        // Pairwise repulsion.
        for i in 0..n {
            for j in (i + 1)..n {
                let d = vector::sub(&pos[j], &pos[i]);
                let d2 = vector::length_sq(&d) + 0.01;
                let dist = d2.sqrt();
                let f = vector::scale(&d, params.repulsion / (d2 * dist));
                vector::sub_in_place(&mut force[i], &f);
                vector::add_in_place(&mut force[j], &f);
            }
        }

        // Directional (or plain) spring per edge.
        for edge in &graph.edges {
            let (tail, head) = (edge.tail, edge.head);

            if let Some(port) = edge.tail_port {
                let target = vector::scale(&port.unit_vector(dim), params.edge_length);
                let actual = vector::sub(&pos[head], &pos[tail]);
                let err = vector::sub(&actual, &target);
                let f = vector::scale(&err, params.spring_k);
                vector::sub_in_place(&mut force[head], &f);
                vector::add_in_place(&mut force[tail], &f);
            }

            if let Some(port) = edge.head_port {
                let target = vector::scale(&port.unit_vector(dim), params.edge_length);
                let actual = vector::sub(&pos[tail], &pos[head]);
                let err = vector::sub(&actual, &target);
                let f = vector::scale(&err, params.spring_k);
                vector::sub_in_place(&mut force[tail], &f);
                vector::add_in_place(&mut force[head], &f);
            }

            if edge.tail_port.is_none() && edge.head_port.is_none() {
                let d = vector::sub(&pos[head], &pos[tail]);
                let dist = vector::length(&d).max(1e-6);
                let stretch = dist - params.edge_length;
                let f = vector::scale(&d, params.spring_k * stretch / dist);
                vector::sub_in_place(&mut force[head], &f);
                vector::add_in_place(&mut force[tail], &f);
            }
        }

        // Centering, then integrate.
        for i in 0..n {
            let centering = vector::scale(&pos[i], params.center_k);
            vector::sub_in_place(&mut force[i], &centering);

            vel[i] = vector::scale(&vector::add(&vel[i], &force[i]), params.damping);
            pos[i] = vector::add(&pos[i], &vel[i]);
        }
    }

    pos
}

/// Builds the graph for the n-dimensional hypercube: `2^n` nodes (one per
/// n-bit binary string), with an edge between every pair of nodes that
/// differ in exactly one bit. The edge's compass port is that bit's axis,
/// signed by which direction the bit flips.
pub fn hypercube(n: usize) -> Graph {
    let node_count = 1usize << n;
    let bits = |index: usize| -> Vec<i32> {
        (0..n).map(|b| ((index >> b) & 1) as i32).collect()
    };

    let mut edges = Vec::new();
    for i in 0..node_count {
        for j in (i + 1)..node_count {
            let a = bits(i);
            let b = bits(j);
            let diff: Vec<i32> = a.iter().zip(&b).map(|(x, y)| y - x).collect();
            let nonzero: Vec<usize> = diff
                .iter()
                .enumerate()
                .filter(|(_, &v)| v != 0)
                .map(|(idx, _)| idx)
                .collect();
            if nonzero.len() == 1 {
                let axis = nonzero[0];
                let positive = diff[axis] > 0;
                edges.push(Edge {
                    tail: i,
                    head: j,
                    tail_port: Some(AxisPort::new(axis, positive)),
                    head_port: Some(AxisPort::new(axis, !positive)),
                });
            }
        }
    }

    Graph {
        dim: n,
        node_count,
        edges,
    }
}

/// The binary-string label for hypercube node `index` in `n` dimensions,
/// e.g. `hypercube_label(5, 4) == "0101"`. Purely a display convenience.
pub fn hypercube_label(index: usize, n: usize) -> String {
    (0..n)
        .rev()
        .map(|b| if (index >> b) & 1 == 1 { '1' } else { '0' })
        .collect()
}

/// Maps node index -> label -> final position, for convenience when you
/// want results keyed by something more readable than a raw index.
pub fn labeled_positions(n: usize, positions: &[Vec<f64>]) -> HashMap<String, Vec<f64>> {
    positions
        .iter()
        .enumerate()
        .map(|(i, p)| (hypercube_label(i, n), p.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypercube_has_right_shape() {
        for n in 1..=6 {
            let g = hypercube(n);
            assert_eq!(g.node_count, 1 << n);
            assert_eq!(g.edges.len(), n * (1 << (n - 1)));
        }
    }

    #[test]
    fn simulated_hypercube_respects_axis_directions() {
        for n in 2..=5 {
            let g = hypercube(n);
            let params = SimParams::default();
            let positions = simulate(&g, &params);

            for edge in &g.edges {
                let port = edge.tail_port.unwrap();
                let delta = positions[edge.head][port.axis] - positions[edge.tail][port.axis];
                if port.positive {
                    assert!(delta > 0.0, "dim {n}: expected positive delta on axis {}", port.axis);
                } else {
                    assert!(delta < 0.0, "dim {n}: expected negative delta on axis {}", port.axis);
                }
            }
        }
    }
}

fn main() {
    let n = 3;
    let g = hypercube(n);
    let params = SimParams::default();
    let positions = simulate(&g, &params);
    let labeled = labeled_positions(n, &positions);

    let mut labels: Vec<&String> = labeled.keys().collect();
    labels.sort();
    for label in labels {
        let p = &labeled[label];
        let coords: Vec<String> = p.iter().map(|v| format!("{v:.1}")).collect();
        println!("{label}: ({})", coords.join(", "));
    }
}
