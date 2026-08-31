use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

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

/// A directed edge with an optional compass constraint at each end, `tail_port` constrains the direction
/// from tail to head, `head_port` constrains head to tail. An edge with
/// neither set falls back to a plain distance-only spring.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub tail: usize,
    pub head: usize,
    pub tail_port: Option<AxisPort>,
    pub head_port: Option<AxisPort>,
    /// Whether this edge joins a point (empty basis) to a line
    /// (one-element basis), so it uses the one-dimensional coefficients.
    pub one_dimensional: bool,
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
    pub one_d_compression_k: f64,
    pub one_d_extension_k: f64,
    pub other_compression_k: f64,
    pub other_extension_k: f64,
    pub one_d_angle_k: f64,
    pub one_d_angle_force: f64,
    pub other_angle_k: f64,
    pub other_angle_force: f64,
    pub repulsion: f64,
    pub damping: f64,
    pub center_k: f64,
    pub steps: usize,
}

/// Cabinet projection angles used for the first six directions.
pub const PRESET_ANGLES_DEGREES: [f64; 6] = [0.0, 90.0, 33.0, 6.0, 80.0, 3.0];

/// Cabinet projection scales used for the first six directions.
pub const PRESET_SCALES: [f64; 6] = [1.5, 1.5, 1.0, 4.1, 4.1, 10.0];

const INITIAL_POSITION_SEED: u64 = 0x5eed_c0de_5eed_c0de;

impl Default for SimParams {
    fn default() -> Self {
        SimParams {
            edge_length: 100.0,
            one_d_compression_k: 0.06,
            one_d_extension_k: 0.01,
            other_compression_k: 0.01,
            other_extension_k: 0.01,
            one_d_angle_k: 0.06,
            one_d_angle_force: 0.1,
            other_angle_k: 0.01,
            other_angle_force: 0.0,
            repulsion: 3500.0,
            damping: 0.5,
            center_k: 0.001,
            steps: 1000,
        }
    }
}

fn spring_length_stiffness(stretch: f64, one_dimensional: bool, params: &SimParams) -> f64 {
    match (one_dimensional, stretch < 0.0) {
        (true, true) => params.one_d_compression_k,
        (true, false) => params.one_d_extension_k,
        (false, true) => params.other_compression_k,
        (false, false) => params.other_extension_k,
    }
}

fn spring_angle_stiffness(one_dimensional: bool, params: &SimParams) -> f64 {
    if one_dimensional {
        params.one_d_angle_k
    } else {
        params.other_angle_k
    }
}

fn spring_constant_angle_force(one_dimensional: bool, params: &SimParams) -> f64 {
    if one_dimensional {
        params.one_d_angle_force
    } else {
        params.other_angle_force
    }
}

/// The signed Euclidean-length error for one edge. Negative stretch uses the
/// compression coefficient and positive stretch uses the extension coefficient.
fn spring_length_error(
    actual: &[f64],
    fallback_direction: &[f64],
    one_dimensional: bool,
    params: &SimParams,
) -> Vec<f64> {
    let distance = vector::length(actual);
    let direction = if distance > 1e-6 {
        vector::scale(actual, 1.0 / distance)
    } else {
        fallback_direction.to_vec()
    };
    let stretch = distance - params.edge_length;
    let stiffness = spring_length_stiffness(stretch, one_dimensional, params);
    vector::scale(&direction, stiffness * stretch)
}

/// The tangential force error that rotates an edge toward one endpoint's port.
/// Its proportional part has magnitude `distance * angle_k * sin(angle)`;
/// its constant part has the configured magnitude whenever the angle is nonzero.
fn directional_angle_error(
    actual: &[f64],
    port: AxisPort,
    dim: usize,
    one_dimensional: bool,
    params: &SimParams,
) -> Vec<f64> {
    let distance = vector::length(actual);
    if distance <= 1e-6 {
        return vector::zero(dim);
    }

    let axis = port.unit_vector(dim);
    let direction = vector::scale(actual, 1.0 / distance);
    let direction_alignment = vector::dot(&direction, &axis);
    let tangent_error = vector::sub(&vector::scale(&direction, direction_alignment), &axis);
    let tangent_magnitude = vector::length(&tangent_error);
    if tangent_magnitude <= 1e-8 {
        return vector::zero(dim);
    }

    let proportional_scale = distance * spring_angle_stiffness(one_dimensional, params);
    let constant_scale = spring_constant_angle_force(one_dimensional, params) / tangent_magnitude;
    vector::scale(&tangent_error, proportional_scale + constant_scale)
}

/// Deterministic random initial layout in every simulation dimension.
fn random_start(node_count: usize, dim: usize, radius: f64) -> Vec<Vec<f64>> {
    let radius = radius.abs();
    let mut rng = SmallRng::seed_from_u64(INITIAL_POSITION_SEED);

    (0..node_count)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    if radius == 0.0 {
                        0.0
                    } else {
                        rng.random_range(-radius..radius)
                    }
                })
                .collect()
        })
        .collect()
}

/// Runs the compass-directed spring simulation in `graph.dim` dimensions
/// and returns the final position of every node as an n-vector, indexed
/// the same way as the input graph.
pub fn simulate(graph: &Graph, params: &SimParams) -> Vec<Vec<f64>> {
    let n = graph.node_count;
    let dim = graph.dim;

    let mut pos = random_start(n, dim, params.edge_length);
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

        // One Euclidean-length spring per edge, plus an angular spring at
        // each endpoint that has a compass port.
        for edge in &graph.edges {
            let (tail, head) = (edge.tail, edge.head);
            let delta = vector::sub(&pos[head], &pos[tail]);

            let fallback_direction = if let Some(port) = edge.tail_port {
                port.unit_vector(dim)
            } else if let Some(port) = edge.head_port {
                vector::scale(&port.unit_vector(dim), -1.0)
            } else {
                vector::zero(dim)
            };
            let length_error =
                spring_length_error(&delta, &fallback_direction, edge.one_dimensional, params);
            vector::sub_in_place(&mut force[head], &length_error);
            vector::add_in_place(&mut force[tail], &length_error);

            if let Some(port) = edge.tail_port {
                let angle_error =
                    directional_angle_error(&delta, port, dim, edge.one_dimensional, params);
                vector::sub_in_place(&mut force[head], &angle_error);
                vector::add_in_place(&mut force[tail], &angle_error);
            }

            if let Some(port) = edge.head_port {
                let actual = vector::sub(&pos[tail], &pos[head]);
                let angle_error =
                    directional_angle_error(&actual, port, dim, edge.one_dimensional, params);
                vector::sub_in_place(&mut force[tail], &angle_error);
                vector::add_in_place(&mut force[head], &angle_error);
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

/// Project n-dimensional points to 2D for Graphviz output.
///
/// One-dimensional layouts use the x-axis, two-dimensional layouts use the
/// first two axes directly, and higher-dimensional layouts spread their axes
/// around the unit circle to avoid simply dropping coordinates.
pub fn project_to_2d(points: &[Vec<f64>]) -> Vec<[f64; 2]> {
    let dim = points.iter().map(Vec::len).max().unwrap_or(0);
    points
        .iter()
        .map(|point| {
            let mut projected = [0.0, 0.0];
            for (axis, &coord) in point.iter().enumerate() {
                let basis = projection_axis(axis, dim);
                projected[0] += coord * basis[0];
                projected[1] += coord * basis[1];
            }
            projected
        })
        .collect()
}

/// Run the simulation and project the resulting positions to 2D.
pub fn simulate_projected_2d(graph: &Graph, params: &SimParams) -> Vec<[f64; 2]> {
    project_to_2d(&simulate(graph, params))
}

fn projection_axis(axis: usize, dim: usize) -> [f64; 2] {
    let (angle, scale) = if axis < PRESET_ANGLES_DEGREES.len() {
        (
            PRESET_ANGLES_DEGREES[axis].to_radians(),
            PRESET_SCALES[axis],
        )
    } else {
        (
            std::f64::consts::TAU * axis as f64 / dim.max(1) as f64,
            PRESET_SCALES[PRESET_SCALES.len() - 1] + (axis + 1 - PRESET_SCALES.len()) as f64,
        )
    };

    [scale * angle.cos(), scale * angle.sin()]
}

/// Builds the graph for the n-dimensional hypercube: `2^n` nodes (one per
/// n-bit binary string), with an edge between every pair of nodes that
/// differ in exactly one bit. The edge's compass port is that bit's axis,
/// signed by which direction the bit flips.
pub fn hypercube(n: usize) -> Graph {
    let node_count = 1usize << n;
    let bits = |index: usize| -> Vec<i32> { (0..n).map(|b| ((index >> b) & 1) as i32).collect() };

    let mut edges = Vec::new();
    for i in 0..node_count {
        for j in (i + 1)..node_count {
            let a = bits(i);
            let b = bits(j);
            let diff: Vec<i32> = a.iter().zip(&b).map(|(x, y)| y - x).collect();
            let nonzero: Vec<usize> = diff
                .iter()
                .enumerate()
                .filter(|(_, v)| **v != 0)
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
                    one_dimensional: false,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuned_defaults_are_stable() {
        let params = SimParams::default();

        assert_eq!(params.edge_length, 100.0);
        assert_eq!(params.one_d_compression_k, 0.06);
        assert_eq!(params.one_d_extension_k, 0.01);
        assert_eq!(params.other_compression_k, 0.01);
        assert_eq!(params.other_extension_k, 0.01);
        assert_eq!(params.one_d_angle_k, 0.06);
        assert_eq!(params.one_d_angle_force, 0.1);
        assert_eq!(params.other_angle_k, 0.01);
        assert_eq!(params.other_angle_force, 0.0);
        assert_eq!(params.repulsion, 3500.0);
        assert_eq!(params.damping, 0.5);
        assert_eq!(params.center_k, 0.001);
        assert_eq!(params.steps, 1000);
    }

    fn assert_vector_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn length_force_uses_dimension_and_load_specific_coefficients() {
        let params = SimParams::default();
        let fallback = [1.0, 0.0];

        assert_vector_close(
            &spring_length_error(&[80.0, 0.0], &fallback, true, &params),
            &[-1.2, 0.0],
        );
        assert_vector_close(
            &spring_length_error(&[130.0, 0.0], &fallback, true, &params),
            &[0.3, 0.0],
        );
        assert_vector_close(
            &spring_length_error(&[80.0, 0.0], &fallback, false, &params),
            &[-0.2, 0.0],
        );
        assert_vector_close(
            &spring_length_error(&[130.0, 0.0], &fallback, false, &params),
            &[0.3, 0.0],
        );
    }

    #[test]
    fn angle_force_is_tangential_and_uses_dimension_specific_coefficients() {
        let params = SimParams::default();
        let actual = [80.0, 60.0];
        let port = AxisPort::new(0, true);

        let one_d = directional_angle_error(&actual, port, 2, true, &params);
        assert_vector_close(&one_d, &[-2.22, 2.96]);
        assert!((vector::dot(&actual, &one_d)).abs() < 1e-12);

        let other = directional_angle_error(&actual, port, 2, false, &params);
        assert_vector_close(&other, &[-0.36, 0.48]);
        assert!((vector::dot(&actual, &other)).abs() < 1e-12);
    }

    #[test]
    fn random_start_is_seeded_and_uses_every_dimension() {
        let first = random_start(8, 4, 100.0);
        let second = random_start(8, 4, 100.0);

        assert_eq!(first, second);
        assert!(
            first
                .iter()
                .flatten()
                .all(|coordinate| coordinate.abs() < 100.0)
        );
        for axis in 0..4 {
            assert!(first.iter().any(|point| point[axis] != 0.0));
        }
        assert_eq!(random_start(2, 3, 0.0), vec![vec![0.0; 3]; 2]);
    }

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
                    assert!(
                        delta > 0.0,
                        "dim {n}: expected positive delta on axis {}",
                        port.axis
                    );
                } else {
                    assert!(
                        delta < 0.0,
                        "dim {n}: expected negative delta on axis {}",
                        port.axis
                    );
                }
            }
        }
    }

    #[test]
    fn projection_keeps_third_axis_visible() {
        let points = vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]];
        let projected = project_to_2d(&points);

        assert_ne!(projected[0], projected[1]);
    }
}
