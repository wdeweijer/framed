use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;

use ofposets::{
    BoundaryMode, DirectionImage, Embedding, FramedPoset, Renderer, Sign, SignedPermutation,
    boundary, embedding_to_dot, to_dot, transform, transform_embedding,
};

const OUTPUT_DIR: &str = "visualizations/smallest_plain_boundary_equivariance_failure";

struct Failure {
    shape_code: usize,
    symmetry_index: usize,
    symmetry: SignedPermutation,
    source_sign: Sign,
    source_direction: usize,
    target_sign: Sign,
    target_direction: usize,
    source: Arc<FramedPoset>,
    transformed: Arc<FramedPoset>,
    source_boundary: Embedding,
    transformed_boundary: Embedding,
    target_boundary: Embedding,
}

fn main() -> io::Result<()> {
    verify_smaller_shapes_are_equivariant()?;

    let symmetries = two_dimensional_symmetries();
    let mut failure = None;
    for shape_code in 0..16 {
        let shape = minimal_two_dimensional_shape(shape_code);
        if let Some(found) = find_failure(shape_code, &shape, &symmetries)? {
            failure = Some(found);
            break;
        }
    }
    let failure = failure
        .ok_or_else(|| io::Error::other("all 16 minimal two-dimensional OFPs were equivariant"))?;

    verify_hat_boundary_is_equivariant(&failure)?;
    write_failure(&failure)?;

    println!("smallest counterexample has 4 cells");
    println!(
        "shape code {}; symmetry {} {:?}; source ({:?}, {}), target ({:?}, {})",
        failure.shape_code,
        failure.symmetry_index,
        failure.symmetry,
        failure.source_sign,
        failure.source_direction,
        failure.target_sign,
        failure.target_direction,
    );
    println!(
        "plain boundary sizes: source {:?}, transformed source {:?}, direct target {:?}",
        failure.source_boundary.dom.sizes(),
        failure.transformed_boundary.dom.sizes(),
        failure.target_boundary.dom.sizes(),
    );
    println!(
        "source OFP: {}",
        serde_json::to_string(failure.source.as_ref()).map_err(io::Error::other)?
    );
    println!("wrote witness diagrams to {OUTPUT_DIR}");
    Ok(())
}

fn verify_smaller_shapes_are_equivariant() -> io::Result<()> {
    for cell_count in 2..4 {
        for shape in one_dimensional_shapes(cell_count) {
            let symmetries = match shape.active_directions().len() {
                1 => one_dimensional_symmetries().to_vec(),
                2 => two_dimensional_symmetries(),
                dimension => unreachable!("enumerated shape has {dimension} active directions"),
            };
            if find_failure(0, &shape, &symmetries)?.is_some() {
                return Err(io::Error::other(format!(
                    "found an unexpected one-dimensional counterexample with {cell_count} cells"
                )));
            }
        }
    }
    Ok(())
}

fn find_failure(
    shape_code: usize,
    shape: &Arc<FramedPoset>,
    symmetries: &[SignedPermutation],
) -> io::Result<Option<Failure>> {
    let dimension = shape.active_directions().len();
    let source_boundaries: Vec<_> = [Sign::Input, Sign::Output]
        .into_iter()
        .flat_map(|sign| {
            (0..dimension).map(move |direction| {
                let (_, embedding) = boundary(BoundaryMode::Plain, sign, direction, shape);
                (sign, direction, embedding)
            })
        })
        .collect();

    for (symmetry_index, symmetry) in symmetries.iter().enumerate() {
        let transformed = Arc::new(transform(shape, symmetry).map_err(io::Error::other)?);

        for (source_sign, source_direction, source_boundary) in &source_boundaries {
            let image = symmetry
                .image_of(*source_direction)
                .expect("the symmetry covers every active direction");
            let target_sign = if image.reflected {
                opposite(*source_sign)
            } else {
                *source_sign
            };
            let transformed_boundary =
                transform_embedding(source_boundary, symmetry).map_err(io::Error::other)?;
            let (_, target_boundary) = boundary(
                BoundaryMode::Plain,
                target_sign,
                image.direction,
                &transformed,
            );

            if !Embedding::equal(&transformed_boundary, &target_boundary) {
                return Ok(Some(Failure {
                    shape_code,
                    symmetry_index,
                    symmetry: symmetry.clone(),
                    source_sign: *source_sign,
                    source_direction: *source_direction,
                    target_sign,
                    target_direction: image.direction,
                    source: Arc::clone(shape),
                    transformed,
                    source_boundary: source_boundary.clone(),
                    transformed_boundary,
                    target_boundary,
                }));
            }
        }
    }

    Ok(None)
}

/// Enumerate the 16 signings of the unique four-cell profile with bases
/// empty, {0}, {1}, and {0, 1}.
fn minimal_two_dimensional_shape(code: usize) -> Arc<FramedPoset> {
    debug_assert!(code < 16);
    let edge_0_input = code & 1 == 0;
    let edge_1_input = code & 2 == 0;
    let face_0_input = code & 4 == 0;
    let face_1_input = code & 8 == 0;

    let mut faces_in = vec![vec![vec![]], vec![vec![], vec![]], vec![vec![]]];
    let mut faces_out = faces_in.clone();
    assign_face(&mut faces_in[1][0], &mut faces_out[1][0], 0, edge_0_input);
    assign_face(&mut faces_in[1][1], &mut faces_out[1][1], 0, edge_1_input);
    assign_face(&mut faces_in[2][0], &mut faces_out[2][0], 0, face_0_input);
    assign_face(&mut faces_in[2][0], &mut faces_out[2][0], 1, face_1_input);

    Arc::new(FramedPoset::from_faces(
        vec![vec![vec![]], vec![vec![0], vec![1]], vec![vec![0, 1]]],
        faces_in,
        faces_out,
    ))
}

fn assign_face(input: &mut Vec<usize>, output: &mut Vec<usize>, face: usize, is_input: bool) {
    if is_input {
        input.push(face);
    } else {
        output.push(face);
    }
}

fn one_dimensional_shapes(cell_count: usize) -> Vec<Arc<FramedPoset>> {
    let mut shapes = Vec::new();
    for vertex_count in 1..cell_count {
        let edge_count = cell_count - vertex_count;
        let face_patterns = signed_nonempty_subsets(vertex_count);
        enumerate_edge_faces(
            vertex_count,
            edge_count,
            &face_patterns,
            &mut Vec::new(),
            &mut shapes,
        );
    }
    shapes
}

fn signed_nonempty_subsets(size: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    (1..3usize.pow(size as u32))
        .map(|mut code| {
            let mut input = Vec::new();
            let mut output = Vec::new();
            for vertex in 0..size {
                match code % 3 {
                    0 => {}
                    1 => input.push(vertex),
                    2 => output.push(vertex),
                    _ => unreachable!(),
                }
                code /= 3;
            }
            (input, output)
        })
        .collect()
}

fn enumerate_edge_faces(
    vertex_count: usize,
    edge_count: usize,
    patterns: &[(Vec<usize>, Vec<usize>)],
    selected: &mut Vec<usize>,
    shapes: &mut Vec<Arc<FramedPoset>>,
) {
    if selected.len() == edge_count {
        let faces_in: Vec<_> = selected
            .iter()
            .map(|&pattern| patterns[pattern].0.clone())
            .collect();
        let faces_out: Vec<_> = selected
            .iter()
            .map(|&pattern| patterns[pattern].1.clone())
            .collect();
        for directions in edge_direction_assignments(edge_count) {
            shapes.push(Arc::new(FramedPoset::from_faces(
                vec![
                    vec![vec![]; vertex_count],
                    directions
                        .into_iter()
                        .map(|direction| vec![direction])
                        .collect(),
                ],
                vec![vec![vec![]; vertex_count], faces_in.clone()],
                vec![vec![vec![]; vertex_count], faces_out.clone()],
            )));
        }
        return;
    }

    for pattern in 0..patterns.len() {
        selected.push(pattern);
        enumerate_edge_faces(vertex_count, edge_count, patterns, selected, shapes);
        selected.pop();
    }
}

/// Direction assignments with active directions either {0} or {0, 1}.
fn edge_direction_assignments(edge_count: usize) -> Vec<Vec<usize>> {
    let mut assignments = vec![vec![0; edge_count]];
    if edge_count >= 2 {
        for code in 0..1usize << edge_count {
            let directions: Vec<_> = (0..edge_count)
                .map(|edge| usize::from(code & (1 << edge) != 0))
                .collect();
            if directions.contains(&0) && directions.contains(&1) {
                assignments.push(directions);
            }
        }
    }
    assignments
}

fn one_dimensional_symmetries() -> [SignedPermutation; 2] {
    [
        SignedPermutation::identity(1),
        SignedPermutation::reflection(1, 0).expect("direction 0 exists"),
    ]
}

fn two_dimensional_symmetries() -> Vec<SignedPermutation> {
    let mut symmetries = Vec::with_capacity(8);
    for permutation in [[0, 1], [1, 0]] {
        for reflections in 0..4 {
            symmetries.push(
                SignedPermutation::try_new(
                    permutation
                        .into_iter()
                        .enumerate()
                        .map(|(source, direction)| DirectionImage {
                            direction,
                            reflected: reflections & (1 << source) != 0,
                        })
                        .collect(),
                )
                .expect("the two-dimensional symmetry table is valid"),
            );
        }
    }
    symmetries
}

fn verify_hat_boundary_is_equivariant(failure: &Failure) -> io::Result<()> {
    let (_, source_boundary) = boundary(
        BoundaryMode::Hat,
        failure.source_sign,
        failure.source_direction,
        &failure.source,
    );
    let transformed_boundary =
        transform_embedding(&source_boundary, &failure.symmetry).map_err(io::Error::other)?;
    let (_, target_boundary) = boundary(
        BoundaryMode::Hat,
        failure.target_sign,
        failure.target_direction,
        &failure.transformed,
    );
    Embedding::equal(&transformed_boundary, &target_boundary)
        .then_some(())
        .ok_or_else(|| io::Error::other("hat boundary unexpectedly failed equivariance"))
}

fn write_failure(failure: &Failure) -> io::Result<()> {
    let output_dir = Path::new(OUTPUT_DIR);
    fs::create_dir_all(output_dir)?;

    write_shape(output_dir, "source", &failure.source)?;
    write_shape(output_dir, "transformed", &failure.transformed)?;
    write_embedding(output_dir, "source_boundary", &failure.source_boundary)?;
    write_embedding(
        output_dir,
        "transformed_source_boundary",
        &failure.transformed_boundary,
    )?;
    write_embedding(
        output_dir,
        "direct_target_boundary",
        &failure.target_boundary,
    )?;
    Ok(())
}

fn write_shape(output_dir: &Path, name: &str, shape: &Arc<FramedPoset>) -> io::Result<()> {
    fs::write(
        output_dir.join(format!("{name}.ofp.json")),
        format!(
            "{}\n",
            serde_json::to_string_pretty(shape.as_ref()).map_err(io::Error::other)?
        ),
    )?;
    for renderer in [Renderer::Ranked, Renderer::CompassSpring] {
        fs::write(
            output_dir.join(format!("{name}_{}.dot", renderer_name(renderer))),
            to_dot(shape, renderer),
        )?;
    }
    Ok(())
}

fn write_embedding(output_dir: &Path, name: &str, embedding: &Embedding) -> io::Result<()> {
    for renderer in [Renderer::Ranked, Renderer::CompassSpring] {
        fs::write(
            output_dir.join(format!("{name}_{}.dot", renderer_name(renderer))),
            embedding_to_dot(embedding, renderer),
        )?;
    }
    Ok(())
}

fn renderer_name(renderer: Renderer) -> &'static str {
    match renderer {
        Renderer::Ranked => "graded",
        Renderer::CompassSpring => "compass_spring",
    }
}

fn opposite(sign: Sign) -> Sign {
    match sign {
        Sign::Input => Sign::Output,
        Sign::Output => Sign::Input,
    }
}
