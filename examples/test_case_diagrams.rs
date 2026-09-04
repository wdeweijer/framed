use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use ofposets::{
    Embedding, FramedPoset, Renderer, Sign, SignedPermutation, boundary, embedding_to_dot, to_dot,
    transform,
};

fn main() -> std::io::Result<()> {
    let output_dir = Path::new("visualizations");
    fs::create_dir_all(output_dir)?;

    let square = Arc::new(two_direction_square());

    fs::write(
        output_dir.join("two_direction_square.dot"),
        to_dot(&square, Renderer::Ranked),
    )?;
    fs::write(
        output_dir.join("two_direction_square_compass_spring.dot"),
        to_dot(&square, Renderer::CompassSpring),
    )?;

    let reflection = SignedPermutation::reflection(2, 0).expect("direction 0 exists");
    let reflected_square =
        Arc::new(transform(&square, &reflection).map_err(std::io::Error::other)?);
    fs::write(
        output_dir.join("two_direction_square_reflected_0.dot"),
        to_dot(&reflected_square, Renderer::Ranked),
    )?;
    fs::write(
        output_dir.join("two_direction_square_reflected_0_compass_spring.dot"),
        to_dot(&reflected_square, Renderer::CompassSpring),
    )?;

    write_boundary(output_dir, &square, Sign::Input, 0, "minus_0")?;
    write_boundary(output_dir, &square, Sign::Output, 0, "plus_0")?;
    write_boundary(output_dir, &square, Sign::Input, 1, "minus_1")?;
    write_boundary(output_dir, &square, Sign::Output, 1, "plus_1")?;

    for dim in 1..=4 {
        let cube = n_cube(dim);
        fs::write(
            output_dir.join(format!("n_cube_{}_compass_spring.dot", dim)),
            to_dot(&cube, Renderer::CompassSpring),
        )?;
    }

    let three_cube = Arc::new(n_cube(3));
    let (minus_0_boundary, minus_0_embedding) = boundary(Sign::Input, 0, &three_cube);
    fs::write(
        output_dir.join("n_cube_3_boundary_minus_0_embedding_compass_spring.dot"),
        embedding_to_dot(&minus_0_embedding, Renderer::CompassSpring),
    )?;

    let (minus_0_minus_1_domain, minus_0_minus_1_embedding) =
        boundary(Sign::Input, 1, &minus_0_boundary);
    let minus_0_minus_1_composite =
        Embedding::compose(&minus_0_minus_1_embedding, &minus_0_embedding);
    fs::write(
        output_dir.join("n_cube_3_boundary_minus_0_minus_1_composite_compass_spring.dot"),
        embedding_to_dot(&minus_0_minus_1_composite, Renderer::CompassSpring),
    )?;

    let (minus_1_boundary, minus_1_embedding) = boundary(Sign::Input, 1, &three_cube);
    let minus_0_union_minus_1 = Embedding::union(&minus_0_embedding, &minus_1_embedding);
    let minus_0_intersection_minus_1 =
        Embedding::intersection(&minus_0_embedding, &minus_1_embedding);
    fs::write(
        output_dir.join("n_cube_3_boundary_minus_0_into_minus_0_union_minus_1_compass_spring.dot"),
        embedding_to_dot(
            &minus_0_union_minus_1.left_into_union,
            Renderer::CompassSpring,
        ),
    )?;

    let (minus_1_minus_0_domain, minus_1_minus_0_embedding) =
        boundary(Sign::Input, 0, &minus_1_boundary);
    let minus_1_minus_0_composite =
        Embedding::compose(&minus_1_minus_0_embedding, &minus_1_embedding);
    let intersection_into_union_via_minus_0 = Embedding::compose(
        &minus_0_intersection_minus_1.into_left,
        &minus_0_union_minus_1.left_into_union,
    );
    let intersection_into_union_via_minus_1 = Embedding::compose(
        &minus_0_intersection_minus_1.into_right,
        &minus_0_union_minus_1.right_into_union,
    );

    assert!(Embedding::same_subobject(
        &minus_0_minus_1_composite,
        &minus_1_minus_0_composite
    ));
    assert!(Embedding::same_subobject(
        &intersection_into_union_via_minus_0,
        &intersection_into_union_via_minus_1
    ));
    assert!(FramedPoset::equal(
        &minus_0_minus_1_domain,
        &minus_1_minus_0_domain
    ));

    fs::write(
        output_dir
            .join("n_cube_3_boundary_minus_0_intersection_minus_1_into_union_compass_spring.dot"),
        embedding_to_dot(
            &intersection_into_union_via_minus_0,
            Renderer::CompassSpring,
        ),
    )?;

    fs::write(
        output_dir.join("n_cube_3_boundary_minus_1_minus_0_composite_compass_spring.dot"),
        embedding_to_dot(&minus_1_minus_0_composite, Renderer::CompassSpring),
    )?;

    let four_cube = Arc::new(n_cube(4));
    let (_, four_minus_0_embedding) = boundary(Sign::Input, 0, &four_cube);
    let (_, four_minus_1_embedding) = boundary(Sign::Input, 1, &four_cube);
    let four_minus_0_union_minus_1 =
        Embedding::union(&four_minus_0_embedding, &four_minus_1_embedding);
    let four_minus_0_intersection_minus_1 =
        Embedding::intersection(&four_minus_0_embedding, &four_minus_1_embedding);
    fs::write(
        output_dir.join("n_cube_4_boundary_minus_0_into_minus_0_union_minus_1_compass_spring.dot"),
        embedding_to_dot(
            &four_minus_0_union_minus_1.left_into_union,
            Renderer::CompassSpring,
        ),
    )?;

    let four_intersection_into_union_via_minus_0 = Embedding::compose(
        &four_minus_0_intersection_minus_1.into_left,
        &four_minus_0_union_minus_1.left_into_union,
    );
    let four_intersection_into_union_via_minus_1 = Embedding::compose(
        &four_minus_0_intersection_minus_1.into_right,
        &four_minus_0_union_minus_1.right_into_union,
    );
    assert!(Embedding::same_subobject(
        &four_intersection_into_union_via_minus_0,
        &four_intersection_into_union_via_minus_1
    ));

    fs::write(
        output_dir
            .join("n_cube_4_boundary_minus_0_intersection_minus_1_into_union_compass_spring.dot"),
        embedding_to_dot(
            &four_intersection_into_union_via_minus_0,
            Renderer::CompassSpring,
        ),
    )?;

    Ok(())
}

fn write_boundary(
    output_dir: &Path,
    shape: &Arc<FramedPoset>,
    sign: Sign,
    direction: usize,
    name: &str,
) -> std::io::Result<()> {
    let (domain, embedding) = boundary(sign, direction, shape);
    fs::write(
        output_dir.join(format!("two_direction_square_boundary_{}.dot", name)),
        to_dot(&domain, Renderer::Ranked),
    )?;
    fs::write(
        output_dir.join(format!(
            "two_direction_square_boundary_{}_embedding.dot",
            name
        )),
        embedding_to_dot(&embedding, Renderer::Ranked),
    )
}

fn two_direction_square() -> FramedPoset {
    n_cube(2)
}

fn n_cube(n: usize) -> FramedPoset {
    let levels: Vec<Vec<CubeCell>> = (0..=n).map(|dim| cube_cells(n, dim)).collect();
    let index = cube_index(&levels);

    let frames = levels
        .iter()
        .map(|level| level.iter().map(|cell| cell.frame.clone()).collect())
        .collect::<Vec<_>>();

    let mut faces_in = empty_adjacency(&levels);
    let mut faces_out = empty_adjacency(&levels);

    for dim in 1..=n {
        for (pos, cell) in levels[dim].iter().enumerate() {
            for &direction in &cell.frame {
                let face_frame = cell
                    .frame
                    .iter()
                    .copied()
                    .filter(|&frame_direction| frame_direction != direction)
                    .collect::<Vec<_>>();
                let input = CubeCell {
                    frame: face_frame.clone(),
                    fixed: cell.fixed,
                };
                let output = CubeCell {
                    frame: face_frame,
                    fixed: cell.fixed | (1usize << direction),
                };

                faces_in[dim][pos].push(index[&(dim - 1, input)]);
                faces_out[dim][pos].push(index[&(dim - 1, output)]);
            }
        }
    }

    sort_adjacency(&mut faces_in);
    sort_adjacency(&mut faces_out);

    FramedPoset::from_faces(frames, faces_in, faces_out)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CubeCell {
    frame: Vec<usize>,
    fixed: usize,
}

fn cube_cells(n: usize, dim: usize) -> Vec<CubeCell> {
    combinations(n, dim)
        .into_iter()
        .flat_map(|frame| {
            fixed_coordinate_masks(n, &frame)
                .into_iter()
                .map(move |fixed| CubeCell {
                    frame: frame.clone(),
                    fixed,
                })
        })
        .collect()
}

fn cube_index(levels: &[Vec<CubeCell>]) -> HashMap<(usize, CubeCell), usize> {
    levels
        .iter()
        .enumerate()
        .flat_map(|(dim, level)| {
            level
                .iter()
                .cloned()
                .enumerate()
                .map(move |(pos, cell)| ((dim, cell), pos))
        })
        .collect()
}

fn empty_adjacency(levels: &[Vec<CubeCell>]) -> Vec<Vec<Vec<usize>>> {
    levels
        .iter()
        .map(|level| vec![vec![]; level.len()])
        .collect()
}

fn sort_adjacency(adjacency: &mut [Vec<Vec<usize>>]) {
    for level in adjacency {
        for faces in level {
            faces.sort_unstable();
            faces.dedup();
        }
    }
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn go(next: usize, n: usize, k: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }

        let remaining = k - current.len();
        for item in next..=n - remaining {
            current.push(item);
            go(item + 1, n, k, current, out);
            current.pop();
        }
    }

    let mut out = Vec::new();
    go(0, n, k, &mut Vec::new(), &mut out);
    out
}

fn fixed_coordinate_masks(n: usize, frame: &[usize]) -> Vec<usize> {
    (0..(1usize << n))
        .filter(|&mask| {
            frame
                .iter()
                .all(|&direction| mask & (1usize << direction) == 0)
        })
        .collect()
}
