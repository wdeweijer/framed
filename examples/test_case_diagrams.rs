use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use ofposets::{FramedPoset, Sign, boundary, embedding_to_dot, to_compass_spring_dot, to_dot};

fn main() -> std::io::Result<()> {
    let output_dir = Path::new("visualizations");
    fs::create_dir_all(output_dir)?;

    let square = Arc::new(two_direction_square());

    fs::write(output_dir.join("two_direction_square.dot"), to_dot(&square))?;
    fs::write(
        output_dir.join("two_direction_square_compass_spring.dot"),
        to_compass_spring_dot(&square),
    )?;

    write_boundary(output_dir, &square, Sign::Input, 0, "minus_0")?;
    write_boundary(output_dir, &square, Sign::Output, 0, "plus_0")?;
    write_boundary(output_dir, &square, Sign::Input, 1, "minus_1")?;
    write_boundary(output_dir, &square, Sign::Output, 1, "plus_1")?;

    for dim in 3..=5 {
        let cube = n_cube(dim);
        fs::write(
            output_dir.join(format!("n_cube_{}_compass_spring.dot", dim)),
            to_compass_spring_dot(&cube),
        )?;
    }

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
        to_dot(&domain),
    )?;
    fs::write(
        output_dir.join(format!(
            "two_direction_square_boundary_{}_embedding.dot",
            name
        )),
        embedding_to_dot(&embedding),
    )
}

fn two_direction_square() -> FramedPoset {
    n_cube(2)
}

fn n_cube(n: usize) -> FramedPoset {
    let levels: Vec<Vec<CubeCell>> = (0..=n).map(|dim| cube_cells(n, dim)).collect();
    let index = cube_index(&levels);

    let basis = levels
        .iter()
        .map(|level| level.iter().map(|cell| cell.basis.clone()).collect())
        .collect::<Vec<_>>();

    let mut faces_in = empty_adjacency(&levels);
    let mut faces_out = empty_adjacency(&levels);

    for dim in 1..=n {
        for (pos, cell) in levels[dim].iter().enumerate() {
            for &direction in &cell.basis {
                let face_basis = cell
                    .basis
                    .iter()
                    .copied()
                    .filter(|&basis_direction| basis_direction != direction)
                    .collect::<Vec<_>>();
                let input = CubeCell {
                    basis: face_basis.clone(),
                    fixed: cell.fixed,
                };
                let output = CubeCell {
                    basis: face_basis,
                    fixed: cell.fixed | (1usize << direction),
                };

                faces_in[dim][pos].push(index[&(dim - 1, input)]);
                faces_out[dim][pos].push(index[&(dim - 1, output)]);
            }
        }
    }

    sort_adjacency(&mut faces_in);
    sort_adjacency(&mut faces_out);

    FramedPoset::from_faces(basis, faces_in, faces_out)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CubeCell {
    basis: Vec<usize>,
    fixed: usize,
}

fn cube_cells(n: usize, dim: usize) -> Vec<CubeCell> {
    combinations(n, dim)
        .into_iter()
        .flat_map(|basis| {
            fixed_coordinate_masks(n, &basis)
                .into_iter()
                .map(move |fixed| CubeCell {
                    basis: basis.clone(),
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

fn fixed_coordinate_masks(n: usize, basis: &[usize]) -> Vec<usize> {
    (0..(1usize << n))
        .filter(|&mask| {
            basis
                .iter()
                .all(|&direction| mask & (1usize << direction) == 0)
        })
        .collect()
}
