use std::collections::HashMap;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ofposets::pushout::{Pushout, pushout};
use ofposets::{BoundaryMode, boundary};
use ofposets::{
    Embedding, FramedPoset, FramedPosetSubset, Renderer, Sign, closure, embedding_to_dot,
    isomorphisms, normalize, to_dot,
};
use serde::{Deserialize, Serialize};

const DEFAULT_FAILURE: &str =
    "visualizations/random_boundary_gluing_failures/seed_933fb66f5574e2f7_pair_14";
const SIGN_PAIRS: [(Sign, Sign); 4] = [
    (Sign::Input, Sign::Input),
    (Sign::Input, Sign::Output),
    (Sign::Output, Sign::Input),
    (Sign::Output, Sign::Output),
];

#[derive(Deserialize)]
struct SavedReport {
    direction: usize,
}

struct Candidate {
    shape: Arc<FramedPoset>,
    cells: usize,
    boundary: Embedding,
    boundary_normal: Arc<FramedPoset>,
}

struct CubularityFailure {
    sign_0: Sign,
    sign_1: Sign,
    zero_then_one: Embedding,
    one_then_zero: Embedding,
}

struct ReducedFailure {
    first: Arc<FramedPoset>,
    second: Arc<FramedPoset>,
    input_boundary: Embedding,
    output_boundary: Embedding,
    boundary_isomorphism: Embedding,
    pushout: Pushout,
    cubularity: CubularityFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Score {
    source_cells: usize,
    pushout_cells: usize,
    largest_source: usize,
}

#[derive(Serialize)]
struct ReducedReport<'a> {
    direction: usize,
    original_first_cells: usize,
    original_second_cells: usize,
    reduced_first_cells: usize,
    reduced_second_cells: usize,
    reduced_pushout_cells: usize,
    failing_sign_0: &'static str,
    failing_sign_1: &'static str,
    boundary_isomorphism_map: &'a [Vec<usize>],
}

fn main() -> io::Result<()> {
    let source_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_FAILURE));
    let report: SavedReport = read_json(&source_dir.join("report.json"))?;
    let first: FramedPoset = read_json(&source_dir.join("first.ofp.json"))?;
    let second: FramedPoset = read_json(&source_dir.join("second.ofp.json"))?;
    let original_first_cells = cell_count(&first);
    let original_second_cells = cell_count(&second);

    let first_subobjects = cubular_closed_subobjects(Arc::new(first))?;
    let second_subobjects = cubular_closed_subobjects(Arc::new(second))?;
    println!(
        "found {} cubular closed subobjects of the first source and {} of the second",
        first_subobjects.len(),
        second_subobjects.len()
    );

    let first_candidates = prepare_candidates(first_subobjects, Sign::Input, report.direction);
    let second_candidates = prepare_candidates(second_subobjects, Sign::Output, report.direction);
    let (failure, compatible_pairs, tested_isomorphisms) =
        minimize(&first_candidates, &second_candidates)?;
    let failure = failure.ok_or_else(|| {
        io::Error::other("no failing gluing was found among the closed subobjects")
    })?;

    let output_dir = unique_output_directory(&source_dir)?;
    write_failure(
        &output_dir,
        report.direction,
        original_first_cells,
        original_second_cells,
        &failure,
    )?;
    println!(
        "tested {compatible_pairs} compatible source pairs and {tested_isomorphisms} boundary isomorphisms"
    );
    println!(
        "reduced source sizes from ({original_first_cells}, {original_second_cells}) to ({}, {}), with a {}-cell pushout; wrote {}",
        cell_count(&failure.first),
        cell_count(&failure.second),
        cell_count(&failure.pushout.tip),
        output_dir.display()
    );
    Ok(())
}

fn cubular_closed_subobjects(shape: Arc<FramedPoset>) -> io::Result<Vec<Arc<FramedPoset>>> {
    let sizes = shape.sizes();
    let cells = sizes.iter().sum::<usize>();
    if cells >= usize::BITS as usize {
        return Err(io::Error::other(format!(
            "cannot exhaustively enumerate subsets of a {cells}-cell OFP"
        )));
    }

    let mut unique = HashMap::<Arc<FramedPoset>, Vec<u8>>::new();
    for mask in 0usize..(1usize << cells) {
        let mut offset = 0usize;
        let keep: Vec<Vec<bool>> = sizes
            .iter()
            .map(|&size| {
                let row = (0..size)
                    .map(|pos| mask & (1usize << (offset + pos)) != 0)
                    .collect();
                offset += size;
                row
            })
            .collect();
        let subset = FramedPosetSubset::make(Arc::clone(&shape), keep);
        let (closed, _) = closure(&subset);
        let normal = Arc::new(normalize(&closed));
        if !is_hat_cubular(&normal) || unique.contains_key(&normal) {
            continue;
        }
        let serialized = serde_json::to_vec(normal.as_ref()).map_err(io::Error::other)?;
        unique.insert(normal, serialized);
    }

    let mut subobjects: Vec<(Arc<FramedPoset>, Vec<u8>)> = unique.into_iter().collect();
    subobjects.sort_unstable_by(|(left, left_json), (right, right_json)| {
        cell_count(left)
            .cmp(&cell_count(right))
            .then_with(|| left_json.cmp(right_json))
    });
    Ok(subobjects.into_iter().map(|(shape, _)| shape).collect())
}

fn prepare_candidates(
    shapes: Vec<Arc<FramedPoset>>,
    sign: Sign,
    direction: usize,
) -> Vec<Candidate> {
    shapes
        .into_iter()
        .map(|shape| {
            let (_, boundary) = boundary(BoundaryMode::Hat, sign, direction, &shape);
            Candidate {
                cells: cell_count(&shape),
                boundary_normal: Arc::new(normalize(&boundary.dom)),
                shape,
                boundary,
            }
        })
        .collect()
}

fn minimize(
    first_candidates: &[Candidate],
    second_candidates: &[Candidate],
) -> io::Result<(Option<ReducedFailure>, u64, u64)> {
    let mut best: Option<(Score, ReducedFailure)> = None;
    let mut compatible_pairs = 0u64;
    let mut tested_isomorphisms = 0u64;

    for first in first_candidates {
        for second in second_candidates {
            let source_cells = first.cells + second.cells;
            if best
                .as_ref()
                .is_some_and(|(score, _)| source_cells > score.source_cells)
                || !FramedPoset::equal(&first.boundary_normal, &second.boundary_normal)
            {
                continue;
            }
            compatible_pairs = compatible_pairs
                .checked_add(1)
                .ok_or_else(|| io::Error::other("compatible pair counter overflow"))?;

            for boundary_isomorphism in isomorphisms(&first.boundary.dom, &second.boundary.dom) {
                tested_isomorphisms = tested_isomorphisms
                    .checked_add(1)
                    .ok_or_else(|| io::Error::other("isomorphism counter overflow"))?;
                let into_second = Embedding::compose(&boundary_isomorphism, &second.boundary);
                let glued = pushout(&first.boundary, &into_second);
                let Some(cubularity) = cubularity_failure(&glued.tip) else {
                    continue;
                };

                let score = Score {
                    source_cells,
                    pushout_cells: cell_count(&glued.tip),
                    largest_source: first.cells.max(second.cells),
                };
                if best
                    .as_ref()
                    .is_none_or(|(best_score, _)| score < *best_score)
                {
                    println!(
                        "found smaller failure with source sizes ({}, {}), pushout size {}",
                        first.cells, second.cells, score.pushout_cells
                    );
                    best = Some((
                        score,
                        ReducedFailure {
                            first: Arc::clone(&first.shape),
                            second: Arc::clone(&second.shape),
                            input_boundary: first.boundary.clone(),
                            output_boundary: second.boundary.clone(),
                            boundary_isomorphism,
                            pushout: glued,
                            cubularity,
                        },
                    ));
                }
            }
        }
    }

    Ok((
        best.map(|(_, failure)| failure),
        compatible_pairs,
        tested_isomorphisms,
    ))
}

fn cubularity_failure(shape: &Arc<FramedPoset>) -> Option<CubularityFailure> {
    SIGN_PAIRS.into_iter().find_map(|(sign_0, sign_1)| {
        let zero_then_one = iterated_hat_boundary(shape, sign_0, 0, sign_1, 1);
        let one_then_zero = iterated_hat_boundary(shape, sign_1, 1, sign_0, 0);
        (!Embedding::equal(&zero_then_one, &one_then_zero)).then_some(CubularityFailure {
            sign_0,
            sign_1,
            zero_then_one,
            one_then_zero,
        })
    })
}

fn is_hat_cubular(shape: &Arc<FramedPoset>) -> bool {
    cubularity_failure(shape).is_none()
}

fn iterated_hat_boundary(
    shape: &Arc<FramedPoset>,
    first_sign: Sign,
    first_direction: usize,
    second_sign: Sign,
    second_direction: usize,
) -> Embedding {
    let (first_boundary, first_embedding) =
        boundary(BoundaryMode::Hat, first_sign, first_direction, shape);
    let (_, second_embedding) = boundary(
        BoundaryMode::Hat,
        second_sign,
        second_direction,
        &first_boundary,
    );
    Embedding::compose(&second_embedding, &first_embedding)
}

fn write_failure(
    output_dir: &Path,
    direction: usize,
    original_first_cells: usize,
    original_second_cells: usize,
    failure: &ReducedFailure,
) -> io::Result<()> {
    let report = ReducedReport {
        direction,
        original_first_cells,
        original_second_cells,
        reduced_first_cells: cell_count(&failure.first),
        reduced_second_cells: cell_count(&failure.second),
        reduced_pushout_cells: cell_count(&failure.pushout.tip),
        failing_sign_0: sign_name(failure.cubularity.sign_0),
        failing_sign_1: sign_name(failure.cubularity.sign_1),
        boundary_isomorphism_map: &failure.boundary_isomorphism.map,
    };
    fs::write(
        output_dir.join("report.json"),
        format!(
            "{}\n",
            serde_json::to_string_pretty(&report).map_err(io::Error::other)?
        ),
    )?;

    write_shape_artifacts(output_dir, "first", &failure.first)?;
    write_shape_artifacts(output_dir, "second", &failure.second)?;
    write_shape_artifacts(output_dir, "pushout", &failure.pushout.tip)?;
    write_embedding_artifacts(output_dir, "first_input_boundary", &failure.input_boundary)?;
    write_embedding_artifacts(
        output_dir,
        "second_output_boundary",
        &failure.output_boundary,
    )?;
    write_embedding_artifacts(
        output_dir,
        "boundary_isomorphism",
        &failure.boundary_isomorphism,
    )?;
    write_embedding_artifacts(output_dir, "first_into_pushout", &failure.pushout.inl)?;
    write_embedding_artifacts(output_dir, "second_into_pushout", &failure.pushout.inr)?;
    write_embedding_artifacts(
        output_dir,
        "failing_zero_then_one",
        &failure.cubularity.zero_then_one,
    )?;
    write_embedding_artifacts(
        output_dir,
        "failing_one_then_zero",
        &failure.cubularity.one_then_zero,
    )
}

fn write_shape_artifacts(output_dir: &Path, name: &str, shape: &FramedPoset) -> io::Result<()> {
    fs::write(
        output_dir.join(format!("{name}.ofp.json")),
        format!(
            "{}\n",
            serde_json::to_string_pretty(shape).map_err(io::Error::other)?
        ),
    )?;
    fs::write(
        output_dir.join(format!("{name}_graded.dot")),
        to_dot(shape, Renderer::Ranked),
    )?;
    fs::write(
        output_dir.join(format!("{name}_compass_spring.dot")),
        to_dot(shape, Renderer::CompassSpring),
    )
}

fn write_embedding_artifacts(
    output_dir: &Path,
    name: &str,
    embedding: &Embedding,
) -> io::Result<()> {
    fs::write(
        output_dir.join(format!("{name}_graded.dot")),
        embedding_to_dot(embedding, Renderer::Ranked),
    )?;
    fs::write(
        output_dir.join(format!("{name}_compass_spring.dot")),
        embedding_to_dot(embedding, Renderer::CompassSpring),
    )
}

fn unique_output_directory(source: &Path) -> io::Result<PathBuf> {
    for suffix in 0usize.. {
        let name = if suffix == 0 {
            "minimized".to_owned()
        } else {
            format!("minimized_{suffix}")
        };
        let path = source.join(name);
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    unreachable!()
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> io::Result<T> {
    let input = fs::read(path)?;
    serde_json::from_slice(&input).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {error}", path.display()),
        )
    })
}

fn cell_count(shape: &FramedPoset) -> usize {
    shape.sizes().into_iter().sum()
}

fn sign_name(sign: Sign) -> &'static str {
    match sign {
        Sign::Input => "input",
        Sign::Output => "output",
    }
}
