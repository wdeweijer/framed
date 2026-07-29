use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use ofposets::poset::boundary_hat;
use ofposets::pushout::{Pushout, pushout};
use ofposets::{
    Embedding, FramedPoset, Renderer, Sign, SignedPermutation, embedding_to_dot, isomorphisms,
    normalize, to_dot, transform,
};
use serde::Deserialize;

const DATASET: &str = "visualizations/random_4_cells_normal_forms_hat_cubular_up_to_symmetry.jsonl";
const OUTPUT_DIR: &str = "visualizations/axial_boundary_gluing_failure_4_cells";
const DIRECTION: usize = 0;

#[derive(Deserialize)]
struct DatasetRecord {
    ofp: FramedPoset,
}

struct AxialFailure {
    first: Arc<FramedPoset>,
    second: Arc<FramedPoset>,
    first_output: Embedding,
    second_input: Embedding,
    boundary_isomorphism: Embedding,
    pasted: Pushout,
    actual_input: Embedding,
    expected_input: Embedding,
    actual_output: Embedding,
    expected_output: Embedding,
}

fn main() -> io::Result<()> {
    let failure = reconstruct_failure()?;
    let output_dir = Path::new(OUTPUT_DIR);
    fs::create_dir_all(output_dir)?;

    write_shape(output_dir, "first", &failure.first)?;
    write_shape(output_dir, "second", &failure.second)?;
    write_shape(output_dir, "pushout", &failure.pasted.tip)?;
    write_embedding(output_dir, "first_output_boundary", &failure.first_output)?;
    write_embedding(output_dir, "second_input_boundary", &failure.second_input)?;
    write_embedding(
        output_dir,
        "boundary_isomorphism",
        &failure.boundary_isomorphism,
    )?;
    write_embedding(
        output_dir,
        "actual_input_boundary_of_pushout",
        &failure.actual_input,
    )?;
    write_embedding(
        output_dir,
        "expected_input_boundary_from_first",
        &failure.expected_input,
    )?;
    write_embedding(
        output_dir,
        "actual_output_boundary_of_pushout",
        &failure.actual_output,
    )?;
    write_embedding(
        output_dir,
        "expected_output_boundary_from_second",
        &failure.expected_output,
    )?;

    let report = serde_json::json!({
        "dataset": DATASET,
        "dataset_line": 1,
        "direction": DIRECTION,
        "first_symmetry": "identity",
        "second_symmetry": "reflection in direction 0",
        "boundary_isomorphism_map": failure.boundary_isomorphism.map,
        "input_boundaries_equal": Embedding::equal(
            &failure.actual_input,
            &failure.expected_input,
        ),
        "output_boundaries_equal": Embedding::equal(
            &failure.actual_output,
            &failure.expected_output,
        ),
    });
    fs::write(
        output_dir.join("report.json"),
        format!(
            "{}\n",
            serde_json::to_string_pretty(&report).map_err(io::Error::other)?
        ),
    )?;

    println!("wrote axial-boundary failure diagrams to {OUTPUT_DIR}");
    Ok(())
}

fn reconstruct_failure() -> io::Result<AxialFailure> {
    let record = read_first_record(Path::new(DATASET))?;
    let source = Arc::new(record.ofp);
    let identity = SignedPermutation::identity(2);
    let reflection = SignedPermutation::reflection(2, 0).expect("direction 0 exists");
    let first = Arc::new(normalize(
        &transform(&source, &identity).map_err(io::Error::other)?,
    ));
    let second = Arc::new(normalize(
        &transform(&source, &reflection).map_err(io::Error::other)?,
    ));
    assert_eq!(first.sizes().iter().sum::<usize>(), 4);
    assert_eq!(second.sizes().iter().sum::<usize>(), 4);

    let (first_output_domain, first_output) = boundary_hat(Sign::Output, DIRECTION, &first);
    let (second_input_domain, second_input) = boundary_hat(Sign::Input, DIRECTION, &second);
    let mut boundary_isomorphisms = isomorphisms(&first_output_domain, &second_input_domain);
    assert_eq!(
        boundary_isomorphisms.len(),
        1,
        "the selected boundaries must have a unique isomorphism"
    );
    let boundary_isomorphism = boundary_isomorphisms.pop().unwrap();
    let first_output_into_second = Embedding::compose(&boundary_isomorphism, &second_input);
    let pasted = pushout(&first_output, &first_output_into_second);

    let (_, actual_input) = boundary_hat(Sign::Input, DIRECTION, &pasted.tip);
    let (_, first_input) = boundary_hat(Sign::Input, DIRECTION, &first);
    let expected_input = Embedding::compose(&first_input, &pasted.inl);
    let (_, actual_output) = boundary_hat(Sign::Output, DIRECTION, &pasted.tip);
    let (_, second_output) = boundary_hat(Sign::Output, DIRECTION, &second);
    let expected_output = Embedding::compose(&second_output, &pasted.inr);

    assert!(actual_input.is_closed());
    assert!(expected_input.is_closed());
    assert!(actual_output.is_closed());
    assert!(expected_output.is_closed());
    assert!(!Embedding::equal(&actual_input, &expected_input));
    assert!(!Embedding::equal(&actual_output, &expected_output));
    assert_transverse_equations(&first, &second, &pasted);

    Ok(AxialFailure {
        first,
        second,
        first_output,
        second_input,
        boundary_isomorphism,
        pasted,
        actual_input,
        expected_input,
        actual_output,
        expected_output,
    })
}

fn assert_transverse_equations(
    first: &Arc<FramedPoset>,
    second: &Arc<FramedPoset>,
    pasted: &Pushout,
) {
    for sign in [Sign::Input, Sign::Output] {
        let (_, actual) = boundary_hat(sign, 1, &pasted.tip);
        let (_, first_boundary) = boundary_hat(sign, 1, first);
        let (_, second_boundary) = boundary_hat(sign, 1, second);
        let first_boundary = Embedding::compose(&first_boundary, &pasted.inl);
        let second_boundary = Embedding::compose(&second_boundary, &pasted.inr);
        let expected = Embedding::union(&first_boundary, &second_boundary).into_codomain;
        assert!(Embedding::equal(&actual, &expected));
    }
}

fn read_first_record(path: &Path) -> io::Result<DatasetRecord> {
    let mut line = String::new();
    BufReader::new(File::open(path)?).read_line(&mut line)?;
    serde_json::from_str(&line).map_err(io::Error::other)
}

fn write_shape(output_dir: &Path, name: &str, shape: &FramedPoset) -> io::Result<()> {
    fs::write(
        output_dir.join(format!("{name}.ofp.json")),
        format!(
            "{}\n",
            serde_json::to_string_pretty(shape).map_err(io::Error::other)?
        ),
    )?;
    write_dot_variants(output_dir, name, |renderer| to_dot(shape, renderer))
}

fn write_embedding(output_dir: &Path, name: &str, embedding: &Embedding) -> io::Result<()> {
    write_dot_variants(output_dir, name, |renderer| {
        embedding_to_dot(embedding, renderer)
    })
}

fn write_dot_variants(
    output_dir: &Path,
    name: &str,
    render: impl Fn(Renderer) -> String,
) -> io::Result<()> {
    for (renderer_name, renderer) in [
        ("graded", Renderer::Ranked),
        ("compass_spring", Renderer::CompassSpring),
    ] {
        fs::write(
            output_dir.join(format!("{name}_{renderer_name}.dot")),
            render(renderer),
        )?;
    }
    Ok(())
}
