use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;

use ofposets::pushout::{Pushout, paste_along_boundary};
use ofposets::{
    BoundaryMode, Embedding, FramedPoset, Renderer, Sign, boundary, embedding_to_dot, to_dot,
};

const OUTPUT_DIR: &str = "visualizations/subdivided_boundary_pasting";

fn main() -> io::Result<()> {
    let square = standard_square();
    let stacked_squares = paste_along_boundary(&square, &square, 1);
    assert_eq!(stacked_squares.tip.sizes(), vec![6, 7, 2]);

    let rectangle = subdivided_input_rectangle();
    let (rectangle_input, _) = boundary(BoundaryMode::Plain, Sign::Input, 0, &rectangle);
    let (rectangle_output, rectangle_output_embedding) =
        boundary(BoundaryMode::Plain, Sign::Output, 0, &rectangle);
    let (rectangle_hat_output, rectangle_hat_output_embedding) =
        boundary(BoundaryMode::Hat, Sign::Output, 0, &rectangle);
    let (rectangle_maximal_output, rectangle_maximal_output_embedding) =
        boundary(BoundaryMode::Maximal, Sign::Output, 0, &rectangle);
    assert_eq!(rectangle_input.sizes(), vec![3, 2]);
    assert_eq!(rectangle_output.sizes(), vec![2, 1]);
    assert_eq!(rectangle_hat_output.sizes(), vec![3, 1]);
    assert_eq!(rectangle_maximal_output.sizes(), vec![2, 1]);
    assert!(Embedding::equal(
        &rectangle_output_embedding,
        &rectangle_maximal_output_embedding,
    ));

    let (stacked_output, _) = boundary(BoundaryMode::Plain, Sign::Output, 0, &stacked_squares.tip);
    assert_eq!(stacked_output.sizes(), vec![3, 2]);

    let pasted = paste_along_boundary(&stacked_squares.tip, &rectangle, 0);
    assert_eq!(pasted.tip.sizes(), vec![8, 10, 3]);

    let expected_output_in_pasted = Embedding::compose(&rectangle_output_embedding, &pasted.inr);
    let (_, pasted_output) = boundary(BoundaryMode::Plain, Sign::Output, 0, &pasted.tip);
    assert!(Embedding::equal(&expected_output_in_pasted, &pasted_output));
    let expected_maximal_output_in_pasted =
        Embedding::compose(&rectangle_maximal_output_embedding, &pasted.inr);
    let (_, pasted_maximal_output) = boundary(BoundaryMode::Maximal, Sign::Output, 0, &pasted.tip);
    assert!(Embedding::equal(
        &expected_maximal_output_in_pasted,
        &pasted_maximal_output,
    ));
    write_diagrams(
        &stacked_squares,
        &rectangle,
        &pasted,
        &pasted_output,
        &rectangle_hat_output_embedding,
        &pasted_maximal_output,
    )?;

    println!(
        "pasted shape sizes {:?}; 0-output boundary sizes {:?}",
        pasted.tip.sizes(),
        rectangle_output.sizes(),
    );
    println!("wrote diagrams to {OUTPUT_DIR}");
    Ok(())
}

fn standard_square() -> Arc<FramedPoset> {
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

/// A single 2-cell whose direction-0 input consists of two direction-1
/// arrows, while its direction-0 output consists of one direction-1 arrow.
fn subdivided_input_rectangle() -> Arc<FramedPoset> {
    Arc::new(FramedPoset::from_faces(
        vec![
            vec![vec![], vec![], vec![], vec![], vec![]],
            vec![vec![0], vec![0], vec![1], vec![1], vec![1]],
            vec![vec![0, 1]],
        ],
        vec![
            vec![vec![], vec![], vec![], vec![], vec![]],
            vec![vec![0], vec![3], vec![0], vec![2], vec![1]],
            vec![vec![0, 2, 3]],
        ],
        vec![
            vec![vec![], vec![], vec![], vec![], vec![]],
            vec![vec![1], vec![4], vec![2], vec![3], vec![4]],
            vec![vec![1, 4]],
        ],
    ))
}

fn write_diagrams(
    stacked_squares: &Pushout,
    rectangle: &Arc<FramedPoset>,
    pasted: &Pushout,
    rectangle_output: &Embedding,
    rectangle_hat_output: &Embedding,
    pasted_maximal_output: &Embedding,
) -> io::Result<()> {
    let output_dir = Path::new(OUTPUT_DIR);
    fs::create_dir_all(output_dir)?;

    write_shape(output_dir, "stacked_squares", &stacked_squares.tip)?;
    write_shape(output_dir, "single_cell_rectangle", rectangle)?;
    write_shape(output_dir, "pasted", &pasted.tip)?;
    write_embedding(output_dir, "pasted_stacked_squares", &pasted.inl)?;
    write_embedding(output_dir, "pasted_single_cell_rectangle", &pasted.inr)?;
    write_embedding(
        output_dir,
        "pasted_plain_0_output_boundary",
        rectangle_output,
    )?;
    write_embedding(
        output_dir,
        "rectangle_hat_0_output_boundary",
        rectangle_hat_output,
    )?;
    write_embedding(
        output_dir,
        "pasted_maximal_0_output_boundary",
        pasted_maximal_output,
    )?;
    Ok(())
}

fn write_shape(output_dir: &Path, name: &str, shape: &Arc<FramedPoset>) -> io::Result<()> {
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
