use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;

use ofposets::{
    BoundaryMode, Embedding, FramedPoset, Renderer, Sign, SignedPermutation, boundary,
    embedding_to_dot, to_dot, transform, transform_embedding,
};

const OUTPUT_DIR: &str = "visualizations/symmetry_failure_000004af538e4949";
const SOURCE_JSON: &str = r#"{"version":1,"basis":[[[],[]],[[0],[0],[1],[1]],[[0,1],[0,1],[0,1]]],"faces_in":[[[],[]],[[],[1],[0],[]],[[3],[1,3],[0,1,2]]],"faces_out":[[[],[]],[[0,1],[],[],[1]],[[0,1,2],[0],[3]]]}"#;

fn main() -> io::Result<()> {
    let output_dir = Path::new(OUTPUT_DIR);
    fs::create_dir_all(output_dir)?;

    let source: FramedPoset = serde_json::from_str(SOURCE_JSON).map_err(io::Error::other)?;
    let reflection = SignedPermutation::reflection(2, 0).expect("direction 0 exists");
    let reflected = transform(&source, &reflection).map_err(io::Error::other)?;

    assert_hat_boundary_equivariant(&source, &reflected, &reflection);

    write_shape_and_boundaries(output_dir, "source", Arc::new(source))?;
    write_shape_and_boundaries(output_dir, "reflected_0", Arc::new(reflected))?;

    println!("wrote ranked diagrams to {}", output_dir.display());
    Ok(())
}

fn assert_hat_boundary_equivariant(
    source: &FramedPoset,
    transformed: &FramedPoset,
    symmetry: &SignedPermutation,
) {
    let source = Arc::new(source.clone());
    let transformed = Arc::new(transformed.clone());

    for source_sign in [Sign::Input, Sign::Output] {
        for source_direction in 0..symmetry.dimension() {
            let direction_image = symmetry
                .image_of(source_direction)
                .expect("source direction lies in the symmetry");
            let target_sign = if direction_image.reflected {
                match source_sign {
                    Sign::Input => Sign::Output,
                    Sign::Output => Sign::Input,
                }
            } else {
                source_sign
            };

            let (_, source_boundary) =
                boundary(BoundaryMode::Hat, source_sign, source_direction, &source);
            let transformed_boundary = transform_embedding(&source_boundary, symmetry)
                .expect("the symmetry covers every direction in the boundary");
            let (_, target_boundary) = boundary(
                BoundaryMode::Hat,
                target_sign,
                direction_image.direction,
                &transformed,
            );

            assert!(
                Embedding::equal(&transformed_boundary, &target_boundary),
                "the hat boundary is not equivariant for sign {source_sign:?} and direction {source_direction}"
            );
        }
    }
}

fn write_shape_and_boundaries(
    output_dir: &Path,
    name: &str,
    shape: Arc<FramedPoset>,
) -> io::Result<()> {
    fs::write(
        output_dir.join(format!("{name}.dot")),
        to_dot(&shape, Renderer::Ranked),
    )?;
    fs::write(
        output_dir.join(format!("{name}.ofp.json")),
        format!(
            "{}\n",
            serde_json::to_string_pretty(shape.as_ref()).map_err(io::Error::other)?
        ),
    )?;

    for sign in [Sign::Input, Sign::Output] {
        for direction in 0..=1 {
            let (_, emb) = boundary(BoundaryMode::Plain, sign, direction, &shape);
            fs::write(
                output_dir.join(format!(
                    "{name}_boundary_{}_{}.dot",
                    sign_name(sign),
                    direction
                )),
                embedding_to_dot(&emb, Renderer::Ranked),
            )?;
        }
    }

    Ok(())
}

fn sign_name(sign: Sign) -> &'static str {
    match sign {
        Sign::Input => "input",
        Sign::Output => "output",
    }
}
