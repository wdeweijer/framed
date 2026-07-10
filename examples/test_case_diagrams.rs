use std::fs;
use std::path::Path;
use std::sync::Arc;

use ofposets::{FramedPoset, Sign, boundary, embedding_to_dot, to_dot};

fn main() -> std::io::Result<()> {
    let output_dir = Path::new("visualizations");
    fs::create_dir_all(output_dir)?;

    let square = Arc::new(two_direction_square());

    fs::write(output_dir.join("two_direction_square.dot"), to_dot(&square))?;

    write_boundary(output_dir, &square, Sign::Input, 0, "minus_0")?;
    write_boundary(output_dir, &square, Sign::Output, 0, "plus_0")?;
    write_boundary(output_dir, &square, Sign::Input, 1, "minus_1")?;
    write_boundary(output_dir, &square, Sign::Output, 1, "plus_1")?;

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
    FramedPoset::from_faces(
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
    )
}
