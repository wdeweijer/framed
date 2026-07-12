use std::fs;
use std::path::Path;
use std::sync::Arc;

use ofposets::{Embedding, FramedPoset, Renderer, Sign, boundary, random_framed_poset, to_dot};
use rand::SeedableRng;
use rand::rngs::SmallRng;

const CELL_COUNT: usize = 10;
const SAMPLE_COUNT: usize = 100;
const SEED: u64 = 0x5eed_0f50_5e75;

fn main() -> std::io::Result<()> {
    let output_dir = Path::new("visualizations/random_10_cells");
    let cubular_output_dir = Path::new("visualizations/random_10_cells_cubular");
    fs::create_dir_all(output_dir)?;
    fs::create_dir_all(cubular_output_dir)?;

    let mut rng = SmallRng::seed_from_u64(SEED);
    let mut cubular_count = 0;
    for sample in 0..SAMPLE_COUNT {
        let poset = Arc::new(random_framed_poset(CELL_COUNT, &mut rng));
        let compass_dot = to_dot(&poset, Renderer::CompassSpring);
        let graded_dot = to_dot(&poset, Renderer::Ranked);
        let compass_file_name = format!("sample_{sample:03}.dot");
        let graded_file_name = format!("sample_{sample:03}_graded.dot");

        fs::write(output_dir.join(&compass_file_name), &compass_dot)?;
        fs::write(output_dir.join(&graded_file_name), &graded_dot)?;
        if is_cubular(&poset) {
            fs::write(cubular_output_dir.join(compass_file_name), compass_dot)?;
            fs::write(cubular_output_dir.join(graded_file_name), graded_dot)?;
            cubular_count += 1;
        }
    }

    println!(
        "wrote {SAMPLE_COUNT} OFPs in compass-spring and graded layouts to {}",
        output_dir.display()
    );
    if cubular_count == 0 {
        println!("no cubular OFPs found");
    } else {
        println!(
            "wrote {cubular_count} cubular diagrams to {}",
            cubular_output_dir.display()
        );
    }
    Ok(())
}

fn is_cubular(shape: &Arc<FramedPoset>) -> bool {
    for sign_0 in [Sign::Input, Sign::Output] {
        for sign_1 in [Sign::Input, Sign::Output] {
            let zero_then_one = iterated_boundary(shape, sign_0, 0, sign_1, 1);
            let one_then_zero = iterated_boundary(shape, sign_1, 1, sign_0, 0);

            if !Embedding::equal(&zero_then_one, &one_then_zero) {
                return false;
            }
        }
    }

    true
}

fn iterated_boundary(
    shape: &Arc<FramedPoset>,
    first_sign: Sign,
    first_direction: usize,
    second_sign: Sign,
    second_direction: usize,
) -> Embedding {
    let (first_boundary, first_embedding) = boundary(first_sign, first_direction, shape);
    let (_, second_embedding) = boundary(second_sign, second_direction, &first_boundary);
    Embedding::compose(&second_embedding, &first_embedding)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_is_cubular() {
        let square = Arc::new(FramedPoset::from_faces(
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
        ));

        assert!(is_cubular(&square));
    }
}
