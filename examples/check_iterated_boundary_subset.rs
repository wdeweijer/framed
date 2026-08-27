use std::io;
use std::sync::Arc;

use ofposets::{
    FramedPoset, FramedPosetSubset, RandomFramedPosetGenerator, Sign, boundary, iterated_boundary,
};
use rand::rngs::{OsRng, SmallRng};
use rand::{SeedableRng, TryRngCore};

const CELL_COUNT: usize = 4;
const SAMPLE_COUNT: usize = 10_000;

#[derive(Debug, Clone)]
struct Failure {
    shape: Arc<FramedPoset>,
    first_direction: usize,
    first_sign: Sign,
    second_direction: usize,
    second_sign: Sign,
}

fn main() -> io::Result<()> {
    let seed = OsRng.try_next_u64().map_err(io::Error::other)?;
    let mut rng = SmallRng::seed_from_u64(seed);
    let generator = RandomFramedPosetGenerator::new(2, CELL_COUNT);
    let mut failures = 0usize;
    let mut first_failure = None;

    for i in 0..SAMPLE_COUNT {
        if i % 100_000 == 0 {
            println!("Checked {i} OFPs, {failures} failures")
        }

        let shape = Arc::new(generator.generate(&mut rng));

        for (first_direction, second_direction) in [(0, 1), (1, 0)] {
            for first_sign in [Sign::Input, Sign::Output] {
                for second_sign in [Sign::Input, Sign::Output] {
                    if !iterated_boundary_is_subset(
                        &shape,
                        first_sign,
                        first_direction,
                        second_sign,
                        second_direction,
                    ) {
                        failures += 1;
                        if first_failure.is_none() {
                            first_failure = Some(Failure {
                                shape: Arc::clone(&shape),
                                first_direction,
                                first_sign,
                                second_direction,
                                second_sign,
                            });
                        }
                    }
                }
            }
        }
    }

    println!(
        "generated {SAMPLE_COUNT}, {CELL_COUNT}-cell \
         two-dimensional OFPs with seed {seed:#018x}"
    );
    println!("iterated-boundary subset failures: {failures}");
    if let Some(failure) = first_failure {
        println!(
            "first failure: {}\nboundaries ({:?}, {}) then ({:?}, {})",
            serde_json::to_string(failure.shape.as_ref()).map_err(io::Error::other)?,
            failure.first_sign,
            failure.first_direction,
            failure.second_sign,
            failure.second_direction
        );
    } else {
        println!("every iterated boundary was a subset of its first boundary");
    }

    Ok(())
}

fn iterated_boundary_is_subset(
    shape: &Arc<FramedPoset>,
    first_sign: Sign,
    first_direction: usize,
    second_sign: Sign,
    second_direction: usize,
) -> bool {
    debug_assert_ne!(first_direction, second_direction);

    let (_, iterated_into_shape) = iterated_boundary(
        &[
            (first_sign, first_direction),
            (second_sign, second_direction),
        ],
        shape,
    );
    let iterated_subset = FramedPosetSubset::from_embedding(&iterated_into_shape);

    let (_, second_into_shape) = boundary(second_sign, second_direction, shape);
    let second_subset = FramedPosetSubset::from_embedding(&second_into_shape);
    iterated_subset.is_subset_of(&second_subset)
}
