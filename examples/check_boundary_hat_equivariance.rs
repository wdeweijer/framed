use std::env;
use std::io;
use std::sync::Arc;
use std::time::Instant;

use ofposets::poset::boundary_hat;
use ofposets::{
    DirectionImage, Embedding, FramedPoset, RandomFramedPosetGenerator, Sign, SignedPermutation,
    transform, transform_embedding,
};
use rand::rngs::{OsRng, SmallRng};
use rand::{SeedableRng, TryRngCore};

const CELL_COUNT: usize = 9;
const DEFAULT_SAMPLE_COUNT: u64 = 100_000;
const REPORT_EVERY: u64 = 10_000;

fn main() -> io::Result<()> {
    let (sample_count, seed) = arguments()?;
    let symmetries = two_dimensional_symmetries();
    let mut rng = SmallRng::seed_from_u64(seed);
    let generator = RandomFramedPosetGenerator::new(2, CELL_COUNT);
    let started = Instant::now();

    println!(
        "checking boundary equivariance on {sample_count} random {CELL_COUNT}-cell two-dimensional OFPs (seed {seed:#018x})"
    );

    for sample in 1..=sample_count {
        let shape = Arc::new(generator.generate(&mut rng));
        check_equivariance(&shape, &symmetries, seed, sample)?;

        if sample.is_multiple_of(REPORT_EVERY) || sample == sample_count {
            println!("checked {sample} OFPs ({:.1?})", started.elapsed());
        }
    }

    println!("boundary was equivariant under all eight symmetries for all {sample_count} samples");
    Ok(())
}

fn check_equivariance(
    shape: &Arc<FramedPoset>,
    symmetries: &[SignedPermutation],
    seed: u64,
    sample: u64,
) -> io::Result<()> {
    let source_boundaries: Vec<(Sign, usize, Embedding)> = [Sign::Input, Sign::Output]
        .into_iter()
        .flat_map(|sign| {
            (0..2).map(move |direction| {
                let (_, embedding) = boundary_hat(sign, direction, shape);
                (sign, direction, embedding)
            })
        })
        .collect();

    for (symmetry_index, symmetry) in symmetries.iter().enumerate() {
        let transformed_shape = Arc::new(transform(shape, symmetry).map_err(io::Error::other)?);

        for (source_sign, source_direction, source_boundary) in &source_boundaries {
            let direction_image = symmetry
                .image_of(*source_direction)
                .expect("every two-dimensional symmetry covers directions 0 and 1");
            let target_sign = if direction_image.reflected {
                opposite(*source_sign)
            } else {
                *source_sign
            };
            let transformed_boundary =
                transform_embedding(source_boundary, symmetry).map_err(io::Error::other)?;
            let (_, target_boundary) =
                boundary_hat(target_sign, direction_image.direction, &transformed_shape);

            if !Embedding::equal(&transformed_boundary, &target_boundary) {
                return Err(equivariance_failure(
                    seed,
                    sample,
                    symmetry_index,
                    symmetry,
                    *source_sign,
                    *source_direction,
                    target_sign,
                    direction_image.direction,
                    shape,
                    &transformed_shape,
                    source_boundary,
                    &transformed_boundary,
                    &target_boundary,
                ));
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn equivariance_failure(
    seed: u64,
    sample: u64,
    symmetry_index: usize,
    symmetry: &SignedPermutation,
    source_sign: Sign,
    source_direction: usize,
    target_sign: Sign,
    target_direction: usize,
    source_shape: &FramedPoset,
    transformed_shape: &FramedPoset,
    source_boundary: &Embedding,
    transformed_boundary: &Embedding,
    target_boundary: &Embedding,
) -> io::Error {
    let source_json = serde_json::to_string(source_shape)
        .unwrap_or_else(|error| format!("<serialization failed: {error}>"));
    let transformed_json = serde_json::to_string(transformed_shape)
        .unwrap_or_else(|error| format!("<serialization failed: {error}>"));
    let source_boundary_json = serde_json::to_string(source_boundary.dom.as_ref())
        .unwrap_or_else(|error| format!("<serialization failed: {error}>"));
    let transformed_boundary_json = serde_json::to_string(transformed_boundary.dom.as_ref())
        .unwrap_or_else(|error| format!("<serialization failed: {error}>"));
    let target_boundary_json = serde_json::to_string(target_boundary.dom.as_ref())
        .unwrap_or_else(|error| format!("<serialization failed: {error}>"));

    io::Error::other(format!(
        "boundary equivariance failed with seed {seed:#018x} at sample {sample}: symmetry {symmetry_index} {symmetry:?}, source ({source_sign:?}, {source_direction}), target ({target_sign:?}, {target_direction}); source boundary: {source_boundary_json}; transformed source boundary: {transformed_boundary_json}; direct target boundary: {target_boundary_json}; transformed image map: {:?}; direct target image map: {:?}; source OFP: {source_json}; transformed OFP: {transformed_json}",
        transformed_boundary.map, target_boundary.map
    ))
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

fn opposite(sign: Sign) -> Sign {
    match sign {
        Sign::Input => Sign::Output,
        Sign::Output => Sign::Input,
    }
}

fn arguments() -> io::Result<(u64, u64)> {
    let mut arguments = env::args().skip(1);
    let sample_count = arguments
        .next()
        .map(|value| parse_u64("sample count", &value))
        .transpose()?
        .unwrap_or(DEFAULT_SAMPLE_COUNT);
    let seed = arguments
        .next()
        .map(|value| parse_u64("seed", &value))
        .transpose()?
        .map_or_else(|| OsRng.try_next_u64().map_err(io::Error::other), Ok)?;

    if arguments.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: check_boundary_equivariance [sample-count] [seed]",
        ));
    }
    if sample_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "sample count must be positive",
        ));
    }

    Ok((sample_count, seed))
}

fn parse_u64(name: &str, value: &str) -> io::Result<u64> {
    let parsed = if let Some(hexadecimal) = value.strip_prefix("0x") {
        u64::from_str_radix(hexadecimal, 16)
    } else {
        value.parse()
    };
    parsed.map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid {name} {value:?}: {error}"),
        )
    })
}
