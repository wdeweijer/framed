use std::env;
use std::io;
use std::sync::Arc;
use std::time::Instant;

use ofposets::{FramedPoset, RandomFramedPosetGenerator, is_volumetric, orthogonal_product, shift};
use rand::rngs::{OsRng, SmallRng};
use rand::{SeedableRng, TryRngCore};
use rayon::prelude::*;

const DEFAULT_EXAMPLE_COUNT: u64 = 100;
const DEFAULT_DIMENSION: usize = 2;
const DEFAULT_CELL_COUNT: usize = 9;
const REPORT_EVERY_CANDIDATES: u64 = 1_000_000;
const CANDIDATES_PER_THREAD: usize = 16;
const PRODUCTS_PER_THREAD: usize = 8;

struct Options {
    example_count: u64,
    dimension: usize,
    cell_count: usize,
    seed: u64,
}

struct Factor {
    candidate: u64,
    shape: Arc<FramedPoset>,
}

#[derive(Debug, Clone, Copy)]
struct ProductCase {
    left: usize,
    right: usize,
}

struct ProductFailure {
    product: usize,
    case: ProductCase,
    left: Arc<FramedPoset>,
    right: Arc<FramedPoset>,
    product_shape: Arc<FramedPoset>,
}

fn main() -> io::Result<()> {
    let options = arguments()?;
    let example_count = usize::try_from(options.example_count)
        .map_err(|_| invalid_input("example count does not fit usize"))?;
    let cases = product_cases(example_count)?;
    let product_count = cases.len();
    let started = Instant::now();

    println!(
        "generating {example_count} random volumetric, {}-dimensional, {}-cell OFPs and checking all {product_count} unordered mutually orthogonal products with {} Rayon threads (seed {:#018x})",
        options.dimension,
        options.cell_count,
        rayon::current_num_threads(),
        options.seed,
    );

    let (factors, candidates) = collect_factors(&options, example_count, started)?;
    let shifted: Vec<_> = factors
        .par_iter()
        .map(|factor| Arc::new(shift_by(&factor.shape, options.dimension)))
        .collect();

    let product_batch_size = rayon::current_num_threads()
        .saturating_mul(PRODUCTS_PER_THREAD)
        .max(1);
    let mut checked = 0usize;
    for batch in cases.chunks(product_batch_size) {
        let failure = batch
            .par_iter()
            .enumerate()
            .find_map_any(|(batch_index, &case)| {
                let left = Arc::clone(&factors[case.left].shape);
                let right = Arc::clone(&shifted[case.right]);
                debug_assert!(ofposets::intset::is_disjoint(
                    &left.total_frame(),
                    &right.total_frame(),
                ));
                let product_shape = Arc::new(orthogonal_product(&left, &right));

                (!is_volumetric(&product_shape)).then_some(ProductFailure {
                    product: checked + batch_index,
                    case,
                    left,
                    right,
                    product_shape,
                })
            });

        if let Some(failure) = failure {
            return Err(volumetricity_failure(
                &options, candidates, &factors, &failure,
            ));
        }

        checked += batch.len();
        println!(
            "checked {checked}/{product_count} products ({:.1?})",
            started.elapsed(),
        );
    }

    println!(
        "all {product_count} orthogonal products were volumetric; generated {candidates} candidates and retained {} volumetric factors ({:.1?})",
        factors.len(),
        started.elapsed(),
    );
    Ok(())
}

fn collect_factors(
    options: &Options,
    example_count: usize,
    started: Instant,
) -> io::Result<(Vec<Factor>, u64)> {
    let generator = RandomFramedPosetGenerator::new(options.dimension, options.cell_count);
    let batch_size = rayon::current_num_threads()
        .saturating_mul(CANDIDATES_PER_THREAD)
        .max(1);
    let mut factors = Vec::with_capacity(example_count);
    let mut candidates = 0u64;
    let mut next_report = REPORT_EVERY_CANDIDATES;

    while factors.len() < example_count {
        let batch_start = candidates;
        let mut found: Vec<_> = (0..batch_size)
            .into_par_iter()
            .filter_map(|offset| {
                let candidate = batch_start + offset as u64 + 1;
                let mut rng = SmallRng::seed_from_u64(candidate_seed(options.seed, candidate));
                let shape = Arc::new(generator.generate(&mut rng));
                is_volumetric(&shape).then_some(Factor { candidate, shape })
            })
            .collect();
        found.sort_unstable_by_key(|factor| factor.candidate);

        candidates = candidates
            .checked_add(batch_size as u64)
            .ok_or_else(|| io::Error::other("candidate counter overflow"))?;
        factors.extend(found);

        if candidates >= next_report || factors.len() >= example_count {
            println!(
                "generated {candidates} candidates; retained {}/{} volumetric factors ({:.1?})",
                factors.len().min(example_count),
                example_count,
                started.elapsed(),
            );
            while next_report <= candidates {
                next_report = next_report.saturating_add(REPORT_EVERY_CANDIDATES);
                if next_report == u64::MAX {
                    break;
                }
            }
        }
    }

    factors.truncate(example_count);
    Ok((factors, candidates))
}

fn product_cases(example_count: usize) -> io::Result<Vec<ProductCase>> {
    let maximum_pairs = example_count
        .checked_mul(
            example_count
                .checked_add(1)
                .ok_or_else(|| invalid_input("example count is too large"))?,
        )
        .map(|pairs| pairs / 2)
        .ok_or_else(|| invalid_input("example count is too large"))?;

    let mut cases = Vec::with_capacity(maximum_pairs);
    for left in 0..example_count {
        for right in left..example_count {
            cases.push(ProductCase { left, right });
        }
    }

    Ok(cases)
}

fn candidate_seed(seed: u64, candidate: u64) -> u64 {
    let mut value = seed ^ candidate.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn shift_by(shape: &FramedPoset, offset: usize) -> FramedPoset {
    if offset == 0 {
        return shape.clone();
    }
    let mut shape = shift(shape);
    for _ in 1..offset {
        shape = shift(&shape);
    }
    shape
}

fn volumetricity_failure(
    options: &Options,
    candidates: u64,
    factors: &[Factor],
    failure: &ProductFailure,
) -> io::Error {
    let serialize = |shape: &FramedPoset| {
        serde_json::to_string(shape)
            .unwrap_or_else(|error| format!("<serialization failed: {error}>"))
    };

    io::Error::other(format!(
        "orthogonal product of volumetric OFPs was not volumetric after {candidates} candidates at product {} (factor indices {} and {}, candidate IDs {} and {}), seed {:#018x}; dimension {}; cell count {}; right direction offset {}; left OFP: {}; shifted right OFP: {}; product OFP: {}",
        failure.product + 1,
        failure.case.left,
        failure.case.right,
        factors[failure.case.left].candidate,
        factors[failure.case.right].candidate,
        options.seed,
        options.dimension,
        options.cell_count,
        options.dimension,
        serialize(&failure.left),
        serialize(&failure.right),
        serialize(&failure.product_shape),
    ))
}

fn arguments() -> io::Result<Options> {
    let mut arguments = env::args().skip(1);
    let example_count = parse_optional(&mut arguments, "example count", DEFAULT_EXAMPLE_COUNT)?;
    let dimension = parse_optional(&mut arguments, "dimension", DEFAULT_DIMENSION)?;
    let cell_count = parse_optional(&mut arguments, "cell count", DEFAULT_CELL_COUNT)?;
    let seed = arguments
        .next()
        .map(|value| parse_u64("seed", &value))
        .transpose()?
        .map_or_else(|| OsRng.try_next_u64().map_err(io::Error::other), Ok)?;

    if arguments.next().is_some() {
        return Err(invalid_input(
            "usage: check_random_volumetric_products [example-count] [dimension] [cell-count] [seed]",
        ));
    }
    if example_count == 0 {
        return Err(invalid_input("example count must be positive"));
    }
    if dimension >= usize::BITS as usize {
        return Err(invalid_input(format!(
            "dimension must be smaller than {}",
            usize::BITS,
        )));
    }

    let minimum_cell_count = 1usize << dimension;
    if cell_count < minimum_cell_count {
        return Err(invalid_input(format!(
            "at least {minimum_cell_count} cells are required in dimension {dimension}",
        )));
    }

    Ok(Options {
        example_count,
        dimension,
        cell_count,
        seed,
    })
}

fn parse_optional<T>(
    arguments: &mut impl Iterator<Item = String>,
    name: &str,
    default: T,
) -> io::Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    arguments
        .next()
        .map(|value| {
            value
                .parse()
                .map_err(|error| invalid_input(format!("invalid {name} {value:?}: {error}")))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn parse_u64(name: &str, value: &str) -> io::Result<u64> {
    let parsed = if let Some(hexadecimal) = value.strip_prefix("0x") {
        u64::from_str_radix(hexadecimal, 16)
    } else {
        value.parse()
    };
    parsed.map_err(|error| invalid_input(format!("invalid {name} {value:?}: {error}")))
}

fn invalid_input(error: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, error.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_plan_contains_every_unordered_pair() {
        let cases = product_cases(4).unwrap();
        let pairs: Vec<_> = cases.iter().map(|case| (case.left, case.right)).collect();

        assert_eq!(
            pairs,
            vec![
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 2),
                (2, 3),
                (3, 3),
            ]
        );
    }

    #[test]
    fn candidate_seeds_are_reproducible_and_distinct() {
        assert_eq!(candidate_seed(7, 11), candidate_seed(7, 11));
        assert_ne!(candidate_seed(7, 11), candidate_seed(7, 12));
    }
}
