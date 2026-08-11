use std::env;
use std::io;
use std::time::Instant;

use ofposets::{FramedPoset, RandomFramedPosetGenerator, orthogonal_product, shift};
use rand::rngs::{OsRng, SmallRng};
use rand::{SeedableRng, TryRngCore};

const DEFAULT_PRODUCT_COUNT: u64 = 100;
const DEFAULT_DIMENSION: usize = 2;
const DEFAULT_CELL_COUNT: usize = 9;
const REPORT_EVERY_CANDIDATES: u64 = 10000;

struct Options {
    product_count: u64,
    dimension: usize,
    cell_count: usize,
    seed: u64,
}

fn main() -> io::Result<()> {
    let options = arguments()?;
    let generator = RandomFramedPosetGenerator::new(options.dimension, options.cell_count);
    let mut rng = SmallRng::seed_from_u64(options.seed);
    let mut pending_factor: Option<FramedPoset> = None;
    let mut candidates = 0u64;
    let mut rigid_factors = 0u64;
    let mut products = 0u64;
    let started = Instant::now();

    println!(
        "checking {} products of random rigid, mutually orthogonal, {}-dimensional, {}-cell OFPs (seed {:#018x})",
        options.product_count, options.dimension, options.cell_count, options.seed,
    );

    while products < options.product_count {
        candidates += 1;
        let candidate = generator.generate(&mut rng);
        if !candidate.is_rigid() {
            report_progress(
                candidates,
                rigid_factors,
                products,
                started,
                options.product_count,
            );
            continue;
        }

        rigid_factors += 1;
        if let Some(left) = pending_factor.take() {
            let right_offset = options.dimension;
            let right = shift_by(candidate, right_offset);
            debug_assert!(ofposets::intset::is_disjoint(
                &left.active_directions(),
                &right.active_directions(),
            ));
            let product = orthogonal_product(&left, &right);
            products += 1;

            if !product.is_rigid() {
                return Err(rigidity_failure(
                    &options,
                    candidates,
                    products,
                    right_offset,
                    &left,
                    &right,
                    &product,
                ));
            }
        } else {
            pending_factor = Some(candidate);
        }

        report_progress(
            candidates,
            rigid_factors,
            products,
            started,
            options.product_count,
        );
    }

    println!(
        "all {products} orthogonal products were rigid; generated {candidates} candidates and retained {rigid_factors} rigid factors ({:.1?})",
        started.elapsed(),
    );
    Ok(())
}

fn report_progress(
    candidates: u64,
    rigid_factors: u64,
    products: u64,
    started: Instant,
    product_count: u64,
) {
    if candidates.is_multiple_of(REPORT_EVERY_CANDIDATES) || products == product_count {
        println!(
            "generated {candidates} candidates; retained {rigid_factors} rigid factors; checked {products}/{product_count} products ({:.1?})",
            started.elapsed(),
        );
    }
}

fn shift_by(mut shape: FramedPoset, offset: usize) -> FramedPoset {
    for _ in 0..offset {
        shape = shift(&shape);
    }
    shape
}

#[allow(clippy::too_many_arguments)]
fn rigidity_failure(
    options: &Options,
    candidates: u64,
    products: u64,
    right_offset: usize,
    left: &FramedPoset,
    right: &FramedPoset,
    product: &FramedPoset,
) -> io::Error {
    let serialize = |shape: &FramedPoset| {
        serde_json::to_string(shape)
            .unwrap_or_else(|error| format!("<serialization failed: {error}>"))
    };

    io::Error::other(format!(
        "orthogonal product of rigid OFPs was not rigid after {candidates} candidates at product {products}, seed {:#018x}; dimension {}; cell count {}; right direction offset {right_offset}; left OFP: {}; right OFP: {}; product OFP: {}",
        options.seed,
        options.dimension,
        options.cell_count,
        serialize(left),
        serialize(right),
        serialize(product),
    ))
}

fn arguments() -> io::Result<Options> {
    let mut arguments = env::args().skip(1);
    let product_count = parse_optional(&mut arguments, "product count", DEFAULT_PRODUCT_COUNT)?;
    let dimension = parse_optional(&mut arguments, "dimension", DEFAULT_DIMENSION)?;
    let cell_count = parse_optional(&mut arguments, "cell count", DEFAULT_CELL_COUNT)?;
    let seed = arguments
        .next()
        .map(|value| parse_u64("seed", &value))
        .transpose()?
        .map_or_else(|| OsRng.try_next_u64().map_err(io::Error::other), Ok)?;

    if arguments.next().is_some() {
        return Err(invalid_input(
            "usage: check_random_rigid_products [product-count] [dimension] [cell-count] [seed]",
        ));
    }
    if product_count == 0 {
        return Err(invalid_input("product count must be positive"));
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
        product_count,
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
