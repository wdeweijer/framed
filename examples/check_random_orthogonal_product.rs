use std::env;
use std::io;
use std::sync::Arc;
use std::time::Instant;

use ofposets::{
    Embedding, FramedPoset, RandomFramedPosetGenerator, orthogonal_product,
    orthogonal_product_associator, orthogonal_product_commutator, orthogonal_product_embedding,
    shift,
};
use rand::rngs::{OsRng, SmallRng};
use rand::{Rng, SeedableRng, TryRngCore};

const DEFAULT_SAMPLE_COUNT: u64 = 1_000;
const DEFAULT_DIMENSION: usize = 2;
const DEFAULT_CELL_COUNT: usize = 9;
const REPORT_EVERY: u64 = 100;

struct Options {
    sample_count: u64,
    dimension: usize,
    cell_count: usize,
    seed: u64,
}

struct RandomTuple {
    factors: [FramedPoset; 4],
    offsets: [usize; 4],
}

fn main() -> io::Result<()> {
    let options = arguments()?;
    let generator = RandomFramedPosetGenerator::new(options.dimension, options.cell_count);
    let mut rng = SmallRng::seed_from_u64(options.seed);
    let started = Instant::now();

    println!(
        "checking {0} random 4-tuples of {1}-dimensional, {2}-cell OFPs (seed {3:#018x})",
        options.sample_count, options.dimension, options.cell_count, options.seed,
    );

    for sample in 1..=options.sample_count {
        let left = generator.generate(&mut rng);
        let middle_offset = rng.random_range(0..=options.dimension);
        let right_offset = rng.random_range(0..=options.dimension.saturating_mul(2));
        let fourth_offset = rng.random_range(0..=options.dimension.saturating_mul(3));
        let middle = shift_by(generator.generate(&mut rng), middle_offset);
        let right = shift_by(generator.generate(&mut rng), right_offset);
        let fourth = shift_by(generator.generate(&mut rng), fourth_offset);
        let tuple = RandomTuple {
            factors: [left, middle, right, fourth],
            offsets: [0, middle_offset, right_offset, fourth_offset],
        };

        check_commutativity(
            &tuple.factors[0],
            &tuple.factors[1],
            "left * middle",
            &options,
            sample,
            &tuple,
        )?;
        check_commutativity(
            &tuple.factors[1],
            &tuple.factors[2],
            "middle * right",
            &options,
            sample,
            &tuple,
        )?;
        check_commutativity(
            &tuple.factors[0],
            &tuple.factors[2],
            "left * right",
            &options,
            sample,
            &tuple,
        )?;
        check_associativity(
            &tuple.factors[0],
            &tuple.factors[1],
            &tuple.factors[2],
            &options,
            sample,
            &tuple,
        )?;
        check_pentagon(&options, sample, &tuple)?;
        check_commutator_symmetry(&options, sample, &tuple)?;
        check_commutator_unit(&options, sample, &tuple)?;
        check_commutator_hexagons(&options, sample, &tuple)?;
        check_commutator_naturality(&options, sample, &tuple)?;

        if sample.is_multiple_of(REPORT_EVERY) || sample == options.sample_count {
            println!("checked {sample} 4-tuples ({:.1?})", started.elapsed());
        }
    }

    println!(
        "orthogonal product was symmetric monoidal, including the pentagon, hexagon, symmetry, unit, and naturality identities, for all {} 4-tuples",
        options.sample_count,
    );
    Ok(())
}

fn check_commutativity(
    first: &FramedPoset,
    second: &FramedPoset,
    expression: &str,
    options: &Options,
    sample: u64,
    tuple: &RandomTuple,
) -> io::Result<()> {
    let commutator = orthogonal_product_commutator(first, second);

    if commutator.is_isomorphism() {
        Ok(())
    } else {
        Err(failure(
            &format!("commutativity failed for {expression}"),
            options,
            sample,
            tuple,
        ))
    }
}

fn check_associativity(
    left: &FramedPoset,
    middle: &FramedPoset,
    right: &FramedPoset,
    options: &Options,
    sample: u64,
    tuple: &RandomTuple,
) -> io::Result<()> {
    let associator = orthogonal_product_associator(left, middle, right);

    if associator.is_isomorphism() {
        Ok(())
    } else {
        Err(failure("associativity failed", options, sample, tuple))
    }
}

fn check_pentagon(options: &Options, sample: u64, tuple: &RandomTuple) -> io::Result<()> {
    let [a, b, c, d] = &tuple.factors;
    let ab = orthogonal_product(a, b);
    let bc = orthogonal_product(b, c);
    let cd = orthogonal_product(c, d);

    let short_first = orthogonal_product_associator(&ab, c, d);
    let short_second = orthogonal_product_associator(a, b, &cd);
    let short_path = Embedding::compose(&short_first, &short_second);

    let alpha_abc = orthogonal_product_associator(a, b, c);
    let identity_d = Embedding::id(Arc::new(d.clone()));
    let long_first = orthogonal_product_embedding(&alpha_abc, &identity_d);
    let long_second = orthogonal_product_associator(a, &bc, d);
    let identity_a = Embedding::id(Arc::new(a.clone()));
    let alpha_bcd = orthogonal_product_associator(b, c, d);
    let long_third = orthogonal_product_embedding(&identity_a, &alpha_bcd);
    let long_first_two = Embedding::compose(&long_first, &long_second);
    let long_path = Embedding::compose(&long_first_two, &long_third);

    let same_endpoints = FramedPoset::equal(&short_path.dom, &long_path.dom)
        && FramedPoset::equal(&short_path.cod, &long_path.cod);
    if short_path.is_isomorphism()
        && long_path.is_isomorphism()
        && same_endpoints
        && short_path.map == long_path.map
    {
        Ok(())
    } else {
        Err(failure("pentagon identity failed", options, sample, tuple))
    }
}

fn check_commutator_symmetry(
    options: &Options,
    sample: u64,
    tuple: &RandomTuple,
) -> io::Result<()> {
    for first in 0..tuple.factors.len() {
        for second in first + 1..tuple.factors.len() {
            let forward =
                orthogonal_product_commutator(&tuple.factors[first], &tuple.factors[second]);
            let backward =
                orthogonal_product_commutator(&tuple.factors[second], &tuple.factors[first]);
            let round_trip = Embedding::compose(&forward, &backward);
            let identity = Embedding::id(Arc::clone(&forward.dom));
            let reason = format!("commutator symmetry failed for factors {first} and {second}");
            check_morphism_equality(&reason, &round_trip, &identity, options, sample, tuple)?;
        }
    }

    Ok(())
}

fn check_commutator_unit(options: &Options, sample: u64, tuple: &RandomTuple) -> io::Result<()> {
    let point = FramedPoset::point();

    for (factor, shape) in tuple.factors.iter().enumerate() {
        for (side, commutator) in [
            ("right", orthogonal_product_commutator(shape, &point)),
            ("left", orthogonal_product_commutator(&point, shape)),
        ] {
            let identity = Embedding::id(Arc::clone(&commutator.dom));
            let reason = format!("commutator {side}-unit law failed for factor {factor}");
            check_morphism_equality(&reason, &commutator, &identity, options, sample, tuple)?;
        }
    }

    Ok(())
}

fn check_commutator_hexagons(
    options: &Options,
    sample: u64,
    tuple: &RandomTuple,
) -> io::Result<()> {
    let [a, b, c, _] = &tuple.factors;
    let ab = orthogonal_product(a, b);
    let bc = orthogonal_product(b, c);

    let first_left_1 = orthogonal_product_associator(a, b, c);
    let first_left_2 = orthogonal_product_commutator(a, &bc);
    let first_left_3 = orthogonal_product_associator(b, c, a);
    let first_left_12 = Embedding::compose(&first_left_1, &first_left_2);
    let first_left = Embedding::compose(&first_left_12, &first_left_3);

    let commutator_ab = orthogonal_product_commutator(a, b);
    let identity_c = Embedding::id(Arc::new(c.clone()));
    let first_right_1 = orthogonal_product_embedding(&commutator_ab, &identity_c);
    let first_right_2 = orthogonal_product_associator(b, a, c);
    let identity_b = Embedding::id(Arc::new(b.clone()));
    let commutator_ac = orthogonal_product_commutator(a, c);
    let first_right_3 = orthogonal_product_embedding(&identity_b, &commutator_ac);
    let first_right_12 = Embedding::compose(&first_right_1, &first_right_2);
    let first_right = Embedding::compose(&first_right_12, &first_right_3);
    check_morphism_equality(
        "first commutator hexagon failed",
        &first_left,
        &first_right,
        options,
        sample,
        tuple,
    )?;

    let second_left_1 = orthogonal_product_associator(a, b, c).inverse_isomorphism();
    let second_left_2 = orthogonal_product_commutator(&ab, c);
    let second_left_3 = orthogonal_product_associator(c, a, b).inverse_isomorphism();
    let second_left_12 = Embedding::compose(&second_left_1, &second_left_2);
    let second_left = Embedding::compose(&second_left_12, &second_left_3);

    let identity_a = Embedding::id(Arc::new(a.clone()));
    let commutator_bc = orthogonal_product_commutator(b, c);
    let second_right_1 = orthogonal_product_embedding(&identity_a, &commutator_bc);
    let second_right_2 = orthogonal_product_associator(a, c, b).inverse_isomorphism();
    let identity_b = Embedding::id(Arc::new(b.clone()));
    let commutator_ac = orthogonal_product_commutator(a, c);
    let second_right_3 = orthogonal_product_embedding(&commutator_ac, &identity_b);
    let second_right_12 = Embedding::compose(&second_right_1, &second_right_2);
    let second_right = Embedding::compose(&second_right_12, &second_right_3);
    check_morphism_equality(
        "second commutator hexagon failed",
        &second_left,
        &second_right,
        options,
        sample,
        tuple,
    )
}

fn check_commutator_naturality(
    options: &Options,
    sample: u64,
    tuple: &RandomTuple,
) -> io::Result<()> {
    let [a, b, c, d] = &tuple.factors;
    let f = orthogonal_product_commutator(a, b);
    let g = orthogonal_product_commutator(c, d);

    let left_1 = orthogonal_product_embedding(&f, &g);
    let left_2 = orthogonal_product_commutator(&f.cod, &g.cod);
    let left = Embedding::compose(&left_1, &left_2);

    let right_1 = orthogonal_product_commutator(&f.dom, &g.dom);
    let right_2 = orthogonal_product_embedding(&g, &f);
    let right = Embedding::compose(&right_1, &right_2);

    check_morphism_equality(
        "commutator naturality failed",
        &left,
        &right,
        options,
        sample,
        tuple,
    )
}

fn check_morphism_equality(
    reason: &str,
    left: &Embedding,
    right: &Embedding,
    options: &Options,
    sample: u64,
    tuple: &RandomTuple,
) -> io::Result<()> {
    if left.is_isomorphism()
        && right.is_isomorphism()
        && FramedPoset::equal(&left.dom, &right.dom)
        && FramedPoset::equal(&left.cod, &right.cod)
        && left.map == right.map
    {
        Ok(())
    } else {
        Err(failure(reason, options, sample, tuple))
    }
}

fn shift_by(mut shape: FramedPoset, offset: usize) -> FramedPoset {
    for _ in 0..offset {
        shape = shift(&shape);
    }
    shape
}

fn failure(reason: &str, options: &Options, sample: u64, tuple: &RandomTuple) -> io::Error {
    let factors = serde_json::to_string(&tuple.factors)
        .unwrap_or_else(|error| format!("<serialization failed: {error}>"));

    io::Error::other(format!(
        "{reason} at sample {sample} with seed {:#018x}; dimension {}; cell count {}; direction offsets {:?}; factor OFPs: {factors}",
        options.seed, options.dimension, options.cell_count, tuple.offsets,
    ))
}

fn arguments() -> io::Result<Options> {
    let mut arguments = env::args().skip(1);
    let sample_count = parse_optional(&mut arguments, "sample count", DEFAULT_SAMPLE_COUNT)?;
    let dimension = parse_optional(&mut arguments, "dimension", DEFAULT_DIMENSION)?;
    let cell_count = parse_optional(&mut arguments, "cell count", DEFAULT_CELL_COUNT)?;
    let seed = arguments
        .next()
        .map(|value| parse_u64("seed", &value))
        .transpose()?
        .map_or_else(|| OsRng.try_next_u64().map_err(io::Error::other), Ok)?;

    if arguments.next().is_some() {
        return Err(invalid_input(
            "usage: check_random_orthogonal_product [sample-count] [dimension] [cell-count] [seed]",
        ));
    }
    if sample_count == 0 {
        return Err(invalid_input("sample count must be positive"));
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
        sample_count,
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
