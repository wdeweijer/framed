use std::collections::{HashSet, hash_map::DefaultHasher};
use std::env;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use ofposets::{
    CubularityMode, FramedPoset, Sign, boundary, is_cubular, normalize, orthogonal_product, shift,
};
use rand::rngs::{OsRng, SmallRng};
use rand::{Rng, SeedableRng, TryRngCore};
use serde::Deserialize;

const DEFAULT_DATASET: &str = "visualizations/random_13_cells_normal_forms_hat_strongly_cubular_connected_3d_up_to_symmetry.jsonl";
const DEFAULT_PRODUCT_COUNT: u64 = 100;
const DIRECTION_COUNT: usize = 3;
const CELL_COUNT: usize = 13;
const SYMMETRY_COUNT: usize = 48;
const BUFFER_CAPACITY: usize = 8 * 1024 * 1024;
const REPORT_EVERY_PRODUCTS: u64 = 10;

struct Options {
    product_count: u64,
    seed: u64,
    dataset: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetRecord {
    hash: String,
    stabilizer_size: usize,
    multiplicity: u64,
    boundary_hashes: [BoundaryHashRecord; DIRECTION_COUNT],
    ofp: FramedPoset,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundaryHashRecord {
    direction: usize,
    input: String,
    output: String,
}

struct DatasetShape {
    line: usize,
    hash: u64,
    shape: Arc<FramedPoset>,
}

fn main() -> io::Result<()> {
    let options = arguments()?;
    let load_started = Instant::now();
    let shapes = load_dataset(&options.dataset)?;
    let maximum_pairs = shapes
        .len()
        .checked_mul(shapes.len() + 1)
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| invalid_data("dataset contains too many shapes"))?;

    if options.product_count > maximum_pairs as u64 {
        return Err(invalid_input(format!(
            "requested {} products, but the dataset has only {maximum_pairs} unordered pairs",
            options.product_count,
        )));
    }

    println!(
        "loaded {} strongly cubular 3D OFPs from {} in {:.1?}",
        shapes.len(),
        options.dataset.display(),
        load_started.elapsed(),
    );
    println!(
        "checking {} randomly selected orthogonal products (seed {:#018x})",
        options.product_count, options.seed,
    );

    let mut rng = SmallRng::seed_from_u64(options.seed);
    let mut selected_pairs = HashSet::with_capacity(options.product_count as usize);
    let started = Instant::now();

    for product_number in 1..=options.product_count {
        let (left_index, right_index) = loop {
            let left = rng.random_range(0..shapes.len());
            let right = rng.random_range(0..shapes.len());
            let pair = if left <= right {
                (left, right)
            } else {
                (right, left)
            };
            if selected_pairs.insert(pair) {
                break pair;
            }
        };

        let left = &shapes[left_index];
        let right_source = &shapes[right_index];
        let right = Arc::new(shift_by(&right_source.shape, DIRECTION_COUNT));
        debug_assert!(ofposets::intset::is_disjoint(
            &left.shape.total_frame(),
            &right.total_frame(),
        ));
        let product = Arc::new(orthogonal_product(&left.shape, &right));

        if !is_cubular(CubularityMode::Strong, &product) {
            return Err(strong_cubularity_failure(
                &options,
                product_number,
                left,
                right_source,
                &right,
                &product,
            ));
        }

        if product_number.is_multiple_of(REPORT_EVERY_PRODUCTS)
            || product_number == options.product_count
        {
            println!(
                "checked {product_number}/{} products ({:.1?})",
                options.product_count,
                started.elapsed(),
            );
        }
    }

    println!(
        "all {} orthogonal products were strongly cubular ({:.1?})",
        options.product_count,
        started.elapsed(),
    );
    Ok(())
}

fn load_dataset(path: &Path) -> io::Result<Vec<DatasetShape>> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(BUFFER_CAPACITY, file);
    let mut shapes = Vec::new();
    let mut previous_hash = None;
    let mut line = String::new();
    let mut line_number = 0usize;

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }
        line_number += 1;
        if !line.ends_with('\n') {
            return Err(invalid_line(
                path,
                line_number,
                "line is not newline-terminated",
            ));
        }

        let record: DatasetRecord = serde_json::from_str(&line).map_err(|error| {
            invalid_line(path, line_number, format!("invalid JSONL record: {error}"))
        })?;
        let hash = parse_hash(path, line_number, &record.hash)?;
        if previous_hash.is_some_and(|previous| previous >= hash) {
            return Err(invalid_line(
                path,
                line_number,
                "hashes must be strictly increasing",
            ));
        }
        previous_hash = Some(hash);
        validate_record(path, line_number, hash, &record)?;

        shapes.push(DatasetShape {
            line: line_number,
            hash,
            shape: Arc::new(normalize(&record.ofp)),
        });
    }

    if shapes.is_empty() {
        return Err(invalid_data(format!("{} is empty", path.display())));
    }
    Ok(shapes)
}

fn validate_record(
    path: &Path,
    line: usize,
    stored_hash: u64,
    record: &DatasetRecord,
) -> io::Result<()> {
    if record.multiplicity == 0 {
        return Err(invalid_line(path, line, "multiplicity must be positive"));
    }
    if record.stabilizer_size == 0 || !SYMMETRY_COUNT.is_multiple_of(record.stabilizer_size) {
        return Err(invalid_line(
            path,
            line,
            format!("stabilizer size must be a positive divisor of {SYMMETRY_COUNT}"),
        ));
    }

    let shape = Arc::new(normalize(&record.ofp));
    if !FramedPoset::equal(&shape, &record.ofp) {
        return Err(invalid_line(
            path,
            line,
            "stored OFP is not in canonical normal form",
        ));
    }
    if shape.sizes().iter().sum::<usize>() != CELL_COUNT {
        return Err(invalid_line(
            path,
            line,
            format!("OFP does not have exactly {CELL_COUNT} cells"),
        ));
    }
    if shape.total_frame() != [0, 1, 2] {
        return Err(invalid_line(
            path,
            line,
            "OFP does not use precisely directions 0, 1, and 2",
        ));
    }
    if !shape.is_connected() {
        return Err(invalid_line(path, line, "OFP is not connected"));
    }
    if !is_cubular(CubularityMode::Strong, &shape) {
        return Err(invalid_line(path, line, "OFP is not strongly cubular"));
    }

    let actual_hash = structural_hash(&shape);
    if actual_hash != stored_hash {
        return Err(invalid_line(
            path,
            line,
            format!(
                "stored hash {stored_hash:016x} does not match recomputed hash {actual_hash:016x}"
            ),
        ));
    }

    for (direction, hashes) in record.boundary_hashes.iter().enumerate() {
        if hashes.direction != direction {
            return Err(invalid_line(
                path,
                line,
                format!(
                    "boundary hash entry {direction} has direction {}",
                    hashes.direction
                ),
            ));
        }
        validate_boundary_hash(path, line, &shape, direction, Sign::Input, &hashes.input)?;
        validate_boundary_hash(path, line, &shape, direction, Sign::Output, &hashes.output)?;
    }
    Ok(())
}

fn validate_boundary_hash(
    path: &Path,
    line: usize,
    shape: &Arc<FramedPoset>,
    direction: usize,
    sign: Sign,
    stored: &str,
) -> io::Result<()> {
    let stored = parse_hash(path, line, stored)?;
    let (boundary, _) = boundary(sign, direction, shape);
    let actual = structural_hash(&normalize(&boundary));
    if actual != stored {
        return Err(invalid_line(
            path,
            line,
            format!(
                "stored {sign:?} boundary hash {stored:016x} in direction {direction} does not match recomputed hash {actual:016x}"
            ),
        ));
    }
    Ok(())
}

fn structural_hash(shape: &FramedPoset) -> u64 {
    let mut hasher = DefaultHasher::new();
    shape.hash(&mut hasher);
    hasher.finish()
}

fn shift_by(shape: &FramedPoset, offset: usize) -> FramedPoset {
    let mut shifted = shape.clone();
    for _ in 0..offset {
        shifted = shift(&shifted);
    }
    shifted
}

fn strong_cubularity_failure(
    options: &Options,
    product_number: u64,
    left: &DatasetShape,
    right_source: &DatasetShape,
    shifted_right: &FramedPoset,
    product: &FramedPoset,
) -> io::Error {
    let serialize = |shape: &FramedPoset| {
        serde_json::to_string(shape)
            .unwrap_or_else(|error| format!("<serialization failed: {error}>"))
    };

    io::Error::other(format!(
        "orthogonal product of strongly cubular dataset OFPs was not strongly cubular at product {product_number}, seed {:#018x}; dataset {}; left line {} hash {:016x}; right line {} hash {:016x}; right direction offset {DIRECTION_COUNT}; left OFP: {}; right OFP before shifting: {}; right OFP after shifting: {}; product OFP: {}",
        options.seed,
        options.dataset.display(),
        left.line,
        left.hash,
        right_source.line,
        right_source.hash,
        serialize(&left.shape),
        serialize(&right_source.shape),
        serialize(shifted_right),
        serialize(product),
    ))
}

fn arguments() -> io::Result<Options> {
    let mut arguments = env::args().skip(1);
    let product_count = parse_optional(&mut arguments, "product count", DEFAULT_PRODUCT_COUNT)?;
    let seed = arguments
        .next()
        .map(|value| parse_u64("seed", &value))
        .transpose()?
        .map_or_else(|| OsRng.try_next_u64().map_err(io::Error::other), Ok)?;
    let dataset = arguments
        .next()
        .map_or_else(|| PathBuf::from(DEFAULT_DATASET), PathBuf::from);

    if arguments.next().is_some() {
        return Err(invalid_input(
            "usage: check_random_strongly_cubular_products [product-count] [seed] [dataset]",
        ));
    }
    if product_count == 0 {
        return Err(invalid_input("product count must be positive"));
    }

    Ok(Options {
        product_count,
        seed,
        dataset,
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

fn parse_hash(path: &Path, line: usize, hash: &str) -> io::Result<u64> {
    let value = u64::from_str_radix(hash, 16)
        .map_err(|_| invalid_line(path, line, "hash is not hexadecimal"))?;
    if hash.len() != 16 || format!("{value:016x}") != hash {
        return Err(invalid_line(
            path,
            line,
            "hash must be exactly 16 lowercase hexadecimal digits",
        ));
    }
    Ok(value)
}

fn invalid_line(path: &Path, line: usize, error: impl std::fmt::Display) -> io::Error {
    invalid_data(format!("{}:{line}: {error}", path.display()))
}

fn invalid_input(error: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, error.into())
}

fn invalid_data(error: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.into())
}
