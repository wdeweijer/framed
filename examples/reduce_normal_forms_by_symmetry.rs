use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use ofposets::{
    DirectionImage, Embedding, FramedPoset, Sign, SignedPermutation, iterated_boundary, normalize,
    transform,
};
use serde::de::IgnoredAny;
use serde::{Deserialize, Serialize};

const INPUT_FILE: &str = "visualizations/random_9_cells_normal_forms_cubular.jsonl";
const OUTPUT_FILE: &str = "visualizations/random_9_cells_normal_forms_cubular_up_to_symmetry.jsonl";
const BUFFER_CAPACITY: usize = 8 * 1024 * 1024;
const REPORT_EVERY_INDEXED: usize = 1_000_000;
const REPORT_EVERY_REDUCED: usize = 100_000;
const SYMMETRY_COUNT: usize = 8;
const SIGN_PAIRS: [(Sign, Sign); 4] = [
    (Sign::Input, Sign::Input),
    (Sign::Input, Sign::Output),
    (Sign::Output, Sign::Input),
    (Sign::Output, Sign::Output),
];

#[derive(Debug, Clone, Copy)]
struct IndexedRecord {
    offset: u64,
    hash: u64,
    multiplicity: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexLine {
    hash: String,
    multiplicity: u64,
    #[serde(rename = "ofp")]
    _ofp: IgnoredAny,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputRecord {
    hash: String,
    multiplicity: u64,
    ofp: FramedPoset,
}

#[derive(Serialize)]
struct OutputRecord<'a> {
    /// Hash of the representative retained from the input dataset.
    hash: String,
    /// Minimum hash in the full symmetry orbit.
    orbit_hash: String,
    /// Sorted hashes of all distinct normalized symmetry images.
    symmetry_hashes: Vec<String>,
    preserving_symmetries: usize,
    /// Number of input isomorphism classes combined into this record.
    dataset_classes: usize,
    /// Sum of their sampling multiplicities.
    multiplicity: u64,
    ofp: &'a FramedPoset,
}

struct Orbit {
    hashes: Vec<u64>,
}

fn main() -> io::Result<()> {
    let input_path = Path::new(INPUT_FILE);
    let output_path = Path::new(OUTPUT_FILE);
    let temporary_path = temporary_path(output_path);
    let started = Instant::now();

    let input_file = File::open(input_path)?;
    let mut input = BufReader::with_capacity(BUFFER_CAPACITY, input_file);
    let (records, hash_to_index, input_multiplicity) = build_index(&mut input, input_path)?;

    println!(
        "indexed {} OFPs from {} in {:.1?}",
        records.len(),
        input_path.display(),
        started.elapsed()
    );

    let output_file = File::create(&temporary_path)?;
    let mut output = BufWriter::with_capacity(BUFFER_CAPACITY, output_file);
    let summary = reduce(
        &mut input,
        &mut output,
        input_path,
        &records,
        &hash_to_index,
    )?;

    if summary.multiplicity != input_multiplicity {
        return Err(invalid_data(format!(
            "output multiplicity {} does not match input multiplicity {input_multiplicity}",
            summary.multiplicity
        )));
    }
    if summary.handled != records.len() {
        return Err(invalid_data(format!(
            "handled {} input records, expected {}",
            summary.handled,
            records.len()
        )));
    }

    output.flush()?;
    let output_file = output.into_inner().map_err(|error| error.into_error())?;
    output_file.sync_all()?;
    fs::rename(&temporary_path, output_path)?;

    println!(
        "wrote {} symmetry classes to {} after combining {} input classes in {:.1?}",
        summary.output_records,
        output_path.display(),
        summary.handled,
        started.elapsed()
    );
    Ok(())
}

fn build_index<R: BufRead>(
    input: &mut R,
    path: &Path,
) -> io::Result<(Vec<IndexedRecord>, HashMap<u64, usize>, u64)> {
    let mut records = Vec::new();
    let mut hash_to_index = HashMap::new();
    let mut total_multiplicity = 0u64;
    let mut previous_hash = None;
    let mut offset = 0u64;
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = input.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }
        let line_number = records.len() + 1;
        if !line.ends_with('\n') {
            return Err(invalid_line(
                path,
                line_number,
                "line is not newline-terminated",
            ));
        }

        let parsed: IndexLine = serde_json::from_str(&line).map_err(|error| {
            invalid_line(path, line_number, format!("invalid JSONL record: {error}"))
        })?;
        let hash = parse_hash(path, line_number, &parsed.hash)?;
        if previous_hash.is_some_and(|previous| previous >= hash) {
            return Err(invalid_line(
                path,
                line_number,
                "hashes must be strictly increasing",
            ));
        }
        if parsed.multiplicity == 0 {
            return Err(invalid_line(
                path,
                line_number,
                "multiplicity must be positive",
            ));
        }

        let index = records.len();
        records.push(IndexedRecord {
            offset,
            hash,
            multiplicity: parsed.multiplicity,
        });
        if hash_to_index.insert(hash, index).is_some() {
            return Err(invalid_line(path, line_number, "duplicate structural hash"));
        }
        total_multiplicity = total_multiplicity
            .checked_add(parsed.multiplicity)
            .ok_or_else(|| invalid_data("input multiplicity overflow"))?;
        previous_hash = Some(hash);
        offset = offset
            .checked_add(bytes_read as u64)
            .ok_or_else(|| invalid_data("input byte offset overflow"))?;

        if records.len().is_multiple_of(REPORT_EVERY_INDEXED) {
            println!("indexed {} OFPs", records.len());
        }
    }

    if records.is_empty() {
        return Err(invalid_data("input dataset is empty"));
    }
    Ok((records, hash_to_index, total_multiplicity))
}

#[derive(Default)]
struct ReductionSummary {
    output_records: usize,
    handled: usize,
    multiplicity: u64,
}

fn reduce<R: BufRead + Seek, W: Write>(
    input: &mut R,
    output: &mut W,
    input_path: &Path,
    records: &[IndexedRecord],
    hash_to_index: &HashMap<u64, usize>,
) -> io::Result<ReductionSummary> {
    let symmetries = two_dimensional_symmetries();
    let mut handled = vec![false; records.len()];
    let mut summary = ReductionSummary::default();
    let mut line = String::new();
    let started = Instant::now();

    for index in 0..records.len() {
        if handled[index] {
            continue;
        }

        let input_record = read_record(input, input_path, index, records[index], &mut line)?;
        validate_dataset_shape(&input_record.ofp, input_path, index + 1)?;
        let actual_hash = structural_hash(&input_record.ofp);
        if actual_hash != records[index].hash {
            return Err(invalid_line(
                input_path,
                index + 1,
                format!(
                    "stored hash {:016x} does not match recomputed hash {actual_hash:016x}",
                    records[index].hash
                ),
            ));
        }

        let orbit = analyze_orbit(&input_record.ofp, &symmetries).map_err(|message| {
            let source = serde_json::to_string(&input_record.ofp)
                .unwrap_or_else(|error| format!("<serialization failed: {error}>"));
            invalid_line(
                input_path,
                index + 1,
                format!(
                    "invalid symmetry orbit for OFP hash {}: {message}; source OFP: {source}",
                    input_record.hash
                ),
            )
        })?;
        let mut multiplicity = 0u64;
        let mut dataset_classes = 0usize;

        for &hash in &orbit.hashes {
            let Some(&member_index) = hash_to_index.get(&hash) else {
                continue;
            };
            if handled[member_index] {
                return Err(invalid_line(
                    input_path,
                    index + 1,
                    format!("symmetry orbit overlaps an earlier orbit at hash {hash:016x}"),
                ));
            }

            handled[member_index] = true;
            summary.handled += 1;
            dataset_classes += 1;
            multiplicity = multiplicity
                .checked_add(records[member_index].multiplicity)
                .ok_or_else(|| invalid_data("orbit multiplicity overflow"))?;
        }

        if !handled[index] {
            return Err(invalid_line(
                input_path,
                index + 1,
                "the OFP's own hash is absent from its symmetry orbit",
            ));
        }

        let distinct_symmetry_images = orbit.hashes.len();
        if !SYMMETRY_COUNT.is_multiple_of(distinct_symmetry_images) {
            return Err(invalid_line(
                input_path,
                index + 1,
                format!("orbit size {distinct_symmetry_images} does not divide {SYMMETRY_COUNT}"),
            ));
        }
        let preserving_symmetries = SYMMETRY_COUNT / distinct_symmetry_images;
        let symmetry_hashes: Vec<String> = orbit
            .hashes
            .iter()
            .map(|hash| format!("{hash:016x}"))
            .collect();

        serde_json::to_writer(
            &mut *output,
            &OutputRecord {
                hash: input_record.hash,
                orbit_hash: symmetry_hashes[0].clone(),
                symmetry_hashes,
                preserving_symmetries,
                dataset_classes,
                multiplicity,
                ofp: &input_record.ofp,
            },
        )
        .map_err(io::Error::other)?;
        output.write_all(b"\n")?;

        summary.output_records += 1;
        summary.multiplicity = summary
            .multiplicity
            .checked_add(multiplicity)
            .ok_or_else(|| invalid_data("output multiplicity overflow"))?;

        if summary.output_records.is_multiple_of(REPORT_EVERY_REDUCED) {
            println!(
                "wrote {} symmetry classes; handled {} of {} input OFPs ({:.1?})",
                summary.output_records,
                summary.handled,
                records.len(),
                started.elapsed()
            );
        }
    }

    Ok(summary)
}

fn read_record<R: BufRead + Seek>(
    input: &mut R,
    path: &Path,
    index: usize,
    indexed: IndexedRecord,
    line: &mut String,
) -> io::Result<InputRecord> {
    input.seek(SeekFrom::Start(indexed.offset))?;
    line.clear();
    if input.read_line(line)? == 0 {
        return Err(invalid_line(
            path,
            index + 1,
            "record offset points past EOF",
        ));
    }

    let record: InputRecord = serde_json::from_str(line)
        .map_err(|error| invalid_line(path, index + 1, format!("invalid JSONL record: {error}")))?;
    let hash = parse_hash(path, index + 1, &record.hash)?;
    if hash != indexed.hash {
        return Err(invalid_line(
            path,
            index + 1,
            "record hash changed between indexing and reduction",
        ));
    }
    if record.multiplicity != indexed.multiplicity {
        return Err(invalid_line(
            path,
            index + 1,
            "record multiplicity changed between indexing and reduction",
        ));
    }
    Ok(record)
}

fn analyze_orbit(shape: &FramedPoset, symmetries: &[SignedPermutation]) -> Result<Orbit, String> {
    let mut images: Vec<(u64, FramedPoset)> = Vec::with_capacity(SYMMETRY_COUNT);

    for (symmetry_index, symmetry) in symmetries.iter().enumerate() {
        let transformed = transform(shape, symmetry)
            .map_err(|error| format!("symmetry {symmetry_index} could not be applied: {error}"))?;
        if let Some((sign_0, sign_1)) = cubularity_failure(&Arc::new(transformed.clone())) {
            let transformed_json = serde_json::to_string(&transformed)
                .unwrap_or_else(|error| format!("<serialization failed: {error}>"));
            return Err(format!(
                "symmetry {symmetry_index} {symmetry:?} produced a non-cubular OFP for signs ({sign_0:?}, {sign_1:?}); transformed OFP: {transformed_json}"
            ));
        }

        let normal = normalize(&transformed);
        if symmetry_index == 0 && !FramedPoset::equal(shape, &normal) {
            return Err("input OFP is not in canonical normal form".to_owned());
        }
        let hash = structural_hash(&normal);
        if let Some((_, existing)) = images
            .iter()
            .find(|(existing_hash, _)| *existing_hash == hash)
        {
            if !FramedPoset::equal(existing, &normal) {
                return Err(format!(
                    "structural hash collision within orbit at {hash:016x}"
                ));
            }
        } else {
            images.push((hash, normal));
        }
    }

    images.sort_unstable_by_key(|(hash, _)| *hash);
    Ok(Orbit {
        hashes: images.into_iter().map(|(hash, _)| hash).collect(),
    })
}

fn two_dimensional_symmetries() -> Vec<SignedPermutation> {
    let mut symmetries = Vec::with_capacity(SYMMETRY_COUNT);

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

    debug_assert_eq!(symmetries.len(), SYMMETRY_COUNT);
    symmetries
}

fn cubularity_failure(shape: &Arc<FramedPoset>) -> Option<(Sign, Sign)> {
    SIGN_PAIRS.into_iter().find(|&(sign_0, sign_1)| {
        let (_, zero_then_one) = iterated_boundary(&[(sign_0, 0), (sign_1, 1)], shape);
        let (_, one_then_zero) = iterated_boundary(&[(sign_1, 1), (sign_0, 0)], shape);
        !Embedding::same_subobject(&zero_then_one, &one_then_zero)
    })
}

fn validate_dataset_shape(shape: &FramedPoset, path: &Path, line: usize) -> io::Result<()> {
    let sizes = shape.sizes();
    if sizes.len() != 3 || sizes.iter().sum::<usize>() != 9 || sizes[2] == 0 {
        return Err(invalid_line(
            path,
            line,
            "OFP is not a nine-cell two-dimensional shape",
        ));
    }
    for (dim, size) in sizes.into_iter().enumerate() {
        for pos in 0..size {
            if shape
                .basis_of(dim, pos)
                .iter()
                .any(|&direction| direction > 1)
            {
                return Err(invalid_line(
                    path,
                    line,
                    "OFP contains a direction outside {0, 1}",
                ));
            }
        }
    }
    Ok(())
}

fn structural_hash(shape: &FramedPoset) -> u64 {
    let mut hasher = DefaultHasher::new();
    shape.hash(&mut hasher);
    hasher.finish()
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

fn temporary_path(output: &Path) -> PathBuf {
    let mut path = output.as_os_str().to_owned();
    path.push(".tmp");
    PathBuf::from(path)
}

fn invalid_line(path: &Path, line: usize, message: impl std::fmt::Display) -> io::Error {
    invalid_data(format!("{}:{line}: {message}", path.display()))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> FramedPoset {
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

    #[test]
    fn enumerates_the_eight_two_dimensional_symmetries() {
        let symmetries = two_dimensional_symmetries();

        assert_eq!(symmetries.len(), 8);
        for (index, symmetry) in symmetries.iter().enumerate() {
            assert!(!symmetries[..index].contains(symmetry));
        }
        assert_eq!(symmetries[0], SignedPermutation::identity(2));
    }

    #[test]
    fn standard_square_is_fixed_up_to_isomorphism_by_every_symmetry() {
        let normal = normalize(&square());
        let orbit = analyze_orbit(&normal, &two_dimensional_symmetries()).unwrap();

        assert_eq!(orbit.hashes.len(), 1);
    }

    #[test]
    fn parses_and_indexes_jsonl_records_strictly() {
        let shape = normalize(&square());
        let hash = structural_hash(&shape);
        let line = serde_json::json!({
            "hash": format!("{hash:016x}"),
            "multiplicity": 3,
            "ofp": shape,
        })
        .to_string()
            + "\n";
        let mut input = io::Cursor::new(line.into_bytes());

        let (records, hashes, multiplicity) =
            build_index(&mut input, Path::new("test.jsonl")).unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].hash, hash);
        assert_eq!(hashes.get(&hash), Some(&0));
        assert_eq!(multiplicity, 3);
    }
}
