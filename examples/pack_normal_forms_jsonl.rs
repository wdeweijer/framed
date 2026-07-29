use std::collections::hash_map::DefaultHasher;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use ofposets::FramedPoset;
use serde::Serialize;

const DATASET_DIR: &str = "visualizations/random_8_cells_normal_forms_cubular";
const OUTPUT_FILE: &str = "visualizations/random_8_cells_normal_forms_cubular.jsonl";
const REPORT_EVERY: usize = 1_000_000;

#[derive(Serialize)]
struct JsonlRecord<'a> {
    hash: &'a str,
    multiplicity: u64,
    ofp: &'a FramedPoset,
}

fn main() -> io::Result<()> {
    let dataset = Path::new(DATASET_DIR);
    let multiplicities = read_multiplicities(&dataset.join("multiplicities.tsv"))?;
    let hashes_path = dataset.join("hashes.tsv");
    let output_path = Path::new(OUTPUT_FILE);
    let temporary_path = temporary_path(output_path);

    println!(
        "packing {} strictly validated OFPs into {}",
        multiplicities.len(),
        output_path.display()
    );

    let hashes_file = File::open(&hashes_path)?;
    let mut hashes = BufReader::with_capacity(8 * 1024 * 1024, hashes_file).lines();
    require_header(&hashes_path, hashes.next(), "hash\tsample")?;

    let output_file = File::create(&temporary_path)?;
    let mut output = BufWriter::with_capacity(8 * 1024 * 1024, output_file);
    let mut seen = vec![false; multiplicities.len()];
    let mut previous_hash: Option<String> = None;
    let mut json = Vec::with_capacity(1024);
    let mut record_count = 0usize;

    for (line_index, line) in hashes.enumerate() {
        let line_number = line_index + 2;
        let line = line?;
        let (hash, sample) = parse_hash_row(&hashes_path, line_number, &line)?;

        if previous_hash
            .as_deref()
            .is_some_and(|previous| previous > hash)
        {
            return Err(invalid_line(
                &hashes_path,
                line_number,
                "hashes are not sorted in nondecreasing order",
            ));
        }
        previous_hash = Some(hash.to_owned());

        let sample = usize::try_from(sample).map_err(|_| {
            invalid_line(
                &hashes_path,
                line_number,
                "sample number does not fit usize",
            )
        })?;
        let multiplicity = multiplicities.get(sample).copied().ok_or_else(|| {
            invalid_line(
                &hashes_path,
                line_number,
                "sample number has no multiplicity",
            )
        })?;
        if std::mem::replace(&mut seen[sample], true) {
            return Err(invalid_line(
                &hashes_path,
                line_number,
                "sample number occurs more than once",
            ));
        }

        let shape_path = dataset.join(format!("sample_{sample:06}.ofp.json"));
        let shape = read_shape(&shape_path, &mut json)?;
        validate_dataset_shape(&shape, &shape_path)?;

        let actual_hash = structural_hash(&shape);
        if actual_hash != hash {
            return Err(invalid_line(
                &hashes_path,
                line_number,
                format!(
                    "stored hash {hash} does not match recomputed hash {actual_hash} for sample {sample}"
                ),
            ));
        }

        serde_json::to_writer(
            &mut output,
            &JsonlRecord {
                hash,
                multiplicity,
                ofp: &shape,
            },
        )
        .map_err(io::Error::other)?;
        output.write_all(b"\n")?;

        record_count += 1;
        if record_count.is_multiple_of(REPORT_EVERY) {
            println!("packed {record_count} OFPs");
        }
    }

    if record_count != multiplicities.len() {
        return Err(invalid_data(format!(
            "hash row count {record_count} does not match multiplicity count {}",
            multiplicities.len()
        )));
    }
    if let Some(missing) = seen.iter().position(|&is_seen| !is_seen) {
        return Err(invalid_data(format!(
            "sample {missing} has a multiplicity but no hash row"
        )));
    }

    validate_source_files(dataset, &seen)?;
    output.flush()?;
    let output_file = output.into_inner().map_err(|error| error.into_error())?;
    output_file.sync_all()?;
    fs::rename(&temporary_path, output_path)?;

    println!("wrote {record_count} OFPs to {}", output_path.display());
    Ok(())
}

fn read_multiplicities(path: &Path) -> io::Result<Vec<u64>> {
    let file = File::open(path)?;
    let mut lines = BufReader::with_capacity(8 * 1024 * 1024, file).lines();
    require_header(path, lines.next(), "representative\tcount")?;

    let mut multiplicities = Vec::new();
    for (line_index, line) in lines.enumerate() {
        let line_number = line_index + 2;
        let line = line?;
        let (representative, count) = exactly_two_fields(path, line_number, &line)?;
        let sample = representative
            .strip_prefix("sample_")
            .ok_or_else(|| invalid_line(path, line_number, "invalid representative name"))?
            .parse::<usize>()
            .map_err(|_| invalid_line(path, line_number, "invalid representative number"))?;
        if sample != multiplicities.len() {
            return Err(invalid_line(
                path,
                line_number,
                format!(
                    "expected contiguous sample {}, found {sample}",
                    multiplicities.len()
                ),
            ));
        }

        let count = count
            .parse::<u64>()
            .map_err(|_| invalid_line(path, line_number, "invalid multiplicity"))?;
        if count == 0 {
            return Err(invalid_line(
                path,
                line_number,
                "multiplicity must be positive",
            ));
        }
        multiplicities.push(count);
    }

    if multiplicities.is_empty() {
        return Err(invalid_data("multiplicity table is empty"));
    }
    Ok(multiplicities)
}

fn parse_hash_row<'a>(
    path: &Path,
    line_number: usize,
    line: &'a str,
) -> io::Result<(&'a str, u64)> {
    let (hash, sample) = exactly_two_fields(path, line_number, line)?;
    let parsed_hash = u64::from_str_radix(hash, 16)
        .map_err(|_| invalid_line(path, line_number, "hash is not hexadecimal"))?;
    if hash.len() != 16 || format!("{parsed_hash:016x}") != hash {
        return Err(invalid_line(
            path,
            line_number,
            "hash must be exactly 16 lowercase hexadecimal digits",
        ));
    }

    let sample = sample
        .parse::<u64>()
        .map_err(|_| invalid_line(path, line_number, "invalid sample number"))?;
    Ok((hash, sample))
}

fn exactly_two_fields<'a>(
    path: &Path,
    line_number: usize,
    line: &'a str,
) -> io::Result<(&'a str, &'a str)> {
    let mut fields = line.split('\t');
    let first = fields
        .next()
        .ok_or_else(|| invalid_line(path, line_number, "missing first field"))?;
    let second = fields
        .next()
        .ok_or_else(|| invalid_line(path, line_number, "missing second field"))?;
    if first.is_empty() || second.is_empty() || fields.next().is_some() {
        return Err(invalid_line(
            path,
            line_number,
            "expected exactly two nonempty tab-separated fields",
        ));
    }
    Ok((first, second))
}

fn require_header(
    path: &Path,
    header: Option<io::Result<String>>,
    expected: &str,
) -> io::Result<()> {
    let header = header.ok_or_else(|| invalid_data(format!("{} is empty", path.display())))??;
    if header != expected {
        return Err(invalid_data(format!(
            "{} has header {header:?}, expected {expected:?}",
            path.display()
        )));
    }
    Ok(())
}

fn read_shape(path: &Path, json: &mut Vec<u8>) -> io::Result<FramedPoset> {
    json.clear();
    File::open(path)?.read_to_end(json)?;
    serde_json::from_slice(json).map_err(|error| {
        invalid_data(format!(
            "could not deserialize framed poset {}: {error}",
            path.display()
        ))
    })
}

fn validate_dataset_shape(shape: &FramedPoset, path: &Path) -> io::Result<()> {
    let sizes = shape.sizes();
    for (dim, size) in sizes.into_iter().enumerate() {
        for pos in 0..size {
            if shape
                .basis_of(dim, pos)
                .iter()
                .any(|&direction| direction > 1)
            {
                return Err(invalid_data(format!(
                    "{} contains a direction outside {{0, 1}}",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

fn validate_source_files(dataset: &Path, seen: &[bool]) -> io::Result<()> {
    let mut source_count = 0usize;
    for entry in fs::read_dir(dataset)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            return Err(invalid_data("dataset contains a non-UTF-8 filename"));
        };
        if !file_name.starts_with("sample_") {
            continue;
        }

        let sample = sample_file_number(OsStr::new(file_name)).ok_or_else(|| {
            invalid_data(format!("invalid sample filename in dataset: {file_name}"))
        })?;
        if sample >= seen.len() || !seen[sample] {
            return Err(invalid_data(format!(
                "source file {file_name} has no matching hash and multiplicity"
            )));
        }
        source_count += 1;
    }

    if source_count != seen.len() {
        return Err(invalid_data(format!(
            "found {source_count} sample files, expected {}",
            seen.len()
        )));
    }
    Ok(())
}

fn sample_file_number(file_name: &OsStr) -> Option<usize> {
    file_name
        .to_str()?
        .strip_prefix("sample_")?
        .strip_suffix(".ofp.json")?
        .parse()
        .ok()
}

fn structural_hash(shape: &FramedPoset) -> String {
    let mut hasher = DefaultHasher::new();
    shape.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn temporary_path(output: &Path) -> PathBuf {
    let mut path = output.as_os_str().to_owned();
    path.push(".tmp");
    PathBuf::from(path)
}

fn invalid_line(path: &Path, line_number: usize, message: impl std::fmt::Display) -> io::Error {
    invalid_data(format!("{}:{line_number}: {message}", path.display()))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_strict_hash_rows() {
        let path = Path::new("hashes.tsv");

        assert_eq!(
            parse_hash_row(path, 2, "0123456789abcdef\t42").unwrap(),
            ("0123456789abcdef", 42)
        );
        assert!(parse_hash_row(path, 2, "123\t42").is_err());
        assert!(parse_hash_row(path, 2, "0123456789ABCDEF\t42").is_err());
        assert!(parse_hash_row(path, 2, "0123456789abcdef\t42\textra").is_err());
    }

    #[test]
    fn jsonl_record_serializes_on_one_line_without_sample_number() {
        let shape = FramedPoset::point();
        let record = JsonlRecord {
            hash: "0123456789abcdef",
            multiplicity: 7,
            ofp: &shape,
        };
        let json = serde_json::to_string(&record).unwrap();

        assert!(!json.contains('\n'));
        assert!(!json.contains("sample"));
        assert!(json.contains("\"multiplicity\":7"));
        assert!(json.contains("\"ofp\""));
    }
}
