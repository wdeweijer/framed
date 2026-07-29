use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ofposets::{FramedPoset, is_cubular, is_strongly_cubular, normalize};
use serde::Deserialize;

const BUFFER_CAPACITY: usize = 8 * 1024 * 1024;
const REPORT_INTERVAL: usize = 100_000;
const SYMMETRY_COUNT: usize = 8;

const DATASETS: [DatasetSpec; 4] = [
    DatasetSpec {
        cells: 4,
        input: "visualizations/random_4_cells_normal_forms_hat_cubular_up_to_symmetry.jsonl",
        output: "visualizations/random_4_cells_normal_forms_hat_strongly_cubular_up_to_symmetry.jsonl",
    },
    DatasetSpec {
        cells: 5,
        input: "visualizations/random_5_cells_normal_forms_hat_cubular_up_to_symmetry.jsonl",
        output: "visualizations/random_5_cells_normal_forms_hat_strongly_cubular_up_to_symmetry.jsonl",
    },
    DatasetSpec {
        cells: 6,
        input: "visualizations/random_6_cells_normal_forms_hat_cubular_up_to_symmetry.jsonl",
        output: "visualizations/random_6_cells_normal_forms_hat_strongly_cubular_up_to_symmetry.jsonl",
    },
    DatasetSpec {
        cells: 9,
        input: "visualizations/random_9_cells_normal_forms_hat_cubular_up_to_symmetry.jsonl",
        output: "visualizations/random_9_cells_normal_forms_hat_strongly_cubular_up_to_symmetry.jsonl",
    },
];

#[derive(Clone, Copy)]
struct DatasetSpec {
    cells: usize,
    input: &'static str,
    output: &'static str,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetRecord {
    hash: String,
    stabilizer_size: usize,
    multiplicity: u64,
    ofp: FramedPoset,
}

#[derive(Default)]
struct FilterStatistics {
    records: u64,
    strongly_cubular_records: u64,
    multiplicity: u128,
    strongly_cubular_multiplicity: u128,
}

fn main() -> io::Result<()> {
    for spec in DATASETS {
        filter_dataset(spec)?;
    }
    Ok(())
}

fn filter_dataset(spec: DatasetSpec) -> io::Result<()> {
    let input_path = Path::new(spec.input);
    let output_path = Path::new(spec.output);
    let temporary_path = temporary_path(output_path);
    let result = filter_to_temporary(spec, input_path, &temporary_path);

    match result {
        Ok(statistics) => {
            fs::rename(&temporary_path, output_path)?;
            print_statistics(spec, &statistics);
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary_path);
            Err(error)
        }
    }
}

fn filter_to_temporary(
    spec: DatasetSpec,
    input_path: &Path,
    temporary_path: &Path,
) -> io::Result<FilterStatistics> {
    let mut reader = BufReader::with_capacity(BUFFER_CAPACITY, File::open(input_path)?);
    let mut writer = BufWriter::with_capacity(BUFFER_CAPACITY, File::create(temporary_path)?);
    let mut statistics = FilterStatistics::default();
    let mut previous_hash = None;
    let mut line = String::new();

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        statistics.records += 1;
        let line_number = usize::try_from(statistics.records)
            .map_err(|_| invalid_data("line number exceeds usize"))?;
        let record: DatasetRecord = serde_json::from_str(line.trim_end())
            .map_err(|error| invalid_line(input_path, line_number, error))?;
        let hash = validate_record(spec, input_path, line_number, &record)?;
        if previous_hash.is_some_and(|previous| previous >= hash) {
            return Err(invalid_line(
                input_path,
                line_number,
                "hashes are not strictly increasing",
            ));
        }
        previous_hash = Some(hash);
        statistics.multiplicity += u128::from(record.multiplicity);

        let shape = Arc::new(record.ofp);
        if is_strongly_cubular(&shape) {
            writer.write_all(line.as_bytes())?;
            if !line.ends_with('\n') {
                writer.write_all(b"\n")?;
            }
            statistics.strongly_cubular_records += 1;
            statistics.strongly_cubular_multiplicity += u128::from(record.multiplicity);
        }

        if statistics.records.is_multiple_of(REPORT_INTERVAL as u64) {
            println!(
                "{} cells: checked {} records; retained {}",
                spec.cells, statistics.records, statistics.strongly_cubular_records
            );
        }
    }

    writer.flush()?;
    if statistics.records == 0 {
        return Err(invalid_data(format!("{} is empty", input_path.display())));
    }
    Ok(statistics)
}

fn validate_record(
    spec: DatasetSpec,
    path: &Path,
    line: usize,
    record: &DatasetRecord,
) -> io::Result<u64> {
    let hash = parse_hash(path, line, &record.hash)?;
    if record.stabilizer_size == 0 || !SYMMETRY_COUNT.is_multiple_of(record.stabilizer_size) {
        return Err(invalid_line(
            path,
            line,
            "stabilizer size does not divide the symmetry-group order",
        ));
    }
    if record.multiplicity == 0 {
        return Err(invalid_line(path, line, "multiplicity is zero"));
    }
    if record.ofp.sizes().iter().sum::<usize>() != spec.cells {
        return Err(invalid_line(
            path,
            line,
            format!("OFP does not have {} cells", spec.cells),
        ));
    }
    validate_directions(path, line, &record.ofp)?;

    let normal = Arc::new(normalize(&record.ofp));
    if !FramedPoset::equal(&normal, &record.ofp) {
        return Err(invalid_line(path, line, "OFP is not normalized"));
    }
    if structural_hash(&normal) != hash {
        return Err(invalid_line(path, line, "stored hash is incorrect"));
    }
    if !is_cubular(&normal) {
        return Err(invalid_line(
            path,
            line,
            "input dataset contains a non-cubular OFP",
        ));
    }
    Ok(hash)
}

fn validate_directions(path: &Path, line: usize, shape: &FramedPoset) -> io::Result<()> {
    for (dim, size) in shape.sizes().into_iter().enumerate() {
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

fn print_statistics(spec: DatasetSpec, statistics: &FilterStatistics) {
    println!(
        "{} cells: retained {} of {} records ({:.4}%) in {}",
        spec.cells,
        statistics.strongly_cubular_records,
        statistics.records,
        percentage(
            u128::from(statistics.strongly_cubular_records),
            u128::from(statistics.records)
        ),
        spec.output
    );
    println!(
        "{} cells: retained multiplicity {} of {} ({:.4}%)",
        spec.cells,
        statistics.strongly_cubular_multiplicity,
        statistics.multiplicity,
        percentage(
            statistics.strongly_cubular_multiplicity,
            statistics.multiplicity
        )
    );
}

fn percentage(part: u128, whole: u128) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
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
            "hash is not canonical hexadecimal",
        ));
    }
    Ok(value)
}

fn temporary_path(output: &Path) -> PathBuf {
    output.with_extension("jsonl.tmp")
}

fn invalid_line(path: &Path, line: usize, error: impl std::fmt::Display) -> io::Error {
    invalid_data(format!("{}:{line}: {error}", path.display()))
}

fn invalid_data(error: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.into())
}
