use std::collections::hash_map::DefaultHasher;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Instant;

use ofposets::FramedPoset;

const DATASET_DIR: &str = "visualizations/random_8_cells_normal_forms_cubular";
const OUTPUT_FILE: &str = "hashes.tsv";
const BATCH_SIZE: usize = 1024;
const REPORT_EVERY: u64 = 1_000_000;

type WorkItem = (u64, PathBuf);
type HashRecord = (u64, u64);

fn main() -> io::Result<()> {
    let worker_count = (thread::available_parallelism()?.get() / 2).max(1);
    let started = Instant::now();

    println!("hashing normalized OFPs in {DATASET_DIR} with {worker_count} workers");
    let mut records = hash_dataset(Path::new(DATASET_DIR), worker_count)?;

    println!("sorting {} hash records", records.len());
    records.sort_unstable();

    let output = Path::new(DATASET_DIR).join(OUTPUT_FILE);
    write_tsv(&output, &records)?;
    println!(
        "wrote {} sorted records to {} in {:.1?}",
        records.len(),
        output.display(),
        started.elapsed()
    );
    Ok(())
}

fn hash_dataset(directory: &Path, worker_count: usize) -> io::Result<Vec<HashRecord>> {
    let checked = Arc::new(AtomicU64::new(0));
    let (work_sender, work_receiver) =
        mpsc::sync_channel::<Vec<WorkItem>>(worker_count.saturating_mul(2).max(1));
    let work_receiver = Arc::new(Mutex::new(work_receiver));
    let (result_sender, result_receiver) = mpsc::channel::<io::Result<Vec<HashRecord>>>();

    thread::scope(|scope| -> io::Result<Vec<HashRecord>> {
        for _ in 0..worker_count {
            let work_receiver = Arc::clone(&work_receiver);
            let result_sender = result_sender.clone();
            let checked = Arc::clone(&checked);
            scope.spawn(move || hash_worker(work_receiver, result_sender, checked));
        }
        drop(result_sender);

        let mut batch = Vec::with_capacity(BATCH_SIZE);
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let Some(sample) = sample_number(&entry.file_name()) else {
                continue;
            };

            batch.push((sample, entry.path()));
            if batch.len() == BATCH_SIZE {
                work_sender
                    .send(std::mem::replace(
                        &mut batch,
                        Vec::with_capacity(BATCH_SIZE),
                    ))
                    .map_err(|_| io::Error::other("hash workers stopped unexpectedly"))?;
            }
        }
        if !batch.is_empty() {
            work_sender
                .send(batch)
                .map_err(|_| io::Error::other("hash workers stopped unexpectedly"))?;
        }
        drop(work_sender);

        let mut records = Vec::new();
        for result in result_receiver {
            records.extend(result?);
        }
        Ok(records)
    })
}

fn hash_worker(
    work_receiver: Arc<Mutex<mpsc::Receiver<Vec<WorkItem>>>>,
    result_sender: mpsc::Sender<io::Result<Vec<HashRecord>>>,
    checked: Arc<AtomicU64>,
) {
    let mut json = Vec::with_capacity(1024);

    loop {
        let batch = match work_receiver.lock().unwrap().recv() {
            Ok(batch) => batch,
            Err(_) => return,
        };

        let result = hash_batch(batch, &checked, &mut json);
        if result_sender.send(result).is_err() {
            return;
        }
    }
}

fn hash_batch(
    batch: Vec<WorkItem>,
    checked: &AtomicU64,
    json: &mut Vec<u8>,
) -> io::Result<Vec<HashRecord>> {
    let mut records = Vec::with_capacity(batch.len());

    for (sample, path) in batch {
        let mut file =
            File::open(&path).map_err(|error| with_path_context(error, &path, "could not open"))?;
        json.clear();
        file.read_to_end(json)
            .map_err(|error| with_path_context(error, &path, "could not read"))?;
        let shape: FramedPoset = serde_json::from_slice(json).map_err(|error| {
            io::Error::other(format!("could not deserialize {}: {error}", path.display()))
        })?;
        records.push((structural_hash(&shape), sample));

        let count = checked.fetch_add(1, Ordering::Relaxed) + 1;
        if count.is_multiple_of(REPORT_EVERY) {
            println!("hashed {count} OFPs");
        }
    }

    Ok(records)
}

fn structural_hash(shape: &FramedPoset) -> u64 {
    let mut hasher = DefaultHasher::new();
    shape.hash(&mut hasher);
    hasher.finish()
}

fn sample_number(file_name: &OsStr) -> Option<u64> {
    let file_name = file_name.to_str()?;
    file_name
        .strip_prefix("sample_")?
        .strip_suffix(".ofp.json")?
        .parse()
        .ok()
}

fn write_tsv(path: &Path, records: &[HashRecord]) -> io::Result<()> {
    let temporary = path.with_extension("tsv.tmp");
    let mut output = BufWriter::with_capacity(8 * 1024 * 1024, File::create(&temporary)?);
    writeln!(output, "hash\tsample")?;
    for &(hash, sample) in records {
        writeln!(output, "{hash:016x}\t{sample}")?;
    }
    output.flush()?;
    drop(output);
    fs::rename(temporary, path)
}

fn with_path_context(error: io::Error, path: &Path, operation: &str) -> io::Error {
    io::Error::new(
        error.kind(),
        format!("{operation} {}: {error}", path.display()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_only_sample_json_names() {
        assert_eq!(
            sample_number(OsStr::new("sample_000042.ofp.json")),
            Some(42)
        );
        assert_eq!(
            sample_number(OsStr::new("sample_12345678.ofp.json")),
            Some(12_345_678)
        );
        assert_eq!(sample_number(OsStr::new("hashes.tsv")), None);
        assert_eq!(sample_number(OsStr::new("sample_42.dot")), None);
    }

    #[test]
    fn structural_hash_uses_framed_poset_equality_data() {
        let first = FramedPoset::point();
        let second = FramedPoset::point();

        assert_eq!(first, second);
        assert_eq!(structural_hash(&first), structural_hash(&second));
    }
}
