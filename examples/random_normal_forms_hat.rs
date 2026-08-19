use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ofposets::{BoundaryMode, boundary};
use ofposets::{
    DirectionImage, Embedding, FramedPoset, RandomFramedPosetGenerator, Sign, SignedPermutation,
    normalize, transform,
};
use rand::rngs::{OsRng, SmallRng};
use rand::{SeedableRng, TryRngCore};
use serde::Serialize;

const CELL_COUNT: usize = 4;
const SYMMETRY_COUNT: usize = 8;
const OUTPUT_FILE: &str =
    "visualizations/random_4_cells_normal_forms_hat_cubular_up_to_symmetry.jsonl";
const BUFFER_CAPACITY: usize = 8 * 1024 * 1024;
const REPORT_INTERVAL: Duration = Duration::from_secs(5);
const INPUT_POLL_INTERVAL: Duration = Duration::from_millis(100);
const SIGN_PAIRS: [(Sign, Sign); 4] = [
    (Sign::Input, Sign::Input),
    (Sign::Input, Sign::Output),
    (Sign::Output, Sign::Input),
    (Sign::Output, Sign::Output),
];

type SharedState = Arc<Mutex<GeneratorState>>;

#[derive(Default)]
struct GeneratorState {
    /// Every distinct normalized symmetry image points to its orbit representative.
    image_to_orbit: HashMap<Arc<FramedPoset>, Arc<FramedPoset>>,
    /// Only the canonical representative of each orbit occurs as a key.
    orbits: HashMap<Arc<FramedPoset>, OrbitRecord>,
}

#[derive(Debug, Clone, Copy)]
struct OrbitRecord {
    stabilizer_size: usize,
    multiplicity: u64,
}

struct OrbitAnalysis {
    representative: Arc<FramedPoset>,
    images: Vec<Arc<FramedPoset>>,
    stabilizer_size: usize,
}

#[derive(Serialize)]
struct OutputRecord<'a> {
    hash: String,
    stabilizer_size: usize,
    multiplicity: u64,
    ofp: &'a FramedPoset,
}

struct OutputRow {
    hash: u64,
    representative: Arc<FramedPoset>,
    record: OrbitRecord,
}

struct WorkerContext<'a> {
    generator: &'a RandomFramedPosetGenerator,
    symmetries: &'a [SignedPermutation],
    state: &'a SharedState,
    generated: &'a AtomicU64,
    accepted: &'a AtomicU64,
    stop: &'a AtomicBool,
}

struct RawModeGuard;

impl RawModeGuard {
    fn enable() -> io::Result<Self> {
        enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

fn main() -> io::Result<()> {
    let seed = OsRng.try_next_u64().map_err(io::Error::other)?;
    let worker_count = (thread::available_parallelism()?.get() / 2).max(1);
    let generator = RandomFramedPosetGenerator::new(2, CELL_COUNT);
    let symmetries = Arc::new(two_dimensional_symmetries());
    let state = Arc::new(Mutex::new(GeneratorState::default()));
    let generated = Arc::new(AtomicU64::new(0));
    let accepted = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let worker_error = Arc::new(Mutex::new(None));

    println!(
        "sampling hat-boundary cubular {CELL_COUNT}-cell two-dimensional OFPs with {worker_count} workers (seed {seed:#018x})"
    );
    println!("press any key to stop and write {OUTPUT_FILE}");
    let raw_mode = RawModeGuard::enable()?;

    let run_result = thread::scope(|scope| {
        for worker in 0..worker_count {
            let generator = &generator;
            let symmetries = Arc::clone(&symmetries);
            let state = Arc::clone(&state);
            let generated = Arc::clone(&generated);
            let accepted = Arc::clone(&accepted);
            let stop = Arc::clone(&stop);
            let worker_error = Arc::clone(&worker_error);

            scope.spawn(move || {
                let context = WorkerContext {
                    generator,
                    symmetries: &symmetries,
                    state: &state,
                    generated: &generated,
                    accepted: &accepted,
                    stop: &stop,
                };
                if let Err(error) = sample_worker(worker, seed, context) {
                    set_worker_error(&worker_error, worker, error);
                    stop.store(true, Ordering::Release);
                }
            });
        }

        let result = monitor(&state, &generated, &accepted, &worker_error);
        stop.store(true, Ordering::Release);
        result
    });

    drop(raw_mode);
    if let Some(error) = worker_error.lock().unwrap().clone() {
        return Err(io::Error::other(error));
    }
    run_result?;

    let generated = generated.load(Ordering::Relaxed);
    let accepted = accepted.load(Ordering::Relaxed);
    let (orbit_count, image_count) = state_counts(&state)?;
    println!(
        "stopped after {generated} candidates and {accepted} cubular samples; writing {orbit_count} symmetry orbits ({image_count} cached images)"
    );
    write_dataset(Path::new(OUTPUT_FILE), &state, accepted)?;
    println!("wrote {orbit_count} symmetry orbits to {OUTPUT_FILE}");
    Ok(())
}

fn sample_worker(worker: usize, seed: u64, context: WorkerContext<'_>) -> Result<(), String> {
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(worker as u64));

    while !context.stop.load(Ordering::Acquire) {
        let shape = Arc::new(context.generator.generate(&mut rng));
        context.generated.fetch_add(1, Ordering::Relaxed);
        if !is_hat_cubular(&shape) {
            continue;
        }

        let normal = Arc::new(normalize(&shape));
        debug_assert!(normal.is_normal());
        context.accepted.fetch_add(1, Ordering::Relaxed);

        if increment_cached_orbit(context.state, &normal)? {
            continue;
        }

        let analysis = analyze_orbit(&normal, context.symmetries)?;
        register_or_increment_orbit(context.state, analysis)?;
    }

    Ok(())
}

fn increment_cached_orbit(state: &SharedState, image: &Arc<FramedPoset>) -> Result<bool, String> {
    let mut state = state
        .lock()
        .map_err(|_| "generator state mutex was poisoned".to_owned())?;
    let Some(representative) = state.image_to_orbit.get(image).cloned() else {
        return Ok(false);
    };
    increment_multiplicity(&mut state, &representative)?;
    Ok(true)
}

fn register_or_increment_orbit(state: &SharedState, analysis: OrbitAnalysis) -> Result<(), String> {
    let mut state = state
        .lock()
        .map_err(|_| "generator state mutex was poisoned".to_owned())?;

    let mut existing_representative: Option<Arc<FramedPoset>> = None;
    for image in &analysis.images {
        let Some(representative) = state.image_to_orbit.get(image) else {
            continue;
        };
        if let Some(existing) = &existing_representative {
            if !FramedPoset::equal(existing, representative) {
                return Err("one symmetry orbit points to two representatives".to_owned());
            }
        } else {
            existing_representative = Some(Arc::clone(representative));
        }
    }

    if let Some(representative) = existing_representative {
        if !FramedPoset::equal(&representative, &analysis.representative) {
            return Err("concurrent orbit analyses chose different representatives".to_owned());
        }
        let record = state
            .orbits
            .get(&representative)
            .ok_or_else(|| "cached image points to a missing orbit".to_owned())?;
        if record.stabilizer_size != analysis.stabilizer_size {
            return Err("concurrent orbit analyses computed different stabilizers".to_owned());
        }
        for image in analysis.images {
            match state.image_to_orbit.get(&image) {
                Some(cached) if !FramedPoset::equal(cached, &representative) => {
                    return Err("symmetry image is already assigned to another orbit".to_owned());
                }
                Some(_) => {}
                None => {
                    state
                        .image_to_orbit
                        .insert(image, Arc::clone(&representative));
                }
            }
        }
        increment_multiplicity(&mut state, &representative)?;
        return Ok(());
    }

    if state.orbits.contains_key(&analysis.representative) {
        return Err("orbit representative exists without a cached symmetry image".to_owned());
    }
    for image in &analysis.images {
        if state.image_to_orbit.contains_key(image) {
            return Err("symmetry image appeared while registering a new orbit".to_owned());
        }
    }

    let representative = Arc::clone(&analysis.representative);
    state.orbits.insert(
        Arc::clone(&representative),
        OrbitRecord {
            stabilizer_size: analysis.stabilizer_size,
            multiplicity: 1,
        },
    );
    for image in analysis.images {
        state
            .image_to_orbit
            .insert(image, Arc::clone(&representative));
    }
    Ok(())
}

fn increment_multiplicity(
    state: &mut GeneratorState,
    representative: &Arc<FramedPoset>,
) -> Result<(), String> {
    let record = state
        .orbits
        .get_mut(representative)
        .ok_or_else(|| "cached image points to a missing orbit".to_owned())?;
    record.multiplicity = record
        .multiplicity
        .checked_add(1)
        .ok_or_else(|| "orbit multiplicity overflow".to_owned())?;
    Ok(())
}

fn analyze_orbit(
    normal: &Arc<FramedPoset>,
    symmetries: &[SignedPermutation],
) -> Result<OrbitAnalysis, String> {
    let mut distinct: Vec<(Vec<u8>, Arc<FramedPoset>)> = Vec::with_capacity(SYMMETRY_COUNT);

    for symmetry in symmetries {
        let transformed = transform(normal, symmetry)
            .map_err(|error| format!("could not apply symmetry {symmetry:?}: {error}"))?;
        debug_assert!(is_hat_cubular(&Arc::new(transformed.clone())));
        let image = Arc::new(normalize(&transformed));
        if distinct
            .iter()
            .any(|(_, existing)| FramedPoset::equal(existing, &image))
        {
            continue;
        }
        let serialized = serde_json::to_vec(image.as_ref())
            .map_err(|error| format!("could not serialize symmetry image: {error}"))?;
        distinct.push((serialized, image));
    }

    if distinct.is_empty() || !SYMMETRY_COUNT.is_multiple_of(distinct.len()) {
        return Err(format!(
            "invalid symmetry orbit with {} distinct images",
            distinct.len()
        ));
    }
    distinct.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
    let stabilizer_size = SYMMETRY_COUNT / distinct.len();
    let representative = Arc::clone(&distinct[0].1);
    let images = distinct.into_iter().map(|(_, image)| image).collect();

    Ok(OrbitAnalysis {
        representative,
        images,
        stabilizer_size,
    })
}

fn is_hat_cubular(shape: &Arc<FramedPoset>) -> bool {
    SIGN_PAIRS.into_iter().all(|(sign_0, sign_1)| {
        let zero_then_one = iterated_hat_boundary(shape, sign_0, 0, sign_1, 1);
        let one_then_zero = iterated_hat_boundary(shape, sign_1, 1, sign_0, 0);
        Embedding::equal(&zero_then_one, &one_then_zero)
    })
}

fn iterated_hat_boundary(
    shape: &Arc<FramedPoset>,
    first_sign: Sign,
    first_direction: usize,
    second_sign: Sign,
    second_direction: usize,
) -> Embedding {
    let (first_boundary, first_embedding) =
        boundary(BoundaryMode::Hat, first_sign, first_direction, shape);
    let (_, second_embedding) = boundary(
        BoundaryMode::Hat,
        second_sign,
        second_direction,
        &first_boundary,
    );
    Embedding::compose(&second_embedding, &first_embedding)
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

fn monitor(
    state: &SharedState,
    generated: &AtomicU64,
    accepted: &AtomicU64,
    worker_error: &Mutex<Option<String>>,
) -> io::Result<()> {
    let mut next_report = Instant::now() + REPORT_INTERVAL;

    loop {
        if let Some(error) = worker_error.lock().unwrap().clone() {
            return Err(io::Error::other(error));
        }

        let now = Instant::now();
        let timeout = INPUT_POLL_INTERVAL.min(next_report.saturating_duration_since(now));
        if event::poll(timeout)? {
            match event::read()? {
                Event::Key(key) if key.kind != KeyEventKind::Release => return Ok(()),
                _ => {}
            }
        }

        if Instant::now() >= next_report {
            let generated = generated.load(Ordering::Relaxed);
            let accepted = accepted.load(Ordering::Relaxed);
            let (orbit_count, image_count) = state_counts(state)?;
            print_raw_line(&format!(
                "OFP candidates: {generated}; cubular samples: {accepted}; symmetry orbits: {orbit_count}; cached images: {image_count}"
            ))?;
            next_report = Instant::now() + REPORT_INTERVAL;
        }
    }
}

fn state_counts(state: &SharedState) -> io::Result<(usize, usize)> {
    let state = state
        .lock()
        .map_err(|_| io::Error::other("generator state mutex was poisoned"))?;
    Ok((state.orbits.len(), state.image_to_orbit.len()))
}

fn write_dataset(path: &Path, state: &SharedState, accepted: u64) -> io::Result<()> {
    let mut rows = snapshot_and_validate(state, accepted)?;
    rows.sort_unstable_by_key(|row| row.hash);
    for pair in rows.windows(2) {
        if pair[0].hash == pair[1].hash {
            return Err(io::Error::other(format!(
                "structural hash collision between {} and {}",
                serde_json::to_string(pair[0].representative.as_ref())
                    .unwrap_or_else(|error| format!("<serialization failed: {error}>")),
                serde_json::to_string(pair[1].representative.as_ref())
                    .unwrap_or_else(|error| format!("<serialization failed: {error}>"))
            )));
        }
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary_path = temporary_path(path);
    let output_file = File::create(&temporary_path)?;
    let mut output = BufWriter::with_capacity(BUFFER_CAPACITY, output_file);

    for row in &rows {
        serde_json::to_writer(
            &mut output,
            &OutputRecord {
                hash: format!("{:016x}", row.hash),
                stabilizer_size: row.record.stabilizer_size,
                multiplicity: row.record.multiplicity,
                ofp: row.representative.as_ref(),
            },
        )
        .map_err(io::Error::other)?;
        output.write_all(b"\n")?;
    }

    output.flush()?;
    let output_file = output.into_inner().map_err(|error| error.into_error())?;
    output_file.sync_all()?;
    fs::rename(&temporary_path, path)?;
    Ok(())
}

fn snapshot_and_validate(state: &SharedState, accepted: u64) -> io::Result<Vec<OutputRow>> {
    let state = state
        .lock()
        .map_err(|_| io::Error::other("generator state mutex was poisoned"))?;
    if state.orbits.is_empty() {
        return Err(io::Error::other(
            "no cubular symmetry orbits were generated",
        ));
    }

    let mut alias_counts: HashMap<Arc<FramedPoset>, usize> = HashMap::new();
    for representative in state.image_to_orbit.values() {
        if !state.orbits.contains_key(representative) {
            return Err(io::Error::other(
                "cached symmetry image points to a missing orbit",
            ));
        }
        *alias_counts.entry(Arc::clone(representative)).or_default() += 1;
    }

    let mut total_multiplicity = 0u64;
    let mut rows = Vec::with_capacity(state.orbits.len());
    for (representative, &record) in &state.orbits {
        if record.multiplicity == 0
            || record.stabilizer_size == 0
            || !SYMMETRY_COUNT.is_multiple_of(record.stabilizer_size)
        {
            return Err(io::Error::other("orbit has invalid stored statistics"));
        }
        let expected_images = SYMMETRY_COUNT / record.stabilizer_size;
        if alias_counts.get(representative).copied() != Some(expected_images) {
            return Err(io::Error::other(format!(
                "orbit has {} cached images, expected {expected_images}",
                alias_counts.get(representative).copied().unwrap_or(0)
            )));
        }
        total_multiplicity = total_multiplicity
            .checked_add(record.multiplicity)
            .ok_or_else(|| io::Error::other("total multiplicity overflow"))?;
        rows.push(OutputRow {
            hash: structural_hash(representative),
            representative: Arc::clone(representative),
            record,
        });
    }

    if total_multiplicity != accepted {
        return Err(io::Error::other(format!(
            "stored multiplicity {total_multiplicity} does not match accepted sample count {accepted}"
        )));
    }
    Ok(rows)
}

fn structural_hash(shape: &FramedPoset) -> u64 {
    let mut hasher = DefaultHasher::new();
    shape.hash(&mut hasher);
    hasher.finish()
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    PathBuf::from(temporary)
}

fn set_worker_error(worker_error: &Mutex<Option<String>>, worker: usize, error: String) {
    let mut first_error = worker_error.lock().unwrap();
    if first_error.is_none() {
        *first_error = Some(format!("worker {}: {error}", worker + 1));
    }
}

fn print_raw_line(message: &str) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    write!(stdout, "{message}\r\n")?;
    stdout.flush()
}
