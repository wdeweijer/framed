use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::env;
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
use ofposets::poset::boundary_hat;
use ofposets::{
    DirectionImage, FramedPoset, RandomFramedPosetGenerator, Sign, SignedPermutation,
    is_strongly_cubular, normalize, transform,
};
use rand::rngs::{OsRng, SmallRng};
use rand::{SeedableRng, TryRngCore};
use serde::Serialize;

const DIRECTION_COUNT: usize = 3;
const SYMMETRY_COUNT: usize = 48;
const DEFAULT_CELL_COUNT: usize = 8;
const BUFFER_CAPACITY: usize = 8 * 1024 * 1024;
const REPORT_INTERVAL: Duration = Duration::from_secs(5);
const INPUT_POLL_INTERVAL: Duration = Duration::from_millis(100);

type SharedState = Arc<Mutex<GeneratorState>>;

struct Options {
    cell_count: usize,
    worker_count: usize,
    seed: u64,
    output: PathBuf,
}

#[derive(Default)]
struct GeneratorState {
    /// Every normalized symmetry image points to its orbit representative.
    image_to_orbit: HashMap<Arc<FramedPoset>, Arc<FramedPoset>>,
    /// Only the lexicographically canonical representative occurs as a key.
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

struct OutputRow {
    hash: u64,
    representative: Arc<FramedPoset>,
    record: OrbitRecord,
    boundary_hashes: [[u64; 2]; DIRECTION_COUNT],
}

#[derive(Serialize)]
struct OutputRecord<'a> {
    hash: String,
    stabilizer_size: usize,
    multiplicity: u64,
    boundary_hashes: [BoundaryHashRecord; DIRECTION_COUNT],
    ofp: &'a FramedPoset,
}

#[derive(Serialize)]
struct BoundaryHashRecord {
    direction: usize,
    input: String,
    output: String,
}

struct WorkerContext<'a> {
    generator: &'a RandomFramedPosetGenerator,
    symmetries: &'a [SignedPermutation],
    state: &'a SharedState,
    next_ticket: &'a AtomicU64,
    generated: &'a AtomicU64,
    strongly_cubular: &'a AtomicU64,
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
    let options = parse_options()?;
    let generator = RandomFramedPosetGenerator::new(DIRECTION_COUNT, options.cell_count);
    let symmetries = Arc::new(three_dimensional_symmetries());
    let state = Arc::new(Mutex::new(GeneratorState::default()));
    let next_ticket = Arc::new(AtomicU64::new(0));
    let generated = Arc::new(AtomicU64::new(0));
    let strongly_cubular = Arc::new(AtomicU64::new(0));
    let accepted = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let worker_error = Arc::new(Mutex::new(None));

    println!(
        "sampling strongly cubular, connected {}-cell 3D OFPs with {} workers \
         (seed {:#018x})",
        options.cell_count, options.worker_count, options.seed
    );
    println!(
        "press any key to stop and write {}",
        options.output.display()
    );
    let raw_mode = RawModeGuard::enable()?;

    let run_result = thread::scope(|scope| {
        for worker in 0..options.worker_count {
            let generator = &generator;
            let symmetries = Arc::clone(&symmetries);
            let state = Arc::clone(&state);
            let next_ticket = Arc::clone(&next_ticket);
            let generated = Arc::clone(&generated);
            let strongly_cubular = Arc::clone(&strongly_cubular);
            let accepted = Arc::clone(&accepted);
            let stop = Arc::clone(&stop);
            let worker_error = Arc::clone(&worker_error);

            scope.spawn(move || {
                let context = WorkerContext {
                    generator,
                    symmetries: &symmetries,
                    state: &state,
                    next_ticket: &next_ticket,
                    generated: &generated,
                    strongly_cubular: &strongly_cubular,
                    accepted: &accepted,
                    stop: &stop,
                };
                if let Err(error) = sample_worker(options.seed, context) {
                    set_worker_error(&worker_error, worker, error);
                    stop.store(true, Ordering::Release);
                }
            });
        }

        let result = monitor(
            &state,
            &generated,
            &strongly_cubular,
            &accepted,
            &worker_error,
        );
        stop.store(true, Ordering::Release);
        result
    });

    drop(raw_mode);
    if let Some(error) = worker_error.lock().unwrap().clone() {
        return Err(io::Error::other(error));
    }
    run_result?;

    let generated = generated.load(Ordering::Relaxed);
    let strongly_cubular = strongly_cubular.load(Ordering::Relaxed);
    let accepted = accepted.load(Ordering::Relaxed);
    let (orbit_count, image_count) = state_counts(&state)?;
    println!(
        "stopped after {generated} candidates, {strongly_cubular} strongly cubular candidates, \
         and {accepted} connected accepted samples; writing {orbit_count} symmetry orbits \
         ({image_count} cached images)"
    );
    write_dataset(&options.output, &state, options.cell_count, accepted)?;
    println!(
        "wrote {orbit_count} symmetry orbits to {}",
        options.output.display()
    );
    Ok(())
}

fn parse_options() -> io::Result<Options> {
    let mut cell_count = DEFAULT_CELL_COUNT;
    let mut worker_count = (thread::available_parallelism()?.get() / 2).max(1);
    let mut seed = OsRng.try_next_u64().map_err(io::Error::other)?;
    let mut output = None;
    let mut arguments = env::args().skip(1);

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--cells" => cell_count = parse_usize_argument(&mut arguments, "--cells")?,
            "--threads" => worker_count = parse_usize_argument(&mut arguments, "--threads")?,
            "--seed" => {
                let value = arguments.next().ok_or_else(|| {
                    invalid_input("--seed requires a decimal or 0x-prefixed hexadecimal value")
                })?;
                seed = parse_seed(&value)?;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    arguments
                        .next()
                        .ok_or_else(|| invalid_input("--output requires a path"))?,
                ));
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --release --example random_normal_forms_3d_hat -- \
                     [--cells N] [--threads N] [--seed N] [--output PATH]"
                );
                std::process::exit(0);
            }
            _ => return Err(invalid_input(format!("unknown argument {argument:?}"))),
        }
    }

    if cell_count < 1usize << DIRECTION_COUNT {
        return Err(invalid_input("--cells must be at least 8"));
    }
    if worker_count == 0 {
        return Err(invalid_input("--threads must be positive"));
    }
    let output = output.unwrap_or_else(|| default_output_path(cell_count));

    Ok(Options {
        cell_count,
        worker_count,
        seed,
        output,
    })
}

fn parse_usize_argument(
    arguments: &mut impl Iterator<Item = String>,
    name: &str,
) -> io::Result<usize> {
    let value = arguments
        .next()
        .ok_or_else(|| invalid_input(format!("{name} requires a value")))?;
    value
        .parse()
        .map_err(|_| invalid_input(format!("invalid value {value:?} for {name}")))
}

fn parse_seed(value: &str) -> io::Result<u64> {
    if let Some(hexadecimal) = value.strip_prefix("0x") {
        u64::from_str_radix(hexadecimal, 16)
    } else {
        value.parse()
    }
    .map_err(|_| invalid_input(format!("invalid seed {value:?}")))
}

fn default_output_path(cell_count: usize) -> PathBuf {
    PathBuf::from(format!(
        "visualizations/random_{cell_count}_cells_normal_forms_hat_\
         strongly_cubular_connected_3d_up_to_symmetry.jsonl"
    ))
}

fn sample_worker(seed: u64, context: WorkerContext<'_>) -> Result<(), String> {
    while !context.stop.load(Ordering::Acquire) {
        let ticket = context.next_ticket.fetch_add(1, Ordering::Relaxed);
        if ticket == u64::MAX {
            return Err("candidate ticket overflow".to_owned());
        }
        let mut rng = SmallRng::seed_from_u64(derived_seed(seed, ticket));
        let shape = Arc::new(context.generator.generate(&mut rng));
        context.generated.fetch_add(1, Ordering::Relaxed);

        if !is_strongly_cubular(&shape) {
            continue;
        }
        context.strongly_cubular.fetch_add(1, Ordering::Relaxed);
        if !shape.is_connected() {
            continue;
        }
        context.accepted.fetch_add(1, Ordering::Relaxed);

        let normal = Arc::new(normalize(&shape));
        debug_assert!(normal.is_normal());
        debug_assert!(normal.is_connected());

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
    let mut distinct = Vec::<(Vec<u8>, Arc<FramedPoset>)>::with_capacity(SYMMETRY_COUNT);

    for symmetry in symmetries {
        let transformed = transform(normal, symmetry)
            .map_err(|error| format!("could not apply symmetry {symmetry:?}: {error}"))?;
        debug_assert!(transformed.is_connected());
        debug_assert!(is_strongly_cubular(&Arc::new(transformed.clone())));
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

fn three_dimensional_symmetries() -> Vec<SignedPermutation> {
    const PERMUTATIONS: [[usize; DIRECTION_COUNT]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    let mut symmetries = Vec::with_capacity(SYMMETRY_COUNT);
    for permutation in PERMUTATIONS {
        for reflections in 0..1usize << DIRECTION_COUNT {
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
                .expect("the three-dimensional symmetry table is valid"),
            );
        }
    }
    debug_assert_eq!(symmetries.len(), SYMMETRY_COUNT);
    symmetries
}

fn monitor(
    state: &SharedState,
    generated: &AtomicU64,
    strongly_cubular: &AtomicU64,
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
            let strongly_cubular = strongly_cubular.load(Ordering::Relaxed);
            let accepted = accepted.load(Ordering::Relaxed);
            let (orbit_count, image_count) = state_counts(state)?;
            print_raw_line(&format!(
                "OFP candidates: {generated}; strongly cubular: {strongly_cubular}; connected: \
                 {accepted}; symmetry orbits: {orbit_count}; cached images: {image_count}"
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

fn write_dataset(
    path: &Path,
    state: &SharedState,
    cell_count: usize,
    accepted: u64,
) -> io::Result<()> {
    let mut rows = snapshot_and_validate(state, cell_count, accepted)?;
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

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let temporary_path = temporary_path(path);
    let result = write_rows(&temporary_path, &rows);
    match result {
        Ok(()) => fs::rename(&temporary_path, path),
        Err(error) => {
            let _ = fs::remove_file(&temporary_path);
            Err(error)
        }
    }
}

fn write_rows(path: &Path, rows: &[OutputRow]) -> io::Result<()> {
    let output_file = File::create(path)?;
    let mut output = BufWriter::with_capacity(BUFFER_CAPACITY, output_file);

    for row in rows {
        let boundary_hashes = std::array::from_fn(|direction| BoundaryHashRecord {
            direction,
            input: format!("{:016x}", row.boundary_hashes[direction][0]),
            output: format!("{:016x}", row.boundary_hashes[direction][1]),
        });
        serde_json::to_writer(
            &mut output,
            &OutputRecord {
                hash: format!("{:016x}", row.hash),
                stabilizer_size: row.record.stabilizer_size,
                multiplicity: row.record.multiplicity,
                boundary_hashes,
                ofp: row.representative.as_ref(),
            },
        )
        .map_err(io::Error::other)?;
        output.write_all(b"\n")?;
    }

    output.flush()?;
    let output_file = output.into_inner().map_err(|error| error.into_error())?;
    output_file.sync_all()
}

fn snapshot_and_validate(
    state: &SharedState,
    cell_count: usize,
    accepted: u64,
) -> io::Result<Vec<OutputRow>> {
    let state = state
        .lock()
        .map_err(|_| io::Error::other("generator state mutex was poisoned"))?;
    if state.orbits.is_empty() {
        return Err(io::Error::other(
            "no connected, strongly cubular symmetry orbits were generated",
        ));
    }

    let mut alias_counts = HashMap::<Arc<FramedPoset>, usize>::new();
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
        validate_representative(representative, record, cell_count, &alias_counts)?;
        total_multiplicity = total_multiplicity
            .checked_add(record.multiplicity)
            .ok_or_else(|| io::Error::other("total multiplicity overflow"))?;
        rows.push(OutputRow {
            hash: structural_hash(representative),
            representative: Arc::clone(representative),
            record,
            boundary_hashes: boundary_hashes(representative),
        });
    }

    if total_multiplicity != accepted {
        return Err(io::Error::other(format!(
            "stored multiplicity {total_multiplicity} does not match accepted sample count \
             {accepted}"
        )));
    }
    Ok(rows)
}

fn validate_representative(
    representative: &Arc<FramedPoset>,
    record: OrbitRecord,
    cell_count: usize,
    alias_counts: &HashMap<Arc<FramedPoset>, usize>,
) -> io::Result<()> {
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
    if !representative.is_normal()
        || !FramedPoset::equal(&normalize(representative), representative)
    {
        return Err(io::Error::other("orbit representative is not normalized"));
    }
    if representative.sizes().iter().sum::<usize>() != cell_count {
        return Err(io::Error::other(format!(
            "orbit representative does not have {cell_count} cells"
        )));
    }
    if representative.active_directions() != [0, 1, 2] {
        return Err(io::Error::other(
            "orbit representative does not use precisely directions 0, 1, and 2",
        ));
    }
    if !representative.is_connected() {
        return Err(io::Error::other("orbit representative is not connected"));
    }
    if !is_strongly_cubular(representative) {
        return Err(io::Error::other(
            "orbit representative is not strongly cubular",
        ));
    }
    Ok(())
}

/// Hash the normalized domains of the six directional hat boundaries.
fn boundary_hashes(shape: &Arc<FramedPoset>) -> [[u64; 2]; DIRECTION_COUNT] {
    std::array::from_fn(|direction| {
        [
            normalized_boundary_hash(shape, Sign::Input, direction),
            normalized_boundary_hash(shape, Sign::Output, direction),
        ]
    })
}

fn normalized_boundary_hash(shape: &Arc<FramedPoset>, sign: Sign, direction: usize) -> u64 {
    let (boundary, _) = boundary_hat(sign, direction, shape);
    structural_hash(&normalize(&boundary))
}

fn structural_hash(shape: &FramedPoset) -> u64 {
    let mut hasher = DefaultHasher::new();
    shape.hash(&mut hasher);
    hasher.finish()
}

fn derived_seed(seed: u64, stream: u64) -> u64 {
    let mut value = seed ^ stream.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
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

fn invalid_input(error: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, error.into())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn enumerates_all_48_three_dimensional_symmetries() {
        let symmetries = three_dimensional_symmetries();
        let distinct: HashSet<_> = symmetries.iter().collect();

        assert_eq!(symmetries.len(), SYMMETRY_COUNT);
        assert_eq!(distinct.len(), SYMMETRY_COUNT);
    }

    #[test]
    fn a_point_has_one_symmetry_image_with_full_stabilizer() {
        let point = Arc::new(normalize(&FramedPoset::point()));
        let analysis = analyze_orbit(&point, &three_dimensional_symmetries()).unwrap();

        assert_eq!(analysis.images.len(), 1);
        assert_eq!(analysis.stabilizer_size, SYMMETRY_COUNT);
        assert!(FramedPoset::equal(&analysis.representative, &point));
    }

    #[test]
    fn boundary_hash_output_names_every_direction_and_sign() {
        let point = Arc::new(FramedPoset::point());
        let hashes = boundary_hashes(&point);
        let boundary_hashes = std::array::from_fn(|direction| BoundaryHashRecord {
            direction,
            input: format!("{:016x}", hashes[direction][0]),
            output: format!("{:016x}", hashes[direction][1]),
        });
        let record = OutputRecord {
            hash: format!("{:016x}", structural_hash(&point)),
            stabilizer_size: SYMMETRY_COUNT,
            multiplicity: 1,
            boundary_hashes,
            ofp: &point,
        };
        let json = serde_json::to_value(record).unwrap();
        let boundaries = json["boundary_hashes"].as_array().unwrap();

        assert_eq!(boundaries.len(), DIRECTION_COUNT);
        for (direction, boundary) in boundaries.iter().enumerate() {
            assert_eq!(boundary["direction"], direction);
            assert_eq!(boundary["input"].as_str().unwrap().len(), 16);
            assert_eq!(boundary["output"].as_str().unwrap().len(), 16);
        }
    }

    #[test]
    fn default_output_path_contains_the_cell_count() {
        assert_eq!(
            default_output_path(27),
            PathBuf::from(
                "visualizations/random_27_cells_normal_forms_hat_strongly_cubular_connected_3d_up_to_symmetry.jsonl"
            )
        );
    }
}
