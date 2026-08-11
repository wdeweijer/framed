use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{self, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ofposets::{
    BoundaryMode, Embedding, FramedPoset, RandomFramedPosetGenerator, Sign, boundary, normalize,
};
use rand::rngs::{OsRng, SmallRng};
use rand::{SeedableRng, TryRngCore};

const CELL_COUNT: usize = 9;
const OUTPUT_DIR: &str = "visualizations/random_9_cells_normal_forms_cubular";
const REPORT_INTERVAL: Duration = Duration::from_secs(2);
const INPUT_POLL_INTERVAL: Duration = Duration::from_millis(100);
const SIGN_PAIRS: [(Sign, Sign); 4] = [
    (Sign::Input, Sign::Input),
    (Sign::Input, Sign::Output),
    (Sign::Output, Sign::Input),
    (Sign::Output, Sign::Output),
];

type Representatives = Arc<Mutex<HashMap<Arc<FramedPoset>, Representative>>>;

#[derive(Debug, Clone, Copy)]
struct Representative {
    id: usize,
    count: u64,
}

struct WorkerContext<'a> {
    generator: &'a RandomFramedPosetGenerator,
    output_dir: &'a Path,
    representatives: &'a Representatives,
    generated: &'a AtomicU64,
    stop: &'a AtomicBool,
    worker_error: &'a Mutex<Option<String>>,
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
    let output_dir = Arc::new(PathBuf::from(OUTPUT_DIR));

    if output_dir.exists() {
        fs::remove_dir_all(output_dir.as_ref())?;
    }
    fs::create_dir_all(output_dir.as_ref())?;

    let representatives: Representatives = Arc::new(Mutex::new(HashMap::new()));
    let generated = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let worker_error = Arc::new(Mutex::new(None));

    println!(
        "sampling cubular {CELL_COUNT}-cell two-directional OFPs with {worker_count} workers (seed {seed:#018x})"
    );
    println!("press any key to stop");
    let raw_mode = RawModeGuard::enable()?;

    let run_result = thread::scope(|scope| {
        for worker in 0..worker_count {
            let generator = &generator;
            let output_dir = Arc::clone(&output_dir);
            let representatives = Arc::clone(&representatives);
            let generated = Arc::clone(&generated);
            let stop = Arc::clone(&stop);
            let worker_error = Arc::clone(&worker_error);

            scope.spawn(move || {
                let context = WorkerContext {
                    generator,
                    output_dir: &output_dir,
                    representatives: &representatives,
                    generated: &generated,
                    stop: &stop,
                    worker_error: &worker_error,
                };
                sample_worker(worker, seed, context);
            });
        }

        let result = monitor(&representatives, &generated, &worker_error);
        stop.store(true, Ordering::Release);
        result
    });

    drop(raw_mode);
    if let Some(error) = worker_error.lock().unwrap().clone() {
        return Err(io::Error::other(error));
    }
    run_result?;
    write_counts(&output_dir, &representatives)?;

    let total = generated.load(Ordering::Relaxed);
    let unique = representatives.lock().unwrap().len();
    println!(
        "stopped after {total} generated OFP candidates; stored {unique} cubular isomorphism-class representatives in {OUTPUT_DIR}"
    );
    Ok(())
}

fn sample_worker(worker: usize, seed: u64, context: WorkerContext<'_>) {
    let mut rng = SmallRng::seed_from_u64(worker_seed(seed, worker));

    while !context.stop.load(Ordering::Acquire) {
        let shape = Arc::new(context.generator.generate(&mut rng));
        context.generated.fetch_add(1, Ordering::Relaxed);
        if !is_cubular(&shape) {
            continue;
        }

        let normal = Arc::new(normalize(&shape));
        debug_assert!(normal.is_normal());

        let new_representative = {
            let mut representatives = context.representatives.lock().unwrap();
            if let Some(representative) = representatives.get_mut(&normal) {
                representative.count = representative.count.saturating_add(1);
                None
            } else {
                let id = representatives.len();
                representatives.insert(Arc::clone(&normal), Representative { id, count: 1 });
                Some((id, normal))
            }
        };

        if let Some((id, normal)) = new_representative
            && let Err(error) = write_representative(context.output_dir, id, &normal)
        {
            let mut first_error = context.worker_error.lock().unwrap();
            if first_error.is_none() {
                *first_error = Some(format!("worker {}: {error}", worker + 1));
            }
            context.stop.store(true, Ordering::Release);
            return;
        }
    }
}

fn is_cubular(shape: &Arc<FramedPoset>) -> bool {
    SIGN_PAIRS.into_iter().all(|(sign_0, sign_1)| {
        let zero_then_one = iterated_boundary(shape, sign_0, 0, sign_1, 1);
        let one_then_zero = iterated_boundary(shape, sign_1, 1, sign_0, 0);
        Embedding::equal(&zero_then_one, &one_then_zero)
    })
}

fn iterated_boundary(
    shape: &Arc<FramedPoset>,
    first_sign: Sign,
    first_direction: usize,
    second_sign: Sign,
    second_direction: usize,
) -> Embedding {
    let (first_boundary, first_embedding) =
        boundary(BoundaryMode::Plain, first_sign, first_direction, shape);
    let (_, second_embedding) = boundary(
        BoundaryMode::Plain,
        second_sign,
        second_direction,
        &first_boundary,
    );
    Embedding::compose(&second_embedding, &first_embedding)
}

fn monitor(
    representatives: &Representatives,
    generated: &AtomicU64,
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
            let total = generated.load(Ordering::Relaxed);
            let unique = representatives.lock().unwrap().len();
            print_raw_line(&format!(
                "OFP candidates generated: {total}; cubular representatives: {unique}"
            ))?;
            next_report = Instant::now() + REPORT_INTERVAL;
        }
    }
}

fn write_representative(output_dir: &Path, id: usize, shape: &FramedPoset) -> io::Result<()> {
    let stem = format!("sample_{id:06}");
    let json = serde_json::to_string_pretty(shape).map_err(io::Error::other)?;

    fs::write(
        output_dir.join(format!("{stem}.ofp.json")),
        format!("{json}\n"),
    )
}

fn write_counts(output_dir: &Path, representatives: &Representatives) -> io::Result<()> {
    let mut rows: Vec<Representative> = representatives.lock().unwrap().values().copied().collect();
    rows.sort_unstable_by_key(|representative| representative.id);

    let mut output = String::from("representative\tcount\n");
    for representative in rows {
        writeln!(
            output,
            "sample_{:06}\t{}",
            representative.id, representative.count
        )
        .unwrap();
    }
    fs::write(output_dir.join("multiplicities.tsv"), output)
}

fn print_raw_line(message: &str) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    write!(stdout, "{message}\r\n")?;
    stdout.flush()
}

fn worker_seed(seed: u64, worker: usize) -> u64 {
    seed ^ (worker as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}
