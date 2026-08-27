use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use ofposets::pushout::pushout;
use ofposets::{CubularityMode, boundary};
use ofposets::{
    DirectionImage, Embedding, FramedPoset, Sign, SignedPermutation, is_cubular, isomorphisms,
    iterated_boundary, normalize, transform,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

const SYMMETRY_COUNT: usize = 8;
const MAX_SAMPLED_ISOMORPHISMS: usize = 10;
const SAMPLE_SEED: u64 = 0x6a09_e667_f3bc_c909;
const REPORT_INTERVAL: Duration = Duration::from_secs(5);
const WORK_BATCH: u64 = 256;
const PROGRESS_BATCH: u64 = 4_096;
const LOAD_REPORT_EVERY: usize = 100_000;
const INDEX_REPORT_EVERY: usize = 100_000;
const LOAD_BATCH_SIZE: usize = 4_096;
const INDEX_BATCH_SIZE: usize = 16_384;
const SIGN_PAIRS: [(Sign, Sign); 4] = [
    (Sign::Input, Sign::Input),
    (Sign::Input, Sign::Output),
    (Sign::Output, Sign::Input),
    (Sign::Output, Sign::Output),
];
const DATASETS: [DatasetSpec; 4] = [
    DatasetSpec::exact(
        4,
        "visualizations/random_4_cells_normal_forms_hat_cubular_up_to_symmetry.jsonl",
    ),
    DatasetSpec::exact(
        5,
        "visualizations/random_5_cells_normal_forms_hat_cubular_up_to_symmetry.jsonl",
    ),
    DatasetSpec::exact(
        6,
        "visualizations/random_6_cells_normal_forms_hat_cubular_up_to_symmetry.jsonl",
    ),
    DatasetSpec {
        cells: 9,
        path: "visualizations/random_9_cells_normal_forms_hat_cubular_up_to_symmetry.jsonl",
        representative_limit: Some(2_048),
        pair_limit: Some(20_000),
    },
];

const STRONG_DATASETS: [DatasetSpec; 4] = [
    DatasetSpec::exact(
        4,
        "visualizations/random_4_cells_normal_forms_hat_strongly_cubular_up_to_symmetry.jsonl",
    ),
    DatasetSpec::exact(
        5,
        "visualizations/random_5_cells_normal_forms_hat_strongly_cubular_up_to_symmetry.jsonl",
    ),
    DatasetSpec::exact(
        6,
        "visualizations/random_6_cells_normal_forms_hat_strongly_cubular_up_to_symmetry.jsonl",
    ),
    DatasetSpec {
        cells: 9,
        path: "visualizations/random_9_cells_normal_forms_hat_strongly_cubular_up_to_symmetry.jsonl",
        representative_limit: Some(10_048),
        pair_limit: Some(200_000),
    },
];

#[derive(Clone, Copy)]
struct DatasetSpec {
    cells: usize,
    path: &'static str,
    representative_limit: Option<usize>,
    pair_limit: Option<usize>,
}

struct Options {
    require_strong: bool,
    cells: Option<usize>,
    worker_count: usize,
}

impl DatasetSpec {
    const fn exact(cells: usize, path: &'static str) -> Self {
        Self {
            cells,
            path,
            representative_limit: None,
            pair_limit: None,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetRecord {
    hash: String,
    stabilizer_size: usize,
    multiplicity: u64,
    ofp: FramedPoset,
}

#[derive(Debug, Clone, Copy)]
struct ShapeSource {
    line: usize,
    symmetry: usize,
}

struct Shape {
    poset: Arc<FramedPoset>,
    source: ShapeSource,
}

struct PendingRecord {
    line_number: usize,
    record: DatasetRecord,
}

struct ProcessedRecord {
    line_number: usize,
    orbit: Vec<(usize, Arc<FramedPoset>)>,
}

struct BoundaryOccurrence {
    shape_id: usize,
    into_shape: Embedding,
    to_canonical: Embedding,
}

struct PreparedBoundaryOccurrence {
    shape_id: usize,
    direction: usize,
    sign: Sign,
    boundary: Arc<FramedPoset>,
    canonical: Arc<FramedPoset>,
    into_shape: Embedding,
}

struct BoundaryClass {
    direction: usize,
    canonical: Arc<FramedPoset>,
    outputs: Vec<BoundaryOccurrence>,
    inputs: Vec<BoundaryOccurrence>,
    automorphisms: Vec<Embedding>,
}

struct BoundaryIndex {
    classes: Vec<BoundaryClass>,
    cumulative_pairs: Vec<u128>,
    total_pairs: u128,
}

#[derive(Debug, Default, Clone, Copy)]
struct FormulaStatistics {
    pairs: u128,
    gluings: u128,
    any_failures: u128,
    axial_input_failures: u128,
    axial_output_failures: u128,
    transverse_input_failures: u128,
    transverse_output_failures: u128,
}

impl FormulaStatistics {
    fn merge(&mut self, other: Self) {
        self.pairs += other.pairs;
        self.gluings += other.gluings;
        self.any_failures += other.any_failures;
        self.axial_input_failures += other.axial_input_failures;
        self.axial_output_failures += other.axial_output_failures;
        self.transverse_input_failures += other.transverse_input_failures;
        self.transverse_output_failures += other.transverse_output_failures;
    }
}

#[derive(Debug, Clone, Copy)]
struct FormulaResults {
    axial_input: bool,
    axial_output: bool,
    transverse_input: bool,
    transverse_output: bool,
}

impl FormulaResults {
    fn all_hold(self) -> bool {
        self.axial_input && self.axial_output && self.transverse_input && self.transverse_output
    }

    fn failing_names(self) -> Vec<&'static str> {
        [
            (!self.axial_input).then_some("axial input"),
            (!self.axial_output).then_some("axial output"),
            (!self.transverse_input).then_some("transverse input"),
            (!self.transverse_output).then_some("transverse output"),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

#[derive(Debug)]
struct FailureWitness {
    pair: u128,
    direction: usize,
    first: ShapeSource,
    second: ShapeSource,
    isomorphism: usize,
    results: FormulaResults,
}

fn main() -> io::Result<()> {
    let options = parse_options()?;
    let datasets = if options.require_strong {
        &STRONG_DATASETS
    } else {
        &DATASETS
    };
    let symmetries = two_dimensional_symmetries();

    for &spec in datasets
        .iter()
        .filter(|spec| options.cells.is_none_or(|cells| spec.cells == cells))
    {
        check_dataset(
            spec,
            &symmetries,
            options.require_strong,
            options.worker_count,
        )?;
    }
    Ok(())
}

fn parse_options() -> io::Result<Options> {
    let mut require_strong = false;
    let mut cells = None;
    let mut worker_count = thread::available_parallelism()?.get();
    let mut arguments = env::args().skip(1);

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--strong" => require_strong = true,
            "--cells" => {
                let value = arguments.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--cells requires a value")
                })?;
                let value = value.parse::<usize>().map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("invalid cell count {value:?}"),
                    )
                })?;
                if !DATASETS.iter().any(|spec| spec.cells == value) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("no configured dataset has {value} cells"),
                    ));
                }
                cells = Some(value);
            }
            "--threads" => {
                let value = arguments.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--threads requires a value")
                })?;
                worker_count = value.parse::<usize>().map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("invalid thread count {value:?}"),
                    )
                })?;
                if worker_count == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "thread count must be positive",
                    ));
                }
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "unknown argument {argument:?}; expected --strong, --cells N, or --threads N"
                    ),
                ));
            }
        }
    }
    Ok(Options {
        require_strong,
        cells,
        worker_count,
    })
}

fn check_dataset(
    spec: DatasetSpec,
    symmetries: &[SignedPermutation],
    require_strong: bool,
    worker_count: usize,
) -> io::Result<()> {
    let path = Path::new(spec.path);
    let selected_lines = selected_lines(path, spec.representative_limit)?;
    let selected_count = selected_lines.as_ref().map(HashSet::len);
    let shapes = load_and_expand_dataset(
        path,
        selected_lines.as_ref(),
        symmetries,
        require_strong,
        worker_count,
    )?;
    let mut boundary_index = build_boundary_index(&shapes, worker_count)?;
    let sampled = spec.pair_limit.is_some();

    for class in &mut boundary_index.classes {
        class.automorphisms = isomorphisms(&class.canonical, &class.canonical);
        if sampled {
            class.automorphisms.truncate(MAX_SAMPLED_ISOMORPHISMS);
        }
        if class.automorphisms.is_empty() {
            return Err(invalid_data("canonical boundary has no automorphisms"));
        }
    }

    let effective_workers = match spec.pair_limit {
        Some(_) => 1,
        None => effective_worker_count(worker_count, boundary_index.total_pairs),
    };
    println!(
        "{}-cell dataset prepared: {} concrete symmetry images, {} compatible ordered pairs, {effective_workers} worker threads",
        spec.cells,
        shapes.len(),
        boundary_index.total_pairs
    );
    let (statistics, witness) = match spec.pair_limit {
        Some(limit) => sample_statistics(&shapes, &boundary_index, limit),
        None => exhaustive_statistics(&shapes, &boundary_index, effective_workers)?,
    };

    println!("{}-cell dataset results:", spec.cells);
    if let Some(selected_count) = selected_count {
        println!("  sampled orbit representatives: {selected_count}");
    } else {
        println!("  checked every orbit representative");
    }
    println!("  concrete symmetry images: {}", shapes.len());
    println!(
        "  generalized/basic cubularity agreements: {}",
        shapes.len()
    );
    println!("  worker threads: {effective_workers}");
    println!(
        "  compatible ordered pairs available: {}",
        boundary_index.total_pairs
    );
    print_statistics(statistics);
    if let Some(witness) = witness {
        println!(
            "  first failure: pair {}; direction {}; first line {} symmetry {}; second line {} symmetry {}; isomorphism {}; laws: {}",
            witness.pair + 1,
            witness.direction,
            witness.first.line,
            witness.first.symmetry,
            witness.second.line,
            witness.second.symmetry,
            witness.isomorphism + 1,
            witness.results.failing_names().join(", ")
        );
    }
    Ok(())
}

fn exhaustive_statistics(
    shapes: &[Shape],
    boundary_index: &BoundaryIndex,
    requested_workers: usize,
) -> io::Result<(FormulaStatistics, Option<FailureWitness>)> {
    let worker_count = effective_worker_count(requested_workers, boundary_index.total_pairs);
    let total_pairs = u64::try_from(boundary_index.total_pairs)
        .map_err(|_| invalid_data("progress counter exceeds u64"))?;
    let next_pair = AtomicU64::new(0);
    let completed = AtomicU64::new(0);
    let finished = AtomicBool::new(false);

    let partials = thread::scope(|scope| {
        let reporter = scope.spawn(|| {
            while !finished.load(Ordering::Acquire) {
                thread::park_timeout(REPORT_INTERVAL);
                if finished.load(Ordering::Acquire) {
                    break;
                }
                let completed = completed.load(Ordering::Relaxed);
                println!(
                    "  progress: {completed}/{total_pairs} compatible pairs ({:.2}%)",
                    percentage(u128::from(completed), u128::from(total_pairs))
                );
            }
        });
        let reporter_thread = reporter.thread().clone();
        let workers: Vec<_> = (0..worker_count)
            .map(|_| {
                scope.spawn({
                    let next_pair = &next_pair;
                    let completed = &completed;
                    move || {
                        exhaustive_worker(shapes, boundary_index, total_pairs, next_pair, completed)
                    }
                })
            })
            .collect();
        let partials: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().expect("formula worker panicked"))
            .collect();
        finished.store(true, Ordering::Release);
        reporter_thread.unpark();
        reporter.join().expect("progress reporter panicked");
        partials
    });

    let mut statistics = FormulaStatistics::default();
    let mut witness = None;
    for (partial_statistics, partial_witness) in partials {
        statistics.merge(partial_statistics);
        if partial_witness
            .as_ref()
            .is_some_and(|candidate| witness_is_before(candidate, witness.as_ref()))
        {
            witness = partial_witness;
        }
    }
    Ok((statistics, witness))
}

fn exhaustive_worker(
    shapes: &[Shape],
    boundary_index: &BoundaryIndex,
    total_pairs: u64,
    next_pair: &AtomicU64,
    completed: &AtomicU64,
) -> (FormulaStatistics, Option<FailureWitness>) {
    let mut statistics = FormulaStatistics::default();
    let mut witness = None;
    let mut pending_progress = 0u64;

    loop {
        let start = next_pair.fetch_add(WORK_BATCH, Ordering::Relaxed);
        if start >= total_pairs {
            break;
        }
        let end = start.saturating_add(WORK_BATCH).min(total_pairs);

        for pair in start..end {
            let pair = u128::from(pair);
            let (class, output, input) = pair_at(boundary_index, pair);
            statistics.pairs += 1;
            check_pair(
                shapes,
                class,
                output,
                input,
                pair,
                &mut statistics,
                &mut witness,
            );
            pending_progress += 1;
            if pending_progress == PROGRESS_BATCH {
                completed.fetch_add(pending_progress, Ordering::Relaxed);
                pending_progress = 0;
            }
        }
    }
    completed.fetch_add(pending_progress, Ordering::Relaxed);
    (statistics, witness)
}

fn effective_worker_count(requested: usize, total_pairs: u128) -> usize {
    usize::try_from(total_pairs)
        .map_or(requested, |pairs| requested.min(pairs))
        .max(1)
}

fn pair_at(
    boundary_index: &BoundaryIndex,
    pair: u128,
) -> (&BoundaryClass, &BoundaryOccurrence, &BoundaryOccurrence) {
    let class_index = boundary_index
        .cumulative_pairs
        .partition_point(|&end| end <= pair);
    let class_start = class_index
        .checked_sub(1)
        .map_or(0, |previous| boundary_index.cumulative_pairs[previous]);
    let local = pair - class_start;
    let class = &boundary_index.classes[class_index];
    let input_count = class.inputs.len() as u128;
    let output_index = usize::try_from(local / input_count)
        .expect("output index comes from a usize-sized collection");
    let input_index = usize::try_from(local % input_count)
        .expect("input index comes from a usize-sized collection");
    (
        class,
        &class.outputs[output_index],
        &class.inputs[input_index],
    )
}

fn witness_is_before(candidate: &FailureWitness, current: Option<&FailureWitness>) -> bool {
    current.is_none_or(|current| {
        (candidate.pair, candidate.isomorphism) < (current.pair, current.isomorphism)
    })
}

fn sample_statistics(
    shapes: &[Shape],
    boundary_index: &BoundaryIndex,
    limit: usize,
) -> (FormulaStatistics, Option<FailureWitness>) {
    let target = u128::min(boundary_index.total_pairs, limit as u128);
    let mut statistics = FormulaStatistics::default();
    let mut witness = None;
    let mut seen = HashSet::with_capacity(usize::try_from(target).unwrap_or(limit));
    let mut rng = SmallRng::seed_from_u64(SAMPLE_SEED);

    while seen.len() as u128 != target {
        let ticket = rng.random_range(0..boundary_index.total_pairs);
        let class_index = boundary_index
            .cumulative_pairs
            .partition_point(|&end| end <= ticket);
        let class_start = class_index
            .checked_sub(1)
            .map_or(0, |previous| boundary_index.cumulative_pairs[previous]);
        let local = ticket - class_start;
        let class = &boundary_index.classes[class_index];
        let input_count = class.inputs.len() as u128;
        let output_index = usize::try_from(local / input_count)
            .expect("output index comes from a usize-sized collection");
        let input_index = usize::try_from(local % input_count)
            .expect("input index comes from a usize-sized collection");

        if !seen.insert((class_index, output_index, input_index)) {
            continue;
        }
        statistics.pairs += 1;
        check_pair(
            shapes,
            class,
            &class.outputs[output_index],
            &class.inputs[input_index],
            ticket,
            &mut statistics,
            &mut witness,
        );
    }
    (statistics, witness)
}

fn check_pair(
    shapes: &[Shape],
    class: &BoundaryClass,
    output: &BoundaryOccurrence,
    input: &BoundaryOccurrence,
    pair: u128,
    statistics: &mut FormulaStatistics,
    witness: &mut Option<FailureWitness>,
) {
    let first = &shapes[output.shape_id];
    let second = &shapes[input.shape_id];
    let from_canonical = input.to_canonical.inverse_isomorphism();

    for (isomorphism, automorphism) in class.automorphisms.iter().enumerate() {
        let through_automorphism = Embedding::compose(&output.to_canonical, automorphism);
        let boundary_isomorphism = Embedding::compose(&through_automorphism, &from_canonical);
        let into_second = Embedding::compose(&boundary_isomorphism, &input.into_shape);
        let pasted = pushout(&output.into_shape, &into_second);
        let results = check_formulas(first, second, class.direction, &pasted);

        statistics.gluings += 1;
        statistics.axial_input_failures += u128::from(!results.axial_input);
        statistics.axial_output_failures += u128::from(!results.axial_output);
        statistics.transverse_input_failures += u128::from(!results.transverse_input);
        statistics.transverse_output_failures += u128::from(!results.transverse_output);
        if !results.all_hold() {
            statistics.any_failures += 1;
            let candidate = FailureWitness {
                pair,
                direction: class.direction,
                first: first.source,
                second: second.source,
                isomorphism,
                results,
            };
            if witness_is_before(&candidate, witness.as_ref()) {
                *witness = Some(candidate);
            }
        }
    }
}

fn check_formulas(
    first: &Shape,
    second: &Shape,
    direction: usize,
    pasted: &ofposets::pushout::Pushout,
) -> FormulaResults {
    let axial_input = compare_boundary_with_side(
        Sign::Input,
        direction,
        &pasted.tip,
        &first.poset,
        &pasted.inl,
    );
    let axial_output = compare_boundary_with_side(
        Sign::Output,
        direction,
        &pasted.tip,
        &second.poset,
        &pasted.inr,
    );
    let transverse_direction = 1 - direction;
    let transverse_input = compare_boundary_with_union(
        Sign::Input,
        transverse_direction,
        &pasted.tip,
        &first.poset,
        &pasted.inl,
        &second.poset,
        &pasted.inr,
    );
    let transverse_output = compare_boundary_with_union(
        Sign::Output,
        transverse_direction,
        &pasted.tip,
        &first.poset,
        &pasted.inl,
        &second.poset,
        &pasted.inr,
    );

    FormulaResults {
        axial_input,
        axial_output,
        transverse_input,
        transverse_output,
    }
}

fn compare_boundary_with_side(
    sign: Sign,
    direction: usize,
    pasted: &Arc<FramedPoset>,
    side: &Arc<FramedPoset>,
    side_into_pasted: &Embedding,
) -> bool {
    let (_, actual) = boundary(sign, direction, pasted);
    let (_, into_side) = boundary(sign, direction, side);
    let expected = Embedding::compose(&into_side, side_into_pasted);
    let actual_closed = actual.is_closed();
    let expected_closed = expected.is_closed();
    debug_assert!(actual_closed);
    debug_assert!(expected_closed);
    actual_closed && expected_closed && Embedding::equal(&actual, &expected)
}

fn compare_boundary_with_union(
    sign: Sign,
    direction: usize,
    pasted: &Arc<FramedPoset>,
    first: &Arc<FramedPoset>,
    first_into_pasted: &Embedding,
    second: &Arc<FramedPoset>,
    second_into_pasted: &Embedding,
) -> bool {
    let (_, actual) = boundary(sign, direction, pasted);
    let (_, first_boundary) = boundary(sign, direction, first);
    let (_, second_boundary) = boundary(sign, direction, second);
    let first_boundary = Embedding::compose(&first_boundary, first_into_pasted);
    let second_boundary = Embedding::compose(&second_boundary, second_into_pasted);

    let actual_closed = actual.is_closed();
    let first_closed = first_boundary.is_closed();
    let second_closed = second_boundary.is_closed();
    debug_assert!(actual_closed);
    debug_assert!(first_closed);
    debug_assert!(second_closed);
    if !actual_closed || !first_closed || !second_closed {
        return false;
    }
    let expected = Embedding::union(&first_boundary, &second_boundary).into_codomain;
    Embedding::equal(&actual, &expected)
}

fn print_statistics(statistics: FormulaStatistics) {
    println!("  compatible ordered pairs checked: {}", statistics.pairs);
    println!("  individual gluings checked: {}", statistics.gluings);
    print_failure_count("any formula", statistics.any_failures, statistics.gluings);
    print_failure_count(
        "axial input",
        statistics.axial_input_failures,
        statistics.gluings,
    );
    print_failure_count(
        "axial output",
        statistics.axial_output_failures,
        statistics.gluings,
    );
    print_failure_count(
        "transverse input",
        statistics.transverse_input_failures,
        statistics.gluings,
    );
    print_failure_count(
        "transverse output",
        statistics.transverse_output_failures,
        statistics.gluings,
    );
}

fn print_failure_count(label: &str, failures: u128, total: u128) {
    println!(
        "  {label} failures: {failures} ({:.4}%)",
        percentage(failures, total)
    );
}

fn percentage(part: u128, whole: u128) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

fn build_boundary_index(shapes: &[Shape], requested_workers: usize) -> io::Result<BoundaryIndex> {
    let mut classes = Vec::<BoundaryClass>::new();
    let mut class_indices = HashMap::<(usize, Arc<FramedPoset>), usize>::new();
    let mut transports = HashMap::<Arc<FramedPoset>, Embedding>::new();
    let mut next_report = INDEX_REPORT_EVERY;

    for batch_start in (0..shapes.len()).step_by(INDEX_BATCH_SIZE) {
        let batch_end = batch_start
            .saturating_add(INDEX_BATCH_SIZE)
            .min(shapes.len());
        let prepared = prepare_boundary_batch(shapes, batch_start, batch_end, requested_workers);

        for prepared in prepared {
            let class = *class_indices
                .entry((prepared.direction, Arc::clone(&prepared.canonical)))
                .or_insert_with(|| {
                    let class = classes.len();
                    classes.push(BoundaryClass {
                        direction: prepared.direction,
                        canonical: Arc::clone(&prepared.canonical),
                        outputs: Vec::new(),
                        inputs: Vec::new(),
                        automorphisms: Vec::new(),
                    });
                    class
                });
            let to_canonical = transport_to_canonical(
                &prepared.boundary,
                &classes[class].canonical,
                &mut transports,
            )?;
            let occurrence = BoundaryOccurrence {
                shape_id: prepared.shape_id,
                into_shape: prepared.into_shape,
                to_canonical,
            };
            match prepared.sign {
                Sign::Input => classes[class].inputs.push(occurrence),
                Sign::Output => classes[class].outputs.push(occurrence),
            }
        }
        if batch_end >= next_report || batch_end == shapes.len() {
            println!(
                "  indexed boundaries of {} concrete symmetry images",
                batch_end
            );
            while next_report <= batch_end {
                next_report = next_report.saturating_add(INDEX_REPORT_EVERY);
            }
        }
    }

    classes.retain(|class| !class.outputs.is_empty() && !class.inputs.is_empty());
    let mut cumulative_pairs = Vec::with_capacity(classes.len());
    let mut total_pairs = 0u128;
    for class in &classes {
        let pairs = (class.outputs.len() as u128)
            .checked_mul(class.inputs.len() as u128)
            .ok_or_else(|| invalid_data("compatible pair count overflow"))?;
        total_pairs = total_pairs
            .checked_add(pairs)
            .ok_or_else(|| invalid_data("total compatible pair count overflow"))?;
        cumulative_pairs.push(total_pairs);
    }
    if total_pairs == 0 {
        return Err(invalid_data("dataset has no compatible boundary pairs"));
    }
    Ok(BoundaryIndex {
        classes,
        cumulative_pairs,
        total_pairs,
    })
}

fn prepare_boundary_batch(
    shapes: &[Shape],
    start: usize,
    end: usize,
    requested_workers: usize,
) -> Vec<PreparedBoundaryOccurrence> {
    let batch = &shapes[start..end];
    let worker_count = requested_workers.min(batch.len()).max(1);
    let chunk_size = batch.len().div_ceil(worker_count);

    thread::scope(|scope| {
        let workers: Vec<_> = batch
            .chunks(chunk_size)
            .enumerate()
            .map(|(chunk, shapes)| {
                let chunk_start = start + chunk * chunk_size;
                scope.spawn(move || {
                    let mut prepared = Vec::with_capacity(shapes.len() * 4);
                    for (offset, shape) in shapes.iter().enumerate() {
                        for direction in 0..2 {
                            for sign in [Sign::Input, Sign::Output] {
                                let (boundary, into_shape) =
                                    boundary(sign, direction, &shape.poset);
                                let canonical = Arc::new(normalize(&boundary));
                                prepared.push(PreparedBoundaryOccurrence {
                                    shape_id: chunk_start + offset,
                                    direction,
                                    sign,
                                    boundary,
                                    canonical,
                                    into_shape,
                                });
                            }
                        }
                    }
                    prepared
                })
            })
            .collect();

        workers
            .into_iter()
            .flat_map(|worker| worker.join().expect("boundary-index worker panicked"))
            .collect()
    })
}

fn transport_to_canonical(
    boundary: &Arc<FramedPoset>,
    canonical: &Arc<FramedPoset>,
    cache: &mut HashMap<Arc<FramedPoset>, Embedding>,
) -> io::Result<Embedding> {
    if let Some(transport) = cache.get(boundary) {
        return Ok(transport.clone());
    }
    let transport = isomorphisms(boundary, canonical)
        .into_iter()
        .next()
        .ok_or_else(|| invalid_data("boundary is not isomorphic to its normal form"))?;
    cache.insert(Arc::clone(boundary), transport.clone());
    Ok(transport)
}

fn selected_lines(path: &Path, limit: Option<usize>) -> io::Result<Option<HashSet<usize>>> {
    let Some(limit) = limit else {
        return Ok(None);
    };
    let total = count_lines(path)?;
    let selected = limit.min(total);
    let mut lines = HashSet::with_capacity(selected);

    if selected == 1 {
        lines.insert(1);
    } else if selected > 1 {
        for index in 0..selected {
            lines.insert(1 + index * (total - 1) / (selected - 1));
        }
    }
    Ok(Some(lines))
}

fn count_lines(path: &Path) -> io::Result<usize> {
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);
    let mut line = Vec::new();
    let mut count = 0usize;
    while reader.read_until(b'\n', &mut line)? != 0 {
        count += 1;
        line.clear();
    }
    Ok(count)
}

fn load_and_expand_dataset(
    path: &Path,
    selected_lines: Option<&HashSet<usize>>,
    symmetries: &[SignedPermutation],
    require_strong: bool,
    requested_workers: usize,
) -> io::Result<Vec<Shape>> {
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);
    let mut shapes = Vec::new();
    let mut seen_images = HashSet::<Arc<FramedPoset>>::new();
    let mut pending = Vec::with_capacity(LOAD_BATCH_SIZE);
    let mut previous_hash = None;
    let mut next_report = LOAD_REPORT_EVERY;
    let mut line = String::new();
    let mut line_number = 0usize;

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        line_number += 1;
        if selected_lines.is_some_and(|selected| !selected.contains(&line_number)) {
            continue;
        }
        let record: DatasetRecord = serde_json::from_str(line.trim_end())
            .map_err(|error| invalid_line(path, line_number, error))?;
        let hash = parse_hash(path, line_number, &record.hash)?;
        if previous_hash.is_some_and(|previous| previous >= hash) {
            return Err(invalid_line(
                path,
                line_number,
                "hashes are not strictly increasing",
            ));
        }
        previous_hash = Some(hash);
        pending.push(PendingRecord {
            line_number,
            record,
        });

        if pending.len() == LOAD_BATCH_SIZE {
            let processed = process_record_batch(
                path,
                &pending,
                symmetries,
                require_strong,
                requested_workers,
            )?;
            merge_processed_records(path, processed, &mut seen_images, &mut shapes)?;
            pending.clear();
            if line_number >= next_report {
                println!(
                    "  loaded {line_number} orbit representatives and {} concrete symmetry images",
                    shapes.len()
                );
                while next_report <= line_number {
                    next_report = next_report.saturating_add(LOAD_REPORT_EVERY);
                }
            }
        }
    }
    if !pending.is_empty() {
        let processed = process_record_batch(
            path,
            &pending,
            symmetries,
            require_strong,
            requested_workers,
        )?;
        merge_processed_records(path, processed, &mut seen_images, &mut shapes)?;
    }
    if shapes.is_empty() {
        return Err(invalid_data("dataset selection is empty"));
    }
    Ok(shapes)
}

fn process_record_batch(
    path: &Path,
    pending: &[PendingRecord],
    symmetries: &[SignedPermutation],
    require_strong: bool,
    requested_workers: usize,
) -> io::Result<Vec<ProcessedRecord>> {
    let worker_count = requested_workers.min(pending.len()).max(1);
    let chunk_size = pending.len().div_ceil(worker_count);

    thread::scope(|scope| {
        let workers: Vec<_> = pending
            .chunks(chunk_size)
            .map(|records| {
                scope.spawn(move || {
                    records
                        .iter()
                        .map(|pending| {
                            validate_record(
                                path,
                                pending.line_number,
                                &pending.record,
                                require_strong,
                            )?;
                            let orbit = symmetry_images(
                                &pending.record.ofp,
                                symmetries,
                                require_strong,
                            )
                            .map_err(|error| {
                                invalid_line(path, pending.line_number, error)
                            })?;
                            let expected_stabilizer = SYMMETRY_COUNT / orbit.len();
                            if pending.record.stabilizer_size != expected_stabilizer {
                                return Err(invalid_line(
                                    path,
                                    pending.line_number,
                                    format!(
                                        "stored stabilizer {} does not match recomputed stabilizer {expected_stabilizer}",
                                        pending.record.stabilizer_size
                                    ),
                                ));
                            }
                            Ok(ProcessedRecord {
                                line_number: pending.line_number,
                                orbit,
                            })
                        })
                        .collect::<io::Result<Vec<_>>>()
                })
            })
            .collect();

        let mut processed = Vec::with_capacity(pending.len());
        for worker in workers {
            processed.extend(worker.join().expect("dataset-validation worker panicked")?);
        }
        Ok(processed)
    })
}

fn merge_processed_records(
    path: &Path,
    processed: Vec<ProcessedRecord>,
    seen_images: &mut HashSet<Arc<FramedPoset>>,
    shapes: &mut Vec<Shape>,
) -> io::Result<()> {
    for record in processed {
        for (symmetry, poset) in record.orbit {
            if !seen_images.insert(Arc::clone(&poset)) {
                return Err(invalid_line(
                    path,
                    record.line_number,
                    "selected symmetry orbits overlap",
                ));
            }
            shapes.push(Shape {
                poset,
                source: ShapeSource {
                    line: record.line_number,
                    symmetry,
                },
            });
        }
    }
    Ok(())
}

fn validate_record(
    path: &Path,
    line: usize,
    record: &DatasetRecord,
    require_strong: bool,
) -> io::Result<()> {
    if record.multiplicity == 0 {
        return Err(invalid_line(path, line, "multiplicity is zero"));
    }
    let stored_hash = parse_hash(path, line, &record.hash)?;
    let normal = normalize(&record.ofp);
    if !FramedPoset::equal(&normal, &record.ofp) {
        return Err(invalid_line(path, line, "OFP is not normalized"));
    }
    if structural_hash(&normal) != stored_hash {
        return Err(invalid_line(path, line, "stored hash is incorrect"));
    }
    if normal.sizes().iter().sum::<usize>() == 0 || normal.dim() != 2 {
        return Err(invalid_line(path, line, "OFP is not two-dimensional"));
    }
    let normal = Arc::new(normal);
    let basic = is_basic_two_dimensional_cubular(&normal);
    let generalized = is_cubular(CubularityMode::Regular, &normal);
    if generalized != basic {
        return Err(invalid_line(
            path,
            line,
            format!(
                "generalized cubularity ({generalized}) disagrees with basic two-dimensional cubularity ({basic})"
            ),
        ));
    }
    if !generalized {
        return Err(invalid_line(path, line, "OFP is not cubular"));
    }
    if require_strong && !is_cubular(CubularityMode::Strong, &normal) {
        return Err(invalid_line(path, line, "OFP is not strongly cubular"));
    }
    Ok(())
}

fn symmetry_images(
    shape: &FramedPoset,
    symmetries: &[SignedPermutation],
    require_strong: bool,
) -> Result<Vec<(usize, Arc<FramedPoset>)>, String> {
    let shape = Arc::new(shape.clone());
    let mut images: Vec<(usize, Arc<FramedPoset>)> = Vec::with_capacity(SYMMETRY_COUNT);

    for (symmetry_index, symmetry) in symmetries.iter().enumerate() {
        let transformed = transform(&shape, symmetry).map_err(|error| error.to_string())?;
        let image = Arc::new(normalize(&transformed));
        let basic = is_basic_two_dimensional_cubular(&image);
        let generalized = is_cubular(CubularityMode::Regular, &image);
        if generalized != basic {
            return Err(format!(
                "symmetry {symmetry_index} has generalized cubularity {generalized} but basic cubularity {basic}"
            ));
        }
        if !generalized {
            return Err(format!("symmetry {symmetry_index} is not cubular"));
        }
        if require_strong && !is_cubular(CubularityMode::Strong, &image) {
            return Err(format!("symmetry {symmetry_index} is not strongly cubular"));
        }
        if images
            .iter()
            .any(|(_, existing)| FramedPoset::equal(existing, &image))
        {
            continue;
        }
        images.push((symmetry_index, image));
    }
    if images.is_empty() || !SYMMETRY_COUNT.is_multiple_of(images.len()) {
        return Err(format!("orbit has {} symmetry images", images.len()));
    }
    Ok(images)
}

fn is_basic_two_dimensional_cubular(shape: &Arc<FramedPoset>) -> bool {
    SIGN_PAIRS.into_iter().all(|(sign_0, sign_1)| {
        let (_, zero_then_one) = iterated_boundary(&[(sign_0, 0), (sign_1, 1)], shape);
        let (_, one_then_zero) = iterated_boundary(&[(sign_1, 1), (sign_0, 0)], shape);
        Embedding::equal(&zero_then_one, &one_then_zero)
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
    symmetries
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

fn invalid_line(path: &Path, line: usize, error: impl std::fmt::Display) -> io::Error {
    invalid_data(format!("{}:{line}: {error}", path.display()))
}

fn invalid_data(error: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.into())
}
