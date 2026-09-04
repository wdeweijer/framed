use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ofposets::pushout::{Pushout, pushout};
use ofposets::{CubularityMode, boundary};
use ofposets::{
    DirectionImage, Embedding, FramedPoset, Renderer, Sign, SignedPermutation, embedding_to_dot,
    is_cubular, isomorphisms, normalize, to_dot, transform,
};
use rand::rngs::{OsRng, SmallRng};
use rand::{Rng, SeedableRng, TryRngCore};
use serde::{Deserialize, Serialize};

const INPUT_FILE: &str =
    "visualizations/random_9_cells_normal_forms_hat_strongly_cubular_up_to_symmetry.jsonl";
const FAILURE_ROOT: &str = "visualizations/random_boundary_gluing_failures";
const SYMMETRY_COUNT: usize = 8;
const MAX_ISOMORPHISMS: usize = 10;
const REPORT_INTERVAL: Duration = Duration::from_secs(5);
const REPORT_EVERY_LOADED: usize = 100_000;
const REPORT_EVERY_INDEXED: usize = 100_000;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetRecord {
    hash: String,
    stabilizer_size: usize,
    multiplicity: u64,
    ofp: FramedPoset,
}

struct BoundaryOccurrence {
    shape_id: usize,
    into_shape: Embedding,
    to_canonical: Embedding,
}

struct BoundaryClass {
    direction: usize,
    canonical: Arc<FramedPoset>,
    inputs: Vec<BoundaryOccurrence>,
    outputs: Vec<BoundaryOccurrence>,
    automorphisms: Vec<Embedding>,
}

struct BoundaryIndex {
    classes: Vec<BoundaryClass>,
    cumulative_pairs: Vec<u128>,
    total_pairs: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PairKey {
    direction: usize,
    first: usize,
    second: usize,
}

struct SampledPair {
    class: usize,
    input: usize,
    output: usize,
    key: PairKey,
}

struct GluingFailure {
    pair_number: u64,
    key: PairKey,
    isomorphism_index: usize,
    first: Arc<FramedPoset>,
    second: Arc<FramedPoset>,
    input_boundary: Embedding,
    output_boundary: Embedding,
    boundary_isomorphism: Embedding,
    pushout: Pushout,
    // cubularity: CubularityFailure,
}

#[derive(Serialize)]
struct FailureReport<'a> {
    seed: String,
    pair_number: u64,
    direction: usize,
    first_shape: usize,
    second_shape: usize,
    isomorphism_number: usize,
    boundary_isomorphism_map: &'a [Vec<usize>],
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
    let statistics = match env::args().nth(1).as_deref() {
        None => false,
        Some("--statistics") => true,
        Some(argument) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown argument {argument:?}; expected --statistics"),
            ));
        }
    };
    let symmetries = two_dimensional_symmetries();
    let shapes = load_and_expand_dataset(Path::new(INPUT_FILE), &symmetries)?;
    let boundary_index = build_boundary_index(&shapes)?;

    println!(
        "loaded {} concrete symmetry images in {} compatible boundary classes",
        shapes.len(),
        boundary_index.classes.len()
    );
    if statistics {
        exhaustive_statistics(&boundary_index)?;
        return Ok(());
    }

    let seed = OsRng.try_next_u64().map_err(io::Error::other)?;
    println!(
        "sampling among {} compatible ordered pairs (seed {seed:#018x}); press any key to stop",
        boundary_index.total_pairs
    );

    let raw_mode = RawModeGuard::enable()?;
    let result = search(&shapes, &boundary_index, seed);
    drop(raw_mode);

    match result? {
        Some(failure) => {
            let output_dir = write_failure(seed, &failure)?;
            println!("found a non-cubular gluing; wrote {}", output_dir.display());
        }
        None => println!("search stopped without finding a counterexample"),
    }
    Ok(())
}

#[derive(Debug, Default, Clone, Copy)]
struct GluingStatistics {
    compatible_pairs: u128,
    pairs_with_failure: u128,
    pairs_with_only_failures: u128,
    gluings: u128,
    failing_gluings: u128,
}

fn exhaustive_statistics(boundary_index: &BoundaryIndex) -> io::Result<()> {
    let mut total = GluingStatistics::default();
    let mut by_direction = [GluingStatistics::default(); 2];

    for class in &boundary_index.classes {
        let automorphisms = isomorphisms(&class.canonical, &class.canonical);
        if automorphisms.is_empty() {
            return Err(invalid_data(
                "canonical boundary has no identity automorphism",
            ));
        }

        for input in &class.inputs {
            for output in &class.outputs {
                let mut pair_has_failure = false;
                let mut pair_has_success = false;
                let from_canonical = output.to_canonical.inverse_isomorphism();

                for automorphism in &automorphisms {
                    let through_automorphism =
                        Embedding::compose(&input.to_canonical, automorphism);
                    let boundary_isomorphism =
                        Embedding::compose(&through_automorphism, &from_canonical);
                    let into_second = Embedding::compose(&boundary_isomorphism, &output.into_shape);
                    let glued = pushout(&input.into_shape, &into_second);
                    let failed = !is_cubular(CubularityMode::Strong, &glued.tip);

                    total.gluings += 1;
                    by_direction[class.direction].gluings += 1;
                    if failed {
                        total.failing_gluings += 1;
                        by_direction[class.direction].failing_gluings += 1;
                        pair_has_failure = true;
                    } else {
                        pair_has_success = true;
                    }
                }

                total.compatible_pairs += 1;
                by_direction[class.direction].compatible_pairs += 1;
                if pair_has_failure {
                    total.pairs_with_failure += 1;
                    by_direction[class.direction].pairs_with_failure += 1;
                }
                if pair_has_failure && !pair_has_success {
                    total.pairs_with_only_failures += 1;
                    by_direction[class.direction].pairs_with_only_failures += 1;
                }
            }
        }
    }

    print_statistics("all directions", total);
    for (direction, statistics) in by_direction.into_iter().enumerate() {
        print_statistics(&format!("direction {direction}"), statistics);
    }
    Ok(())
}

fn print_statistics(label: &str, statistics: GluingStatistics) {
    println!("{label}:");
    println!(
        "  compatible ordered pairs: {}",
        statistics.compatible_pairs
    );
    println!(
        "  pairs with at least one failing isomorphism: {} ({:.2}%)",
        statistics.pairs_with_failure,
        percentage(statistics.pairs_with_failure, statistics.compatible_pairs)
    );
    println!(
        "  pairs whose every isomorphism fails: {} ({:.2}%)",
        statistics.pairs_with_only_failures,
        percentage(
            statistics.pairs_with_only_failures,
            statistics.compatible_pairs
        )
    );
    println!("  individual gluings: {}", statistics.gluings);
    println!(
        "  non-cubular gluings: {} ({:.2}%)",
        statistics.failing_gluings,
        percentage(statistics.failing_gluings, statistics.gluings)
    );
}

fn percentage(part: u128, whole: u128) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

fn load_and_expand_dataset(
    path: &Path,
    symmetries: &[SignedPermutation],
) -> io::Result<Vec<Arc<FramedPoset>>> {
    let file = File::open(path)?;
    let mut input = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut shapes = Vec::new();
    let mut image_sources: HashMap<Arc<FramedPoset>, usize> = HashMap::new();
    let mut previous_hash = None;
    let mut line = String::new();
    let mut line_number = 0usize;

    loop {
        line.clear();
        let bytes_read = input.read_line(&mut line)?;
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
        let stored_hash = parse_hash(path, line_number, &record.hash)?;
        if previous_hash.is_some_and(|previous| previous >= stored_hash) {
            return Err(invalid_line(
                path,
                line_number,
                "hashes must be strictly increasing",
            ));
        }
        previous_hash = Some(stored_hash);
        if record.multiplicity == 0 {
            return Err(invalid_line(
                path,
                line_number,
                "multiplicity must be positive",
            ));
        }
        validate_dataset_shape(&record.ofp, path, line_number)?;

        let normal = Arc::new(normalize(&record.ofp));
        if !FramedPoset::equal(&normal, &record.ofp) {
            return Err(invalid_line(
                path,
                line_number,
                "stored OFP is not in canonical normal form",
            ));
        }
        let actual_hash = structural_hash(&normal);
        if actual_hash != stored_hash {
            return Err(invalid_line(
                path,
                line_number,
                format!(
                    "stored hash {stored_hash:016x} does not match recomputed hash {actual_hash:016x}"
                ),
            ));
        }
        if !is_cubular(CubularityMode::Strong, &normal) {
            return Err(invalid_line(
                path,
                line_number,
                "stored OFP is not strongly cubular",
            ));
        }

        let orbit = normalized_symmetry_images(&normal, symmetries).map_err(|message| {
            invalid_line(
                path,
                line_number,
                format!("invalid symmetry orbit: {message}"),
            )
        })?;
        let expected_stabilizer = SYMMETRY_COUNT / orbit.len();
        if record.stabilizer_size != expected_stabilizer {
            return Err(invalid_line(
                path,
                line_number,
                format!(
                    "stored stabilizer size {} does not match recomputed size {expected_stabilizer}",
                    record.stabilizer_size
                ),
            ));
        }
        if !FramedPoset::equal(&normal, &orbit[0]) {
            return Err(invalid_line(
                path,
                line_number,
                "stored OFP is not the canonical symmetry-orbit representative",
            ));
        }

        for image in orbit {
            if let Some(previous_line) = image_sources.insert(Arc::clone(&image), line_number) {
                return Err(invalid_line(
                    path,
                    line_number,
                    format!("symmetry orbit overlaps the record on line {previous_line}"),
                ));
            }
            shapes.push(image);
        }

        if line_number.is_multiple_of(REPORT_EVERY_LOADED) {
            println!(
                "loaded {line_number} orbit representatives and {} symmetry images",
                shapes.len()
            );
        }
    }

    if line_number == 0 {
        return Err(invalid_data("input dataset is empty"));
    }
    Ok(shapes)
}

fn normalized_symmetry_images(
    shape: &Arc<FramedPoset>,
    symmetries: &[SignedPermutation],
) -> Result<Vec<Arc<FramedPoset>>, String> {
    let mut images: Vec<(Vec<u8>, Arc<FramedPoset>)> = Vec::with_capacity(SYMMETRY_COUNT);

    for symmetry in symmetries {
        let transformed = transform(shape, symmetry)
            .map_err(|error| format!("could not apply {symmetry:?}: {error}"))?;
        let image = Arc::new(normalize(&transformed));
        if !is_cubular(CubularityMode::Strong, &image) {
            return Err(format!("{symmetry:?} produced a non-cubular image"));
        }
        if images
            .iter()
            .any(|(_, existing)| FramedPoset::equal(existing, &image))
        {
            continue;
        }
        let serialized = serde_json::to_vec(image.as_ref())
            .map_err(|error| format!("could not serialize symmetry image: {error}"))?;
        images.push((serialized, image));
    }

    if images.is_empty() || !SYMMETRY_COUNT.is_multiple_of(images.len()) {
        return Err(format!(
            "orbit has {} distinct symmetry images",
            images.len()
        ));
    }
    images.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
    Ok(images.into_iter().map(|(_, image)| image).collect())
}

fn build_boundary_index(shapes: &[Arc<FramedPoset>]) -> io::Result<BoundaryIndex> {
    let mut classes = Vec::<BoundaryClass>::new();
    let mut class_indices = HashMap::<(usize, Arc<FramedPoset>), usize>::new();
    let mut transports = HashMap::<Arc<FramedPoset>, Embedding>::new();

    for (shape_id, shape) in shapes.iter().enumerate() {
        for direction in 0..2 {
            for sign in [Sign::Input, Sign::Output] {
                let (boundary, into_shape) = boundary(sign, direction, shape);
                let canonical = Arc::new(normalize(&boundary));
                let class = *class_indices
                    .entry((direction, Arc::clone(&canonical)))
                    .or_insert_with(|| {
                        let class = classes.len();
                        classes.push(BoundaryClass {
                            direction,
                            canonical: Arc::clone(&canonical),
                            inputs: Vec::new(),
                            outputs: Vec::new(),
                            automorphisms: Vec::new(),
                        });
                        class
                    });
                let to_canonical =
                    transport_to_canonical(&boundary, &classes[class].canonical, &mut transports)?;
                let occurrence = BoundaryOccurrence {
                    shape_id,
                    into_shape,
                    to_canonical,
                };
                match sign {
                    Sign::Input => classes[class].inputs.push(occurrence),
                    Sign::Output => classes[class].outputs.push(occurrence),
                }
            }
        }

        if (shape_id + 1).is_multiple_of(REPORT_EVERY_INDEXED) {
            println!(
                "indexed boundaries of {} OFPs into {} classes",
                shape_id + 1,
                classes.len()
            );
        }
    }

    classes.retain(|class| !class.inputs.is_empty() && !class.outputs.is_empty());
    let mut cumulative_pairs = Vec::with_capacity(classes.len());
    let mut total_pairs = 0u128;
    for class in &mut classes {
        class.automorphisms = isomorphisms(&class.canonical, &class.canonical)
            .into_iter()
            .take(MAX_ISOMORPHISMS)
            .collect();
        if class.automorphisms.is_empty() {
            return Err(invalid_data(
                "canonical boundary has no identity automorphism",
            ));
        }

        let pair_count = (class.inputs.len() as u128)
            .checked_mul(class.outputs.len() as u128)
            .ok_or_else(|| invalid_data("compatible pair count overflow"))?;
        total_pairs = total_pairs
            .checked_add(pair_count)
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

fn transport_to_canonical(
    boundary: &Arc<FramedPoset>,
    canonical: &Arc<FramedPoset>,
    cache: &mut HashMap<Arc<FramedPoset>, Embedding>,
) -> io::Result<Embedding> {
    if let Some(transport) = cache.get(boundary) {
        if !FramedPoset::equal(&transport.cod, canonical) {
            return Err(invalid_data(
                "equal boundary domains have unequal canonical forms",
            ));
        }
        return Ok(transport.clone());
    }

    let transport = isomorphisms(boundary, canonical)
        .into_iter()
        .next()
        .ok_or_else(|| invalid_data("boundary is not isomorphic to its normal form"))?;
    cache.insert(Arc::clone(boundary), transport.clone());
    Ok(transport)
}

fn search(
    shapes: &[Arc<FramedPoset>],
    boundary_index: &BoundaryIndex,
    seed: u64,
) -> io::Result<Option<GluingFailure>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut seen = HashSet::<PairKey>::new();
    let mut pair_count = 0u64;
    let mut isomorphism_count = 0u64;
    let mut pushout_count = 0u64;
    let mut next_report = Instant::now() + REPORT_INTERVAL;

    loop {
        if seen.len() as u128 == boundary_index.total_pairs {
            println!("tested every compatible ordered pair");
            return Ok(None);
        }
        if event::poll(Duration::ZERO)? {
            match event::read()? {
                Event::Key(key) if key.kind != KeyEventKind::Release => return Ok(None),
                _ => {}
            }
        }

        let sampled = sample_unseen_pair(boundary_index, &mut seen, &mut rng);
        pair_count = pair_count
            .checked_add(1)
            .ok_or_else(|| invalid_data("sampled pair counter overflow"))?;
        let class = &boundary_index.classes[sampled.class];
        let input = &class.inputs[sampled.input];
        let output = &class.outputs[sampled.output];
        let from_canonical = output.to_canonical.inverse_isomorphism();

        for (isomorphism_index, automorphism) in class.automorphisms.iter().enumerate() {
            let through_automorphism = Embedding::compose(&input.to_canonical, automorphism);
            let boundary_isomorphism = Embedding::compose(&through_automorphism, &from_canonical);
            let into_second = Embedding::compose(&boundary_isomorphism, &output.into_shape);
            let glued = pushout(&input.into_shape, &into_second);

            isomorphism_count = isomorphism_count
                .checked_add(1)
                .ok_or_else(|| invalid_data("isomorphism counter overflow"))?;
            pushout_count = pushout_count
                .checked_add(1)
                .ok_or_else(|| invalid_data("pushout counter overflow"))?;

            if !is_cubular(CubularityMode::Strong, &glued.tip) {
                return Ok(Some(GluingFailure {
                    pair_number: pair_count,
                    key: sampled.key,
                    isomorphism_index,
                    first: Arc::clone(&shapes[input.shape_id]),
                    second: Arc::clone(&shapes[output.shape_id]),
                    input_boundary: input.into_shape.clone(),
                    output_boundary: output.into_shape.clone(),
                    boundary_isomorphism,
                    pushout: glued,
                }));
            }
        }

        if Instant::now() >= next_report {
            print_raw_line(&format!(
                "compatible pairs: {pair_count}; isomorphisms: {isomorphism_count}; pushouts: {pushout_count}; remaining pairs: {}",
                boundary_index.total_pairs - seen.len() as u128
            ))?;
            next_report = Instant::now() + REPORT_INTERVAL;
        }
    }
}

fn sample_unseen_pair<R: Rng + ?Sized>(
    boundary_index: &BoundaryIndex,
    seen: &mut HashSet<PairKey>,
    rng: &mut R,
) -> SampledPair {
    loop {
        let ticket = rng.random_range(0..boundary_index.total_pairs);
        let class_index = boundary_index
            .cumulative_pairs
            .partition_point(|&end| end <= ticket);
        let class_start = class_index
            .checked_sub(1)
            .map_or(0, |previous| boundary_index.cumulative_pairs[previous]);
        let local = ticket - class_start;
        let class = &boundary_index.classes[class_index];
        let output_count = class.outputs.len() as u128;
        let input_index = usize::try_from(local / output_count)
            .expect("input index was derived from a usize-sized collection");
        let output_index = usize::try_from(local % output_count)
            .expect("output index was derived from a usize-sized collection");
        let key = PairKey {
            direction: class.direction,
            first: class.inputs[input_index].shape_id,
            second: class.outputs[output_index].shape_id,
        };

        if seen.insert(key) {
            return SampledPair {
                class: class_index,
                input: input_index,
                output: output_index,
                key,
            };
        }
    }
}

fn write_failure(seed: u64, failure: &GluingFailure) -> io::Result<PathBuf> {
    let output_dir = unique_failure_directory(seed, failure.pair_number)?;
    let report = FailureReport {
        seed: format!("{seed:#018x}"),
        pair_number: failure.pair_number,
        direction: failure.key.direction,
        first_shape: failure.key.first,
        second_shape: failure.key.second,
        isomorphism_number: failure.isomorphism_index + 1,
        boundary_isomorphism_map: &failure.boundary_isomorphism.map,
    };
    fs::write(
        output_dir.join("report.json"),
        format!(
            "{}\n",
            serde_json::to_string_pretty(&report).map_err(io::Error::other)?
        ),
    )?;

    write_shape_artifacts(&output_dir, "first", &failure.first)?;
    write_shape_artifacts(&output_dir, "second", &failure.second)?;
    write_shape_artifacts(&output_dir, "pushout", &failure.pushout.tip)?;
    write_embedding_artifacts(&output_dir, "first_input_boundary", &failure.input_boundary)?;
    write_embedding_artifacts(
        &output_dir,
        "second_output_boundary",
        &failure.output_boundary,
    )?;
    write_embedding_artifacts(
        &output_dir,
        "boundary_isomorphism",
        &failure.boundary_isomorphism,
    )?;
    write_embedding_artifacts(&output_dir, "first_into_pushout", &failure.pushout.inl)?;
    write_embedding_artifacts(&output_dir, "second_into_pushout", &failure.pushout.inr)?;

    Ok(output_dir)
}

fn write_shape_artifacts(output_dir: &Path, name: &str, shape: &FramedPoset) -> io::Result<()> {
    fs::write(
        output_dir.join(format!("{name}.ofp.json")),
        format!(
            "{}\n",
            serde_json::to_string_pretty(shape).map_err(io::Error::other)?
        ),
    )?;
    fs::write(
        output_dir.join(format!("{name}_graded.dot")),
        to_dot(shape, Renderer::Ranked),
    )?;
    fs::write(
        output_dir.join(format!("{name}_compass_spring.dot")),
        to_dot(shape, Renderer::CompassSpring),
    )
}

fn write_embedding_artifacts(
    output_dir: &Path,
    name: &str,
    embedding: &Embedding,
) -> io::Result<()> {
    fs::write(
        output_dir.join(format!("{name}_graded.dot")),
        embedding_to_dot(embedding, Renderer::Ranked),
    )?;
    fs::write(
        output_dir.join(format!("{name}_compass_spring.dot")),
        embedding_to_dot(embedding, Renderer::CompassSpring),
    )
}

fn unique_failure_directory(seed: u64, pair_number: u64) -> io::Result<PathBuf> {
    let root = Path::new(FAILURE_ROOT);
    fs::create_dir_all(root)?;

    for suffix in 0usize.. {
        let suffix = if suffix == 0 {
            String::new()
        } else {
            format!("_{suffix}")
        };
        let path = root.join(format!("seed_{seed:016x}_pair_{pair_number}{suffix}"));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    unreachable!()
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

fn validate_dataset_shape(shape: &FramedPoset, path: &Path, line: usize) -> io::Result<()> {
    let sizes = shape.sizes();
    if sizes.len() != 3 || sizes[2] == 0 {
        return Err(invalid_line(
            path,
            line,
            "OFP is not a two-dimensional shape",
        ));
    }
    for (dim, size) in sizes.into_iter().enumerate() {
        for pos in 0..size {
            if shape
                .frame_of(dim, pos)
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

fn print_raw_line(message: &str) -> io::Result<()> {
    use std::io::Write as _;

    let mut stdout = io::stdout().lock();
    write!(stdout, "{message}\r\n")?;
    stdout.flush()
}

fn invalid_line(path: &Path, line: usize, message: impl std::fmt::Display) -> io::Error {
    invalid_data(format!("{}:{line}: {message}", path.display()))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
