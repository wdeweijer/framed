use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ofposets::pushout::{Pushout, pushout};
use ofposets::{
    CubularityMode, Embedding, FramedPoset, RandomFramedPosetGenerator, Renderer, Sign, boundary,
    embedding_to_dot, is_cubular, isomorphisms, normalize, to_dot,
};
use rand::rngs::{OsRng, SmallRng};
use rand::{Rng, SeedableRng, TryRngCore};

const MAX_ISOMORPHISMS: usize = 10;
const GENERATION_BATCH_SIZE: usize = 4_096;
const REPORT_INTERVAL: Duration = Duration::from_secs(5);
const FAILURE_ROOT: &str = "visualizations/random_boundary_gluing_failures";

#[derive(Debug, Clone, Copy)]
struct Options {
    cell_count: usize,
    dimension: usize,
    shape_count: usize,
    pair_count: usize,
    worker_count: usize,
    cubularity_mode: CubularityMode,
    seed: u64,
}

struct BoundaryOccurrence {
    shape: usize,
    into_shape: Embedding,
    to_canonical: Embedding,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SampledPair {
    class: usize,
    output: usize,
    input: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EquationKind {
    Axial,
    Transverse,
}

struct EquationCheck {
    kind: EquationKind,
    direction: usize,
    sign: Sign,
    holds: bool,
    actual: Embedding,
    expected: Option<Embedding>,
    expected_parts: Option<[Embedding; 2]>,
}

#[derive(Clone, Copy)]
struct PastedSide<'a> {
    shape: &'a Arc<FramedPoset>,
    into_pasted: &'a Embedding,
}

struct FailureWitness {
    pair: usize,
    gluing_direction: usize,
    first_shape: usize,
    second_shape: usize,
    isomorphism: usize,
    first: Arc<FramedPoset>,
    second: Arc<FramedPoset>,
    first_output: Embedding,
    second_input: Embedding,
    boundary_isomorphism: Embedding,
    pushout: Pushout,
    glued_passes_cubularity: bool,
    checks: Vec<EquationCheck>,
}

#[derive(Debug, Default)]
struct Statistics {
    pairs: u128,
    gluings: u128,
    failing_gluings: u128,
    cubularity_failures: u128,
    axial_failures: [u128; 2],
    transverse_failures: [u128; 2],
    nonempty_axial_boundary_intersections: u128,
    axial_boundary_intersection_dimensions: BTreeMap<usize, u128>,
}

fn main() -> io::Result<()> {
    let options = parse_options()?;
    let generator = RandomFramedPosetGenerator::new(options.dimension, options.cell_count);
    println!(
        "generating {} {}cubular {}-cell {}D OFPs on {} threads (seed {:#018x})",
        options.shape_count,
        if options.cubularity_mode == CubularityMode::Strong {
            "strongly "
        } else {
            ""
        },
        options.cell_count,
        options.dimension,
        options.worker_count,
        options.seed
    );

    let (shapes, candidates) = generate_shapes(options, &generator)?;
    println!(
        "retained {} shapes from {candidates} generated candidates",
        shapes.len()
    );

    let mut boundary_index = build_boundary_index(&shapes, options.dimension)?;
    prepare_automorphisms(&mut boundary_index)?;
    println!(
        "indexed {} compatible boundary classes containing {} ordered pairs",
        boundary_index.classes.len(),
        boundary_index.total_pairs
    );

    let sample_count = options
        .pair_count
        .min(usize::try_from(boundary_index.total_pairs).unwrap_or(options.pair_count));
    let mut rng = SmallRng::seed_from_u64(options.seed.wrapping_sub(1));
    let pairs = sample_pairs(&boundary_index, sample_count, &mut rng);
    let (statistics, first_failure) = check_pairs(
        &shapes,
        &boundary_index,
        &pairs,
        options.dimension,
        options.cubularity_mode,
    );
    print_statistics(&statistics, options.dimension, options.cubularity_mode);

    if let Some(failure) = first_failure {
        let output_dir = write_failure(options, &failure)?;
        println!(
            "first failure: pair {}; gluing direction {}; first shape {}; second shape {}; \
             isomorphism {}",
            failure.pair + 1,
            failure.gluing_direction,
            failure.first_shape,
            failure.second_shape,
            failure.isomorphism + 1
        );
        if !failure.glued_passes_cubularity {
            println!("glued OFP failed {:?} cubularity", options.cubularity_mode);
        }
        println!("wrote failure artifacts to {}", output_dir.display());
    } else {
        println!(
            "all glued OFPs passed {:?} cubularity and all sampled axial and transverse boundary equations held",
            options.cubularity_mode
        );
    }

    Ok(())
}

fn parse_options() -> io::Result<Options> {
    let mut options = Options {
        cell_count: 8,
        dimension: 3,
        shape_count: 1_000,
        pair_count: 1_000,
        worker_count: thread::available_parallelism()?.get(),
        cubularity_mode: CubularityMode::Regular,
        seed: OsRng.try_next_u64().map_err(io::Error::other)?,
    };
    let mut arguments = env::args().skip(1);

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--cells" => options.cell_count = parse_usize_argument(&mut arguments, "--cells")?,
            "--dim" => options.dimension = parse_usize_argument(&mut arguments, "--dim")?,
            "--shapes" => options.shape_count = parse_usize_argument(&mut arguments, "--shapes")?,
            "--pairs" => options.pair_count = parse_usize_argument(&mut arguments, "--pairs")?,
            "--threads" => {
                options.worker_count = parse_usize_argument(&mut arguments, "--threads")?;
            }
            "--seed" => {
                let value = arguments.next().ok_or_else(|| {
                    invalid_input("--seed requires a decimal or 0x-prefixed hexadecimal value")
                })?;
                options.seed = parse_seed(&value)?;
            }
            "--strong" => options.cubularity_mode = CubularityMode::Strong,
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --release --example \
                     check_random_boundary_gluing -- \
                     [--cells N] [--shapes N] [--pairs N] \
                     [--threads N] [--seed N] [--strong]"
                );
                std::process::exit(0);
            }
            _ => return Err(invalid_input(format!("unknown argument {argument:?}"))),
        }
    }

    if options.dimension >= usize::BITS as usize {
        return Err(invalid_input(format!(
            "--dim must be smaller than {}",
            usize::BITS
        )));
    }
    let minimum_cell_count = 1usize << options.dimension;
    if options.cell_count < minimum_cell_count {
        return Err(invalid_input(format!(
            "--cells must be at least {minimum_cell_count} for dimension {}",
            options.dimension
        )));
    }
    if options.shape_count == 0 {
        return Err(invalid_input("--shapes must be positive"));
    }
    if options.pair_count == 0 {
        return Err(invalid_input("--pairs must be positive"));
    }
    if options.worker_count == 0 {
        return Err(invalid_input("--threads must be positive"));
    }
    Ok(options)
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

fn generate_shapes(
    options: Options,
    generator: &RandomFramedPosetGenerator,
) -> io::Result<(Vec<Arc<FramedPoset>>, usize)> {
    let mut shapes = Vec::with_capacity(options.shape_count);
    let mut candidates = 0usize;
    let mut next_report = Instant::now() + REPORT_INTERVAL;

    while shapes.len() < options.shape_count {
        let batch_end = candidates
            .checked_add(GENERATION_BATCH_SIZE)
            .ok_or_else(|| invalid_data("candidate counter overflow"))?;
        let mut accepted = generate_candidate_batch(options, generator, candidates, batch_end);
        accepted.sort_unstable_by_key(|(ticket, _)| *ticket);
        shapes.extend(
            accepted
                .into_iter()
                .map(|(_, shape)| shape)
                .take(options.shape_count - shapes.len()),
        );
        candidates = batch_end;

        if Instant::now() >= next_report {
            println!(
                "  generation progress: {candidates} candidates; {} retained",
                shapes.len()
            );
            next_report = Instant::now() + REPORT_INTERVAL;
        }
    }

    Ok((shapes, candidates))
}

fn generate_candidate_batch(
    options: Options,
    generator: &RandomFramedPosetGenerator,
    start: usize,
    end: usize,
) -> Vec<(usize, Arc<FramedPoset>)> {
    let candidate_count = end - start;
    let worker_count = options.worker_count.min(candidate_count).max(1);
    let chunk_size = candidate_count.div_ceil(worker_count);

    thread::scope(|scope| {
        let workers: Vec<_> = (start..end)
            .step_by(chunk_size)
            .map(|chunk_start| {
                let chunk_end = chunk_start.saturating_add(chunk_size).min(end);
                scope.spawn(move || {
                    let mut accepted = Vec::new();
                    for ticket in chunk_start..chunk_end {
                        let stream = u64::try_from(ticket)
                            .expect("candidate ticket must fit into a u64 seed");
                        let mut rng = SmallRng::seed_from_u64(options.seed.wrapping_add(stream));
                        let shape = Arc::new(generator.generate(&mut rng));
                        let is_accepted = is_cubular(options.cubularity_mode, &shape);
                        if is_accepted {
                            accepted.push((ticket, shape));
                        }
                    }
                    accepted
                })
            })
            .collect();

        workers
            .into_iter()
            .flat_map(|worker| worker.join().expect("generation worker panicked"))
            .collect()
    })
}

fn build_boundary_index(
    shapes: &[Arc<FramedPoset>],
    direction_count: usize,
) -> io::Result<BoundaryIndex> {
    let mut classes = Vec::<BoundaryClass>::new();
    let mut class_indices = HashMap::<(usize, Arc<FramedPoset>), usize>::new();
    let mut transports = HashMap::<Arc<FramedPoset>, Embedding>::new();

    for (shape, poset) in shapes.iter().enumerate() {
        for direction in 0..direction_count {
            for sign in [Sign::Input, Sign::Output] {
                let (boundary, into_shape) = boundary(sign, direction, poset);
                let canonical = Arc::new(normalize(&boundary));
                let class = *class_indices
                    .entry((direction, Arc::clone(&canonical)))
                    .or_insert_with(|| {
                        let class = classes.len();
                        classes.push(BoundaryClass {
                            direction,
                            canonical: Arc::clone(&canonical),
                            outputs: Vec::new(),
                            inputs: Vec::new(),
                            automorphisms: Vec::new(),
                        });
                        class
                    });
                let to_canonical =
                    transport_to_canonical(&boundary, &classes[class].canonical, &mut transports)?;
                let occurrence = BoundaryOccurrence {
                    shape,
                    into_shape,
                    to_canonical,
                };
                match sign {
                    Sign::Input => classes[class].inputs.push(occurrence),
                    Sign::Output => classes[class].outputs.push(occurrence),
                }
            }
        }
    }

    classes.retain(|class| !class.outputs.is_empty() && !class.inputs.is_empty());
    let mut cumulative_pairs = Vec::with_capacity(classes.len());
    let mut total_pairs = 0u128;
    for class in &classes {
        let class_pairs = (class.outputs.len() as u128)
            .checked_mul(class.inputs.len() as u128)
            .ok_or_else(|| invalid_data("compatible pair count overflow"))?;
        total_pairs = total_pairs
            .checked_add(class_pairs)
            .ok_or_else(|| invalid_data("total compatible pair count overflow"))?;
        cumulative_pairs.push(total_pairs);
    }
    if total_pairs == 0 {
        return Err(invalid_data(
            "generated shapes have no compatible input/output boundary pairs",
        ));
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
        debug_assert!(FramedPoset::equal(&transport.cod, canonical));
        return Ok(transport.clone());
    }

    let transport = isomorphisms(boundary, canonical)
        .into_iter()
        .next()
        .ok_or_else(|| invalid_data("boundary is not isomorphic to its normal form"))?;
    cache.insert(Arc::clone(boundary), transport.clone());
    Ok(transport)
}

fn prepare_automorphisms(index: &mut BoundaryIndex) -> io::Result<()> {
    for class in &mut index.classes {
        class.automorphisms = isomorphisms(&class.canonical, &class.canonical)
            .into_iter()
            .take(MAX_ISOMORPHISMS)
            .collect();
        if class.automorphisms.is_empty() {
            return Err(invalid_data(
                "canonical boundary has no identity automorphism",
            ));
        }
    }
    Ok(())
}

fn sample_pairs<R: Rng + ?Sized>(
    index: &BoundaryIndex,
    sample_count: usize,
    rng: &mut R,
) -> Vec<SampledPair> {
    let mut seen = HashSet::with_capacity(sample_count);
    let mut pairs = Vec::with_capacity(sample_count);

    while pairs.len() < sample_count {
        let ticket = rng.random_range(0..index.total_pairs);
        let class_index = index.cumulative_pairs.partition_point(|&end| end <= ticket);
        let class_start = class_index
            .checked_sub(1)
            .map_or(0, |previous| index.cumulative_pairs[previous]);
        let local = ticket - class_start;
        let class = &index.classes[class_index];
        let input_count = class.inputs.len() as u128;
        let pair = SampledPair {
            class: class_index,
            output: usize::try_from(local / input_count)
                .expect("output index comes from a usize-sized collection"),
            input: usize::try_from(local % input_count)
                .expect("input index comes from a usize-sized collection"),
        };
        if seen.insert(pair) {
            pairs.push(pair);
        }
    }
    pairs
}

fn check_pairs(
    shapes: &[Arc<FramedPoset>],
    index: &BoundaryIndex,
    pairs: &[SampledPair],
    direction_count: usize,
    cubularity_mode: CubularityMode,
) -> (Statistics, Option<FailureWitness>) {
    let mut statistics = Statistics::default();
    let mut first_failure = None;

    for (pair_number, pair) in pairs.iter().copied().enumerate() {
        let class = &index.classes[pair.class];
        let output = &class.outputs[pair.output];
        let input = &class.inputs[pair.input];
        let first = &shapes[output.shape];
        let second = &shapes[input.shape];
        let from_canonical = input.to_canonical.inverse_isomorphism();
        statistics.pairs += 1;
        statistics.record_axial_boundary_intersection(axial_boundary_intersection_dimension(
            first,
            class.direction,
            &output.into_shape,
        ));

        for (isomorphism, automorphism) in class.automorphisms.iter().enumerate() {
            let through_automorphism = Embedding::compose(&output.to_canonical, automorphism);
            let boundary_isomorphism = Embedding::compose(&through_automorphism, &from_canonical);
            let first_output_into_second =
                Embedding::compose(&boundary_isomorphism, &input.into_shape);
            let pasted = pushout(&output.into_shape, &first_output_into_second);
            let glued_passes_cubularity = is_cubular(cubularity_mode, &pasted.tip);
            let checks = check_formulas(direction_count, first, second, class.direction, &pasted);
            let failed = !glued_passes_cubularity || checks.iter().any(|check| !check.holds);

            statistics.record(glued_passes_cubularity, &checks);
            if failed && first_failure.is_none() {
                first_failure = Some(FailureWitness {
                    pair: pair_number,
                    gluing_direction: class.direction,
                    first_shape: output.shape,
                    second_shape: input.shape,
                    isomorphism,
                    first: Arc::clone(first),
                    second: Arc::clone(second),
                    first_output: output.into_shape.clone(),
                    second_input: input.into_shape.clone(),
                    boundary_isomorphism,
                    pushout: pasted,
                    glued_passes_cubularity,
                    checks,
                });
            }
        }
    }

    (statistics, first_failure)
}

impl Statistics {
    fn record(&mut self, glued_passes_cubularity: bool, checks: &[EquationCheck]) {
        self.gluings += 1;
        let mut failed = !glued_passes_cubularity;
        self.cubularity_failures += u128::from(!glued_passes_cubularity);

        for check in checks {
            if check.holds {
                continue;
            }
            failed = true;
            let sign = sign_index(check.sign);
            match check.kind {
                EquationKind::Axial => {
                    self.axial_failures[sign] += 1;
                }
                EquationKind::Transverse => {
                    self.transverse_failures[sign] += 1;
                }
            }
        }
        self.failing_gluings += u128::from(failed);
    }

    fn record_axial_boundary_intersection(&mut self, dimension: Option<usize>) {
        let Some(dimension) = dimension else {
            return;
        };
        self.nonempty_axial_boundary_intersections += 1;
        *self
            .axial_boundary_intersection_dimensions
            .entry(dimension)
            .or_default() += 1;
    }
}

fn axial_boundary_intersection_dimension(
    first: &Arc<FramedPoset>,
    direction: usize,
    output_boundary: &Embedding,
) -> Option<usize> {
    let (_, input_boundary) = boundary(Sign::Input, direction, first);
    debug_assert!(input_boundary.is_closed());
    debug_assert!(output_boundary.is_closed());
    let intersection = Embedding::intersection(&input_boundary, output_boundary).into_codomain;

    (!intersection.is_empty()).then(|| intersection.dom.active_directions().len())
}

fn check_formulas(
    direction_count: usize,
    first: &Arc<FramedPoset>,
    second: &Arc<FramedPoset>,
    gluing_direction: usize,
    pasted: &Pushout,
) -> Vec<EquationCheck> {
    let mut checks = Vec::with_capacity(2 * direction_count);
    let first = PastedSide {
        shape: first,
        into_pasted: &pasted.inl,
    };
    let second = PastedSide {
        shape: second,
        into_pasted: &pasted.inr,
    };
    checks.push(compare_boundary_with_side(
        Sign::Input,
        gluing_direction,
        &pasted.tip,
        first,
    ));
    checks.push(compare_boundary_with_side(
        Sign::Output,
        gluing_direction,
        &pasted.tip,
        second,
    ));

    for direction in 0..direction_count {
        if direction == gluing_direction {
            continue;
        }
        for sign in [Sign::Input, Sign::Output] {
            checks.push(compare_boundary_with_union(
                sign,
                direction,
                &pasted.tip,
                first,
                second,
            ));
        }
    }
    checks
}

fn compare_boundary_with_side(
    sign: Sign,
    direction: usize,
    pasted: &Arc<FramedPoset>,
    side: PastedSide<'_>,
) -> EquationCheck {
    let (_, actual) = boundary(sign, direction, pasted);
    let (_, into_side) = boundary(sign, direction, side.shape);
    let expected = Embedding::compose(&into_side, side.into_pasted);
    let holds = actual.is_closed() && expected.is_closed() && Embedding::equal(&actual, &expected);

    EquationCheck {
        kind: EquationKind::Axial,
        direction,
        sign,
        holds,
        actual,
        expected: Some(expected),
        expected_parts: None,
    }
}

fn compare_boundary_with_union(
    sign: Sign,
    direction: usize,
    pasted: &Arc<FramedPoset>,
    first: PastedSide<'_>,
    second: PastedSide<'_>,
) -> EquationCheck {
    let (_, actual) = boundary(sign, direction, pasted);
    let (_, first_boundary) = boundary(sign, direction, first.shape);
    let (_, second_boundary) = boundary(sign, direction, second.shape);
    let first_boundary = Embedding::compose(&first_boundary, first.into_pasted);
    let second_boundary = Embedding::compose(&second_boundary, second.into_pasted);

    let expected = if first_boundary.is_closed() && second_boundary.is_closed() {
        Some(Embedding::union(&first_boundary, &second_boundary).into_codomain)
    } else {
        None
    };
    let holds = actual.is_closed()
        && expected
            .as_ref()
            .is_some_and(|expected| Embedding::equal(&actual, expected));

    EquationCheck {
        kind: EquationKind::Transverse,
        direction,
        sign,
        holds,
        actual,
        expected,
        expected_parts: Some([first_boundary, second_boundary]),
    }
}

fn print_statistics(
    statistics: &Statistics,
    direction_count: usize,
    cubularity_mode: CubularityMode,
) {
    println!("sampled compatible ordered pairs: {}", statistics.pairs);
    println!("individual gluings checked: {}", statistics.gluings);
    println!(
        "gluings failing {:?} cubularity: {} ({:.4}%)",
        cubularity_mode,
        statistics.cubularity_failures,
        percentage(statistics.cubularity_failures, statistics.gluings)
    );
    println!(
        "gluings with at least one failure: {} ({:.4}%)",
        statistics.failing_gluings,
        percentage(statistics.failing_gluings, statistics.gluings)
    );
    for sign in [Sign::Input, Sign::Output] {
        let failures = statistics.axial_failures[sign_index(sign)];
        println!(
            "axial {} failures: {failures} ({:.4}%)",
            sign_name(sign),
            percentage(failures, statistics.gluings)
        );
    }
    let transverse_checks = statistics
        .gluings
        .saturating_mul(direction_count.saturating_sub(1) as u128);
    for sign in [Sign::Input, Sign::Output] {
        let failures = statistics.transverse_failures[sign_index(sign)];
        println!(
            "transverse {} failures: {failures} ({:.4}%)",
            sign_name(sign),
            percentage(failures, transverse_checks)
        );
    }

    let empty_intersections = statistics.pairs - statistics.nonempty_axial_boundary_intersections;
    println!(
        "input/output gluing-boundary intersections: {} empty ({:.4}%); {} nonempty ({:.4}%)",
        empty_intersections,
        percentage(empty_intersections, statistics.pairs),
        statistics.nonempty_axial_boundary_intersections,
        percentage(
            statistics.nonempty_axial_boundary_intersections,
            statistics.pairs
        )
    );
    for (&dimension, &count) in &statistics.axial_boundary_intersection_dimensions {
        println!(
            "  nonempty intersection with {dimension} active directions: {count} ({:.4}% of \
             nonempty intersections)",
            percentage(count, statistics.nonempty_axial_boundary_intersections)
        );
    }
}

fn write_failure(options: Options, failure: &FailureWitness) -> io::Result<PathBuf> {
    let output_dir = unique_failure_directory(options.seed, failure.pair + 1)?;
    write_shape(&output_dir, "first", &failure.first)?;
    write_shape(&output_dir, "second", &failure.second)?;
    write_shape(&output_dir, "pushout", &failure.pushout.tip)?;
    write_embedding(&output_dir, "first_output_boundary", &failure.first_output)?;
    write_embedding(&output_dir, "second_input_boundary", &failure.second_input)?;
    write_embedding(
        &output_dir,
        "boundary_isomorphism",
        &failure.boundary_isomorphism,
    )?;
    write_embedding(&output_dir, "first_into_pushout", &failure.pushout.inl)?;
    write_embedding(&output_dir, "second_into_pushout", &failure.pushout.inr)?;

    let mut equations = Vec::new();
    for check in &failure.checks {
        let name = equation_name(check);
        equations.push(serde_json::json!({
            "name": name,
            "holds": check.holds,
            "actual_closed": check.actual.is_closed(),
            "expected_closed": check.expected.as_ref().is_some_and(Embedding::is_closed),
        }));
        if check.holds {
            continue;
        }

        write_embedding(&output_dir, &format!("{name}_actual"), &check.actual)?;
        if let Some(expected) = &check.expected {
            write_embedding(&output_dir, &format!("{name}_expected"), expected)?;
        } else if let Some([first_part, second_part]) = &check.expected_parts {
            write_embedding(
                &output_dir,
                &format!("{name}_expected_first_part"),
                first_part,
            )?;
            write_embedding(
                &output_dir,
                &format!("{name}_expected_second_part"),
                second_part,
            )?;
        }
    }

    let report = serde_json::json!({
        "seed": format!("{:#018x}", options.seed),
        "cubularity_mode": format!("{:?}", options.cubularity_mode),
        "sampled_pair": failure.pair + 1,
        "gluing_direction": failure.gluing_direction,
        "first_shape": failure.first_shape,
        "second_shape": failure.second_shape,
        "isomorphism": failure.isomorphism + 1,
        "glued_passes_cubularity": failure.glued_passes_cubularity,
        "boundary_isomorphism_map": failure.boundary_isomorphism.map,
        "equations": equations,
    });
    fs::write(
        output_dir.join("report.json"),
        format!(
            "{}\n",
            serde_json::to_string_pretty(&report).map_err(io::Error::other)?
        ),
    )?;
    Ok(output_dir)
}

fn equation_name(check: &EquationCheck) -> String {
    match check.kind {
        EquationKind::Axial => format!("axial_{}", sign_name(check.sign)),
        EquationKind::Transverse => {
            format!(
                "transverse_direction_{}_{}",
                check.direction,
                sign_name(check.sign)
            )
        }
    }
}

fn write_shape(output_dir: &Path, name: &str, shape: &FramedPoset) -> io::Result<()> {
    fs::write(
        output_dir.join(format!("{name}.ofp.json")),
        format!(
            "{}\n",
            serde_json::to_string_pretty(shape).map_err(io::Error::other)?
        ),
    )?;
    write_dot_variants(output_dir, name, |renderer| to_dot(shape, renderer))
}

fn write_embedding(output_dir: &Path, name: &str, embedding: &Embedding) -> io::Result<()> {
    write_dot_variants(output_dir, name, |renderer| {
        embedding_to_dot(embedding, renderer)
    })
}

fn write_dot_variants(
    output_dir: &Path,
    name: &str,
    render: impl Fn(Renderer) -> String,
) -> io::Result<()> {
    for (renderer_name, renderer) in [
        ("graded", Renderer::Ranked),
        ("compass_spring", Renderer::CompassSpring),
    ] {
        fs::write(
            output_dir.join(format!("{name}_{renderer_name}.dot")),
            render(renderer),
        )?;
    }
    Ok(())
}

fn unique_failure_directory(seed: u64, pair: usize) -> io::Result<PathBuf> {
    let root = Path::new(FAILURE_ROOT);
    fs::create_dir_all(root)?;

    for suffix in 0usize.. {
        let suffix = if suffix == 0 {
            String::new()
        } else {
            format!("_{suffix}")
        };
        let path = root.join(format!("seed_{seed:016x}_pair_{pair}{suffix}"));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    unreachable!()
}

fn sign_index(sign: Sign) -> usize {
    match sign {
        Sign::Input => 0,
        Sign::Output => 1,
    }
}

fn sign_name(sign: Sign) -> &'static str {
    match sign {
        Sign::Input => "input",
        Sign::Output => "output",
    }
}

fn percentage(part: u128, whole: u128) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

fn invalid_input(error: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, error.into())
}

fn invalid_data(error: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.into())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct CubeCell(Vec<Option<bool>>);

    #[test]
    fn parallel_generation_is_independent_of_worker_count() {
        let options = Options {
            cell_count: 8,
            dimension: 3,
            shape_count: 1,
            pair_count: 1,
            worker_count: 1,
            cubularity_mode: CubularityMode::Regular,
            seed: 1,
        };
        let generator = RandomFramedPosetGenerator::new(options.dimension, options.cell_count);
        let serial = generate_candidate_batch(options, &generator, 0, 512);
        let parallel = generate_candidate_batch(
            Options {
                worker_count: 4,
                ..options
            },
            &generator,
            0,
            512,
        );

        assert!(!serial.is_empty());
        assert_eq!(serial.len(), parallel.len());
        for ((serial_ticket, serial_shape), (parallel_ticket, parallel_shape)) in
            serial.iter().zip(&parallel)
        {
            assert_eq!(serial_ticket, parallel_ticket);
            assert!(FramedPoset::equal(serial_shape, parallel_shape));
        }
    }

    #[test]
    fn standard_three_cube_pastings_satisfy_every_formula() {
        let first = standard_cube(3);
        let second = standard_cube(3);

        for direction in 0..3 {
            let (output_domain, output_into_first) = boundary(Sign::Output, direction, &first);
            let (input_domain, input_into_second) = boundary(Sign::Input, direction, &second);
            assert_eq!(
                axial_boundary_intersection_dimension(&first, direction, &output_into_first,),
                None
            );
            let boundary_isomorphisms = isomorphisms(&output_domain, &input_domain);
            assert_eq!(boundary_isomorphisms.len(), 1);

            let output_into_second =
                Embedding::compose(&boundary_isomorphisms[0], &input_into_second);
            let pasted = pushout(&output_into_first, &output_into_second);
            assert!(is_cubular(CubularityMode::Regular, &pasted.tip));
            let checks = check_formulas(3, &first, &second, direction, &pasted);
            let failures: Vec<_> = checks
                .iter()
                .filter(|check| !check.holds)
                .map(|check| (check.kind, check.direction, check.sign))
                .collect();

            assert_eq!(checks.len(), 6);
            assert!(
                failures.is_empty(),
                "direction {direction} failed equations {failures:?}"
            );
        }
    }

    fn standard_cube(dimension: usize) -> Arc<FramedPoset> {
        let mut levels = vec![Vec::new(); dimension + 1];
        for code in 0..3usize.pow(dimension as u32) {
            let mut code = code;
            let mut coordinates = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                coordinates.push(match code % 3 {
                    0 => None,
                    1 => Some(false),
                    2 => Some(true),
                    _ => unreachable!(),
                });
                code /= 3;
            }
            let cell = CubeCell(coordinates);
            let dim = cell
                .0
                .iter()
                .filter(|coordinate| coordinate.is_none())
                .count();
            levels[dim].push(cell);
        }

        let index: HashMap<CubeCell, usize> = levels
            .iter()
            .flat_map(|level| {
                level
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(position, cell)| (cell, position))
            })
            .collect();
        let basis = levels
            .iter()
            .map(|level| {
                level
                    .iter()
                    .map(|cell| {
                        cell.0
                            .iter()
                            .enumerate()
                            .filter_map(|(direction, coordinate)| {
                                coordinate.is_none().then_some(direction)
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let mut faces_in: Vec<Vec<Vec<usize>>> = levels
            .iter()
            .map(|level| vec![vec![]; level.len()])
            .collect();
        let mut faces_out = faces_in.clone();

        for dim in 1..=dimension {
            for (position, cell) in levels[dim].iter().enumerate() {
                for direction in cell
                    .0
                    .iter()
                    .enumerate()
                    .filter_map(|(direction, coordinate)| coordinate.is_none().then_some(direction))
                {
                    let mut input = cell.clone();
                    input.0[direction] = Some(false);
                    let mut output = cell.clone();
                    output.0[direction] = Some(true);
                    faces_in[dim][position].push(index[&input]);
                    faces_out[dim][position].push(index[&output]);
                }
                faces_in[dim][position].sort_unstable();
                faces_out[dim][position].sort_unstable();
            }
        }

        Arc::new(FramedPoset::from_faces(basis, faces_in, faces_out))
    }
}
