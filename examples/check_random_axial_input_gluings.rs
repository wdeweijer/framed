use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;

use ofposets::pushout::pushout;
use ofposets::{BoundaryMode, boundary};
use ofposets::{
    Embedding, FramedPoset, RandomFramedPosetGenerator, Renderer, Sign, embedding_to_dot,
    isomorphisms, normalize, to_dot,
};
use rand::rngs::{OsRng, SmallRng};
use rand::{Rng, SeedableRng, TryRngCore};

const OUTPUT_DIR: &str = "visualizations/transverse_noncubular_gluing_failure/";
const CELL_COUNT: usize = 4;
const SHAPE_COUNT: usize = 1_000;
const PAIR_SAMPLE_COUNT: usize = 1_000;
const MAX_ISOMORPHISMS: usize = 10;

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
struct Pair {
    class: usize,
    output: usize,
    input: usize,
}

#[derive(Debug, Clone, Copy)]
struct Failure {
    pair: usize,
    sampled: Pair,
    direction: usize,
    first: usize,
    second: usize,
    isomorphism: usize,
}

fn main() -> io::Result<()> {
    let seed = OsRng.try_next_u64().map_err(io::Error::other)?;
    let mut rng = SmallRng::seed_from_u64(seed);
    let generator = RandomFramedPosetGenerator::new(2, CELL_COUNT);
    let shapes: Vec<_> = (0..SHAPE_COUNT)
        .map(|_| Arc::new(generator.generate(&mut rng)))
        .collect();
    let mut boundary_index = build_boundary_index(&shapes)?;

    for class in &mut boundary_index.classes {
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

    let sample_count = PAIR_SAMPLE_COUNT
        .min(usize::try_from(boundary_index.total_pairs).unwrap_or(PAIR_SAMPLE_COUNT));
    let pairs = sample_pairs(&boundary_index, sample_count, &mut rng);
    let mut gluing_count = 0usize;
    let mut failing_gluings = 0usize;
    let mut failing_pairs = 0usize;
    let mut first_failure = None;

    for (pair_number, pair) in pairs.into_iter().enumerate() {
        let class = &boundary_index.classes[pair.class];
        let output = &class.outputs[pair.output];
        let input = &class.inputs[pair.input];
        let first = &shapes[output.shape];
        let from_canonical = input.to_canonical.inverse_isomorphism();
        let (_, input_into_first) =
            boundary(BoundaryMode::Hat, Sign::Input, class.direction, first);
        let mut pair_failed = false;

        for (isomorphism, automorphism) in class.automorphisms.iter().enumerate() {
            let through_automorphism = Embedding::compose(&output.to_canonical, automorphism);
            let boundary_isomorphism = Embedding::compose(&through_automorphism, &from_canonical);
            let into_second = Embedding::compose(&boundary_isomorphism, &input.into_shape);
            let pasted = pushout(&output.into_shape, &into_second);

            let (_, actual) =
                boundary(BoundaryMode::Hat, Sign::Input, class.direction, &pasted.tip);
            let expected = Embedding::compose(&input_into_first, &pasted.inl);
            let holds = Embedding::equal(&actual, &expected);

            gluing_count += 1;
            if !holds {
                failing_gluings += 1;
                pair_failed = true;
                first_failure.get_or_insert(Failure {
                    pair: pair_number,
                    sampled: pair,
                    direction: class.direction,
                    first: output.shape,
                    second: input.shape,
                    isomorphism,
                });
            }
        }

        failing_pairs += usize::from(pair_failed);
    }

    println!(
        "generated {SHAPE_COUNT} unrestricted {CELL_COUNT}-cell two-dimensional OFPs \
         with seed {seed:#018x}"
    );
    println!(
        "indexed {} compatible boundary classes containing {} ordered pairs",
        boundary_index.classes.len(),
        boundary_index.total_pairs
    );
    println!("sampled {sample_count} distinct compatible ordered pairs");
    println!(
        "checked {gluing_count} gluings using at most {MAX_ISOMORPHISMS} isomorphisms per pair"
    );
    println!(
        "pairs with an axial input failure: {failing_pairs} ({:.2}%)",
        percentage(failing_pairs, sample_count)
    );
    println!(
        "individual axial input failures: {failing_gluings} ({:.2}%)",
        percentage(failing_gluings, gluing_count)
    );
    if let Some(failure) = first_failure {
        println!(
            "first failure: sampled pair {}; direction {}; first OFP {}; second OFP {}; \
             isomorphism {}",
            failure.pair + 1,
            failure.direction,
            failure.first,
            failure.second,
            failure.isomorphism + 1
        );

        let output_dir = Path::new(OUTPUT_DIR);
        fs::create_dir_all(output_dir)?;
        write_failure(output_dir, &shapes, &boundary_index, failure)?;
        println!("wrote failure diagrams to {}", output_dir.display());
    } else {
        println!("no axial input failures found");
    }

    Ok(())
}

fn build_boundary_index(shapes: &[Arc<FramedPoset>]) -> io::Result<BoundaryIndex> {
    let mut classes = Vec::<BoundaryClass>::new();
    let mut class_indices = HashMap::<(usize, Arc<FramedPoset>), usize>::new();
    let mut transports = HashMap::<Arc<FramedPoset>, Embedding>::new();

    for (shape, poset) in shapes.iter().enumerate() {
        for direction in 0..2 {
            for sign in [Sign::Input, Sign::Output] {
                let (boundary, into_shape) = boundary(BoundaryMode::Hat, sign, direction, poset);
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
            "generated OFPs have no compatible boundary pairs",
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

fn sample_pairs<R: Rng + ?Sized>(
    index: &BoundaryIndex,
    sample_count: usize,
    rng: &mut R,
) -> Vec<Pair> {
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
        let output = usize::try_from(local / input_count)
            .expect("output index comes from a usize-sized collection");
        let input = usize::try_from(local % input_count)
            .expect("input index comes from a usize-sized collection");
        let pair = Pair {
            class: class_index,
            output,
            input,
        };
        if seen.insert(pair) {
            pairs.push(pair);
        }
    }
    pairs
}

fn percentage(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

fn invalid_data(error: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.into())
}

fn write_failure(
    output_dir: &Path,
    shapes: &[Arc<FramedPoset>],
    boundary_index: &BoundaryIndex,
    failure: Failure,
) -> io::Result<()> {
    let class = &boundary_index.classes[failure.sampled.class];
    let output = &class.outputs[failure.sampled.output];
    let input = &class.inputs[failure.sampled.input];
    let first = &shapes[output.shape];
    let second = &shapes[input.shape];
    let automorphism = &class.automorphisms[failure.isomorphism];

    let from_canonical = input.to_canonical.inverse_isomorphism();
    let through_automorphism = Embedding::compose(&output.to_canonical, automorphism);
    let boundary_isomorphism = Embedding::compose(&through_automorphism, &from_canonical);
    let glued_boundary_into_second = Embedding::compose(&boundary_isomorphism, &input.into_shape);
    let pasted = pushout(&output.into_shape, &glued_boundary_into_second);

    let (_, actual) = boundary(BoundaryMode::Hat, Sign::Input, class.direction, &pasted.tip);
    let (_, input_into_first) = boundary(BoundaryMode::Hat, Sign::Input, class.direction, first);
    let expected = Embedding::compose(&input_into_first, &pasted.inl);
    debug_assert!(!Embedding::equal(&actual, &expected));

    write_shape(output_dir, "first_shape", first)?;
    write_shape(output_dir, "second_shape", second)?;
    write_shape(output_dir, "pushout_shape", &pasted.tip)?;
    write_embedding(output_dir, "first_output_boundary", &output.into_shape)?;
    write_embedding(output_dir, "second_input_boundary", &input.into_shape)?;
    write_embedding(output_dir, "boundary_isomorphism", &boundary_isomorphism)?;
    write_embedding(
        output_dir,
        "glued_boundary_into_second",
        &glued_boundary_into_second,
    )?;
    write_embedding(output_dir, "first_into_pushout", &pasted.inl)?;
    write_embedding(output_dir, "second_into_pushout", &pasted.inr)?;
    write_embedding(output_dir, "actual_pushout_input_boundary", &actual)?;
    write_embedding(output_dir, "expected_first_input_boundary", &expected)
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
