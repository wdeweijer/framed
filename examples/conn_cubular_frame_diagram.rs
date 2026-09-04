use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;

use ofposets::{
    CubularityMode, Embedding, FramedPoset, Renderer, Sign, boundary, embedding_to_dot, is_cubular,
    to_dot,
};

const OUTPUT_DIR: &str = "visualizations/conn_cubular_frame";
const SOURCE_JSON: &str = r#"{"version":1,"basis":[[[],[],[]],[[0],[0],[1],[1],[1],[2],[2]],[[0,1],[1,2]]],"faces_in":[[[],[],[]],[[0],[0],[1],[1],[1],[2],[2]],[[1,2],[4,5]]],"faces_out":[[[],[],[]],[[],[],[],[],[],[],[]],[[],[]]]}"#;

#[derive(Clone)]
struct BoundaryState {
    word: Vec<(Sign, usize)>,
    into_source: Embedding,
}

fn main() -> io::Result<()> {
    let output_dir = Path::new(OUTPUT_DIR);
    fs::create_dir_all(output_dir)?;

    let source =
        Arc::new(serde_json::from_str::<FramedPoset>(SOURCE_JSON).map_err(io::Error::other)?);
    let total_frame = source.total_frame();
    let dimension = usize::try_from(source.dim()).expect("the counterexample is non-empty");

    assert!(
        source.is_connected(),
        "the counterexample must be connected"
    );
    assert!(
        is_cubular(CubularityMode::Strong, &source,),
        "the counterexample must be strongly cubular"
    );
    assert_ne!(
        total_frame.len(),
        dimension,
        "the total-frame cardinality must differ from the dimension"
    );
    assert_eq!(total_frame, vec![0, 1, 2]);
    assert_eq!(dimension, 2);

    write_shape_layouts(output_dir, "source", &source)?;
    fs::write(
        output_dir.join("source.ofp.json"),
        format!(
            "{}\n",
            serde_json::to_string_pretty(source.as_ref()).map_err(io::Error::other)?
        ),
    )?;

    let states = boundary_states(&source, &total_frame);
    assert_eq!(states.len(), 3_usize.pow(total_frame.len() as u32) - 1);

    let mut intersection_count = 0;
    for state in &states {
        let label = boundary_word_name(&state.word);
        write_embedding_layouts(output_dir, &format!("{label}_iterated"), &state.into_source)?;

        if state.word.len() >= 2 {
            let intersection = direct_boundary_intersection(&source, &state.word);
            assert!(
                Embedding::same_subobject(&state.into_source, &intersection),
                "iterated boundary {label} differs from its direct-boundary intersection"
            );
            write_embedding_layouts(output_dir, &format!("{label}_intersection"), &intersection)?;
            intersection_count += 1;
        }
    }

    assert_eq!(intersection_count, 20);
    println!(
        "verified a connected, strongly cubular OFP with total frame {total_frame:?} and dimension {dimension}"
    );
    println!(
        "wrote {} iterated boundaries and {intersection_count} intersections to {}",
        states.len(),
        output_dir.display()
    );
    Ok(())
}

/// Enumerate one representative for every non-empty signed subset of the
/// total frame. Directions are applied in increasing order; strong cubularity says
/// that every other ordering gives the same embedding into the source.
fn boundary_states(source: &Arc<FramedPoset>, directions: &[usize]) -> Vec<BoundaryState> {
    let mut states = Vec::new();
    let mut word = Vec::new();
    collect_boundary_states(
        directions,
        0,
        source,
        &Embedding::id(Arc::clone(source)),
        &mut word,
        &mut states,
    );
    states
}

fn collect_boundary_states(
    directions: &[usize],
    next: usize,
    current: &Arc<FramedPoset>,
    into_source: &Embedding,
    word: &mut Vec<(Sign, usize)>,
    states: &mut Vec<BoundaryState>,
) {
    if next == directions.len() {
        if !word.is_empty() {
            states.push(BoundaryState {
                word: word.clone(),
                into_source: into_source.clone(),
            });
        }
        return;
    }

    collect_boundary_states(directions, next + 1, current, into_source, word, states);

    let direction = directions[next];
    for sign in [Sign::Input, Sign::Output] {
        let (domain, into_current) = boundary(sign, direction, current);
        let next_into_source = Embedding::compose(&into_current, into_source);

        word.push((sign, direction));
        collect_boundary_states(
            directions,
            next + 1,
            &domain,
            &next_into_source,
            word,
            states,
        );
        word.pop();
    }
}

/// Intersect the one-step source boundaries named by `word`.
fn direct_boundary_intersection(source: &Arc<FramedPoset>, word: &[(Sign, usize)]) -> Embedding {
    let mut boundaries = word
        .iter()
        .map(|&(sign, direction)| boundary(sign, direction, source).1);
    let mut intersection = boundaries
        .next()
        .expect("a boundary word passed to intersection must be non-empty");

    for next in boundaries {
        intersection = Embedding::intersection(&intersection, &next).into_codomain;
    }
    intersection
}

fn write_shape_layouts(output_dir: &Path, name: &str, shape: &FramedPoset) -> io::Result<()> {
    for renderer in [Renderer::Ranked] {
        fs::write(
            output_dir.join(format!("{name}_{}.dot", renderer_name(renderer))),
            to_dot(shape, renderer),
        )?;
    }
    Ok(())
}

fn write_embedding_layouts(output_dir: &Path, name: &str, embedding: &Embedding) -> io::Result<()> {
    for renderer in [Renderer::Ranked] {
        fs::write(
            output_dir.join(format!("{name}_{}.dot", renderer_name(renderer))),
            embedding_to_dot(embedding, renderer),
        )?;
    }
    Ok(())
}

fn boundary_word_name(word: &[(Sign, usize)]) -> String {
    word.iter()
        .map(|&(sign, direction)| format!("{}_{}", sign_name(sign), direction))
        .collect::<Vec<_>>()
        .join("__")
}

fn sign_name(sign: Sign) -> &'static str {
    match sign {
        Sign::Input => "minus",
        Sign::Output => "plus",
    }
}

fn renderer_name(renderer: Renderer) -> &'static str {
    match renderer {
        Renderer::Ranked => "graded",
        Renderer::CompassSpring => "compass_spring",
    }
}
