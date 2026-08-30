use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use ofposets::embedding::{Embedding, NO_PREIMAGE};
use ofposets::intset::{self, IntSet};
use ofposets::isomorphism::isomorphisms;
use ofposets::orthogonal::orthogonal_product;
use ofposets::poset::{FramedPoset, Sign, boundary};
use ofposets::pushout::{MultiPushout, Pushout, Span, paste_along_boundary, pushout};

const SAMPLES: usize = 9;

fn main() {
    println!("binary pushout with precomputed boundary embeddings");
    println!("median of {SAMPLES} alternating batches");
    println!("dim\tcells/object\titerations\tlegacy\tfinite_colimit\tratio");

    for dimension in 1..=6 {
        let (left, right) = boundary_span(dimension);
        assert_same_result(&left, &right);

        let iterations = pushout_iterations(dimension);
        let (legacy_ns, current_ns) = compare_operations(
            iterations,
            || legacy_pushout(&left, &right),
            || pushout(&left, &right),
        );

        println!(
            "{dimension}\t{}\t{iterations}\t{}\t{}\t{:.3}x",
            left.cod.sizes().iter().sum::<usize>(),
            format_time(legacy_ns),
            format_time(current_ns),
            current_ns / legacy_ns,
        );
    }

    println!();
    println!("end-to-end paste_along_boundary");
    println!("median of {SAMPLES} alternating batches");
    println!("dim\tcells/object\titerations\tlegacy\tfinite_colimit\tratio");

    for dimension in 1..=6 {
        let first = standard_cube(dimension);
        let second = standard_cube(dimension);
        let legacy = legacy_paste_along_boundary(&first, &second, 0);
        let current = paste_along_boundary(&first, &second, 0);
        assert_pushouts_equal(&legacy, &current);

        let iterations = pasting_iterations(dimension);
        let (legacy_ns, current_ns) = compare_operations(
            iterations,
            || legacy_paste_along_boundary(&first, &second, 0),
            || paste_along_boundary(&first, &second, 0),
        );

        println!(
            "{dimension}\t{}\t{iterations}\t{}\t{}\t{:.3}x",
            first.sizes().iter().sum::<usize>(),
            format_time(legacy_ns),
            format_time(current_ns),
            current_ns / legacy_ns,
        );
    }
}

fn compare_operations(
    iterations: usize,
    mut legacy_operation: impl FnMut() -> Pushout,
    mut current_operation: impl FnMut() -> Pushout,
) -> (f64, f64) {
    for _ in 0..(iterations / 20).max(5) {
        black_box(legacy_operation());
        black_box(current_operation());
    }

    let mut legacy = Vec::with_capacity(SAMPLES);
    let mut current = Vec::with_capacity(SAMPLES);

    for sample in 0..SAMPLES {
        if sample % 2 == 0 {
            legacy.push(time_batch(iterations, &mut legacy_operation));
            current.push(time_batch(iterations, &mut current_operation));
        } else {
            current.push(time_batch(iterations, &mut current_operation));
            legacy.push(time_batch(iterations, &mut legacy_operation));
        }
    }

    (median(&mut legacy), median(&mut current))
}

fn time_batch(iterations: usize, mut operation: impl FnMut() -> Pushout) -> f64 {
    let started = Instant::now();
    for _ in 0..iterations {
        black_box(operation());
    }
    started.elapsed().as_secs_f64() * 1_000_000_000.0 / iterations as f64
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_unstable_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn format_time(nanoseconds: f64) -> String {
    if nanoseconds < 1_000.0 {
        format!("{nanoseconds:.1} ns")
    } else if nanoseconds < 1_000_000.0 {
        format!("{:.2} us", nanoseconds / 1_000.0)
    } else {
        format!("{:.2} ms", nanoseconds / 1_000_000.0)
    }
}

fn pushout_iterations(dimension: usize) -> usize {
    match dimension {
        1 => 50_000,
        2 => 20_000,
        3 => 5_000,
        4 => 1_000,
        5 => 200,
        6 => 50,
        _ => unreachable!(),
    }
}

fn pasting_iterations(dimension: usize) -> usize {
    match dimension {
        1 => 2_000,
        2 => 1_000,
        3 => 300,
        4 => 100,
        5 => 30,
        6 => 10,
        _ => unreachable!(),
    }
}

fn boundary_span(dimension: usize) -> (Embedding, Embedding) {
    let first = standard_cube(dimension);
    let second = standard_cube(dimension);
    let (output, into_first) = boundary(Sign::Output, 0, &first);
    let (input, into_second) = boundary(Sign::Input, 0, &second);
    let mut candidates = isomorphisms(&output, &input);
    assert_eq!(candidates.len(), 1);
    let into_second = Embedding::compose(&candidates.pop().unwrap(), &into_second);
    (into_first, into_second)
}

fn standard_cube(dimension: usize) -> Arc<FramedPoset> {
    let mut cube = FramedPoset::point();
    for direction in 0..dimension {
        cube = orthogonal_product(&cube, &tight_arrow(direction));
    }
    Arc::new(cube)
}

fn tight_arrow(direction: usize) -> FramedPoset {
    FramedPoset::from_faces(
        vec![vec![vec![], vec![]], vec![vec![direction]]],
        vec![vec![vec![], vec![]], vec![vec![0]]],
        vec![vec![vec![], vec![]], vec![vec![1]]],
    )
}

fn assert_same_result(left: &Embedding, right: &Embedding) {
    let legacy = legacy_pushout(left, right);
    let current = pushout(left, right);
    assert_pushouts_equal(&legacy, &current);
}

fn assert_pushouts_equal(legacy: &Pushout, current: &Pushout) {
    assert!(FramedPoset::equal(&legacy.tip, &current.tip));
    assert!(Embedding::equal_as_morphisms(&legacy.inl, &current.inl));
    assert!(Embedding::equal_as_morphisms(&legacy.inr, &current.inr));
}

fn legacy_paste_along_boundary(
    first: &Arc<FramedPoset>,
    second: &Arc<FramedPoset>,
    direction: usize,
) -> Pushout {
    let (output_boundary, output_into_first) = boundary(Sign::Output, direction, first);
    let (input_boundary, input_into_second) = boundary(Sign::Input, direction, second);
    let mut candidates = isomorphisms(&output_boundary, &input_boundary);
    assert_eq!(candidates.len(), 1);
    let output_into_second = Embedding::compose(&candidates.pop().unwrap(), &input_into_second);
    legacy_pushout(&output_into_first, &output_into_second)
}

// Pre-finite-colimit algorithm, with raw-table cloning expressed through the
// public cell accessors so it can live in an example crate.
fn legacy_pushout(f: &Embedding, g: &Embedding) -> Pushout {
    let size_sum = |shape: &FramedPoset| shape.sizes().iter().sum::<usize>();
    let (base_embedding, extension_embedding, swapped) = if size_sum(&f.cod) >= size_sum(&g.cod) {
        (f, g, false)
    } else {
        (g, f, true)
    };
    let result = legacy_multi_pushout(
        &base_embedding.cod,
        &[Span {
            into_base: base_embedding,
            into_ext: extension_embedding,
        }],
    );
    let inr = result.inrs.into_iter().next().unwrap();

    if swapped {
        Pushout {
            tip: result.tip,
            inl: inr,
            inr: result.inl,
        }
    } else {
        Pushout {
            tip: result.tip,
            inl: result.inl,
            inr,
        }
    }
}

fn legacy_multi_pushout(base: &Arc<FramedPoset>, spans: &[Span<'_>]) -> MultiPushout {
    let base_sizes = base.sizes();
    let tip_dimension = spans
        .iter()
        .map(|span| span.into_ext.cod.dim())
        .fold(base.dim(), isize::max);
    let levels = if tip_dimension < 0 {
        0
    } else {
        tip_dimension as usize + 1
    };
    let base_level_sizes: Vec<usize> = (0..levels)
        .map(|dimension| base_sizes.get(dimension).copied().unwrap_or(0))
        .collect();

    let mut extra_counts = vec![0usize; levels];
    for span in spans {
        let extension_sizes = span.into_ext.cod.sizes();
        for dimension in 0..extension_sizes.len().min(levels) {
            for position in 0..extension_sizes[dimension] {
                if preimage_at(&span.into_ext.inv, dimension, position) == NO_PREIMAGE {
                    extra_counts[dimension] += 1;
                }
            }
        }
    }
    let total_sizes: Vec<usize> = (0..levels)
        .map(|dimension| base_level_sizes[dimension] + extra_counts[dimension])
        .collect();

    let mut tip_basis = alloc_rows(
        base,
        &base_sizes,
        &total_sizes,
        |shape, dimension, position| shape.basis_of(dimension, position).clone(),
    );
    let mut tip_faces_in = alloc_rows(
        base,
        &base_sizes,
        &total_sizes,
        |shape, dimension, position| shape.faces_of(Sign::Input, dimension, position).clone(),
    );
    let mut tip_faces_out = alloc_rows(
        base,
        &base_sizes,
        &total_sizes,
        |shape, dimension, position| shape.faces_of(Sign::Output, dimension, position).clone(),
    );
    let mut tip_cofaces_in = alloc_rows(
        base,
        &base_sizes,
        &total_sizes,
        |shape, dimension, position| shape.cofaces_of(Sign::Input, dimension, position).clone(),
    );
    let mut tip_cofaces_out = alloc_rows(
        base,
        &base_sizes,
        &total_sizes,
        |shape, dimension, position| shape.cofaces_of(Sign::Output, dimension, position).clone(),
    );
    let mut counters = base_level_sizes.clone();
    let mut inr_data = Vec::with_capacity(spans.len());

    for span in spans {
        let extension = &span.into_ext.cod;
        let extension_sizes = extension.sizes();
        let mut map: Vec<Vec<usize>> = extension_sizes.iter().map(|&size| vec![0; size]).collect();
        let mut inv: Vec<Vec<usize>> = total_sizes
            .iter()
            .map(|&size| vec![NO_PREIMAGE; size])
            .collect();

        for dimension in 0..extension_sizes.len().min(levels) {
            for position in 0..extension_sizes[dimension] {
                let preimage = preimage_at(&span.into_ext.inv, dimension, position);
                if preimage != NO_PREIMAGE {
                    let target = span.into_base.map[dimension][preimage];
                    map[dimension][position] = target;
                    inv[dimension][target] = position;
                    continue;
                }

                let target = counters[dimension];
                counters[dimension] += 1;
                map[dimension][position] = target;
                inv[dimension][target] = position;
                tip_basis[dimension][target] = extension.basis_of(dimension, position).clone();

                if dimension > 0 {
                    let input_faces = intset::collect_sorted(
                        extension
                            .faces_of(Sign::Input, dimension, position)
                            .iter()
                            .map(|&face| map[dimension - 1][face]),
                    );
                    let output_faces = intset::collect_sorted(
                        extension
                            .faces_of(Sign::Output, dimension, position)
                            .iter()
                            .map(|&face| map[dimension - 1][face]),
                    );
                    for &face in &input_faces {
                        intset::insert(&mut tip_cofaces_in[dimension - 1][face], target);
                    }
                    for &face in &output_faces {
                        intset::insert(&mut tip_cofaces_out[dimension - 1][face], target);
                    }
                    tip_faces_in[dimension][target] = input_faces;
                    tip_faces_out[dimension][target] = output_faces;
                }
            }
        }
        inr_data.push((map, inv));
    }

    let tip = Arc::new(FramedPoset::make(
        tip_basis,
        tip_faces_in,
        tip_faces_out,
        tip_cofaces_in,
        tip_cofaces_out,
    ));
    let inl_map: Vec<Vec<usize>> = base_sizes.iter().map(|&size| (0..size).collect()).collect();
    let inl_inv: Vec<Vec<usize>> = (0..levels)
        .map(|dimension| {
            let mut row = vec![NO_PREIMAGE; total_sizes[dimension]];
            for (position, value) in row.iter_mut().enumerate().take(base_level_sizes[dimension]) {
                *value = position;
            }
            row
        })
        .collect();
    let inl = Embedding::make(Arc::clone(base), Arc::clone(&tip), inl_map, inl_inv);
    let inrs = spans
        .iter()
        .zip(inr_data)
        .map(|(span, (map, inv))| {
            Embedding::make(Arc::clone(&span.into_ext.cod), Arc::clone(&tip), map, inv)
        })
        .collect();

    MultiPushout { tip, inl, inrs }
}

fn preimage_at(inv: &[Vec<usize>], dimension: usize, position: usize) -> usize {
    inv.get(dimension)
        .and_then(|row| row.get(position))
        .copied()
        .unwrap_or(NO_PREIMAGE)
}

fn alloc_rows(
    base: &FramedPoset,
    base_sizes: &[usize],
    total_sizes: &[usize],
    mut value: impl FnMut(&FramedPoset, usize, usize) -> IntSet,
) -> Vec<Vec<IntSet>> {
    (0..total_sizes.len())
        .map(|dimension| {
            let mut row = vec![vec![]; total_sizes[dimension]];
            for (position, cell) in row
                .iter_mut()
                .enumerate()
                .take(base_sizes.get(dimension).copied().unwrap_or(0))
            {
                *cell = value(base, dimension, position);
            }
            row
        })
        .collect()
}
