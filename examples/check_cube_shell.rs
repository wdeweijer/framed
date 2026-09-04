use std::sync::Arc;

use ofposets::{CubularityMode, FramedPoset, RandomFramedPosetGenerator, is_cubular};
use rand::SeedableRng;
use rand::rngs::SmallRng;

fn main() {
    if let Some(missing_outer) = search_missing_outer_shadow_orientations() {
        println!(
            "FOUND MISSING-OUTER-SHADOW COUNTEREXAMPLE: sizes={:?}",
            missing_outer.sizes()
        );
        println!("{}", serde_json::to_string_pretty(&*missing_outer).unwrap());
        return;
    }

    if let Some(shadowless) = search_shadowless_orientations() {
        println!(
            "FOUND SHADOWLESS COUNTEREXAMPLE: sizes={:?}",
            shadowless.sizes()
        );
        println!("{}", serde_json::to_string_pretty(&*shadowless).unwrap());
        return;
    }

    if let Some(two_vertex) = search_two_vertex_one_sided_counterexample() {
        println!(
            "FOUND TWO-VERTEX COUNTEREXAMPLE: sizes={:?}",
            two_vertex.sizes()
        );
        println!("{}", serde_json::to_string_pretty(&*two_vertex).unwrap());
        return;
    }

    let vertex_reduced = joined_one_sided_cells_with_two_vertices();
    println!(
        "two-vertex joined cells: sizes={:?}, connected={}, strong={}",
        vertex_reduced.sizes(),
        vertex_reduced.is_connected(),
        is_cubular(CubularityMode::Strong, &vertex_reduced),
    );
    if is_cubular(CubularityMode::Strong, &vertex_reduced) {
        println!("TWO-VERTEX COUNTEREXAMPLE");
        println!(
            "{}",
            serde_json::to_string_pretty(&*vertex_reduced).unwrap()
        );
        return;
    }

    let still_smaller = joined_one_sided_cells_without_common_shadow();
    println!(
        "shadowless joined cells: sizes={:?}, connected={}, strong={}",
        still_smaller.sizes(),
        still_smaller.is_connected(),
        is_cubular(CubularityMode::Strong, &still_smaller),
    );
    if is_cubular(CubularityMode::Strong, &still_smaller) {
        println!("STILL SMALLER COUNTEREXAMPLE");
        println!("{}", serde_json::to_string_pretty(&*still_smaller).unwrap());
        return;
    }

    let smaller = joined_one_sided_cells_with_one_common_shadow();
    println!(
        "smaller joined one-sided cells: sizes={:?}, total_frame={:?}, dim={}, connected={}, strong={}",
        smaller.sizes(),
        smaller.total_frame(),
        smaller.dim(),
        smaller.is_connected(),
        is_cubular(CubularityMode::Strong, &smaller),
    );
    if is_cubular(CubularityMode::Strong, &smaller) {
        println!("SMALLER COUNTEREXAMPLE");
        println!("{}", serde_json::to_string_pretty(&*smaller).unwrap());
        return;
    }

    let candidate = joined_one_sided_cells();
    println!(
        "joined one-sided cells: sizes={:?}, total_frame={:?}, dim={}, connected={}, strong={}",
        candidate.sizes(),
        candidate.total_frame(),
        candidate.dim(),
        candidate.is_connected(),
        is_cubular(CubularityMode::Strong, &candidate),
    );
    if is_cubular(CubularityMode::Strong, &candidate) {
        println!("COUNTEREXAMPLE");
        println!("{}", serde_json::to_string_pretty(&*candidate).unwrap());
        return;
    }

    if std::env::var_os("OFP_ONLY_CYCLE").is_some() {
        let shape = cycle_bridged_squares();
        println!(
            "cycle bridge strong={}",
            is_cubular(CubularityMode::Strong, &shape)
        );
        return;
    }

    if std::env::var_os("OFP_ONLY_MATCH").is_some() {
        let shape = matched_bridged_squares([0, 1, 2, 3], 0);
        println!(
            "matched bridge strong={}",
            is_cubular(CubularityMode::Strong, &shape)
        );
        return;
    }

    let generator = RandomFramedPosetGenerator::new_without_full_frame(3, 9);
    for sample in 1..=200_000u64 {
        let mut rng = SmallRng::seed_from_u64(0x0a11ce55u64.wrapping_add(sample));
        let shape = Arc::new(generator.generate(&mut rng));
        if is_cubular(CubularityMode::Strong, &shape) {
            println!(
                "first random strong shape: sample={sample}, sizes={:?}, connected={}",
                shape.sizes(),
                shape.is_connected()
            );
            println!("{}", serde_json::to_string_pretty(&*shape).unwrap());
            break;
        }
    }

    for first_vertex in 0..4 {
        for second_vertex in 0..4 {
            for reversed in [false, true] {
                let bridge = bridged_squares(first_vertex, second_vertex, reversed);
                if is_cubular(CubularityMode::Strong, &bridge) {
                    println!(
                        "STRONG bridged squares: first={first_vertex}, second={second_vertex}, +                         reversed={reversed}"
                    );
                    println!("{}", serde_json::to_string_pretty(&*bridge).unwrap());
                    return;
                }
            }
        }
    }
    println!("all 32 one-arrow bridged-square configurations fail strong cubularity");

    for first_input in 0..4 {
        for first_output in 0..4 {
            if first_input == first_output {
                continue;
            }
            for second_input in 0..4 {
                for second_output in 0..4 {
                    if second_input == second_output {
                        continue;
                    }
                    for reversed in [false, true] {
                        let bridge = doubly_attached_bridged_squares(
                            first_input,
                            first_output,
                            second_input,
                            second_output,
                            reversed,
                        );
                        if is_cubular(CubularityMode::Strong, &bridge) {
                            println!(
                                "STRONG double bridge: first=({first_input},{first_output}), +                                 second=({second_input},{second_output}), reversed={reversed}"
                            );
                            println!("{}", serde_json::to_string_pretty(&*bridge).unwrap());
                            return;
                        }
                    }
                }
            }
        }
    }
    println!("all 288 two-endpoint bridged-square configurations fail strong cubularity");

    for first_vertex in 0..4 {
        for second_vertex in 0..4 {
            let bridge = two_sided_bridged_squares(first_vertex, second_vertex);
            if is_cubular(CubularityMode::Strong, &bridge) {
                println!("STRONG two-sided bridge: first={first_vertex}, second={second_vertex}");
                println!("{}", serde_json::to_string_pretty(&*bridge).unwrap());
                return;
            }
        }
    }
    println!("all 16 two-sided bridged-square configurations fail strong cubularity");

    let mut permutation = [0, 1, 2, 3];
    loop {
        for orientations in 0..16 {
            let shape = matched_bridged_squares(permutation, orientations);
            if is_cubular(CubularityMode::Strong, &shape) {
                println!("COUNTEREXAMPLE matching={permutation:?}, orientations={orientations:#x}");
                println!("{}", serde_json::to_string_pretty(&*shape).unwrap());
                return;
            }
        }
        if !next_permutation(&mut permutation) {
            break;
        }
    }
    println!("all 384 perfectly matched bridged-square configurations fail strong cubularity");

    let cycle = cycle_bridged_squares();
    println!(
        "cycle-bridged squares: strong={}",
        is_cubular(CubularityMode::Strong, &cycle)
    );
    if is_cubular(CubularityMode::Strong, &cycle) {
        println!("COUNTEREXAMPLE");
        println!("{}", serde_json::to_string_pretty(&*cycle).unwrap());
        return;
    }

    let profiles: [&[usize]; 8] = [
        // Counts indexed by the frame bit mask: empty, 0, 1, 01, 2, 02, 12.
        &[1, 2, 2, 1, 0, 0, 0],
        &[2, 2, 2, 1, 0, 0, 0],
        &[1, 1, 1, 1, 1, 0, 0],
        &[1, 1, 1, 1, 1, 1, 0],
        &[1, 1, 1, 1, 1, 1, 1],
        &[1, 2, 2, 1, 1, 0, 0],
        &[1, 2, 2, 1, 2, 0, 0],
        &[1, 1, 3, 1, 1, 0, 1],
    ];

    for profile in profiles {
        let (cells, groups) = incidence_groups(profile);
        let total = groups
            .iter()
            .map(|group| 3usize.pow(group.faces.len() as u32) - 1)
            .product::<usize>();
        let mut choices = vec![0; groups.len()];
        let mut checked = 0usize;
        let witness = enumerate(0, &cells, &groups, &mut choices, &mut checked);
        println!(
            "profile={profile:?}: checked {checked}/{total}; strong={}",
            witness.is_some()
        );
        if let Some(shape) = witness {
            println!("{}", serde_json::to_string_pretty(&*shape).unwrap());
            return;
        }
    }
}

fn search_missing_outer_shadow_orientations() -> Option<Arc<FramedPoset>> {
    for reflected in [false, true] {
        let frames = if reflected {
            vec![vec![0], vec![0], vec![1], vec![1], vec![1], vec![2]]
        } else {
            vec![vec![0], vec![1], vec![1], vec![1], vec![2], vec![2]]
        };
        let edge_vertices = if reflected {
            [0, 0, 1, 1, 1, 2]
        } else {
            [0, 1, 1, 1, 2, 2]
        };
        let top_faces = if reflected {
            [[0, 2], [3, 5]]
        } else {
            [[0, 1], [2, 4]]
        };
        let mut relations = Vec::new();
        for (edge, &vertex) in edge_vertices.iter().enumerate() {
            relations.push((1, edge, vertex));
        }
        for (top, faces) in top_faces.iter().enumerate() {
            for &face in faces {
                relations.push((2, top, face));
            }
        }
        for signs in 0..1usize << relations.len() {
            let mut faces = [
                vec![vec![vec![]; 3], vec![vec![]; 6], vec![vec![]; 2]],
                vec![vec![vec![]; 3], vec![vec![]; 6], vec![vec![]; 2]],
            ];
            for (relation, &(dim, cell, face)) in relations.iter().enumerate() {
                faces[(signs >> relation) & 1][dim][cell].push(face);
            }
            let shape = Arc::new(FramedPoset::from_faces(
                vec![
                    vec![vec![], vec![], vec![]],
                    frames.clone(),
                    vec![vec![0, 1], vec![1, 2]],
                ],
                faces[0].clone(),
                faces[1].clone(),
            ));
            if shape.is_connected() && is_cubular(CubularityMode::Strong, &shape) {
                return Some(shape);
            }
        }
    }
    println!("no counterexample after removing either outer shadow (2,048 orientations)");
    None
}

fn search_shadowless_orientations() -> Option<Arc<FramedPoset>> {
    let relations = [
        (1, 0, 0),
        (1, 1, 0),
        (1, 2, 1),
        (1, 3, 1),
        (1, 4, 2),
        (1, 5, 2),
        (2, 0, 0),
        (2, 0, 2),
        (2, 1, 3),
        (2, 1, 4),
    ];
    for signs in 0..1usize << relations.len() {
        let mut faces = [
            vec![vec![vec![]; 3], vec![vec![]; 6], vec![vec![]; 2]],
            vec![vec![vec![]; 3], vec![vec![]; 6], vec![vec![]; 2]],
        ];
        for (relation, &(dim, cell, face)) in relations.iter().enumerate() {
            faces[(signs >> relation) & 1][dim][cell].push(face);
        }
        let shape = Arc::new(FramedPoset::from_faces(
            vec![
                vec![vec![], vec![], vec![]],
                vec![vec![0], vec![0], vec![1], vec![1], vec![2], vec![2]],
                vec![vec![0, 1], vec![1, 2]],
            ],
            faces[0].clone(),
            faces[1].clone(),
        ));
        if shape.is_connected() && is_cubular(CubularityMode::Strong, &shape) {
            return Some(shape);
        }
    }
    println!("no shadowless counterexample among all 1,024 orientations");
    None
}

fn search_two_vertex_one_sided_counterexample() -> Option<Arc<FramedPoset>> {
    for attachments in 0..1usize << 7 {
        let edge_faces = (0..7)
            .map(|edge| vec![(attachments >> edge) & 1])
            .collect::<Vec<_>>();
        for first_zero in 0..2 {
            for first_one in 2..5 {
                for second_one in 2..5 {
                    for second_two in 5..7 {
                        let shape = Arc::new(FramedPoset::from_faces(
                            vec![
                                vec![vec![], vec![]],
                                vec![
                                    vec![0],
                                    vec![0],
                                    vec![1],
                                    vec![1],
                                    vec![1],
                                    vec![2],
                                    vec![2],
                                ],
                                vec![vec![0, 1], vec![1, 2]],
                            ],
                            vec![
                                vec![vec![]; 2],
                                edge_faces.clone(),
                                vec![vec![first_zero, first_one], vec![second_one, second_two]],
                            ],
                            vec![vec![vec![]; 2], vec![vec![]; 7], vec![vec![], vec![]]],
                        ));
                        if shape.is_connected() && is_cubular(CubularityMode::Strong, &shape) {
                            return Some(shape);
                        }
                    }
                }
            }
        }
    }
    println!("no two-vertex one-sided counterexample in 4,608 configurations");
    None
}

fn joined_one_sided_cells_with_two_vertices() -> Arc<FramedPoset> {
    Arc::new(FramedPoset::from_faces(
        vec![
            vec![vec![], vec![]],
            vec![
                vec![0],
                vec![0],
                vec![1],
                vec![1],
                vec![1],
                vec![2],
                vec![2],
            ],
            vec![vec![0, 1], vec![1, 2]],
        ],
        vec![
            vec![vec![]; 2],
            vec![
                vec![0],
                vec![0],
                vec![1],
                vec![1],
                vec![1],
                vec![0],
                vec![0],
            ],
            vec![vec![0, 2], vec![3, 5]],
        ],
        vec![vec![vec![]; 2], vec![vec![]; 7], vec![vec![], vec![]]],
    ))
}

fn joined_one_sided_cells_without_common_shadow() -> Arc<FramedPoset> {
    Arc::new(FramedPoset::from_faces(
        vec![
            vec![vec![], vec![], vec![]],
            vec![vec![0], vec![0], vec![1], vec![1], vec![2], vec![2]],
            vec![vec![0, 1], vec![1, 2]],
        ],
        vec![
            vec![vec![]; 3],
            vec![vec![0], vec![0], vec![1], vec![1], vec![2], vec![2]],
            vec![vec![0, 2], vec![3, 4]],
        ],
        vec![vec![vec![]; 3], vec![vec![]; 6], vec![vec![], vec![]]],
    ))
}

fn joined_one_sided_cells_with_one_common_shadow() -> Arc<FramedPoset> {
    Arc::new(FramedPoset::from_faces(
        vec![
            vec![vec![], vec![], vec![]],
            vec![
                vec![0],
                vec![0],
                vec![1],
                vec![1],
                vec![1],
                vec![2],
                vec![2],
            ],
            vec![vec![0, 1], vec![1, 2]],
        ],
        vec![
            vec![vec![]; 3],
            vec![
                vec![0],
                vec![0],
                vec![1],
                vec![1],
                vec![1],
                vec![2],
                vec![2],
            ],
            vec![vec![0, 2], vec![3, 5]],
        ],
        vec![vec![vec![]; 3], vec![vec![]; 7], vec![vec![], vec![]]],
    ))
}

fn joined_one_sided_cells() -> Arc<FramedPoset> {
    Arc::new(FramedPoset::from_faces(
        vec![
            vec![vec![], vec![], vec![]],
            vec![
                vec![0],
                vec![0],
                vec![1],
                vec![1],
                vec![1],
                vec![1],
                vec![2],
                vec![2],
            ],
            vec![vec![0, 1], vec![1, 2]],
        ],
        vec![
            vec![vec![]; 3],
            vec![
                vec![0],
                vec![0],
                vec![1],
                vec![1],
                vec![1],
                vec![1],
                vec![2],
                vec![2],
            ],
            vec![vec![0, 2], vec![4, 6]],
        ],
        vec![vec![vec![]; 3], vec![vec![]; 8], vec![vec![], vec![]]],
    ))
}

fn cycle_bridged_squares() -> Arc<FramedPoset> {
    let cycle = [0, 4, 1, 5, 3, 7, 2, 6];
    let bridge_inputs = cycle.into_iter().map(|vertex| vec![vertex]);
    let bridge_outputs = cycle
        .into_iter()
        .cycle()
        .skip(1)
        .take(cycle.len())
        .map(|vertex| vec![vertex]);

    Arc::new(FramedPoset::from_faces(
        vec![
            vec![vec![]; 8],
            vec![
                vec![0],
                vec![0],
                vec![1],
                vec![1],
                vec![1],
                vec![1],
                vec![2],
                vec![2],
            ]
            .into_iter()
            .chain((0..8).map(|_| vec![1]))
            .collect(),
            vec![vec![0, 1], vec![1, 2]],
        ],
        vec![
            vec![vec![]; 8],
            vec![
                vec![0],
                vec![2],
                vec![0],
                vec![1],
                vec![4],
                vec![6],
                vec![4],
                vec![5],
            ]
            .into_iter()
            .chain(bridge_inputs)
            .collect(),
            vec![vec![0, 2], vec![4, 6]],
        ],
        vec![
            vec![vec![]; 8],
            vec![
                vec![1],
                vec![3],
                vec![2],
                vec![3],
                vec![5],
                vec![7],
                vec![6],
                vec![7],
            ]
            .into_iter()
            .chain(bridge_outputs)
            .collect(),
            vec![vec![1, 3], vec![5, 7]],
        ],
    ))
}

fn matched_bridged_squares(matching: [usize; 4], orientations: usize) -> Arc<FramedPoset> {
    let mut bridge_inputs = Vec::new();
    let mut bridge_outputs = Vec::new();
    for (first, &second) in matching.iter().enumerate() {
        let endpoints = [first, 4 + second];
        bridge_inputs.push(vec![endpoints[(orientations >> first) & 1]]);
        bridge_outputs.push(vec![endpoints[((orientations >> first) & 1) ^ 1]]);
    }

    Arc::new(FramedPoset::from_faces(
        vec![
            vec![vec![]; 8],
            vec![
                vec![0],
                vec![0],
                vec![1],
                vec![1],
                vec![1],
                vec![1],
                vec![2],
                vec![2],
                vec![1],
                vec![1],
                vec![1],
                vec![1],
            ],
            vec![vec![0, 1], vec![1, 2]],
        ],
        vec![
            vec![vec![]; 8],
            vec![
                vec![0],
                vec![2],
                vec![0],
                vec![1],
                vec![4],
                vec![6],
                vec![4],
                vec![5],
            ]
            .into_iter()
            .chain(bridge_inputs)
            .collect(),
            vec![vec![0, 2], vec![4, 6]],
        ],
        vec![
            vec![vec![]; 8],
            vec![
                vec![1],
                vec![3],
                vec![2],
                vec![3],
                vec![5],
                vec![7],
                vec![6],
                vec![7],
            ]
            .into_iter()
            .chain(bridge_outputs)
            .collect(),
            vec![vec![1, 3], vec![5, 7]],
        ],
    ))
}

fn next_permutation(values: &mut [usize]) -> bool {
    let Some(pivot) = (0..values.len() - 1)
        .rev()
        .find(|&index| values[index] < values[index + 1])
    else {
        return false;
    };
    let successor = (pivot + 1..values.len())
        .rev()
        .find(|&index| values[pivot] < values[index])
        .unwrap();
    values.swap(pivot, successor);
    values[pivot + 1..].reverse();
    true
}

fn two_sided_bridged_squares(first_vertex: usize, second_vertex: usize) -> Arc<FramedPoset> {
    Arc::new(FramedPoset::from_faces(
        vec![
            vec![
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            ],
            vec![
                vec![0],
                vec![0],
                vec![1],
                vec![1],
                vec![1],
                vec![1],
                vec![2],
                vec![2],
                vec![1],
                vec![1],
            ],
            vec![vec![0, 1], vec![1, 2]],
        ],
        vec![
            vec![vec![]; 8],
            vec![
                vec![0],
                vec![2],
                vec![0],
                vec![1],
                vec![4],
                vec![6],
                vec![4],
                vec![5],
                vec![first_vertex],
                vec![4 + second_vertex],
            ],
            vec![vec![0, 2], vec![4, 6]],
        ],
        vec![
            vec![vec![]; 8],
            vec![
                vec![1],
                vec![3],
                vec![2],
                vec![3],
                vec![5],
                vec![7],
                vec![6],
                vec![7],
                vec![4 + second_vertex],
                vec![first_vertex],
            ],
            vec![vec![1, 3], vec![5, 7]],
        ],
    ))
}

fn doubly_attached_bridged_squares(
    first_input: usize,
    first_output: usize,
    second_input: usize,
    second_output: usize,
    reversed: bool,
) -> Arc<FramedPoset> {
    let mut vertex_map = [usize::MAX; 8];
    for vertex in 0..4 {
        vertex_map[vertex] = vertex;
    }
    vertex_map[4 + second_input] = first_input;
    vertex_map[4 + second_output] = first_output;
    let mut next = 4;
    for vertex in 4..8 {
        if vertex_map[vertex] == usize::MAX {
            vertex_map[vertex] = next;
            next += 1;
        }
    }

    let first_faces_in = [0, 2, 0, 1];
    let first_faces_out = [1, 3, 2, 3];
    let second_faces_in = [4, 6, 4, 5];
    let second_faces_out = [5, 7, 6, 7];
    let (bridge_input, bridge_output) = if reversed {
        (first_output, first_input)
    } else {
        (first_input, first_output)
    };
    let edge_inputs = first_faces_in
        .into_iter()
        .chain(second_faces_in)
        .map(|vertex| vec![vertex_map[vertex]])
        .chain([vec![bridge_input]])
        .collect::<Vec<_>>();
    let edge_outputs = first_faces_out
        .into_iter()
        .chain(second_faces_out)
        .map(|vertex| vec![vertex_map[vertex]])
        .chain([vec![bridge_output]])
        .collect::<Vec<_>>();

    Arc::new(FramedPoset::from_faces(
        vec![
            vec![vec![]; next],
            vec![
                vec![0],
                vec![0],
                vec![1],
                vec![1],
                vec![1],
                vec![1],
                vec![2],
                vec![2],
                vec![1],
            ],
            vec![vec![0, 1], vec![1, 2]],
        ],
        vec![
            vec![vec![]; next],
            edge_inputs,
            vec![vec![0, 2], vec![4, 6]],
        ],
        vec![
            vec![vec![]; next],
            edge_outputs,
            vec![vec![1, 3], vec![5, 7]],
        ],
    ))
}

fn bridged_squares(first_vertex: usize, second_vertex: usize, reversed: bool) -> Arc<FramedPoset> {
    let (bridge_input, bridge_output) = if reversed {
        (4 + second_vertex, first_vertex)
    } else {
        (first_vertex, 4 + second_vertex)
    };
    Arc::new(FramedPoset::from_faces(
        vec![
            vec![
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            ],
            vec![
                vec![0],
                vec![0],
                vec![1],
                vec![1],
                vec![1],
                vec![1],
                vec![2],
                vec![2],
                vec![1],
            ],
            vec![vec![0, 1], vec![1, 2]],
        ],
        vec![
            vec![vec![]; 8],
            vec![
                vec![0],
                vec![2],
                vec![0],
                vec![1],
                vec![4],
                vec![6],
                vec![4],
                vec![5],
                vec![bridge_input],
            ],
            vec![vec![0, 2], vec![4, 6]],
        ],
        vec![
            vec![vec![]; 8],
            vec![
                vec![1],
                vec![3],
                vec![2],
                vec![3],
                vec![5],
                vec![7],
                vec![6],
                vec![7],
                vec![bridge_output],
            ],
            vec![vec![1, 3], vec![5, 7]],
        ],
    ))
}

#[derive(Clone, Copy)]
struct Cell {
    mask: usize,
    pos: usize,
}

struct IncidenceGroup {
    dim: usize,
    cell: usize,
    faces: Vec<usize>,
}

fn incidence_groups(profile: &[usize]) -> (Vec<Cell>, Vec<IncidenceGroup>) {
    let mut cells = Vec::new();
    let mut by_mask = vec![Vec::new(); profile.len()];
    let mut level_sizes = [0usize; 3];
    for (mask, &count) in profile.iter().enumerate() {
        let dim = mask.count_ones() as usize;
        for _ in 0..count {
            let index = cells.len();
            let pos = level_sizes[dim];
            level_sizes[dim] += 1;
            cells.push(Cell { mask, pos });
            by_mask[mask].push(index);
        }
    }

    let mut groups = Vec::new();
    for (cell, entry) in cells.iter().enumerate() {
        for direction in 0..3 {
            if entry.mask & (1 << direction) != 0 {
                groups.push(IncidenceGroup {
                    dim: entry.mask.count_ones() as usize,
                    cell,
                    faces: by_mask[entry.mask & !(1 << direction)].clone(),
                });
            }
        }
    }
    (cells, groups)
}

fn enumerate(
    group: usize,
    cells: &[Cell],
    groups: &[IncidenceGroup],
    choices: &mut [usize],
    checked: &mut usize,
) -> Option<Arc<FramedPoset>> {
    if group == groups.len() {
        *checked += 1;
        let shape = make_shape(cells, groups, choices);
        return (is_cubular(CubularityMode::Strong, &shape)
            && shape.is_connected()
            && has_incomparable_maximal_frames(&shape))
        .then_some(shape);
    }

    let radix = 3usize.pow(groups[group].faces.len() as u32);
    for choice in 1..radix {
        choices[group] = choice;
        if let Some(witness) = enumerate(group + 1, cells, groups, choices, checked) {
            return Some(witness);
        }
    }
    None
}

fn has_incomparable_maximal_frames(shape: &FramedPoset) -> bool {
    let frames = shape
        .sizes()
        .into_iter()
        .enumerate()
        .flat_map(|(dim, _)| {
            shape
                .maximal(dim)
                .into_iter()
                .map(move |pos| shape.frame_of(dim, pos))
        })
        .collect::<Vec<_>>();
    frames.iter().enumerate().any(|(left, a)| {
        frames[left + 1..]
            .iter()
            .any(|b| !is_subset(a, b) && !is_subset(b, a))
    })
}

fn is_subset(left: &[usize], right: &[usize]) -> bool {
    left.iter()
        .all(|direction| right.binary_search(direction).is_ok())
}

fn make_shape(cells: &[Cell], groups: &[IncidenceGroup], choices: &[usize]) -> Arc<FramedPoset> {
    let mut frames = vec![Vec::new(); 3];
    for cell in cells {
        frames[cell.mask.count_ones() as usize]
            .push((0..3).filter(|&i| cell.mask & (1 << i) != 0).collect());
    }
    while frames.last().is_some_and(Vec::is_empty) {
        frames.pop();
    }

    let mut faces_in: Vec<Vec<Vec<usize>>> = frames
        .iter()
        .map(|level| vec![vec![]; level.len()])
        .collect();
    let mut faces_out = faces_in.clone();
    for (group, &choice) in groups.iter().zip(choices) {
        let cell_pos = cells[group.cell].pos;
        let mut choice = choice;
        for &face in &group.faces {
            let face_pos = cells[face].pos;
            match choice % 3 {
                0 => {}
                1 => faces_in[group.dim][cell_pos].push(face_pos),
                2 => faces_out[group.dim][cell_pos].push(face_pos),
                _ => unreachable!(),
            }
            choice /= 3;
        }
    }
    for row in faces_in.iter_mut().chain(&mut faces_out).flatten() {
        row.sort_unstable();
    }

    Arc::new(FramedPoset::from_faces(frames, faces_in, faces_out))
}
