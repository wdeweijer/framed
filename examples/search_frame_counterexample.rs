use std::sync::Arc;

use ofposets::{Embedding, FramedPoset, Sign, boundary};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const PROFILE: [usize; 7] = [12, 4, 4, 1, 4, 1, 1];
const RESTARTS: usize = 20;
const STEPS: usize = 100_000;

fn main() {
    let (cells, groups) = incidence_groups(&PROFILE);
    let mut rng = SmallRng::seed_from_u64(0x5ea4_c0de);
    let mut global_best = usize::MAX;
    let mut best_shape = None;

    for restart in 0..RESTARTS {
        let mut genes = disjoint_square_genes(&groups);
        for _ in 0..restart * 4 {
            mutate(&mut genes, &groups, &mut rng);
        }
        let mut fast_shape = FastShape::new(&cells, &groups, &genes);
        let mut current_score = fast_shape.score();
        debug_assert_eq!(
            fast_shape.boundary_failures(),
            score(&make_shape(&cells, &groups, &genes))
                - usize::from(!fast_shape.is_connected()) * 100
        );

        for step in 0..STEPS {
            if current_score < global_best {
                global_best = current_score;
                println!(
                    "best={current_score} at restart={restart}, step={step}, connected={}",
                    fast_shape.is_connected()
                );
                best_shape = Some(make_shape(&cells, &groups, &genes));
            }
            if current_score == 0 && fast_shape.is_connected() {
                let shape = make_shape(&cells, &groups, &genes);
                assert_eq!(score(&shape), 0);
                println!("COUNTEREXAMPLE");
                println!("{}", serde_json::to_string_pretty(&*shape).unwrap());
                return;
            }

            let gene = rng.random_range(0..genes.len());
            let old = genes[gene];
            mutate_gene(&mut genes[gene], &groups[gene], &mut rng);
            let candidate = FastShape::new(&cells, &groups, &genes);
            let candidate_score = candidate.score();
            let temperature = 1.0 - step as f64 / STEPS as f64;
            let accept = candidate_score <= current_score
                || rng.random_bool(
                    (-(candidate_score as f64 - current_score as f64)
                        / (1.0 + 4.0 * temperature))
                        .exp(),
                );
            if accept {
                fast_shape = candidate;
                current_score = candidate_score;
            } else {
                genes[gene] = old;
            }
        }
    }

    println!("no counterexample found; best failure count={global_best}");
    if let Some(best_shape) = best_shape {
        println!("identity failures={}", check_state(&best_shape, &[0, 1, 2]));
        for first_direction in 0..3 {
            let remaining = (0..3)
                .filter(|&direction| direction != first_direction)
                .collect::<Vec<_>>();
            for sign in [Sign::Input, Sign::Output] {
                let (state, _) = boundary(sign, first_direction, &best_shape);
                println!(
                    "after ({sign:?},{first_direction}): failures={}",
                    check_state(&state, &remaining)
                );
            }
        }
        println!("{}", serde_json::to_string_pretty(&*best_shape).unwrap());
    }
}

fn mutate(genes: &mut [usize], groups: &[IncidenceGroup], rng: &mut SmallRng) {
    let gene = rng.random_range(0..genes.len());
    mutate_gene(&mut genes[gene], &groups[gene], rng);
}

fn mutate_gene(gene: &mut usize, group: &IncidenceGroup, rng: &mut SmallRng) {
    loop {
        let face = rng.random_range(0..group.faces.len());
        let power = 3usize.pow(face as u32);
        let old = (*gene / power) % 3;
        let replacement = (old + rng.random_range(1..3)) % 3;
        let candidate = *gene - old * power + replacement * power;
        if candidate != 0 {
            *gene = candidate;
            return;
        }
    }
}

fn disjoint_square_genes(groups: &[IncidenceGroup]) -> Vec<usize> {
    let edge_relations: [[(usize, usize); 2]; 12] = [
        [(0, 1), (1, 2)],
        [(2, 1), (3, 2)],
        [(4, 1), (5, 2)],
        [(6, 1), (7, 2)],
        [(0, 1), (2, 2)],
        [(1, 1), (3, 2)],
        [(8, 1), (9, 2)],
        [(10, 1), (11, 2)],
        [(4, 1), (6, 2)],
        [(5, 1), (7, 2)],
        [(8, 1), (10, 2)],
        [(9, 1), (11, 2)],
    ];
    let top_relations: [[(usize, usize); 4]; 3] = [
        [(12, 1), (13, 2), (16, 1), (17, 2)],
        [(14, 1), (15, 2), (21, 1), (22, 2)],
        [(18, 1), (19, 2), (23, 1), (24, 2)],
    ];

    groups
        .iter()
        .map(|group| {
            let relations: &[(usize, usize)] = match group.cell {
                12..=19 => &edge_relations[group.cell - 12],
                20 => &top_relations[0],
                21..=24 => &edge_relations[group.cell - 13],
                25 => &top_relations[1],
                26 => &top_relations[2],
                _ => unreachable!(),
            };
            group
                .faces
                .iter()
                .enumerate()
                .map(|(index, face)| {
                    relations
                        .iter()
                        .find_map(|&(candidate, sign)| (candidate == *face).then_some(sign))
                        .unwrap_or(0)
                        * 3usize.pow(index as u32)
                })
                .sum()
        })
        .collect()
}

fn score(shape: &Arc<FramedPoset>) -> usize {
    let mut failures = check_state(shape, &[0, 1, 2]);
    for first_direction in 0..3 {
        let remaining = (0..3)
            .filter(|&direction| direction != first_direction)
            .collect::<Vec<_>>();
        for sign in [Sign::Input, Sign::Output] {
            let (state, _) = boundary(sign, first_direction, shape);
            failures += check_state(&state, &remaining);
        }
    }
    failures + usize::from(!shape.is_connected()) * 100
}

fn check_state(shape: &Arc<FramedPoset>, directions: &[usize]) -> usize {
    let mut failures = 0;
    let boundaries = directions
        .iter()
        .map(|&direction| {
            [Sign::Input, Sign::Output].map(|sign| boundary(sign, direction, shape))
        })
        .collect::<Vec<_>>();
    for left in 0..directions.len() {
        for right in left + 1..directions.len() {
            for (alpha_index, alpha) in [Sign::Input, Sign::Output].into_iter().enumerate() {
                for (beta_index, beta) in [Sign::Input, Sign::Output].into_iter().enumerate() {
                    let (alpha_shape, alpha_into_shape) = &boundaries[left][alpha_index];
                    let (beta_shape, beta_into_shape) = &boundaries[right][beta_index];
                    let (_, alpha_into_beta) =
                        boundary(alpha, directions[left], beta_shape);
                    let (_, beta_into_alpha) =
                        boundary(beta, directions[right], alpha_shape);
                    let alpha_after_beta =
                        Embedding::compose(&alpha_into_beta, &beta_into_shape);
                    let beta_after_alpha =
                        Embedding::compose(&beta_into_alpha, &alpha_into_shape);
                    let intersection =
                        Embedding::intersection(&alpha_into_shape, &beta_into_shape).into_codomain;
                    failures += usize::from(!Embedding::same_subobject(
                        &alpha_after_beta,
                        &intersection,
                    ));
                    failures += usize::from(!Embedding::same_subobject(
                        &beta_after_alpha,
                        &intersection,
                    ));
                }
            }
        }
    }
    failures
}

struct FastShape {
    basis: Vec<usize>,
    down: Vec<u64>,
    cofaces: [Vec<u64>; 2],
    neighbours: Vec<u64>,
}

impl FastShape {
    fn new(cells: &[Cell], groups: &[IncidenceGroup], genes: &[usize]) -> Self {
        assert!(cells.len() <= 64);
        let basis = cells.iter().map(|cell| cell.mask).collect::<Vec<_>>();
        let mut direct_faces = vec![0u64; cells.len()];
        let mut cofaces = [vec![0u64; cells.len()], vec![0u64; cells.len()]];
        let mut neighbours = vec![0u64; cells.len()];

        for (group, &gene) in groups.iter().zip(genes) {
            let mut gene = gene;
            for &face in &group.faces {
                let sign = gene % 3;
                gene /= 3;
                if sign == 0 {
                    continue;
                }
                direct_faces[group.cell] |= 1 << face;
                cofaces[sign - 1][face] |= 1 << group.cell;
                neighbours[group.cell] |= 1 << face;
                neighbours[face] |= 1 << group.cell;
            }
        }

        let mut down = (0..cells.len())
            .map(|cell| (1 << cell) | direct_faces[cell])
            .collect::<Vec<_>>();
        loop {
            let previous = down.clone();
            for cell in 0..cells.len() {
                let mut faces = direct_faces[cell];
                while faces != 0 {
                    let face = faces.trailing_zeros() as usize;
                    faces &= faces - 1;
                    down[cell] |= previous[face];
                }
            }
            if down == previous {
                break;
            }
        }

        Self {
            basis,
            down,
            cofaces,
            neighbours,
        }
    }

    fn score(&self) -> usize {
        self.boundary_failures() + self.component_count().saturating_sub(1) * 4
    }

    fn boundary_failures(&self) -> usize {
        let full = if self.basis.len() == 64 {
            u64::MAX
        } else {
            (1 << self.basis.len()) - 1
        };
        let mut failures = self.check_state(full, &[0, 1, 2]);
        for first_direction in 0..3 {
            let remaining = (0..3)
                .filter(|&direction| direction != first_direction)
                .collect::<Vec<_>>();
            for sign in 0..2 {
                failures +=
                    self.check_state(self.boundary(full, sign, first_direction), &remaining);
            }
        }
        failures
    }

    fn check_state(&self, state: u64, directions: &[usize]) -> usize {
        let boundaries = directions
            .iter()
            .map(|&direction| {
                [
                    self.boundary(state, 0, direction),
                    self.boundary(state, 1, direction),
                ]
            })
            .collect::<Vec<_>>();
        let mut failures = 0;
        for left in 0..directions.len() {
            for right in left + 1..directions.len() {
                for alpha in 0..2 {
                    for beta in 0..2 {
                        let intersection = boundaries[left][alpha] & boundaries[right][beta];
                        failures += usize::from(
                            self.boundary(
                                boundaries[right][beta],
                                alpha,
                                directions[left],
                            ) != intersection,
                        );
                        failures += usize::from(
                            self.boundary(
                                boundaries[left][alpha],
                                beta,
                                directions[right],
                            ) != intersection,
                        );
                    }
                }
            }
        }
        failures
    }

    fn boundary(&self, state: u64, sign: usize, direction: usize) -> u64 {
        let direction_bit = 1 << direction;
        let orthogonal = self
            .basis
            .iter()
            .enumerate()
            .fold(0u64, |mask, (cell, &basis)| {
                mask | u64::from(basis & direction_bit == 0) << cell
            });
        let mut candidates = state & orthogonal;
        let mut result = 0u64;
        while candidates != 0 {
            let cell = candidates.trailing_zeros() as usize;
            candidates &= candidates - 1;
            let all_cofaces = (self.cofaces[0][cell] | self.cofaces[1][cell]) & state;
            if self.cofaces[sign ^ 1][cell] & state == 0
                && all_cofaces & orthogonal == 0
            {
                result |= self.down[cell];
            }
        }
        result & state
    }

    fn is_connected(&self) -> bool {
        self.component_count() == 1
    }

    fn component_count(&self) -> usize {
        if self.basis.is_empty() {
            return 0;
        }
        let full = if self.basis.len() == 64 {
            u64::MAX
        } else {
            (1 << self.basis.len()) - 1
        };
        let mut unseen = full;
        let mut components = 0;
        while unseen != 0 {
            components += 1;
            let start = unseen.trailing_zeros() as usize;
            let mut frontier = 1u64 << start;
            unseen &= !(1 << start);
            while frontier != 0 {
                let cell = frontier.trailing_zeros() as usize;
                frontier &= frontier - 1;
                let new = self.neighbours[cell] & unseen;
                unseen &= !new;
                frontier |= new;
            }
        }
        components
    }
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

fn make_shape(
    cells: &[Cell],
    groups: &[IncidenceGroup],
    genes: &[usize],
) -> Arc<FramedPoset> {
    let mut basis = vec![Vec::new(); 3];
    for cell in cells {
        basis[cell.mask.count_ones() as usize]
            .push((0..3).filter(|&i| cell.mask & (1 << i) != 0).collect());
    }

    let mut faces_in: Vec<Vec<Vec<usize>>> = basis
        .iter()
        .map(|level| vec![vec![]; level.len()])
        .collect();
    let mut faces_out = faces_in.clone();
    for (group, &gene) in groups.iter().zip(genes) {
        let cell_pos = cells[group.cell].pos;
        let mut gene = gene;
        for &face in &group.faces {
            let face_pos = cells[face].pos;
            match gene % 3 {
                0 => {}
                1 => faces_in[group.dim][cell_pos].push(face_pos),
                2 => faces_out[group.dim][cell_pos].push(face_pos),
                _ => unreachable!(),
            }
            gene /= 3;
        }
    }
    for row in faces_in.iter_mut().chain(&mut faces_out).flatten() {
        row.sort_unstable();
    }
    Arc::new(FramedPoset::from_faces(basis, faces_in, faces_out))
}
