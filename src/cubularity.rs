//! Cubularity conditions defined by directional boundaries.

use std::sync::Arc;

use crate::embedding::Embedding;
use crate::poset::{FramedPoset, Sign, boundary_hat};

/// True when every ordering of distinct directional boundaries agrees.
///
/// Equivalently, boundary operators commute in every iterated boundary of
/// `shape`. In particular, every boundary of a cubular poset is cubular.
pub fn is_cubular(shape: &Arc<FramedPoset>) -> bool {
    check_all_boundary_states(shape, Check::Cubular)
}

/// True when every pair of boundaries intersects strongly at every depth.
///
/// In each iterated boundary `B`, this requires both orders of two further
/// boundaries to equal their intersection in `B`. Strong cubularity therefore
/// implies [`is_cubular`].
pub fn is_strongly_cubular(shape: &Arc<FramedPoset>) -> bool {
    check_all_boundary_states(shape, Check::Strong)
}

#[derive(Debug, Clone, Copy)]
enum Check {
    Cubular,
    Strong,
}

struct Boundary {
    domain: Arc<FramedPoset>,
    into_parent: Embedding,
}

fn check_all_boundary_states(shape: &Arc<FramedPoset>, check: Check) -> bool {
    let directions = shape.active_directions();
    let trace = std::env::var_os("OFP_CUBULARITY_TRACE").is_some();
    let mut selected = vec![false; directions.len()];
    let mut boundary_word = Vec::new();

    if trace {
        eprintln!(
            "[cubularity] start {check:?}: directions={directions:?}, cells={:?}",
            shape.sizes()
        );
    }

    let result = check_states(
        shape,
        &directions,
        &mut selected,
        0,
        &mut boundary_word,
        check,
        trace,
    );

    if trace {
        eprintln!("[cubularity] finish {check:?}: {result}");
    }
    result
}

fn check_states(
    shape: &Arc<FramedPoset>,
    directions: &[usize],
    selected: &mut [bool],
    next: usize,
    boundary_word: &mut Vec<(Sign, usize)>,
    check: Check,
    trace: bool,
) -> bool {
    if next == directions.len() {
        let remaining: Vec<usize> = directions
            .iter()
            .zip(selected.iter())
            .filter_map(|(&direction, &is_selected)| (!is_selected).then_some(direction))
            .collect();
        if trace {
            eprintln!(
                "[cubularity] state {}: cells={:?}, remaining={remaining:?}",
                format_boundary_word(boundary_word),
                shape.sizes()
            );
        }
        return check_last_two(shape, &remaining, boundary_word, check, trace);
    }

    selected[next] = false;
    if trace {
        eprintln!(
            "[cubularity] {}leave direction {} available",
            "  ".repeat(boundary_word.len()),
            directions[next]
        );
    }
    if !check_states(
        shape,
        directions,
        selected,
        next + 1,
        boundary_word,
        check,
        trace,
    ) {
        return false;
    }

    selected[next] = true;
    for sign in [Sign::Input, Sign::Output] {
        if trace {
            eprintln!(
                "[cubularity] {}take {sign:?} boundary in direction {}",
                "  ".repeat(boundary_word.len()),
                directions[next]
            );
        }
        let (boundary, _) = boundary_hat(sign, directions[next], shape);
        boundary_word.push((sign, directions[next]));
        if trace {
            eprintln!(
                "[cubularity] {}reached {} with cells={:?}",
                "  ".repeat(boundary_word.len()),
                format_boundary_word(boundary_word),
                boundary.sizes()
            );
        }
        if !check_states(
            &boundary,
            directions,
            selected,
            next + 1,
            boundary_word,
            check,
            trace,
        ) {
            boundary_word.pop();
            selected[next] = false;
            return false;
        }
        boundary_word.pop();
    }
    selected[next] = false;
    true
}

fn check_last_two(
    shape: &Arc<FramedPoset>,
    directions: &[usize],
    boundary_word: &[(Sign, usize)],
    check: Check,
    trace: bool,
) -> bool {
    if directions.len() < 2 {
        if trace {
            eprintln!(
                "[cubularity]   no last-two check needed for {}",
                format_boundary_word(boundary_word)
            );
        }
        return true;
    }

    let boundaries: Vec<[Boundary; 2]> = directions
        .iter()
        .map(|&direction| {
            [Sign::Input, Sign::Output].map(|sign| {
                let (domain, into_parent) = boundary_hat(sign, direction, shape);
                Boundary {
                    domain,
                    into_parent,
                }
            })
        })
        .collect();

    if trace {
        for (index, &direction) in directions.iter().enumerate() {
            eprintln!(
                "[cubularity]   direct boundaries at {}: Input={:?}, Output={:?}",
                direction,
                boundaries[index][0].domain.sizes(),
                boundaries[index][1].domain.sizes()
            );
        }
    }

    for left in 0..directions.len() {
        for right in left + 1..directions.len() {
            for alpha in 0..2 {
                for beta in 0..2 {
                    let alpha_sign = [Sign::Input, Sign::Output][alpha];
                    let beta_sign = [Sign::Input, Sign::Output][beta];
                    let alpha_boundary = &boundaries[left][alpha];
                    let beta_boundary = &boundaries[right][beta];
                    let alpha_after_beta =
                        boundary_after(beta_boundary, alpha_sign, directions[left]);
                    let beta_after_alpha =
                        boundary_after(alpha_boundary, beta_sign, directions[right]);

                    let equal = match check {
                        Check::Cubular => {
                            let equal = Embedding::equal(&alpha_after_beta, &beta_after_alpha);
                            if trace {
                                eprintln!(
                                    "[cubularity]   commute ({alpha_sign:?}, {}) after \
                                     ({beta_sign:?}, {}) with ({beta_sign:?}, {}) after \
                                     ({alpha_sign:?}, {}): {equal}; cells={:?} vs {:?}",
                                    directions[left],
                                    directions[right],
                                    directions[right],
                                    directions[left],
                                    alpha_after_beta.dom.sizes(),
                                    beta_after_alpha.dom.sizes()
                                );
                            }
                            equal
                        }
                        Check::Strong => {
                            let intersection = Embedding::intersection(
                                &alpha_boundary.into_parent,
                                &beta_boundary.into_parent,
                            )
                            .into_codomain;
                            let alpha_equal = Embedding::equal(&alpha_after_beta, &intersection);
                            let beta_equal = Embedding::equal(&beta_after_alpha, &intersection);
                            if trace {
                                eprintln!(
                                    "[cubularity]   strong ({alpha_sign:?}, {}) / \
                                     ({beta_sign:?}, {}): first={alpha_equal}, \
                                     second={beta_equal}; cells={:?}, {:?}, intersection={:?}",
                                    directions[left],
                                    directions[right],
                                    alpha_after_beta.dom.sizes(),
                                    beta_after_alpha.dom.sizes(),
                                    intersection.dom.sizes()
                                );
                            }
                            alpha_equal && beta_equal
                        }
                    };
                    if !equal {
                        if trace {
                            eprintln!(
                                "[cubularity]   FAILED at state {}",
                                format_boundary_word(boundary_word)
                            );
                        }
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn boundary_after(first: &Boundary, second_sign: Sign, second_direction: usize) -> Embedding {
    let (_, second_into_first) = boundary_hat(second_sign, second_direction, &first.domain);
    Embedding::compose(&second_into_first, &first.into_parent)
}

fn format_boundary_word(boundaries: &[(Sign, usize)]) -> String {
    if boundaries.is_empty() {
        return "identity".to_owned();
    }

    boundaries
        .iter()
        .map(|&(sign, direction)| {
            format!(
                "{}{direction}",
                match sign {
                    Sign::Input => "-",
                    Sign::Output => "+",
                }
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn square() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![
                vec![vec![], vec![], vec![], vec![]],
                vec![vec![0], vec![0], vec![1], vec![1]],
                vec![vec![0, 1]],
            ],
            vec![
                vec![vec![], vec![], vec![], vec![]],
                vec![vec![0], vec![2], vec![0], vec![1]],
                vec![vec![0, 2]],
            ],
            vec![
                vec![vec![], vec![], vec![], vec![]],
                vec![vec![1], vec![3], vec![2], vec![3]],
                vec![vec![1, 3]],
            ],
        ))
    }

    fn weakly_but_not_strongly_cubular() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![vec![vec![]], vec![vec![0], vec![1]], vec![vec![0, 1]]],
            vec![vec![vec![]], vec![vec![0], vec![0]], vec![vec![]]],
            vec![vec![vec![]], vec![vec![], vec![]], vec![vec![0, 1]]],
        ))
    }

    fn non_cubular_two_dimensional() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![
                vec![vec![], vec![]],
                vec![vec![0], vec![0], vec![1], vec![1]],
                vec![vec![0, 1], vec![0, 1], vec![0, 1]],
            ],
            vec![
                vec![vec![], vec![]],
                vec![vec![], vec![1], vec![0], vec![]],
                vec![vec![3], vec![1, 3], vec![0, 1, 2]],
            ],
            vec![
                vec![vec![], vec![]],
                vec![vec![0, 1], vec![], vec![], vec![1]],
                vec![vec![0, 1, 2], vec![0], vec![3]],
            ],
        ))
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct CubeCell(Vec<Option<bool>>);

    fn cube(n: usize) -> Arc<FramedPoset> {
        let mut levels = vec![Vec::new(); n + 1];
        for code in 0..3usize.pow(n as u32) {
            let mut code = code;
            let mut coordinates = Vec::with_capacity(n);
            for _ in 0..n {
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

        for dim in 1..=n {
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

    fn is_top_level_cubular(shape: &Arc<FramedPoset>) -> bool {
        let directions = shape.active_directions();
        check_last_two(shape, &directions, &[], Check::Cubular, false)
    }

    #[test]
    fn standard_square_is_strongly_cubular() {
        let shape = square();
        assert!(is_strongly_cubular(&shape));
        assert!(is_cubular(&shape));
    }

    #[test]
    fn strong_cubularity_is_stricter_than_cubularity() {
        let shape = weakly_but_not_strongly_cubular();
        assert!(is_cubular(&shape));
        assert!(!is_strongly_cubular(&shape));
    }

    #[test]
    fn one_directional_shapes_are_vacuously_strongly_cubular() {
        let arrow = Arc::new(FramedPoset::from_faces(
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![1]]],
        ));
        assert!(is_strongly_cubular(&arrow));
    }

    #[test]
    fn two_dimensional_check_is_exactly_the_top_level_check() {
        for shape in [
            square(),
            weakly_but_not_strongly_cubular(),
            non_cubular_two_dimensional(),
        ] {
            assert_eq!(is_cubular(&shape), is_top_level_cubular(&shape));
        }
    }

    #[test]
    fn standard_three_cube_and_all_its_boundaries_are_strongly_cubular() {
        let cube = cube(3);
        assert!(is_strongly_cubular(&cube));

        for direction in 0..3 {
            for sign in [Sign::Input, Sign::Output] {
                let (boundary, _) = boundary_hat(sign, direction, &cube);
                assert!(is_strongly_cubular(&boundary));
                assert!(is_cubular(&boundary));
            }
        }
    }

    #[test]
    fn standard_three_cube_is_cubular() {
        assert!(is_cubular(&cube(3)));
    }
}
