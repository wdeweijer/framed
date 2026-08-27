//! Box constructions and elementary cylinders.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use petgraph::unionfind::UnionFind;

use crate::cubularity::{CubularityMode, is_cubular};
use crate::embedding::Embedding;
use crate::intset::{self, IntSet};
use crate::isomorphism::{isomorphic, isomorphisms};
use crate::orthogonal::orthogonal_product;
use crate::poset::{FramedPoset, Sign, boundary, iterated_boundary, shift};
use crate::volumetric::is_volumetric;

/// One signed directional face supplied to a box construction.
#[derive(Debug, Clone)]
pub struct BoxFace {
    pub direction: usize,
    pub sign: Sign,
    pub shape: Arc<FramedPoset>,
}

impl BoxFace {
    pub fn new(direction: usize, sign: Sign, shape: Arc<FramedPoset>) -> Self {
        Self {
            direction,
            sign,
            shape,
        }
    }
}

/// An isomorphism identifying the overlap of two box faces.
///
/// For a left face `U_i^alpha` and right face `U_j^beta`, `isomorphism`
/// has type
///
/// `boundary(beta, j, U_i^alpha) -> boundary(alpha, i, U_j^beta)`.
#[derive(Debug, Clone)]
pub struct BoxGluing {
    pub left_direction: usize,
    pub left_sign: Sign,
    pub right_direction: usize,
    pub right_sign: Sign,
    pub isomorphism: Embedding,
}

/// The inclusion of one supplied face into a completed box.
#[derive(Debug, Clone)]
pub struct BoxFaceEmbedding {
    pub direction: usize,
    pub sign: Sign,
    pub embedding: Embedding,
}

/// A box together with the inclusions of all its signed directional faces.
#[derive(Debug, Clone)]
pub struct BoxConstruction {
    pub shape: Arc<FramedPoset>,
    pub faces: Vec<BoxFaceEmbedding>,
}

impl BoxConstruction {
    /// Inclusion of the requested signed directional face.
    pub fn face(&self, sign: Sign, direction: usize) -> Option<&Embedding> {
        self.faces
            .iter()
            .find(|face| face.sign == sign && face.direction == direction)
            .map(|face| &face.embedding)
    }
}

/// Construct a box from signed directional faces and overlap isomorphisms.
///
/// There must be one face for each sign and direction, and one gluing for each
/// pair of faces in unequal directions. Gluings may be supplied in either
/// direction. The quotient construction checks that every face still embeds
/// injectively and that identified cells have compatible signed incidence.
pub fn box_construction(faces: &[BoxFace], gluings: &[BoxGluing]) -> BoxConstruction {
    let faces = canonical_faces(faces);
    let frame: IntSet = faces
        .iter()
        .map(|face| face.direction)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let mut frame = frame;
    frame.sort_unstable();

    let spans = gluing_spans(&faces, gluings);
    let (boundary_shape, into_boundary) = quotient_boundary(&faces, &spans);
    attach_top(&frame, &faces, boundary_shape, into_boundary)
}

/// Construct a box whose overlap isomorphisms are forced by rigidity.
///
/// Panics unless every pair of required directional boundaries has exactly
/// one isomorphism.
pub fn rigid_box_construction(faces: &[BoxFace]) -> BoxConstruction {
    let faces = canonical_faces(faces);
    debug_assert!(faces.iter().all(|face| face.shape.is_rigid()));

    let mut gluings = Vec::new();
    for left in 0..faces.len() {
        for right in left + 1..faces.len() {
            if faces[left].direction == faces[right].direction {
                continue;
            }

            let (left_boundary, _) = boundary(
                faces[right].sign,
                faces[right].direction,
                &faces[left].shape,
            );
            let (right_boundary, _) =
                boundary(faces[left].sign, faces[left].direction, &faces[right].shape);
            let mut candidates = isomorphisms(&left_boundary, &right_boundary);
            assert_eq!(
                candidates.len(),
                1,
                "box faces ({:?}, {}) and ({:?}, {}) must have exactly one boundary isomorphism",
                faces[left].sign,
                faces[left].direction,
                faces[right].sign,
                faces[right].direction,
            );

            gluings.push(BoxGluing {
                left_direction: faces[left].direction,
                left_sign: faces[left].sign,
                right_direction: faces[right].direction,
                right_sign: faces[right].sign,
                isomorphism: candidates.pop().unwrap(),
            });
        }
    }

    box_construction(&faces, &gluings)
}

/// Construct the elementary cylinder of two compatible volumetric posets.
///
/// The first argument is the input and the second is the output. Their
/// direction-zero boundaries must be isomorphic for both signs, and the output
/// frame must be contained in the input frame.
pub fn elementary_cylinder(
    input: &Arc<FramedPoset>,
    output: &Arc<FramedPoset>,
) -> Arc<FramedPoset> {
    debug_assert!(is_volumetric(input));
    debug_assert!(is_volumetric(output));

    let input_frame = input.active_directions();
    let output_frame = output.active_directions();
    assert!(
        intset::is_subset(&output_frame, &input_frame),
        "the elementary-cylinder output frame must be contained in the input frame",
    );
    for sign in [Sign::Input, Sign::Output] {
        let (input_boundary, _) = boundary(sign, 0, input);
        let (output_boundary, _) = boundary(sign, 0, output);
        assert!(
            isomorphic(&input_boundary, &output_boundary),
            "the direction-zero {sign:?} boundaries must be isomorphic",
        );
    }

    elementary_cylinder_recursive(input, output)
}

#[derive(Debug)]
struct GluingSpan {
    left: usize,
    right: usize,
    left_into_face: Embedding,
    right_into_face: Embedding,
    isomorphism: Embedding,
}

fn canonical_faces(faces: &[BoxFace]) -> Vec<BoxFace> {
    assert!(!faces.is_empty(), "a box frame must be non-empty");

    let mut faces = faces.to_vec();
    faces.sort_by_key(|face| (face.direction, sign_index(face.sign)));

    let mut labels = HashSet::with_capacity(faces.len());
    for face in &faces {
        assert!(
            labels.insert((face.direction, face.sign)),
            "duplicate {:?} face in direction {}",
            face.sign,
            face.direction,
        );
    }

    let frame = intset::collect_sorted(faces.iter().map(|face| face.direction));
    assert_eq!(
        faces.len(),
        2 * frame.len(),
        "a box needs one input and one output face in every direction",
    );

    let expected_input_dimension = frame.len() as isize - 1;
    for &direction in &frame {
        for sign in [Sign::Input, Sign::Output] {
            let face = faces
                .iter()
                .find(|face| face.direction == direction && face.sign == sign)
                .expect("a signed face is missing");
            let face_frame = face.shape.active_directions();
            assert!(
                intset::is_subset(&face_frame, &frame),
                "a box face uses a direction outside the box frame",
            );
            assert!(
                face_frame.binary_search(&direction).is_err(),
                "a box face must be orthogonal to its own direction",
            );
            if sign == Sign::Input {
                assert_eq!(
                    face.shape.dim(),
                    expected_input_dimension,
                    "an input box face must have codimension one",
                );
            }
            debug_assert!(is_cubular(CubularityMode::Strong, &face.shape));
        }
    }

    faces
}

fn gluing_spans(faces: &[BoxFace], gluings: &[BoxGluing]) -> Vec<GluingSpan> {
    let indices: HashMap<_, _> = faces
        .iter()
        .enumerate()
        .map(|(index, face)| ((face.direction, face.sign), index))
        .collect();
    let mut seen = HashSet::with_capacity(gluings.len());
    let mut spans = Vec::with_capacity(gluings.len());

    for gluing in gluings {
        let mut left = *indices
            .get(&(gluing.left_direction, gluing.left_sign))
            .expect("a gluing refers to a missing left box face");
        let mut right = *indices
            .get(&(gluing.right_direction, gluing.right_sign))
            .expect("a gluing refers to a missing right box face");
        assert_ne!(
            faces[left].direction, faces[right].direction,
            "box faces in the same direction are not directly glued",
        );

        let mut isomorphism = gluing.isomorphism.clone();
        if faces[left].direction > faces[right].direction {
            std::mem::swap(&mut left, &mut right);
            isomorphism = isomorphism.inverse_isomorphism();
        }
        assert!(
            seen.insert((left, right)),
            "duplicate gluing between two signed box faces",
        );

        let (left_boundary, left_into_face) = boundary(
            faces[right].sign,
            faces[right].direction,
            &faces[left].shape,
        );
        let (right_boundary, right_into_face) =
            boundary(faces[left].sign, faces[left].direction, &faces[right].shape);
        assert!(
            isomorphism.is_isomorphism(),
            "a box gluing must be an isomorphism"
        );
        assert!(
            FramedPoset::equal(&isomorphism.dom, &left_boundary),
            "a box gluing has the wrong domain boundary",
        );
        assert!(
            FramedPoset::equal(&isomorphism.cod, &right_boundary),
            "a box gluing has the wrong codomain boundary",
        );

        spans.push(GluingSpan {
            left,
            right,
            left_into_face,
            right_into_face,
            isomorphism,
        });
    }

    for left in 0..faces.len() {
        for right in left + 1..faces.len() {
            if faces[left].direction != faces[right].direction {
                assert!(
                    seen.contains(&(left, right)),
                    "missing gluing between ({:?}, {}) and ({:?}, {})",
                    faces[left].sign,
                    faces[left].direction,
                    faces[right].sign,
                    faces[right].direction,
                );
            }
        }
    }
    assert_eq!(
        seen.len(),
        gluings.len(),
        "every gluing must identify a distinct required overlap",
    );

    spans.sort_by_key(|span| (span.left, span.right));
    spans
}

fn quotient_boundary(
    faces: &[BoxFace],
    spans: &[GluingSpan],
) -> (Arc<FramedPoset>, Vec<Embedding>) {
    let mut next_global = 0usize;
    let global_cells: Vec<Vec<Vec<usize>>> = faces
        .iter()
        .map(|face| {
            face.shape
                .sizes()
                .into_iter()
                .map(|size| {
                    (0..size)
                        .map(|_| {
                            let global = next_global;
                            next_global += 1;
                            global
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let mut classes = UnionFind::<usize>::new(next_global);

    for span in spans {
        for (dim, level) in span.left_into_face.map.iter().enumerate() {
            for (boundary_pos, &left_pos) in level.iter().enumerate() {
                let right_boundary_pos = span.isomorphism.map[dim][boundary_pos];
                let right_pos = span.right_into_face.map[dim][right_boundary_pos];
                classes.union(
                    global_cells[span.left][dim][left_pos],
                    global_cells[span.right][dim][right_pos],
                );
            }
        }
    }

    let levels = faces
        .iter()
        .map(|face| face.shape.sizes().len())
        .max()
        .unwrap_or(0);
    let mut quotient_basis: Vec<Vec<IntSet>> = vec![Vec::new(); levels];
    let mut quotient_cells = HashMap::<usize, (usize, usize)>::new();
    let mut face_maps: Vec<Vec<Vec<usize>>> = faces
        .iter()
        .map(|face| {
            face.shape
                .sizes()
                .into_iter()
                .map(|size| vec![0; size])
                .collect()
        })
        .collect();

    for dim in 0..levels {
        for (face_index, face) in faces.iter().enumerate() {
            let Some(size) = face.shape.sizes().get(dim).copied() else {
                continue;
            };
            for pos in 0..size {
                let root = classes.find_mut(global_cells[face_index][dim][pos]);
                let quotient = if let Some(&(root_dim, root_pos)) = quotient_cells.get(&root) {
                    assert_eq!(
                        root_dim, dim,
                        "a gluing identified cells of unequal dimension"
                    );
                    assert_eq!(
                        quotient_basis[dim][root_pos],
                        *face.shape.basis_of(dim, pos),
                        "a gluing identified cells with unequal bases",
                    );
                    root_pos
                } else {
                    let quotient = quotient_basis[dim].len();
                    quotient_basis[dim].push(face.shape.basis_of(dim, pos).clone());
                    quotient_cells.insert(root, (dim, quotient));
                    quotient
                };
                face_maps[face_index][dim][pos] = quotient;
            }
        }
    }

    for (face_index, map) in face_maps.iter().enumerate() {
        for (dim, level) in map.iter().enumerate() {
            let unique: HashSet<_> = level.iter().copied().collect();
            assert_eq!(
                unique.len(),
                level.len(),
                "incoherent box gluings identify distinct cells of face ({:?}, {})",
                faces[face_index].sign,
                faces[face_index].direction,
            );
            debug_assert!(level.iter().all(|&pos| pos < quotient_basis[dim].len()));
        }
    }

    let mut quotient_faces_in: Vec<Vec<IntSet>> = quotient_basis
        .iter()
        .map(|level| vec![vec![]; level.len()])
        .collect();
    let mut quotient_faces_out = quotient_faces_in.clone();
    let mut defined: Vec<Vec<bool>> = quotient_basis
        .iter()
        .map(|level| vec![false; level.len()])
        .collect();

    for (face_index, face) in faces.iter().enumerate() {
        for dim in 0..face.shape.sizes().len() {
            for pos in 0..face.shape.sizes()[dim] {
                let quotient = face_maps[face_index][dim][pos];
                let mapped_in = if dim == 0 {
                    vec![]
                } else {
                    intset::collect_sorted(
                        face.shape
                            .faces_of(Sign::Input, dim, pos)
                            .iter()
                            .map(|&face_pos| face_maps[face_index][dim - 1][face_pos]),
                    )
                };
                let mapped_out = if dim == 0 {
                    vec![]
                } else {
                    intset::collect_sorted(
                        face.shape
                            .faces_of(Sign::Output, dim, pos)
                            .iter()
                            .map(|&face_pos| face_maps[face_index][dim - 1][face_pos]),
                    )
                };

                if defined[dim][quotient] {
                    assert_eq!(
                        quotient_faces_in[dim][quotient], mapped_in,
                        "identified cells have incompatible input faces",
                    );
                    assert_eq!(
                        quotient_faces_out[dim][quotient], mapped_out,
                        "identified cells have incompatible output faces",
                    );
                } else {
                    quotient_faces_in[dim][quotient] = mapped_in;
                    quotient_faces_out[dim][quotient] = mapped_out;
                    defined[dim][quotient] = true;
                }
            }
        }
    }

    while quotient_basis.last().is_some_and(Vec::is_empty) {
        quotient_basis.pop();
        quotient_faces_in.pop();
        quotient_faces_out.pop();
    }
    let boundary_shape = Arc::new(FramedPoset::from_faces(
        quotient_basis,
        quotient_faces_in,
        quotient_faces_out,
    ));
    let embeddings = faces
        .iter()
        .zip(face_maps)
        .map(|(face, map)| {
            let embedding =
                Embedding::from_map(Arc::clone(&face.shape), Arc::clone(&boundary_shape), map);
            debug_assert!(embedding.is_closed());
            embedding
        })
        .collect();

    (boundary_shape, embeddings)
}

fn attach_top(
    frame: &IntSet,
    faces: &[BoxFace],
    boundary_shape: Arc<FramedPoset>,
    into_boundary: Vec<Embedding>,
) -> BoxConstruction {
    let top_dim = frame.len();
    let mut basis = boundary_shape.basis.clone();
    let mut faces_in = boundary_shape.faces_in.clone();
    let mut faces_out = boundary_shape.faces_out.clone();
    basis.resize_with(top_dim + 1, Vec::new);
    faces_in.resize_with(top_dim + 1, Vec::new);
    faces_out.resize_with(top_dim + 1, Vec::new);
    assert!(
        basis[top_dim].is_empty(),
        "the boundary colimit unexpectedly contains a full-frame cell",
    );

    let top_faces = |sign| {
        intset::collect_sorted(
            faces
                .iter()
                .zip(&into_boundary)
                .filter(move |(face, _)| face.sign == sign)
                .flat_map(|(face, embedding)| {
                    let face_basis: IntSet = frame
                        .iter()
                        .copied()
                        .filter(|&direction| direction != face.direction)
                        .collect();
                    let dim = face_basis.len();
                    face.shape.basis[dim]
                        .iter()
                        .enumerate()
                        .filter(move |(_, basis)| **basis == face_basis)
                        .map(move |(pos, _)| embedding.map[dim][pos])
                }),
        )
    };
    let top_input_faces = top_faces(Sign::Input);
    let top_output_faces = top_faces(Sign::Output);
    basis[top_dim].push(frame.clone());
    faces_in[top_dim].push(top_input_faces);
    faces_out[top_dim].push(top_output_faces);

    let shape = Arc::new(FramedPoset::from_faces(basis, faces_in, faces_out));
    let boundary_map: Vec<Vec<usize>> = boundary_shape
        .sizes()
        .into_iter()
        .map(|size| (0..size).collect())
        .collect();
    let boundary_into_box = Embedding::from_map(
        Arc::clone(&boundary_shape),
        Arc::clone(&shape),
        boundary_map,
    );
    let faces = faces
        .iter()
        .zip(into_boundary)
        .map(|(face, into_boundary)| {
            let embedding = Embedding::compose(&into_boundary, &boundary_into_box);
            debug_assert!(embedding.is_closed());
            BoxFaceEmbedding {
                direction: face.direction,
                sign: face.sign,
                embedding,
            }
        })
        .collect();

    BoxConstruction { shape, faces }
}

fn elementary_cylinder_recursive(
    input: &Arc<FramedPoset>,
    output: &Arc<FramedPoset>,
) -> Arc<FramedPoset> {
    let frame = input.active_directions();
    if frame.is_empty() {
        return Arc::new(tight_arrow());
    }

    let mut faces = Vec::with_capacity(2 * (frame.len() + 1));
    for sign in [Sign::Input, Sign::Output] {
        let shape = match sign {
            Sign::Input => input,
            Sign::Output => output,
        };
        faces.push(BoxFace::new(0, sign, Arc::new(shift(shape))));
    }

    for (index, &direction) in frame.iter().enumerate() {
        let cylinder_direction = direction
            .checked_add(1)
            .expect("cannot shift direction usize::MAX");
        for sign in [Sign::Input, Sign::Output] {
            let signed_shape = match sign {
                Sign::Input => input,
                Sign::Output => output,
            };
            let left = Arc::new(shift(&boundary_block(sign, &frame[..=index], signed_shape)));
            let inner_input = boundary_block(sign, &frame[index..], input);
            let inner_output = boundary_block(sign, &frame[index..], output);
            let inner_cylinder = elementary_cylinder_recursive(&inner_input, &inner_output);
            let shape = Arc::new(orthogonal_product(&left, &inner_cylinder));
            faces.push(BoxFace::new(cylinder_direction, sign, shape));
        }
    }

    rigid_box_construction(&faces).shape
}

fn boundary_block(sign: Sign, directions: &[usize], shape: &Arc<FramedPoset>) -> Arc<FramedPoset> {
    let word: Vec<_> = directions
        .iter()
        .map(|&direction| (sign, direction))
        .collect();
    iterated_boundary(&word, shape).0
}

fn tight_arrow() -> FramedPoset {
    FramedPoset::from_faces(
        vec![vec![vec![], vec![]], vec![vec![0]]],
        vec![vec![vec![], vec![]], vec![vec![0]]],
        vec![vec![vec![], vec![]], vec![vec![1]]],
    )
}

fn sign_index(sign: Sign) -> usize {
    match sign {
        Sign::Input => 0,
        Sign::Output => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point() -> Arc<FramedPoset> {
        Arc::new(FramedPoset::point())
    }

    fn arrow(direction: usize) -> Arc<FramedPoset> {
        Arc::new(FramedPoset::from_faces(
            vec![vec![vec![], vec![]], vec![vec![direction]]],
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![1]]],
        ))
    }

    fn cube(dimension: usize) -> Arc<FramedPoset> {
        let mut result = FramedPoset::point();
        for direction in 0..dimension {
            result = orthogonal_product(&result, &arrow(direction));
        }
        Arc::new(result)
    }

    #[test]
    fn one_direction_rigid_box_is_the_tight_arrow() {
        let construction = rigid_box_construction(&[
            BoxFace::new(0, Sign::Input, point()),
            BoxFace::new(0, Sign::Output, point()),
        ]);
        let expected = Arc::new(tight_arrow());

        assert!(isomorphic(&construction.shape, &expected));
        assert_eq!(construction.shape.sizes(), vec![2, 1]);
        for face in &construction.faces {
            let (_, actual) = boundary(face.sign, face.direction, &construction.shape);
            assert!(Embedding::same_subobject(&actual, &face.embedding));
        }
    }

    #[test]
    fn rigid_box_of_four_edges_is_the_standard_square() {
        let construction = rigid_box_construction(&[
            BoxFace::new(0, Sign::Input, arrow(1)),
            BoxFace::new(0, Sign::Output, arrow(1)),
            BoxFace::new(1, Sign::Input, arrow(0)),
            BoxFace::new(1, Sign::Output, arrow(0)),
        ]);
        let expected = cube(2);

        assert!(isomorphic(&construction.shape, &expected));
        assert_eq!(construction.shape.sizes(), vec![4, 4, 1]);
        for face in &construction.faces {
            let (_, actual) = boundary(face.sign, face.direction, &construction.shape);
            assert!(Embedding::same_subobject(&actual, &face.embedding));
        }
    }

    #[test]
    fn elementary_cylinder_advances_standard_cubes_by_one_dimension() {
        for dimension in 0..=2 {
            let shape = cube(dimension);
            let cylinder = elementary_cylinder(&shape, &shape);
            let expected = cube(dimension + 1);

            assert!(
                isomorphic(&cylinder, &expected),
                "failed in dimension {dimension}",
            );
        }
    }
}
