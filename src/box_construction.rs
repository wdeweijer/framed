//! Box constructions and elementary cylinders.

use std::collections::HashSet;
use std::ops::{Index, IndexMut};
use std::sync::Arc;

use crate::cubularity::{CubularityMode, is_cubular};
use crate::embedding::Embedding;
use crate::intset::{self, IntSet};
use crate::isomorphism::{isomorphic, isomorphisms};
use crate::orthogonal::orthogonal_product;
use crate::poset::{FramedPoset, Sign, boundary, iterated_boundary, shift};
use crate::pushout::{ColimitSpan, finite_colimit};
use crate::volumetric::is_volumetric;

/// A value for each input/output sign.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedPair<T> {
    pub input: T,
    pub output: T,
}

impl<T> SignedPair<T> {
    fn map<U>(self, mut map: impl FnMut(T) -> U) -> SignedPair<U> {
        SignedPair {
            input: map(self.input),
            output: map(self.output),
        }
    }
}

impl<T> Index<Sign> for SignedPair<T> {
    type Output = T;

    fn index(&self, sign: Sign) -> &Self::Output {
        match sign {
            Sign::Input => &self.input,
            Sign::Output => &self.output,
        }
    }
}

impl<T> IndexMut<Sign> for SignedPair<T> {
    fn index_mut(&mut self, sign: Sign) -> &mut Self::Output {
        match sign {
            Sign::Input => &mut self.input,
            Sign::Output => &mut self.output,
        }
    }
}

/// Signed box faces indexed by position in a sorted frame.
#[derive(Debug, Clone)]
pub struct BoxFaces {
    pub frame: IntSet,
    pub by_direction: Vec<SignedPair<Arc<FramedPoset>>>,
}

impl BoxFaces {
    /// The signed face in an actual frame direction.
    pub fn face(&self, sign: Sign, direction: usize) -> Option<&Arc<FramedPoset>> {
        self.frame
            .binary_search(&direction)
            .ok()
            .map(|index| &self.by_direction[index][sign])
    }

    fn well_formed(&self) -> bool {
        if self.frame.is_empty()
            || !intset::is_sorted_unique(&self.frame)
            || self.frame.len() != self.by_direction.len()
        {
            return false;
        }

        let expected_input_dimension = self.frame.len() as isize - 1;
        self.frame
            .iter()
            .copied()
            .zip(&self.by_direction)
            .all(|(direction, signed_faces)| {
                [Sign::Input, Sign::Output].into_iter().all(|sign| {
                    let face = &signed_faces[sign];
                    let face_frame = face.active_directions();
                    intset::is_subset(&face_frame, &self.frame)
                        && face_frame.binary_search(&direction).is_err()
                        && (sign != Sign::Input || face.dim() == expected_input_dimension)
                        && is_cubular(CubularityMode::Strong, face)
                })
            })
    }
}

/// An isomorphism identifying the overlap of two box faces.
///
/// `left_direction` and `right_direction` are positions in [`BoxFaces::frame`],
/// not the direction values themselves. The isomorphism has type
///
/// `boundary(right_sign, frame[right_direction], left_face)
///     -> boundary(left_sign, frame[left_direction], right_face)`.
#[derive(Debug, Clone)]
pub struct BoxGluing {
    pub left_direction: usize,
    pub left_sign: Sign,
    pub right_direction: usize,
    pub right_sign: Sign,
    pub isomorphism: Embedding,
}

/// A box together with the inclusions of all its signed directional faces.
#[derive(Debug, Clone)]
pub struct BoxConstruction {
    pub shape: Arc<FramedPoset>,
    pub frame: IntSet,
    pub face_embeddings: Vec<SignedPair<Embedding>>,
}

impl BoxConstruction {
    /// Inclusion of the requested signed directional face.
    pub fn face(&self, sign: Sign, direction: usize) -> Option<&Embedding> {
        self.frame
            .binary_search(&direction)
            .ok()
            .map(|index| &self.face_embeddings[index][sign])
    }
}

/// Construct a box from signed directional faces and overlap isomorphisms.
///
/// There must be one gluing for every pair of signed faces in unequal frame
/// directions. Gluings may be supplied in either direction.
pub fn box_construction(faces: &BoxFaces, gluings: &[BoxGluing]) -> BoxConstruction {
    debug_assert!(faces.well_formed());
    let prepared = prepare_gluings(faces, gluings);
    debug_assert!(prepared_gluings_well_formed(faces, &prepared));
    construct_prepared_box(faces, prepared)
}

/// Construct a box whose overlap isomorphisms are forced by rigidity.
///
/// Panics unless every pair of required directional boundaries has exactly
/// one isomorphism.
pub fn rigid_box_construction(faces: &BoxFaces) -> BoxConstruction {
    debug_assert!(faces.well_formed());
    debug_assert!(
        faces
            .by_direction
            .iter()
            .all(|faces| faces.input.is_rigid() && faces.output.is_rigid())
    );

    let mut prepared = Vec::new();
    for left in 0..faces.frame.len() {
        for right in left + 1..faces.frame.len() {
            for left_sign in [Sign::Input, Sign::Output] {
                for right_sign in [Sign::Input, Sign::Output] {
                    let left_face = &faces.by_direction[left][left_sign];
                    let right_face = &faces.by_direction[right][right_sign];
                    let (left_boundary, left_into_face) =
                        boundary(right_sign, faces.frame[right], left_face);
                    let (right_boundary, right_into_face) =
                        boundary(left_sign, faces.frame[left], right_face);
                    let mut candidates = isomorphisms(&left_boundary, &right_boundary);
                    assert_eq!(
                        candidates.len(),
                        1,
                        "box faces ({left_sign:?}, {}) and ({right_sign:?}, {}) must have exactly one boundary isomorphism",
                        faces.frame[left],
                        faces.frame[right],
                    );

                    prepared.push(PreparedGluing {
                        left_direction: left,
                        left_sign,
                        right_direction: right,
                        right_sign,
                        left_into_face,
                        right_into_face,
                        isomorphism: candidates.pop().unwrap(),
                    });
                }
            }
        }
    }

    debug_assert!(prepared_gluings_well_formed(faces, &prepared));
    construct_prepared_box(faces, prepared)
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
struct PreparedGluing {
    left_direction: usize,
    left_sign: Sign,
    right_direction: usize,
    right_sign: Sign,
    left_into_face: Embedding,
    right_into_face: Embedding,
    isomorphism: Embedding,
}

fn prepare_gluings(faces: &BoxFaces, gluings: &[BoxGluing]) -> Vec<PreparedGluing> {
    gluings
        .iter()
        .map(|gluing| {
            let mut left_direction = gluing.left_direction;
            let mut left_sign = gluing.left_sign;
            let mut right_direction = gluing.right_direction;
            let mut right_sign = gluing.right_sign;
            let mut isomorphism = gluing.isomorphism.clone();
            if left_direction > right_direction {
                std::mem::swap(&mut left_direction, &mut right_direction);
                std::mem::swap(&mut left_sign, &mut right_sign);
                isomorphism = isomorphism.inverse_isomorphism();
            }

            let left_face = &faces.by_direction[left_direction][left_sign];
            let right_face = &faces.by_direction[right_direction][right_sign];
            let (_, left_into_face) = boundary(right_sign, faces.frame[right_direction], left_face);
            let (_, right_into_face) = boundary(left_sign, faces.frame[left_direction], right_face);
            PreparedGluing {
                left_direction,
                left_sign,
                right_direction,
                right_sign,
                left_into_face,
                right_into_face,
                isomorphism,
            }
        })
        .collect()
}

fn prepared_gluings_well_formed(faces: &BoxFaces, gluings: &[PreparedGluing]) -> bool {
    let mut labels = HashSet::with_capacity(gluings.len());
    for gluing in gluings {
        if gluing.left_direction >= gluing.right_direction
            || gluing.right_direction >= faces.frame.len()
            || !labels.insert((
                gluing.left_direction,
                gluing.left_sign,
                gluing.right_direction,
                gluing.right_sign,
            ))
            || !gluing.isomorphism.is_isomorphism()
            || !FramedPoset::equal(&gluing.isomorphism.dom, &gluing.left_into_face.dom)
            || !FramedPoset::equal(&gluing.isomorphism.cod, &gluing.right_into_face.dom)
        {
            return false;
        }
    }

    for left in 0..faces.frame.len() {
        for right in left + 1..faces.frame.len() {
            for left_sign in [Sign::Input, Sign::Output] {
                for right_sign in [Sign::Input, Sign::Output] {
                    if !labels.contains(&(left, left_sign, right, right_sign)) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn construct_prepared_box(faces: &BoxFaces, prepared: Vec<PreparedGluing>) -> BoxConstruction {
    let mut objects = Vec::with_capacity(2 * faces.by_direction.len());
    let object_indices: Vec<_> = faces
        .by_direction
        .iter()
        .map(|faces| {
            let input = objects.len();
            objects.push(Arc::clone(&faces.input));
            let output = objects.len();
            objects.push(Arc::clone(&faces.output));
            SignedPair { input, output }
        })
        .collect();
    let legs: Vec<_> = prepared
        .iter()
        .map(|gluing| {
            (
                gluing.left_into_face.clone(),
                Embedding::compose(&gluing.isomorphism, &gluing.right_into_face),
            )
        })
        .collect();
    let spans: Vec<_> = prepared
        .iter()
        .zip(&legs)
        .map(|(gluing, (into_left, into_right))| ColimitSpan {
            left: object_indices[gluing.left_direction][gluing.left_sign],
            right: object_indices[gluing.right_direction][gluing.right_sign],
            into_left,
            into_right,
        })
        .collect();
    let colimit = finite_colimit(&objects, &spans);
    let mut injections = colimit.injections.into_iter();
    let into_boundary = (0..faces.frame.len())
        .map(|_| SignedPair {
            input: injections.next().unwrap(),
            output: injections.next().unwrap(),
        })
        .collect();
    debug_assert!(injections.next().is_none());

    attach_top(faces, colimit.tip, into_boundary)
}

fn attach_top(
    faces: &BoxFaces,
    boundary_shape: Arc<FramedPoset>,
    into_boundary: Vec<SignedPair<Embedding>>,
) -> BoxConstruction {
    let top_dim = faces.frame.len();
    let mut basis = boundary_shape.basis.clone();
    let mut faces_in = boundary_shape.faces_in.clone();
    let mut faces_out = boundary_shape.faces_out.clone();
    basis.resize_with(top_dim + 1, Vec::new);
    faces_in.resize_with(top_dim + 1, Vec::new);
    faces_out.resize_with(top_dim + 1, Vec::new);
    debug_assert!(basis[top_dim].is_empty());

    let top_faces = |sign| {
        intset::collect_sorted(
            faces
                .frame
                .iter()
                .copied()
                .zip(&faces.by_direction)
                .zip(&into_boundary)
                .flat_map(|((direction, signed_faces), signed_embedding)| {
                    let face_basis: IntSet = faces
                        .frame
                        .iter()
                        .copied()
                        .filter(|&candidate| candidate != direction)
                        .collect();
                    let dim = face_basis.len();
                    signed_faces[sign]
                        .basis
                        .get(dim)
                        .into_iter()
                        .flatten()
                        .enumerate()
                        .filter(move |(_, basis)| **basis == face_basis)
                        .map(move |(pos, _)| signed_embedding[sign].map[dim][pos])
                }),
        )
    };
    let top_input_faces = top_faces(Sign::Input);
    let top_output_faces = top_faces(Sign::Output);
    basis[top_dim].push(faces.frame.clone());
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
    let face_embeddings: Vec<SignedPair<Embedding>> = into_boundary
        .into_iter()
        .map(|embeddings| {
            embeddings.map(|embedding| Embedding::compose(&embedding, &boundary_into_box))
        })
        .collect();
    debug_assert!(
        face_embeddings
            .iter()
            .all(|embeddings: &SignedPair<Embedding>| {
                embeddings.input.is_closed() && embeddings.output.is_closed()
            })
    );

    BoxConstruction {
        shape,
        frame: faces.frame.clone(),
        face_embeddings,
    }
}

fn elementary_cylinder_recursive(
    input: &Arc<FramedPoset>,
    output: &Arc<FramedPoset>,
) -> Arc<FramedPoset> {
    let frame = input.active_directions();
    if frame.is_empty() {
        return Arc::new(tight_arrow());
    }

    let cylinder_frame: IntSet = std::iter::once(0)
        .chain(frame.iter().map(|direction| {
            direction
                .checked_add(1)
                .expect("cannot shift direction usize::MAX")
        }))
        .collect();
    let mut by_direction = Vec::with_capacity(cylinder_frame.len());
    by_direction.push(SignedPair {
        input: Arc::new(shift(input)),
        output: Arc::new(shift(output)),
    });

    for index in 0..frame.len() {
        by_direction.push(SignedPair {
            input: elementary_cylinder_face(Sign::Input, index, &frame, input, input, output),
            output: elementary_cylinder_face(Sign::Output, index, &frame, output, input, output),
        });
    }

    rigid_box_construction(&BoxFaces {
        frame: cylinder_frame,
        by_direction,
    })
    .shape
}

fn elementary_cylinder_face(
    sign: Sign,
    index: usize,
    frame: &IntSet,
    signed_shape: &Arc<FramedPoset>,
    input: &Arc<FramedPoset>,
    output: &Arc<FramedPoset>,
) -> Arc<FramedPoset> {
    let left = Arc::new(shift(&boundary_block(sign, &frame[..=index], signed_shape)));
    let inner_input = boundary_block(sign, &frame[index..], input);
    let inner_output = boundary_block(sign, &frame[index..], output);
    let inner_cylinder = elementary_cylinder_recursive(&inner_input, &inner_output);
    Arc::new(orthogonal_product(&left, &inner_cylinder))
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

    fn unique_gluings(faces: &BoxFaces) -> Vec<BoxGluing> {
        let mut gluings = Vec::new();
        for left in 0..faces.frame.len() {
            for right in left + 1..faces.frame.len() {
                for left_sign in [Sign::Input, Sign::Output] {
                    for right_sign in [Sign::Input, Sign::Output] {
                        let (left_boundary, _) = boundary(
                            right_sign,
                            faces.frame[right],
                            &faces.by_direction[left][left_sign],
                        );
                        let (right_boundary, _) = boundary(
                            left_sign,
                            faces.frame[left],
                            &faces.by_direction[right][right_sign],
                        );
                        let mut candidates = isomorphisms(&left_boundary, &right_boundary);
                        assert_eq!(candidates.len(), 1);
                        gluings.push(BoxGluing {
                            left_direction: left,
                            left_sign,
                            right_direction: right,
                            right_sign,
                            isomorphism: candidates.pop().unwrap(),
                        });
                    }
                }
            }
        }
        gluings
    }

    fn assert_faces_are_boundaries(construction: &BoxConstruction) {
        for (&direction, embeddings) in construction.frame.iter().zip(&construction.face_embeddings)
        {
            for sign in [Sign::Input, Sign::Output] {
                let (_, actual) = boundary(sign, direction, &construction.shape);
                assert!(Embedding::same_subobject(&actual, &embeddings[sign]));
            }
        }
    }

    #[test]
    fn one_direction_rigid_box_is_the_tight_arrow() {
        let construction = rigid_box_construction(&BoxFaces {
            frame: vec![0],
            by_direction: vec![SignedPair {
                input: point(),
                output: point(),
            }],
        });
        let expected = Arc::new(tight_arrow());

        assert!(isomorphic(&construction.shape, &expected));
        assert_eq!(construction.shape.sizes(), vec![2, 1]);
        assert_faces_are_boundaries(&construction);
    }

    #[test]
    fn explicit_and_rigid_boxes_of_four_edges_are_the_standard_square() {
        let faces = BoxFaces {
            frame: vec![0, 1],
            by_direction: vec![
                SignedPair {
                    input: arrow(1),
                    output: arrow(1),
                },
                SignedPair {
                    input: arrow(0),
                    output: arrow(0),
                },
            ],
        };
        let explicit = box_construction(&faces, &unique_gluings(&faces));
        let rigid = rigid_box_construction(&faces);
        let expected = cube(2);

        for construction in [explicit, rigid] {
            assert!(isomorphic(&construction.shape, &expected));
            assert_eq!(construction.shape.sizes(), vec![4, 4, 1]);
            assert_faces_are_boundaries(&construction);
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

    #[test]
    fn elementary_cylinder_allows_a_lower_dimensional_output() {
        let input = arrow(0);
        let output = point();
        let cylinder = elementary_cylinder(&input, &output);

        assert_eq!(cylinder.active_directions(), vec![0, 1]);
        assert!(is_volumetric(&cylinder));
    }
}
