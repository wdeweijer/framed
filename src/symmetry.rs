//! Hyperoctahedral symmetries of oriented framed posets.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::embedding::Embedding;
use crate::intset;
use crate::poset::{FramedPoset, Sign};

/// The image of one source direction under a signed permutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DirectionImage {
    /// Target direction.
    pub direction: usize,
    /// Whether orientation in this source direction is reflected.
    pub reflected: bool,
}

/// A permutation and independent reflection of finitely many directions.
///
/// Entry `images[i]` records the image of source direction `i`. The length is
/// the ambient dimension on which the symmetry acts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SignedPermutation {
    images: Vec<DirectionImage>,
}

/// Failure to construct, combine, or apply a signed permutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymmetryError {
    /// A target direction lies outside the symmetry's ambient dimension.
    TargetOutOfRange {
        source: usize,
        target: usize,
        dimension: usize,
    },
    /// Two source directions have the same target direction.
    DuplicateTarget { target: usize },
    /// Two symmetries of different ambient dimensions were composed.
    DimensionMismatch { first: usize, second: usize },
    /// The poset uses a direction outside the symmetry's ambient dimension.
    DirectionNotCovered { direction: usize, dimension: usize },
}

impl fmt::Display for SymmetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TargetOutOfRange {
                source,
                target,
                dimension,
            } => write!(
                f,
                "source direction {source} maps to {target}, outside ambient dimension {dimension}"
            ),
            Self::DuplicateTarget { target } => {
                write!(f, "target direction {target} occurs more than once")
            }
            Self::DimensionMismatch { first, second } => {
                write!(f, "cannot compose ambient dimensions {first} and {second}")
            }
            Self::DirectionNotCovered {
                direction,
                dimension,
            } => write!(
                f,
                "poset direction {direction} is outside ambient dimension {dimension}"
            ),
        }
    }
}

impl Error for SymmetryError {}

impl SignedPermutation {
    /// Construct a signed permutation from source-indexed direction images.
    pub fn try_new(images: Vec<DirectionImage>) -> Result<Self, SymmetryError> {
        let dimension = images.len();
        let mut seen = vec![false; dimension];

        for (source, image) in images.iter().enumerate() {
            if image.direction >= dimension {
                return Err(SymmetryError::TargetOutOfRange {
                    source,
                    target: image.direction,
                    dimension,
                });
            }
            if seen[image.direction] {
                return Err(SymmetryError::DuplicateTarget {
                    target: image.direction,
                });
            }
            seen[image.direction] = true;
        }

        Ok(Self { images })
    }

    /// The identity symmetry in the given ambient dimension.
    pub fn identity(dimension: usize) -> Self {
        Self {
            images: (0..dimension)
                .map(|direction| DirectionImage {
                    direction,
                    reflected: false,
                })
                .collect(),
        }
    }

    /// Construct an unreflected permutation of directions.
    pub fn from_permutation(permutation: Vec<usize>) -> Result<Self, SymmetryError> {
        Self::try_new(
            permutation
                .into_iter()
                .map(|direction| DirectionImage {
                    direction,
                    reflected: false,
                })
                .collect(),
        )
    }

    /// Reflect one direction and fix all other directions.
    pub fn reflection(dimension: usize, direction: usize) -> Result<Self, SymmetryError> {
        if direction >= dimension {
            return Err(SymmetryError::TargetOutOfRange {
                source: direction,
                target: direction,
                dimension,
            });
        }

        let mut symmetry = Self::identity(dimension);
        symmetry.images[direction].reflected = true;
        Ok(symmetry)
    }

    /// Ambient dimension on which this symmetry acts.
    pub fn dimension(&self) -> usize {
        self.images.len()
    }

    /// Image of a source direction, if it lies in the ambient dimension.
    pub fn image_of(&self, direction: usize) -> Option<DirectionImage> {
        self.images.get(direction).copied()
    }

    /// Compose two symmetries, applying `first` and then `second`.
    pub fn compose(first: &Self, second: &Self) -> Result<Self, SymmetryError> {
        if first.dimension() != second.dimension() {
            return Err(SymmetryError::DimensionMismatch {
                first: first.dimension(),
                second: second.dimension(),
            });
        }

        let images = first
            .images
            .iter()
            .map(|first_image| {
                let second_image = second.images[first_image.direction];
                DirectionImage {
                    direction: second_image.direction,
                    reflected: first_image.reflected ^ second_image.reflected,
                }
            })
            .collect();

        Ok(Self { images })
    }

    /// Invert this signed permutation.
    pub fn inverse(&self) -> Self {
        let mut images = vec![
            DirectionImage {
                direction: 0,
                reflected: false,
            };
            self.dimension()
        ];

        for (source, image) in self.images.iter().enumerate() {
            images[image.direction] = DirectionImage {
                direction: source,
                reflected: image.reflected,
            };
        }

        Self { images }
    }
}

/// Apply a signed permutation of directions to an oriented framed poset.
///
/// Cell indices and unsigned cover relations are preserved. Bases are
/// permuted, and a cover relation changes sign exactly when its missing source
/// direction is reflected. The result is not marked as normalized.
pub fn transform(
    shape: &FramedPoset,
    symmetry: &SignedPermutation,
) -> Result<FramedPoset, SymmetryError> {
    debug_assert!(shape.well_formed());

    let basis = shape
        .basis
        .iter()
        .map(|level| {
            level
                .iter()
                .map(|cell_basis| {
                    cell_basis
                        .iter()
                        .map(|&direction| {
                            symmetry.image_of(direction).map_or_else(
                                || {
                                    Err(SymmetryError::DirectionNotCovered {
                                        direction,
                                        dimension: symmetry.dimension(),
                                    })
                                },
                                |image| Ok(image.direction),
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map(|mut transformed| {
                            transformed.sort_unstable();
                            transformed
                        })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let sizes = shape.sizes();
    let mut faces_in: Vec<Vec<Vec<usize>>> = sizes.iter().map(|&size| vec![vec![]; size]).collect();
    let mut faces_out: Vec<Vec<Vec<usize>>> =
        sizes.iter().map(|&size| vec![vec![]; size]).collect();

    for dim in 1..sizes.len() {
        for cell in 0..sizes[dim] {
            for sign in [Sign::Input, Sign::Output] {
                for &face in shape.faces_of(sign, dim, cell) {
                    let direction = intset::cover_direction(
                        shape.basis_of(dim - 1, face),
                        shape.basis_of(dim, cell),
                    )
                    .expect("a cover in a well-formed OFP removes one direction");
                    let image =
                        symmetry
                            .image_of(direction)
                            .ok_or(SymmetryError::DirectionNotCovered {
                                direction,
                                dimension: symmetry.dimension(),
                            })?;
                    let target_sign = if image.reflected {
                        sign.opposite()
                    } else {
                        sign
                    };
                    let target = match target_sign {
                        Sign::Input => &mut faces_in[dim][cell],
                        Sign::Output => &mut faces_out[dim][cell],
                    };
                    intset::insert(target, face);
                }
            }
        }
    }

    let transformed = FramedPoset::from_faces(basis, faces_in, faces_out);
    debug_assert!(transformed.well_formed());
    debug_assert!(!transformed.is_normal());
    Ok(transformed)
}

/// Apply a signed permutation of directions to an embedding.
///
/// The domain and codomain are transformed by the same symmetry. Since the
/// action preserves cell indices, the forward and inverse cell maps are
/// unchanged.
pub fn transform_embedding(
    embedding: &Embedding,
    symmetry: &SignedPermutation,
) -> Result<Embedding, SymmetryError> {
    debug_assert!(embedding.well_formed());

    let dom = Arc::new(transform(&embedding.dom, symmetry)?);
    let cod = Arc::new(transform(&embedding.cod, symmetry)?);
    let transformed = Embedding::make(dom, cod, embedding.map.clone(), embedding.inv.clone());

    debug_assert_eq!(embedding.is_closed(), transformed.is_closed());
    Ok(transformed)
}

#[cfg(test)]
mod tests {
    use crate::poset::boundary;

    use super::*;

    fn arrow(direction: usize) -> FramedPoset {
        FramedPoset::from_faces(
            vec![vec![vec![], vec![]], vec![vec![direction]]],
            vec![vec![vec![], vec![]], vec![vec![0]]],
            vec![vec![vec![], vec![]], vec![vec![1]]],
        )
    }

    fn square() -> FramedPoset {
        FramedPoset::from_faces(
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
        )
    }

    #[test]
    fn rejects_invalid_direction_images() {
        assert_eq!(
            SignedPermutation::from_permutation(vec![0, 0]),
            Err(SymmetryError::DuplicateTarget { target: 0 })
        );
        assert_eq!(
            SignedPermutation::from_permutation(vec![0, 2]),
            Err(SymmetryError::TargetOutOfRange {
                source: 1,
                target: 2,
                dimension: 2,
            })
        );
    }

    #[test]
    fn identity_preserves_a_poset_without_normalizing_it() {
        let shape = square();
        let transformed = transform(&shape, &SignedPermutation::identity(2)).unwrap();

        assert!(FramedPoset::equal(&shape, &transformed));
        assert!(!transformed.is_normal());
    }

    #[test]
    fn reflection_reverses_only_its_direction() {
        let reflected =
            transform(&arrow(0), &SignedPermutation::reflection(1, 0).unwrap()).unwrap();

        assert_eq!(reflected.faces_of(Sign::Input, 1, 0), &vec![1]);
        assert_eq!(reflected.faces_of(Sign::Output, 1, 0), &vec![0]);
    }

    #[test]
    fn permutation_relabels_bases_without_moving_cells() {
        let transformed = transform(
            &square(),
            &SignedPermutation::from_permutation(vec![1, 0]).unwrap(),
        )
        .unwrap();

        assert_eq!(transformed.basis_of(1, 0), &vec![1]);
        assert_eq!(transformed.basis_of(1, 1), &vec![1]);
        assert_eq!(transformed.basis_of(1, 2), &vec![0]);
        assert_eq!(transformed.basis_of(2, 0), &vec![0, 1]);
        assert_eq!(
            transformed.faces_of(Sign::Input, 2, 0),
            square().faces_of(Sign::Input, 2, 0)
        );
    }

    #[test]
    fn composition_matches_successive_actions() {
        let first = SignedPermutation::try_new(vec![
            DirectionImage {
                direction: 1,
                reflected: true,
            },
            DirectionImage {
                direction: 0,
                reflected: false,
            },
        ])
        .unwrap();
        let second = SignedPermutation::reflection(2, 0).unwrap();
        let composite = SignedPermutation::compose(&first, &second).unwrap();

        let successive = transform(&transform(&square(), &first).unwrap(), &second).unwrap();
        let direct = transform(&square(), &composite).unwrap();

        assert!(FramedPoset::equal(&successive, &direct));
    }

    #[test]
    fn inverse_undoes_the_action() {
        let symmetry = SignedPermutation::try_new(vec![
            DirectionImage {
                direction: 1,
                reflected: true,
            },
            DirectionImage {
                direction: 0,
                reflected: false,
            },
        ])
        .unwrap();
        let transformed = transform(&square(), &symmetry).unwrap();
        let restored = transform(&transformed, &symmetry.inverse()).unwrap();

        assert!(FramedPoset::equal(&square(), &restored));
        assert_eq!(
            SignedPermutation::compose(&symmetry, &symmetry.inverse()).unwrap(),
            SignedPermutation::identity(2)
        );
    }

    #[test]
    fn boundary_is_equivariant_for_direction_permutations() {
        let shape = Arc::new(square());
        let symmetry = SignedPermutation::from_permutation(vec![1, 0]).unwrap();
        let transformed_shape = Arc::new(transform(&shape, &symmetry).unwrap());

        for source_sign in [Sign::Input, Sign::Output] {
            for source_direction in 0..2 {
                let (source_boundary, _) = boundary(source_sign, source_direction, &shape);
                let transformed_boundary = transform(&source_boundary, &symmetry).unwrap();
                let image = symmetry.image_of(source_direction).unwrap();
                let (target_boundary, _) =
                    boundary(source_sign, image.direction, &transformed_shape);

                assert!(FramedPoset::equal(&transformed_boundary, &target_boundary));
            }
        }
    }

    #[test]
    fn ambient_dimension_is_explicit() {
        assert!(transform(&FramedPoset::empty(), &SignedPermutation::identity(4)).is_ok());
        assert!(transform(&FramedPoset::point(), &SignedPermutation::identity(4)).is_ok());
        assert_eq!(
            transform(&arrow(5), &SignedPermutation::identity(5)),
            Err(SymmetryError::DirectionNotCovered {
                direction: 5,
                dimension: 5,
            })
        );
        assert!(transform(&arrow(5), &SignedPermutation::identity(6)).is_ok());
    }

    #[test]
    fn embedding_action_preserves_maps_and_closedness() {
        let square = Arc::new(square());
        let (_, embedding) = boundary(Sign::Input, 0, &square);
        let symmetry = SignedPermutation::try_new(vec![
            DirectionImage {
                direction: 1,
                reflected: true,
            },
            DirectionImage {
                direction: 0,
                reflected: false,
            },
        ])
        .unwrap();

        let transformed = transform_embedding(&embedding, &symmetry).unwrap();

        assert_eq!(transformed.map, embedding.map);
        assert_eq!(transformed.inv, embedding.inv);
        assert!(FramedPoset::equal(
            &transformed.dom,
            &transform(&embedding.dom, &symmetry).unwrap()
        ));
        assert!(FramedPoset::equal(
            &transformed.cod,
            &transform(&embedding.cod, &symmetry).unwrap()
        ));
        assert!(embedding.is_closed());
        assert!(transformed.is_closed());
        assert!(!transformed.dom.is_normal());
        assert!(!transformed.cod.is_normal());
    }

    #[test]
    fn embedding_action_respects_composition() {
        let square = Arc::new(square());
        let (edge, edge_into_square) = boundary(Sign::Input, 0, &square);
        let (_, point_into_edge) = boundary(Sign::Output, 1, &edge);
        let point_into_square = Embedding::compose(&point_into_edge, &edge_into_square);
        let symmetry = SignedPermutation::from_permutation(vec![1, 0]).unwrap();

        let transformed_composite = transform_embedding(&point_into_square, &symmetry).unwrap();
        let transformed_point_into_edge = transform_embedding(&point_into_edge, &symmetry).unwrap();
        let transformed_edge_into_square =
            transform_embedding(&edge_into_square, &symmetry).unwrap();
        let composite_of_transforms =
            Embedding::compose(&transformed_point_into_edge, &transformed_edge_into_square);

        assert!(FramedPoset::equal(
            &transformed_composite.dom,
            &composite_of_transforms.dom
        ));
        assert!(FramedPoset::equal(
            &transformed_composite.cod,
            &composite_of_transforms.cod
        ));
        assert_eq!(transformed_composite.map, composite_of_transforms.map);
        assert_eq!(transformed_composite.inv, composite_of_transforms.inv);
    }

    #[test]
    fn inverse_symmetry_restores_an_embedding() {
        let square = Arc::new(square());
        let (_, embedding) = boundary(Sign::Input, 1, &square);
        let symmetry = SignedPermutation::try_new(vec![
            DirectionImage {
                direction: 1,
                reflected: false,
            },
            DirectionImage {
                direction: 0,
                reflected: true,
            },
        ])
        .unwrap();

        let transformed = transform_embedding(&embedding, &symmetry).unwrap();
        let restored = transform_embedding(&transformed, &symmetry.inverse()).unwrap();

        assert!(FramedPoset::equal(&restored.dom, &embedding.dom));
        assert!(FramedPoset::equal(&restored.cod, &embedding.cod));
        assert_eq!(restored.map, embedding.map);
        assert_eq!(restored.inv, embedding.inv);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "assertion failed: embedding.well_formed()")]
    fn embedding_action_debug_asserts_maps_are_well_formed() {
        let point = Arc::new(FramedPoset::point());
        let malformed = Embedding {
            dom: Arc::clone(&point),
            cod: point,
            map: vec![vec![1]],
            inv: vec![vec![0]],
        };

        let _ = transform_embedding(&malformed, &SignedPermutation::identity(0));
    }
}
