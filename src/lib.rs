//! Oriented framed posets.
//!
//! This crate is intentionally small and experimental.  It implements the core
//! finite-table data structures for oriented framed posets, their embeddings,
//! and pushouts of embeddings, without any diagram language or normalisation
//! machinery.

pub mod compass_spring_nd;
pub mod cubularity;
pub mod dot;
pub mod embedding;
pub mod intset;
pub mod isomorphism;
pub mod poset;
pub mod pushout;
pub mod random;
pub mod symmetry;

pub use cubularity::{is_cubular, is_strongly_cubular};
pub use dot::{
    Renderer, compass_spring_debug_json, embedding_to_dot, embedding_to_dot_with_params, to_dot,
    to_dot_with_params,
};
pub use embedding::{Embedding, EmbeddingIntersection, EmbeddingUnion, NO_PREIMAGE};
pub use isomorphism::{isomorphic, isomorphisms, normalize};
pub use poset::{FramedPoset, FramedPosetSubset, Sign, boundary, closure};
pub use random::RandomFramedPosetGenerator;
pub use symmetry::{
    DirectionImage, SignedPermutation, SymmetryError, transform, transform_embedding,
};
