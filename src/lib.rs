//! Oriented framed posets.
//!
//! This crate is intentionally small and experimental.  It implements the core
//! finite-table data structures for oriented framed posets, their embeddings,
//! and pushouts of embeddings, without any diagram language or normalisation
//! machinery.

pub mod compass_spring_nd;
pub mod dot;
pub mod embedding;
pub mod intset;
pub mod poset;
pub mod pushout;
pub mod random;

pub use dot::{
    Renderer, embedding_to_dot, embedding_to_dot_with_params, to_dot, to_dot_with_params,
};
pub use embedding::{Embedding, EmbeddingIntersection, EmbeddingUnion, NO_PREIMAGE};
pub use poset::{FramedPoset, FramedPosetSubset, Sign, boundary, closure};
pub use random::random_framed_poset;
