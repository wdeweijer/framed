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

pub use dot::{embedding_to_dot, to_compass_spring_dot, to_compass_spring_dot_with_params, to_dot};
pub use embedding::{Embedding, NO_PREIMAGE};
pub use poset::{FramedPoset, Sign, boundary};
