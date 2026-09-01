//! Dataset loading for `main.rs`. Binary-only: not part of the library
//! crate (`lib.rs` never declares this module), so none of it is exposed
//! as framework API — it only produces plain `Vec<f32>`/`Vec<usize>`
//! buffers that `main.rs` wraps into `Tensor`s via the existing public
//! constructors.

pub mod housing;
pub mod mnist;
