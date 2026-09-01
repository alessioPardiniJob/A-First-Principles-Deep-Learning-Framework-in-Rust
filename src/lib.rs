//! A first-principles deep learning framework.
//!
//! The four core abstractions, as specified in `docs/architecture.pdf`
//! (Table 1), are [`Tensor`], [`Module`], [`Loss`], and [`Optimizer`].

pub mod loss;
pub mod module;
pub mod optimizer;
pub mod tensor;

pub use loss::Loss;
pub use module::Module;
pub use optimizer::Optimizer;
pub use tensor::Tensor;
