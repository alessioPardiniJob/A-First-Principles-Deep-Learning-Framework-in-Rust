use crate::Tensor;

pub mod mse;
pub mod softmax_cross_entropy;

pub use mse::MSELoss;
pub use softmax_cross_entropy::SoftmaxCrossEntropy;

/// Terminal objective evaluation, external to the chain of `Module`s
/// (architecture.pdf, section 2.3).
///
/// Given the final network output `z_L` and a target, computes both the
/// scalar sample-wise loss and the seed of the reverse accumulation,
/// `a_L = grad_{z_L} loss`, with the same shape as `z_L`.
///
/// Each implementation fixes its own `Target` type via an associated
/// type, rather than threading a generic parameter through the call
/// chain: `SoftmaxCrossEntropy` accepts discrete class labels, `MSELoss`
/// accepts a continuous tensor.
pub trait Loss {
    type Target: ?Sized;

    fn forward(&self, z_l: &Tensor, target: &Self::Target) -> (f32, Tensor);
}
