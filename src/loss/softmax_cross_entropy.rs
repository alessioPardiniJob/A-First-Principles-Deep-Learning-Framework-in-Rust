use super::Loss;
use crate::Tensor;

/// Classification loss: accepts a slice of discrete class labels, avoiding
/// the semantic and memory-overhead trade-offs of forcing labels through
/// a `Tensor`-typed target (architecture.pdf, section 2.3).
pub struct SoftmaxCrossEntropy;

impl Loss for SoftmaxCrossEntropy {
    type Target = [usize];

    fn forward(&self, _z_l: &Tensor, _target: &Self::Target) -> (f32, Tensor) {
        todo!("loss value and gradient, per architecture.pdf section 2.3")
    }
}
