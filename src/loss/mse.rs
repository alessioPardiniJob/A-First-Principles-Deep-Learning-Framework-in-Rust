use super::Loss;
use crate::Tensor;

/// Regression loss: accepts a continuous target tensor matching the shape
/// of the network's final output `z_L` (architecture.pdf, section 2.3).
pub struct MSELoss;

impl Loss for MSELoss {
    type Target = Tensor;

    fn forward(&self, _z_l: &Tensor, _target: &Self::Target) -> (f32, Tensor) {
        todo!("loss value and gradient, per architecture.pdf section 2.3")
    }
}
