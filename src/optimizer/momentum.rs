use super::Optimizer;
use crate::module::Module;
use crate::Tensor;

/// Gradient descent with a velocity term:
/// `v_l <- beta * v_l + grad_theta_l`, `theta_l <- theta_l - lr * v_l`
/// (architecture.pdf, section 2.4).
///
/// `velocity` holds `s_l`, one tensor per parameter block, lazily
/// allocated on the first call to `step` since the optimizer does not know
/// the model's architecture ahead of time. Positionally aligned with
/// `params_mut()` and `grads()`.
pub struct Momentum {
    pub lr: f32,
    pub beta: f32,
    velocity: Vec<Tensor>,
}

impl Momentum {
    pub fn new(lr: f32, beta: f32) -> Self {
        Self {
            lr,
            beta,
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for Momentum {
    fn step(&mut self, _module: &mut dyn Module) {
        todo!("update rule and lazy velocity allocation, per architecture.pdf section 2.4")
    }
}
