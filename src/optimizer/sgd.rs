use super::Optimizer;
use crate::module::Module;

/// Stateless gradient descent: `theta_l <- theta_l - lr * grad_theta_l`
/// (architecture.pdf, section 2.4). `s_l = ∅` for every parameter block.
pub struct Sgd {
    pub lr: f32,
}

impl Optimizer for Sgd {
    fn step(&mut self, _module: &mut dyn Module) {
        todo!("update rule, per architecture.pdf section 2.4")
    }
}
