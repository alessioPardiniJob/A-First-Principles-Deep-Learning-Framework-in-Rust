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
    fn step(&mut self, module: &mut dyn Module) {
        let grads: Vec<Tensor> = module.grads().into_iter().cloned().collect();

        if self.velocity.is_empty() {
            self.velocity = grads.iter().map(Tensor::zeros_like).collect();
        }

        for ((theta, grad), v) in module
            .params_mut()
            .into_iter()
            .zip(grads.iter())
            .zip(self.velocity.iter_mut())
        {
            *v = &(&*v * self.beta) + grad;
            *theta -= &(&*v * self.lr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::Linear;

    #[test]
    fn velocity_is_lazily_allocated_and_moves_parameters() {
        let mut lin = Linear::new(2, 1);
        lin.zero_grad();
        let _ = lin.forward(Tensor::from_vec(vec![1.0, 1.0], &[1, 2]));
        lin.backward(Tensor::from_vec(vec![1.0], &[1, 1]));

        let mut momentum = Momentum::new(0.1, 0.9);
        assert!(momentum.velocity.is_empty());
        momentum.step(&mut lin);
        assert_eq!(momentum.velocity.len(), 2);
        assert_eq!(momentum.velocity[0].shape(), lin.params()[0].shape());
    }

    /// Hand-computed reference across two steps, with weight/input fixed
    /// via the public API so `grad_weight = grad_output^T . input = 1.0`
    /// on every step regardless of the current weight value:
    /// v1 = beta*0 + 1 = 1;      w1 = 1.0 - lr*v1 = 1.0 - 0.1*1   = 0.9
    /// v2 = beta*v1 + 1 = 1.5;   w2 = w1  - lr*v2 = 0.9 - 0.1*1.5 = 0.75
    #[test]
    fn multi_step_update_matches_hand_computation() {
        let mut lin = Linear::new(1, 1);
        {
            let mut params = lin.params_mut();
            *params[0] = Tensor::from_vec(vec![1.0], &[1, 1]);
            *params[1] = Tensor::zeros(&[1, 1]);
        }
        let mut momentum = Momentum::new(0.1, 0.5);

        lin.zero_grad();
        let _ = lin.forward(Tensor::from_vec(vec![1.0], &[1, 1]));
        lin.backward(Tensor::from_vec(vec![1.0], &[1, 1]));
        momentum.step(&mut lin);
        assert!((lin.params()[0].get(&[0, 0]) - 0.9).abs() < 1e-6);

        lin.zero_grad();
        let _ = lin.forward(Tensor::from_vec(vec![1.0], &[1, 1]));
        lin.backward(Tensor::from_vec(vec![1.0], &[1, 1]));
        momentum.step(&mut lin);
        assert!((lin.params()[0].get(&[0, 0]) - 0.75).abs() < 1e-6);
    }
}
