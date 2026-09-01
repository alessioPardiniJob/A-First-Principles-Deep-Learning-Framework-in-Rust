use super::Optimizer;
use crate::module::Module;
use crate::Tensor;

/// Stateless gradient descent: `theta_l <- theta_l - lr * grad_theta_l`
/// (architecture.pdf, section 2.4). `s_l = ∅` for every parameter block.
pub struct Sgd {
    pub lr: f32,
}

impl Optimizer for Sgd {
    fn step(&mut self, module: &mut dyn Module) {
        // Clone gradients into owned tensors (O(1): Rc::clone) to close the
        // shared borrow before requesting the exclusive borrow that
        // params_mut() needs (architecture.pdf, section 2.2.6/2.4).
        let grads: Vec<Tensor> = module.grads().into_iter().cloned().collect();
        for (theta, grad) in module.params_mut().into_iter().zip(grads.iter()) {
            *theta -= &(grad * self.lr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::{Linear, Sequential};

    /// Overrides `Linear`'s randomly-initialized weight/bias with known
    /// values (through the public `params_mut` accessor, per the
    /// architecture's ownership model) so the update can be checked
    /// against a hand computation rather than only its direction.
    #[test]
    fn step_computes_exact_update() {
        let mut lin = Linear::new(2, 1);
        {
            let mut params = lin.params_mut();
            *params[0] = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]); // weight [out=1, in=2]
            *params[1] = Tensor::zeros(&[1, 1]); // bias
        }

        lin.zero_grad();
        let _ = lin.forward(Tensor::from_vec(vec![1.0, 1.0], &[1, 2]));
        lin.backward(Tensor::from_vec(vec![1.0], &[1, 1]));

        // grad_weight = grad_output^T . input = [1] outer [1,1] = [[1, 1]]
        // grad_bias   = sum_batch(grad_output)  = [[1]]
        assert_eq!(lin.grads()[0].get(&[0, 0]), 1.0);
        assert_eq!(lin.grads()[0].get(&[0, 1]), 1.0);
        assert_eq!(lin.grads()[1].get(&[0, 0]), 1.0);

        let mut sgd = Sgd { lr: 0.1 };
        sgd.step(&mut lin);

        // theta <- theta - lr * grad
        assert!((lin.params()[0].get(&[0, 0]) - 0.9).abs() < 1e-6); // 1.0 - 0.1*1
        assert!((lin.params()[0].get(&[0, 1]) - 1.9).abs() < 1e-6); // 2.0 - 0.1*1
        assert!((lin.params()[1].get(&[0, 0]) - (-0.1)).abs() < 1e-6); // 0.0 - 0.1*1
    }

    /// Edge case flagged by the PDF itself (section 2.4): `zip`-based
    /// pairing of params/grads only stays correct if traversal order is
    /// stable across calls. Runs `step` over a two-layer `Sequential` and
    /// checks each layer's own parameter block keeps its own shape
    /// (which, since the two layers here have different shapes, would
    /// only accidentally hold if the pairing were shuffled).
    #[test]
    fn optimizer_step_on_multi_layer_sequential_preserves_shapes_and_ordering() {
        let mut model = Sequential::new();
        model.add(Box::new(Linear::new(2, 3)));
        model.add(Box::new(Linear::new(3, 1)));

        model.zero_grad();
        let _ = model.forward(Tensor::from_vec(vec![1.0, 1.0], &[1, 2]));
        model.backward(Tensor::from_vec(vec![1.0], &[1, 1]));

        let shapes_before: Vec<Vec<usize>> =
            model.params().iter().map(|t| t.shape().to_vec()).collect();
        assert_eq!(
            shapes_before,
            vec![vec![3, 2], vec![1, 3], vec![1, 3], vec![1, 1]]
        );

        let mut sgd = Sgd { lr: 0.01 };
        sgd.step(&mut model);

        let shapes_after: Vec<Vec<usize>> =
            model.params().iter().map(|t| t.shape().to_vec()).collect();
        assert_eq!(shapes_before, shapes_after, "step must not alter parameter shapes");
    }
}
