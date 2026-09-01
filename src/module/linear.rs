use super::Module;
use crate::Tensor;

/// Fully connected layer: `z_l = z_{l-1} W^T + b` (architecture.pdf,
/// section 2.2.6, the running example used to justify the ownership
/// model).
///
/// `weight` has shape `[out_features, in_features]`, so `weight.t()` is
/// `[in_features, out_features]` and admits `input.dot(&weight.t())` for
/// `input: [batch, in_features]`. `bias` is kept as a row `[1,
/// out_features]` so it broadcasts over the batch dimension.
pub struct Linear {
    weight: Tensor,
    bias: Tensor,
    grad_weight: Tensor,
    grad_bias: Tensor,
    input_cache: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let scale = 1.0 / (in_features as f32).sqrt();
        Self {
            weight: Tensor::rand(&[out_features, in_features], scale),
            bias: Tensor::zeros(&[1, out_features]),
            grad_weight: Tensor::zeros(&[out_features, in_features]),
            grad_bias: Tensor::zeros(&[1, out_features]),
            input_cache: None,
        }
    }
}

impl Module for Linear {
    fn forward(&mut self, input: Tensor) -> Tensor {
        self.input_cache = Some(input);
        let cached = self.input_cache.as_ref().unwrap();
        cached.dot(&self.weight.t()) + &self.bias
    }

    fn backward(&mut self, grad_output: Tensor) -> Tensor {
        let input = self
            .input_cache
            .as_ref()
            .expect("forward must be called before backward");

        self.grad_weight += &grad_output.t().dot(input);
        self.grad_bias += &grad_output.sum_axis(0);

        grad_output.dot(&self.weight)
    }

    fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn grads(&self) -> Vec<&Tensor> {
        vec![&self.grad_weight, &self.grad_bias]
    }

    fn zero_grad(&mut self) {
        self.grad_weight = Tensor::zeros_like(&self.weight);
        self.grad_bias = Tensor::zeros_like(&self.bias);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_shape() {
        let mut lin = Linear::new(4, 3);
        let out = lin.forward(Tensor::zeros(&[5, 4]));
        assert_eq!(out.shape(), &[5, 3]);
    }

    #[test]
    fn backward_shapes_and_gradient_accumulation() {
        let mut lin = Linear::new(4, 3);
        let _ = lin.forward(Tensor::zeros(&[5, 4]));
        let grad_in = lin.backward(Tensor::zeros(&[5, 3]));
        assert_eq!(grad_in.shape(), &[5, 4]);
        assert_eq!(lin.grads()[0].shape(), &[3, 4]);
        assert_eq!(lin.grads()[1].shape(), &[1, 3]);
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let mut lin = Linear::new(3, 2);
        let input = Tensor::from_vec(vec![0.5, -1.0, 2.0], &[1, 3]);

        let _ = lin.forward(input.clone());
        let grad_output = Tensor::from_vec(vec![1.0, 1.0], &[1, 2]);
        let grad_input = lin.backward(grad_output.clone());

        let eps = 1e-3;
        for i in 0..3 {
            let mut plus = input.clone();
            let mut minus = input.clone();
            let mut data_plus: Vec<f32> = plus.iter().collect();
            data_plus[i] += eps;
            plus = Tensor::from_vec(data_plus, &[1, 3]);
            let mut data_minus: Vec<f32> = minus.iter().collect();
            data_minus[i] -= eps;
            minus = Tensor::from_vec(data_minus, &[1, 3]);

            let mut probe = Linear::new(3, 2);
            probe.weight = lin.params()[0].clone();
            probe.bias = lin.params()[1].clone();

            let out_plus = probe.forward(plus).iter().sum::<f32>();
            let out_minus = probe.forward(minus).iter().sum::<f32>();
            let numerical = (out_plus - out_minus) / (2.0 * eps);
            let analytical = grad_input.get(&[0, i]);
            assert!(
                (numerical - analytical).abs() < 1e-2,
                "grad mismatch at {i}: numerical={numerical}, analytical={analytical}"
            );
        }
    }
}
