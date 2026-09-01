use super::Module;
use crate::Tensor;

/// Elementwise rectified linear activation, `max(0, z)`. Parameter-free:
/// `params`/`grads` are empty, and `zero_grad` is a no-op — a layer is
/// structurally agnostic to how it is optimized (architecture.pdf,
/// section 2.4).
pub struct ReLU {
    input_cache: Option<Tensor>,
}

impl ReLU {
    pub fn new() -> Self {
        Self { input_cache: None }
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ReLU {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let out = input.map(|x| x.max(0.0));
        self.input_cache = Some(input);
        out
    }

    fn backward(&mut self, grad_output: Tensor) -> Tensor {
        let input = self
            .input_cache
            .as_ref()
            .expect("forward must be called before backward");
        grad_output.zip_map(input, |g, x| if x > 0.0 { g } else { 0.0 })
    }

    fn params(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn grads(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn zero_grad(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroes_negative_and_passes_through_positive() {
        let mut relu = ReLU::new();
        let out = relu.forward(Tensor::from_vec(vec![-1.0, 0.0, 2.0], &[1, 3]));
        assert_eq!(out.get(&[0, 0]), 0.0);
        assert_eq!(out.get(&[0, 1]), 0.0);
        assert_eq!(out.get(&[0, 2]), 2.0);
    }

    #[test]
    fn blocks_gradient_where_input_was_non_positive() {
        let mut relu = ReLU::new();
        let _ = relu.forward(Tensor::from_vec(vec![-1.0, 0.0, 2.0], &[1, 3]));
        let grad_in = relu.backward(Tensor::from_vec(vec![1.0, 1.0, 1.0], &[1, 3]));
        assert_eq!(grad_in.get(&[0, 0]), 0.0);
        assert_eq!(grad_in.get(&[0, 1]), 0.0);
        assert_eq!(grad_in.get(&[0, 2]), 1.0);
    }
}
