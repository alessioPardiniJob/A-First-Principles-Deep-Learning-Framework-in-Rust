use super::Module;
use crate::Tensor;

/// Reshapes `[B, ...]` to `[B, prod(...)]`, so a `Conv2d`'s `[B, C, H, W]`
/// output can feed a `Linear`'s `[B, F]` input.
///
/// Justified as a fundamental layer rather than a `Tensor`-level
/// convenience: it is the only place in the architecture where two
/// concrete `Module`s with structurally incompatible tensor ranks need to
/// be composed, so it earns its place as a `Module` in the composition,
/// on the same footing as `Linear` or `ReLU`. Both directions are O(1)
/// through [`Tensor::reshape`], since a `Conv2d`/`Linear` output is always
/// contiguous.
pub struct Flatten {
    input_shape: Option<Vec<usize>>,
}

impl Flatten {
    pub fn new() -> Self {
        Self { input_shape: None }
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Flatten {
    fn forward(&mut self, input: Tensor) -> Tensor {
        self.input_shape = Some(input.shape().to_vec());
        let batch = input.shape()[0];
        let rest: usize = input.shape()[1..].iter().product();
        input.reshape(&[batch, rest])
    }

    fn backward(&mut self, grad_output: Tensor) -> Tensor {
        let shape = self
            .input_shape
            .as_ref()
            .expect("forward must be called before backward");
        grad_output.reshape(shape)
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
    fn flatten_and_unflatten_round_trip_shapes() {
        let mut flatten = Flatten::new();
        let out = flatten.forward(Tensor::zeros(&[2, 3, 4, 4]));
        assert_eq!(out.shape(), &[2, 48]);

        let grad = flatten.backward(Tensor::zeros(&[2, 48]));
        assert_eq!(grad.shape(), &[2, 3, 4, 4]);
    }
}
