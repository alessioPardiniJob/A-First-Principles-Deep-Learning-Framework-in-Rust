use super::Module;
use crate::Tensor;

/// Owns a run-time extensible, heterogeneous list of layers via
/// `Vec<Box<dyn Module>>`, and itself satisfies the `Module` contract, so
/// an entire network can be handled from the outside exactly like a single
/// layer, nested inside another `Sequential`, or passed to an `Optimizer`
/// written solely against the `Module` interface (architecture.pdf,
/// section 2.1.5).
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add(&mut self, layer: Box<dyn Module>) -> &mut Self {
        self.layers.push(layer);
        self
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let mut current = input;
        for layer in self.layers.iter_mut() {
            current = layer.forward(current);
        }
        current
    }

    fn backward(&mut self, grad_output: Tensor) -> Tensor {
        let mut current = grad_output;
        for layer in self.layers.iter_mut().rev() {
            current = layer.backward(current);
        }
        current
    }

    fn params(&self) -> Vec<&Tensor> {
        self.layers.iter().flat_map(|l| l.params()).collect()
    }

    fn params_mut(&mut self) -> Vec<&mut Tensor> {
        self.layers.iter_mut().flat_map(|l| l.params_mut()).collect()
    }

    fn grads(&self) -> Vec<&Tensor> {
        self.layers.iter().flat_map(|l| l.grads()).collect()
    }

    fn zero_grad(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.zero_grad();
        }
    }
}
