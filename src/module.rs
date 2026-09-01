use crate::Tensor;

pub mod sequential;

pub use sequential::Sequential;

/// A single parameterized transformation `g_l(z_{l-1}; theta_l)`, and the
/// unit of composition for an arbitrarily long, ordered, backward-traversable
/// network (architecture.pdf, section 2.1).
///
/// Implemented as a trait object (`dyn Module`, section 2.1.5) rather than
/// through generics: no method introduces an extra generic type parameter,
/// and none takes or returns `Self` by value, which keeps `Module`
/// object-safe.
pub trait Module {
    /// Computes the output tensor `z_l` from the input tensor `z_{l-1}`.
    ///
    /// Takes `input` by value: the layer caches it for the backward pass
    /// with zero data duplication (section 2.2.6).
    fn forward(&mut self, input: Tensor) -> Tensor;

    /// Given the upstream gradient `a_l = d loss / d z_l`, returns the
    /// downstream gradient `a_{l-1} = d loss / d z_{l-1}`, using whatever
    /// intermediate state was retained during `forward`.
    fn backward(&mut self, grad_output: Tensor) -> Tensor;

    /// Lends read-only access to the layer's parameters `theta_l`.
    fn params(&self) -> Vec<&Tensor>;

    /// Lends mutable access to the layer's parameters, so an optimizer can
    /// update them in place.
    fn params_mut(&mut self) -> Vec<&mut Tensor>;

    /// Lends read-only access to the accumulated parameter gradients,
    /// `grad_theta_l L`.
    fn grads(&self) -> Vec<&Tensor>;

    /// Clears the accumulated parameter gradients before a new
    /// forward/backward accumulation cycle (section 2.4).
    fn zero_grad(&mut self);
}
