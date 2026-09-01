use crate::module::Module;

pub mod momentum;
pub mod sgd;

pub use momentum::Momentum;
pub use sgd::Sgd;

/// Consumes the per-example parameter gradients exposed through the
/// `Module` mechanism, accumulates them into a batch gradient, and applies
/// a parameterized update rule to each parameter block, maintaining any
/// auxiliary state the rule requires (architecture.pdf, section 2.4).
///
/// `step` takes `&mut dyn Module` rather than a generic `M: Module`, so the
/// optimizer is decoupled from the concrete model type: it only ever sees
/// the parameters and gradients re-exposed through the `Module` interface.
pub trait Optimizer {
    fn step(&mut self, module: &mut dyn Module);
}
