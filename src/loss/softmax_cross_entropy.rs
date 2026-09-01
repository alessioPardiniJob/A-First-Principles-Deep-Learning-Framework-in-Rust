use super::Loss;
use crate::Tensor;

/// Classification loss: accepts a slice of discrete class labels, avoiding
/// the semantic and memory-overhead trade-offs of forcing labels through
/// a `Tensor`-typed target (architecture.pdf, section 2.3).
///
/// Fuses softmax and cross-entropy: `z_L` is the raw logits, not a
/// post-softmax distribution, so no separate `Softmax` `Module` is needed
/// (it would only duplicate the numerically-stable log-sum-exp already
/// required here, and the fused gradient `(softmax(z) - onehot(y)) / B` is
/// simpler than differentiating through a standalone softmax layer).
/// Reduction is a mean over the `B` examples in the batch.
pub struct SoftmaxCrossEntropy;

impl Loss for SoftmaxCrossEntropy {
    type Target = [usize];

    fn forward(&self, z_l: &Tensor, target: &Self::Target) -> (f32, Tensor) {
        assert_eq!(z_l.ndim(), 2, "SoftmaxCrossEntropy expects logits of shape [B, C]");
        let (b, c) = (z_l.shape()[0], z_l.shape()[1]);
        assert_eq!(target.len(), b, "one target label per batch example");

        let mut grad_data = vec![0f32; b * c];
        let mut loss = 0f32;
        for bi in 0..b {
            let mut max_logit = f32::NEG_INFINITY;
            for ci in 0..c {
                max_logit = max_logit.max(z_l.get(&[bi, ci]));
            }
            let mut sum_exp = 0f32;
            for ci in 0..c {
                sum_exp += (z_l.get(&[bi, ci]) - max_logit).exp();
            }
            let log_sum_exp = sum_exp.ln();

            for ci in 0..c {
                let log_softmax = (z_l.get(&[bi, ci]) - max_logit) - log_sum_exp;
                let p = log_softmax.exp();
                let is_target = ci == target[bi];
                grad_data[bi * c + ci] = (p - if is_target { 1.0 } else { 0.0 }) / (b as f32);
                if is_target {
                    loss += -log_softmax;
                }
            }
        }
        loss /= b as f32;

        (loss, Tensor::from_vec(grad_data, &[b, c]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confident_correct_prediction_has_low_loss() {
        let z = Tensor::from_vec(vec![10.0, -10.0, -10.0], &[1, 3]);
        let (loss, _) = SoftmaxCrossEntropy.forward(&z, &[0]);
        assert!(loss < 1e-3, "loss={loss}");
    }

    /// Hand-computed reference: uniform logits [0,0] over 2 classes, N=1.
    /// softmax = [0.5, 0.5]; loss = -ln(0.5) = ln(2); grad = softmax - onehot.
    #[test]
    fn forward_matches_hand_computed_value_uniform_logits() {
        let z = Tensor::from_vec(vec![0.0, 0.0], &[1, 2]);
        let (loss, grad) = SoftmaxCrossEntropy.forward(&z, &[0]);
        assert!((loss - std::f32::consts::LN_2).abs() < 1e-5, "loss={loss}");
        assert!((grad.get(&[0, 0]) - (0.5 - 1.0)).abs() < 1e-5);
        assert!((grad.get(&[0, 1]) - 0.5).abs() < 1e-5);
    }

    /// Edge case: normalization is `1/N` (batch size). Duplicating one
    /// (logits, label) pair `N` times must leave the *mean* loss
    /// unchanged, and must leave the *sum* of the per-position gradient
    /// over those `N` duplicates unchanged too — each duplicate carries
    /// `1/N` of the single-example gradient, and there are `N` of them.
    /// That sum is exactly what a `Module`'s backward contracts over the
    /// batch axis (architecture.pdf, section 2.4).
    #[test]
    fn normalizes_by_batch_size() {
        let z1 = Tensor::from_vec(vec![0.0, 0.0], &[1, 2]);
        let (loss1, grad1) = SoftmaxCrossEntropy.forward(&z1, &[0]);

        let z3 = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[3, 2]);
        let (loss3, grad3) = SoftmaxCrossEntropy.forward(&z3, &[0, 0, 0]);

        assert!(
            (loss1 - loss3).abs() < 1e-5,
            "mean loss must not grow with N under identical duplication: loss1={loss1}, loss3={loss3}"
        );
        let summed: f32 = (0..3).map(|i| grad3.get(&[i, 0])).sum();
        assert!(
            (summed - grad1.get(&[0, 0])).abs() < 1e-5,
            "sum over duplicated positions must reproduce the single-example gradient: {summed} vs {}",
            grad1.get(&[0, 0])
        );
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let z = Tensor::from_vec(vec![0.5, -1.0, 2.0, 1.0, 0.0, -0.5], &[2, 3]);
        let target = [2usize, 0usize];
        let (_, grad) = SoftmaxCrossEntropy.forward(&z, &target);

        let eps = 1e-3;
        for i in 0..6 {
            let mut plus: Vec<f32> = z.iter().collect();
            plus[i] += eps;
            let mut minus: Vec<f32> = z.iter().collect();
            minus[i] -= eps;
            let (loss_plus, _) = SoftmaxCrossEntropy.forward(&Tensor::from_vec(plus, &[2, 3]), &target);
            let (loss_minus, _) = SoftmaxCrossEntropy.forward(&Tensor::from_vec(minus, &[2, 3]), &target);
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            let analytical = grad.get(&[i / 3, i % 3]);
            assert!(
                (numerical - analytical).abs() < 1e-3,
                "mismatch at {i}: numerical={numerical}, analytical={analytical}"
            );
        }
    }
}
