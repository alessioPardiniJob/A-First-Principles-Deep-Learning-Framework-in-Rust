use super::Loss;
use crate::Tensor;

/// Regression loss: accepts a continuous target tensor matching the shape
/// of the network's final output `z_L` (architecture.pdf, section 2.3).
///
/// The per-example loss is the sum of squared errors over that example's
/// features, `||z_i - y_i||^2`; the batch loss is the mean of these over
/// the `N` examples, `(1/N) sum_i ||z_i - y_i||^2` — matching the PDF's
/// requirement that "both quantities are computed as a mean over the N
/// examples in the batch" (section 2.3). The normalization is `1/N`, the
/// batch size alone, not `1/(N*F)`: averaging over features too would
/// silently shrink the effective learning rate whenever the feature
/// dimension changes, which is not what "mean over the N examples" means.
pub struct MSELoss;

impl Loss for MSELoss {
    type Target = Tensor;

    fn forward(&self, z_l: &Tensor, target: &Self::Target) -> (f32, Tensor) {
        assert_eq!(
            z_l.shape(),
            target.shape(),
            "MSELoss: prediction and target shapes must match"
        );
        let n = z_l.shape()[0] as f32; // batch size N, not numel()
        let diff = z_l.zip_map(target, |a, b| a - b);
        let loss = diff.map(|d| d * d).sum() / n;
        let grad = diff.map(|d| 2.0 * d / n);
        (loss, grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_error_gives_zero_loss_and_gradient() {
        let z = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
        let y = z.clone();
        let (loss, grad) = MSELoss.forward(&z, &y);
        assert_eq!(loss, 0.0);
        assert_eq!(grad.sum(), 0.0);
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let z = Tensor::from_vec(vec![1.0, -2.0], &[1, 2]);
        let y = Tensor::from_vec(vec![0.5, 0.5], &[1, 2]);
        let (_, grad) = MSELoss.forward(&z, &y);

        let eps = 1e-3;
        for i in 0..2 {
            let mut plus: Vec<f32> = z.iter().collect();
            plus[i] += eps;
            let mut minus: Vec<f32> = z.iter().collect();
            minus[i] -= eps;
            let (loss_plus, _) = MSELoss.forward(&Tensor::from_vec(plus, &[1, 2]), &y);
            let (loss_minus, _) = MSELoss.forward(&Tensor::from_vec(minus, &[1, 2]), &y);
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            assert!((numerical - grad.get(&[0, i])).abs() < 1e-3);
        }
    }

    /// Hand-computed reference: z = [[1, 2]], y = [[0, 0]], N = 1.
    /// loss = (1^2 + 2^2) / 1 = 5; grad = 2*(z-y)/1 = [2, 4].
    #[test]
    fn forward_matches_hand_computed_value_single_example() {
        let z = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]);
        let y = Tensor::from_vec(vec![0.0, 0.0], &[1, 2]);
        let (loss, grad) = MSELoss.forward(&z, &y);
        assert!((loss - 5.0).abs() < 1e-6, "loss={loss}");
        assert!((grad.get(&[0, 0]) - 2.0).abs() < 1e-6);
        assert!((grad.get(&[0, 1]) - 4.0).abs() < 1e-6);
    }

    /// Edge case: the normalization factor is `1/N` (batch size), not
    /// `1/(N*F)` (batch size times feature count).
    ///
    /// Part 1 — the loss is a *sum* over features, only *averaged* over
    /// examples: doubling `F` at fixed `N=1` must double the loss (under
    /// the old, buggy `1/(N*F)` normalization it would not: both would
    /// come out to 1.0, hiding the extra error terms).
    ///
    /// Part 2 — duplicating one example `N` times must leave the loss
    /// unchanged (mean over identical values), and must leave the *sum*
    /// of the per-position gradient over those `N` duplicates unchanged
    /// too (each duplicate carries `1/N` of the single-example gradient,
    /// and there are `N` of them) — that sum is exactly what a `Module`'s
    /// backward contracts over the batch axis to get, e.g.,
    /// `grad_weight` (architecture.pdf, section 2.4).
    #[test]
    fn normalizes_by_batch_size_not_by_total_element_count() {
        let z_f2 = Tensor::from_vec(vec![1.0, 1.0], &[1, 2]);
        let (loss_f2, _) = MSELoss.forward(&z_f2, &Tensor::zeros(&[1, 2]));
        assert!((loss_f2 - 2.0).abs() < 1e-6, "loss_f2={loss_f2}");

        let z_f4 = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 4]);
        let (loss_f4, _) = MSELoss.forward(&z_f4, &Tensor::zeros(&[1, 4]));
        assert!((loss_f4 - 4.0).abs() < 1e-6, "loss_f4={loss_f4}");

        let (_, grad1) = MSELoss.forward(&z_f2, &Tensor::zeros(&[1, 2]));

        let z_rep = Tensor::from_vec(vec![1.0; 8], &[4, 2]);
        let (loss_rep, grad_rep) = MSELoss.forward(&z_rep, &Tensor::zeros(&[4, 2]));
        assert!(
            (loss_rep - loss_f2).abs() < 1e-6,
            "loss must not change under identical duplication: {loss_f2} vs {loss_rep}"
        );
        let summed: f32 = (0..4).map(|i| grad_rep.get(&[i, 0])).sum();
        assert!(
            (summed - grad1.get(&[0, 0])).abs() < 1e-6,
            "sum over duplicated positions must reproduce the single-example gradient: {summed} vs {}",
            grad1.get(&[0, 0])
        );
    }
}
