use super::Module;
use crate::Tensor;

/// 2D convolution over tensors of shape `[B, C, H, W]`, implemented as an
/// im2col matrix multiplication: the receptive field of every output
/// position is gathered into a row of an `[B*OH*OW, C*KH*KW]` matrix,
/// which is then multiplied by the kernel reshaped to
/// `[out_channels, C*KH*KW]` — reusing [`Tensor::dot`] exactly as `Linear`
/// does. Kernel reshape/transpose are O(1) (architecture.pdf, section
/// 2.2.5); only the gather/scatter into the `[B,C,H,W]` layout needs an
/// explicit loop, since that is a genuine data movement, not a
/// reinterpretation of an existing buffer.
///
/// No padding is applied; `stride` applies to both spatial axes.
pub struct Conv2d {
    weight: Tensor, // [out_channels, in_channels, kh, kw]
    bias: Tensor,   // [1, out_channels, 1, 1]
    grad_weight: Tensor,
    grad_bias: Tensor,
    kernel_size: (usize, usize),
    stride: usize,
    input_shape: Option<(usize, usize, usize, usize)>,
    /// The im2col matrix from the last forward pass: exactly the
    /// intermediate value `backward` needs (architecture.pdf, section
    /// 2.1's "retain the intermediate values z_l"); the raw input itself
    /// is not needed once it has been gathered.
    col_cache: Option<Tensor>,
}

impl Conv2d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize), stride: usize) -> Self {
        let (kh, kw) = kernel_size;
        let scale = 1.0 / ((in_channels * kh * kw) as f32).sqrt();
        Self {
            weight: Tensor::rand(&[out_channels, in_channels, kh, kw], scale),
            bias: Tensor::zeros(&[1, out_channels, 1, 1]),
            grad_weight: Tensor::zeros(&[out_channels, in_channels, kh, kw]),
            grad_bias: Tensor::zeros(&[1, out_channels, 1, 1]),
            kernel_size,
            stride,
            input_shape: None,
            col_cache: None,
        }
    }

    fn out_channels(&self) -> usize {
        self.weight.shape()[0]
    }

    /// Kernel reshaped to `[out_channels, in_channels * kh * kw]`; O(1)
    /// since `weight` is always contiguous.
    fn weight_matrix(&self) -> Tensor {
        let (out_c, in_c, kh, kw) = (
            self.weight.shape()[0],
            self.weight.shape()[1],
            self.weight.shape()[2],
            self.weight.shape()[3],
        );
        self.weight.reshape(&[out_c, in_c * kh * kw])
    }
}

impl Module for Conv2d {
    fn forward(&mut self, input: Tensor) -> Tensor {
        assert_eq!(input.ndim(), 4, "Conv2d expects input of shape [B, C, H, W]");
        let (b, c, h, w) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (kh, kw) = self.kernel_size;
        let oh = (h - kh) / self.stride + 1;
        let ow = (w - kw) / self.stride + 1;
        let k = c * kh * kw;

        let mut col = vec![0f32; b * oh * ow * k];
        for bi in 0..b {
            for oh_i in 0..oh {
                for ow_i in 0..ow {
                    let row = (bi * oh + oh_i) * ow + ow_i;
                    for ci in 0..c {
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let hi = oh_i * self.stride + khi;
                                let wi = ow_i * self.stride + kwi;
                                let col_idx = (ci * kh + khi) * kw + kwi;
                                col[row * k + col_idx] = input.get(&[bi, ci, hi, wi]);
                            }
                        }
                    }
                }
            }
        }
        let col = Tensor::from_vec(col, &[b * oh * ow, k]);

        let out_mat = col.dot(&self.weight_matrix().t()); // [B*OH*OW, out_c]
        let out_c = self.out_channels();

        let mut out = vec![0f32; b * out_c * oh * ow];
        for bi in 0..b {
            for oh_i in 0..oh {
                for ow_i in 0..ow {
                    let row = (bi * oh + oh_i) * ow + ow_i;
                    for oc in 0..out_c {
                        out[((bi * out_c + oc) * oh + oh_i) * ow + ow_i] = out_mat.get(&[row, oc]);
                    }
                }
            }
        }

        self.input_shape = Some((b, c, h, w));
        self.col_cache = Some(col);

        Tensor::from_vec(out, &[b, out_c, oh, ow]) + &self.bias
    }

    fn backward(&mut self, grad_output: Tensor) -> Tensor {
        let (b, c, h, w) = self
            .input_shape
            .expect("forward must be called before backward");
        let col = self
            .col_cache
            .as_ref()
            .expect("forward must be called before backward")
            .clone(); // O(1): Rc::clone
        let (kh, kw) = self.kernel_size;
        let out_c = self.out_channels();
        let oh = grad_output.shape()[2];
        let ow = grad_output.shape()[3];

        let mut grad_out_mat = vec![0f32; b * oh * ow * out_c];
        for bi in 0..b {
            for oh_i in 0..oh {
                for ow_i in 0..ow {
                    let row = (bi * oh + oh_i) * ow + ow_i;
                    for oc in 0..out_c {
                        grad_out_mat[row * out_c + oc] = grad_output.get(&[bi, oc, oh_i, ow_i]);
                    }
                }
            }
        }
        let grad_out_mat = Tensor::from_vec(grad_out_mat, &[b * oh * ow, out_c]);

        self.grad_bias += &grad_out_mat.sum_axis(0).reshape(&[1, out_c, 1, 1]);

        let grad_weight_mat = grad_out_mat.t().dot(&col); // [out_c, k]
        self.grad_weight += &grad_weight_mat.reshape(&[out_c, c, kh, kw]);

        let grad_col = grad_out_mat.dot(&self.weight_matrix()); // [B*OH*OW, k]

        let mut grad_input = vec![0f32; b * c * h * w];
        for bi in 0..b {
            for oh_i in 0..oh {
                for ow_i in 0..ow {
                    let row = (bi * oh + oh_i) * ow + ow_i;
                    for ci in 0..c {
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let hi = oh_i * self.stride + khi;
                                let wi = ow_i * self.stride + kwi;
                                let col_idx = (ci * kh + khi) * kw + kwi;
                                grad_input[((bi * c + ci) * h + hi) * w + wi] +=
                                    grad_col.get(&[row, col_idx]);
                            }
                        }
                    }
                }
            }
        }
        Tensor::from_vec(grad_input, &[b, c, h, w])
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
    fn forward_output_shape_no_padding() {
        let mut conv = Conv2d::new(3, 4, (3, 3), 1);
        let out = conv.forward(Tensor::zeros(&[2, 3, 8, 8]));
        assert_eq!(out.shape(), &[2, 4, 6, 6]);
    }

    #[test]
    fn backward_shapes() {
        let mut conv = Conv2d::new(2, 3, (2, 2), 1);
        let _ = conv.forward(Tensor::zeros(&[1, 2, 4, 4]));
        let grad_in = conv.backward(Tensor::zeros(&[1, 3, 3, 3]));
        assert_eq!(grad_in.shape(), &[1, 2, 4, 4]);
        assert_eq!(conv.grads()[0].shape(), &[3, 2, 2, 2]);
        assert_eq!(conv.grads()[1].shape(), &[1, 3, 1, 1]);
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let mut conv = Conv2d::new(1, 1, (2, 2), 1);
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        );

        let _ = conv.forward(input.clone());
        let grad_output = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]);
        let grad_input = conv.backward(grad_output);

        let eps = 1e-3;
        for i in 0..9 {
            let mut data_plus: Vec<f32> = input.iter().collect();
            data_plus[i] += eps;
            let plus = Tensor::from_vec(data_plus, &[1, 1, 3, 3]);

            let mut data_minus: Vec<f32> = input.iter().collect();
            data_minus[i] -= eps;
            let minus = Tensor::from_vec(data_minus, &[1, 1, 3, 3]);

            let mut probe = Conv2d::new(1, 1, (2, 2), 1);
            probe.weight = conv.params()[0].clone();
            probe.bias = conv.params()[1].clone();

            let out_plus: f32 = probe.forward(plus).iter().sum();
            let out_minus: f32 = probe.forward(minus).iter().sum();
            let numerical = (out_plus - out_minus) / (2.0 * eps);
            let analytical = grad_input.get(&[0, 0, i / 3, i % 3]);
            assert!(
                (numerical - analytical).abs() < 1e-2,
                "grad mismatch at {i}: numerical={numerical}, analytical={analytical}"
            );
        }
    }
}
