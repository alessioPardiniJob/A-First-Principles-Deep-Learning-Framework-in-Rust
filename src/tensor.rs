use std::cell::Cell;
use std::ops::{Add, AddAssign, Mul, SubAssign};
use std::rc::Rc;

/// Dense numerical container exchanged between successive [`Module`](crate::Module)s.
///
/// `data` is a single flat buffer of scalars. `shape` and `strides` layer
/// the logical multi-dimensional view on top of it, so that operations
/// which only rearrange which multi-index maps to which offset (permuting
/// axes, reshaping a contiguous buffer) can be expressed as metadata edits
/// alone, leaving `data` untouched (architecture.pdf, section 2.2).
///
/// By the dimensional-compatibility argument of section 2.2.1, the same
/// `Tensor` type carries both activations `z_l` and adjoints `a_l`, since
/// they always live in the same space.
///
/// The buffer is `Rc<Vec<f32>>` rather than a bare `Vec<f32>`: this is what
/// lets [`Tensor::t`] be genuinely O(1) (share the buffer, swap the
/// metadata) as required, rather than the PDF's own sketch, which — under a
/// plain owned `Vec<f32>` — could only reach O(1) by consuming `self`.
/// Mutation (`AddAssign`, `SubAssign`) still behaves like exclusive
/// ownership: it goes through `Rc::make_mut`, which clones the buffer only
/// if another view is still sharing it, so a transposed view of a tensor is
/// never corrupted by mutating the original in place.
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Rc<Vec<f32>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

// A small xorshift64 generator, used only by [`Tensor::rand`] for
// parameter initialization. Written out here rather than pulled in from
// the `rand` crate so the whole project builds with zero external
// dependencies (and therefore offline); the observable behaviour is the
// same — a fresh stream per process, uniform over `[-1, 1)`.
thread_local! {
    static RNG_STATE: Cell<u64> = Cell::new(seed_from_clock());
}

fn seed_from_clock() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    nanos | 1 // xorshift must never be seeded with zero
}

/// Uniform in `[-1, 1)`.
fn next_uniform() -> f32 {
    RNG_STATE.with(|state| {
        let mut x = state.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state.set(x);
        // Top 24 bits give a value in [0, 1) with full f32 mantissa
        // precision, rescaled to [-1, 1).
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    })
}

fn strides_for_shape(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn offset(strides: &[usize], idx: &[usize]) -> usize {
    idx.iter().zip(strides).map(|(&i, &s)| i * s).sum()
}

/// Enumerates every logical multi-index of `shape`, in row-major order.
fn each_index(shape: &[usize]) -> impl Iterator<Item = Vec<usize>> {
    let shape = shape.to_vec();
    let total: usize = shape.iter().product();
    let rank = shape.len();
    (0..total).map(move |mut linear| {
        let mut idx = vec![0usize; rank];
        for d in (0..rank).rev() {
            idx[d] = linear % shape[d];
            linear /= shape[d];
        }
        idx
    })
}

/// Numpy-style broadcast shape: shapes are aligned from the right, and each
/// pair of dimensions must be equal or one of them must be 1.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];
    for i in 0..rank {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        let d = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            panic!("cannot broadcast shapes {a:?} and {b:?}");
        };
        out[rank - 1 - i] = d;
    }
    out
}

/// Maps a multi-index in broadcast-output space down to `src_shape`'s own
/// space: dimensions of size 1 in `src_shape` always read index 0.
fn align_index(out_idx: &[usize], src_shape: &[usize]) -> Vec<usize> {
    let rank_offset = out_idx.len() - src_shape.len();
    src_shape
        .iter()
        .enumerate()
        .map(|(i, &d)| if d == 1 { 0 } else { out_idx[rank_offset + i] })
        .collect()
}

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Tensor {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "from_vec: data length {} does not match shape {:?}",
            data.len(),
            shape
        );
        Tensor {
            data: Rc::new(data),
            shape: shape.to_vec(),
            strides: strides_for_shape(shape),
        }
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::from_vec(vec![0.0; n], shape)
    }

    pub fn zeros_like(t: &Tensor) -> Tensor {
        Tensor::zeros(&t.shape)
    }

    /// Uniform initialization in `[-scale, scale)`, for parameter tensors
    /// (e.g. Linear/Conv2d weights); not part of the PDF's Table 1
    /// abstractions, but required to make those layers trainable.
    pub fn rand(shape: &[usize], scale: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| next_uniform() * scale).collect();
        Tensor::from_vec(data, shape)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == strides_for_shape(&self.shape)
    }

    pub fn get(&self, idx: &[usize]) -> f32 {
        debug_assert_eq!(idx.len(), self.shape.len(), "get: index rank mismatch");
        self.data[offset(&self.strides, idx)]
    }

    /// Elements in logical row-major order, respecting `strides` so a
    /// non-contiguous view (e.g. the result of [`Tensor::t`]) reads
    /// correctly.
    pub fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        each_index(&self.shape).map(move |idx| self.data[offset(&self.strides, &idx)])
    }

    pub fn sum(&self) -> f32 {
        self.iter().sum()
    }

    /// Full axis reversal: for a 2D tensor this is exactly "swap the two
    /// entries of `shape` and `strides`" (architecture.pdf, section 2.2.5).
    /// O(1): the buffer is shared via `Rc::clone`, only the small
    /// `shape`/`strides` metadata is copied.
    pub fn t(&self) -> Tensor {
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.reverse();
        strides.reverse();
        Tensor {
            data: Rc::clone(&self.data),
            shape,
            strides,
        }
    }

    /// Returns a contiguous tensor with the same logical contents: `self`
    /// (an `Rc::clone`, O(1)) if already contiguous, otherwise a freshly
    /// materialized copy.
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            self.clone()
        } else {
            Tensor::from_vec(self.iter().collect(), &self.shape)
        }
    }

    /// Reinterprets the same element count under `new_shape`. O(1) when
    /// `self` is contiguous (shares the buffer via `Rc::clone`); otherwise
    /// materializes a contiguous copy first.
    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            new_numel,
            self.numel(),
            "reshape: element count mismatch ({:?} -> {:?})",
            self.shape,
            new_shape
        );
        if self.is_contiguous() {
            Tensor {
                data: Rc::clone(&self.data),
                shape: new_shape.to_vec(),
                strides: strides_for_shape(new_shape),
            }
        } else {
            self.contiguous().reshape(new_shape)
        }
    }

    /// Applies `f` elementwise, respecting `strides`; always returns a
    /// fresh, contiguous tensor of the same shape.
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Tensor {
        let data: Vec<f32> = self.iter().map(f).collect();
        Tensor::from_vec(data, &self.shape)
    }

    /// Applies `f` elementwise across two same-shaped tensors, respecting
    /// each one's own `strides`.
    pub fn zip_map(&self, other: &Tensor, f: impl Fn(f32, f32) -> f32) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "zip_map requires matching shapes, got {:?} and {:?}",
            self.shape, other.shape
        );
        let mut out = Vec::with_capacity(self.numel());
        for idx in each_index(&self.shape) {
            let a = self.data[offset(&self.strides, &idx)];
            let b = other.data[offset(&other.strides, &idx)];
            out.push(f(a, b));
        }
        Tensor::from_vec(out, &self.shape)
    }

    /// Sums along `axis`, keeping it as a size-1 dimension (so the result
    /// stays broadcast-compatible with the un-reduced tensor, matching the
    /// PDF's `sum_axis(...).insert_axis(...)` pattern for `grad_bias` in a
    /// single call).
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        assert!(axis < self.ndim(), "sum_axis: axis out of range");
        let mut out_shape = self.shape.clone();
        out_shape[axis] = 1;
        let out_strides = strides_for_shape(&out_shape);
        let mut out = vec![0f32; out_shape.iter().product()];
        for idx in each_index(&self.shape) {
            let mut out_idx = idx.clone();
            out_idx[axis] = 0;
            out[offset(&out_strides, &out_idx)] += self.data[offset(&self.strides, &idx)];
        }
        Tensor::from_vec(out, &out_shape)
    }

    /// 2D matrix multiplication. Reads both operands through `get`, so it
    /// is correct regardless of whether either side is a transposed
    /// (non-contiguous) view — the kernel consults `strides` rather than
    /// assuming a fixed layout (architecture.pdf, section 2.2.5).
    pub fn dot(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "dot: lhs must be 2D, got shape {:?}", self.shape);
        assert_eq!(other.ndim(), 2, "dot: rhs must be 2D, got shape {:?}", other.shape);
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k, k2, "dot: inner dimensions must match ({k} vs {k2})");
        let mut out = vec![0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let a = self.get(&[i, p]);
                for j in 0..n {
                    out[i * n + j] += a * other.get(&[p, j]);
                }
            }
        }
        Tensor::from_vec(out, &[m, n])
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    /// Numpy-style broadcasting add (dimensions of size 1 stretch to match
    /// the other operand), used for bias addition in `Linear`/`Conv2d`
    /// (e.g. `[B, F] + [1, F]`) and for optimizer state updates.
    fn add(self, rhs: &Tensor) -> Tensor {
        let out_shape = broadcast_shape(&self.shape, &rhs.shape);
        let out_strides = strides_for_shape(&out_shape);
        let mut out = vec![0f32; out_shape.iter().product()];
        for idx in each_index(&out_shape) {
            let a = self.data[offset(&self.strides, &align_index(&idx, &self.shape))];
            let b = rhs.data[offset(&rhs.strides, &align_index(&idx, &rhs.shape))];
            out[offset(&out_strides, &idx)] = a + b;
        }
        Tensor::from_vec(out, &out_shape)
    }
}

impl Add<&Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        &self + rhs
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f32) -> Tensor {
        self.map(|x| x * scalar)
    }
}

impl AddAssign<&Tensor> for Tensor {
    /// In-place, same-shape accumulation (e.g. `grad_weight += ...`).
    /// Goes through `Rc::make_mut`, so a shared buffer (e.g. still
    /// referenced by a live transposed view) is copied before mutation.
    fn add_assign(&mut self, rhs: &Tensor) {
        assert_eq!(
            self.shape, rhs.shape,
            "add_assign requires matching shapes, got {:?} and {:?}",
            self.shape, rhs.shape
        );
        let shape = self.shape.clone();
        let strides = self.strides.clone();
        let rhs_strides = rhs.strides.clone();
        let data = Rc::make_mut(&mut self.data);
        for idx in each_index(&shape) {
            data[offset(&strides, &idx)] += rhs.data[offset(&rhs_strides, &idx)];
        }
    }
}

impl SubAssign<&Tensor> for Tensor {
    /// In-place, same-shape update (e.g. `*theta -= &(grad * lr)`).
    fn sub_assign(&mut self, rhs: &Tensor) {
        assert_eq!(
            self.shape, rhs.shape,
            "sub_assign requires matching shapes, got {:?} and {:?}",
            self.shape, rhs.shape
        );
        let shape = self.shape.clone();
        let strides = self.strides.clone();
        let rhs_strides = rhs.strides.clone();
        let data = Rc::make_mut(&mut self.data);
        for idx in each_index(&shape) {
            data[offset(&strides, &idx)] -= rhs.data[offset(&rhs_strides, &idx)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_is_zero_copy_view() {
        let t = Tensor::from_vec((0..6).map(|x| x as f32).collect(), &[2, 3]);
        let tt = t.t();
        assert_eq!(tt.shape(), &[3, 2]);
        assert!(Rc::ptr_eq(&t.data, &tt.data), "t() must share the buffer");
        assert_eq!(tt.get(&[1, 0]), 1.0); // original [0,1]
        assert_eq!(tt.get(&[0, 1]), 3.0); // original [1,0]
    }

    #[test]
    fn reshape_is_zero_copy_when_contiguous() {
        let t = Tensor::from_vec((0..6).map(|x| x as f32).collect(), &[2, 3]);
        let r = t.reshape(&[3, 2]);
        assert!(Rc::ptr_eq(&t.data, &r.data));
        assert_eq!(r.get(&[1, 1]), 3.0);
    }

    #[test]
    fn mutating_original_does_not_affect_shared_view() {
        let mut t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let view = t.t();
        t += &Tensor::from_vec(vec![10.0, 10.0, 10.0, 10.0], &[2, 2]);
        assert_eq!(t.get(&[0, 0]), 11.0);
        assert_eq!(view.get(&[0, 0]), 1.0, "shared view must be unaffected (copy-on-write)");
    }

    #[test]
    fn dot_reads_transposed_operand_correctly() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]); // [B=2,in=3]
        let w = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]); // [out=2,in=3]
        let out = a.dot(&w.t()); // [2,3] dot [3,2] -> [2,2]
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.get(&[0, 0]), 1.0);
        assert_eq!(out.get(&[0, 1]), 2.0);
        assert_eq!(out.get(&[1, 0]), 4.0);
        assert_eq!(out.get(&[1, 1]), 5.0);
    }

    #[test]
    fn broadcasting_add_matches_bias_pattern() {
        let z = Tensor::zeros(&[2, 3]);
        let bias = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
        let out = z + &bias;
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.get(&[0, 2]), 3.0);
        assert_eq!(out.get(&[1, 2]), 3.0);
    }

    #[test]
    fn sum_axis_keeps_dim() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let s = t.sum_axis(0);
        assert_eq!(s.shape(), &[1, 2]);
        assert_eq!(s.get(&[0, 0]), 4.0);
        assert_eq!(s.get(&[0, 1]), 6.0);
    }
}
