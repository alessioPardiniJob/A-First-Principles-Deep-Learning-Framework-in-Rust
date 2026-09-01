/// Dense numerical container exchanged between successive [`Module`](crate::Module)s.
///
/// `data` is a single flat, contiguous buffer of scalars. `shape` and
/// `strides` layer the logical multi-dimensional view on top of it, so
/// that operations which only rearrange which multi-index maps to which
/// offset (permuting axes, slicing, reversing) can be expressed as
/// metadata edits alone, leaving `data` untouched (architecture.pdf,
/// section 2.2).
///
/// By the dimensional-compatibility argument of section 2.2.1, the same
/// `Tensor` type carries both activations `z_l` and adjoints `a_l`, since
/// they always live in the same space.
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}
