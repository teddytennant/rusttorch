//! Backward (gradient) implementations for differentiable operations.
//!
//! Each operation has:
//! 1. A `GradFn` implementation that computes gradients given the output gradient
//! 2. A `*_forward` function that performs the forward pass and records the graph

use crate::autograd::variable::Variable;
use crate::error::{Result, TensorError};
use crate::tensor::Tensor;
use ndarray::{Array2, ArrayView2};

// ---- im2col / col2im utilities for efficient convolution ----

/// im2col: Extract sliding windows from a single image into a column matrix.
/// Input: flat slice for one image [C_in, H, W]
/// Output: column matrix [C_in*kH*kW, OH*OW] (row-major flat)
#[allow(clippy::too_many_arguments)]
fn im2col(
    input: &[f32],
    c_in: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let oh = (ih + 2 * padding - kh) / stride + 1;
    let ow = (iw + 2 * padding - kw) / stride + 1;
    let col_h = c_in * kh * kw;
    let col_w = oh * ow;
    let mut col = vec![0.0f32; col_h * col_w];

    for ci in 0..c_in {
        for ky in 0..kh {
            for kx in 0..kw {
                let col_row = ci * kh * kw + ky * kw + kx;
                let row_offset = col_row * col_w;
                for oy in 0..oh {
                    let iy = (oy * stride + ky) as isize - padding as isize;
                    if iy < 0 || iy >= ih as isize {
                        // Entire row of ox values is zero (padding)
                        continue;
                    }
                    let iy = iy as usize;
                    let input_row_offset = ci * ih * iw + iy * iw;
                    for ox in 0..ow {
                        let ix = (ox * stride + kx) as isize - padding as isize;
                        if ix >= 0 && ix < iw as isize {
                            col[row_offset + oy * ow + ox] = input[input_row_offset + ix as usize];
                        }
                    }
                }
            }
        }
    }
    col
}

/// col2im: Scatter-add column matrix back to image shape (inverse of im2col).
/// col: flat [C_in*kH*kW, OH*OW], output: flat [C_in, H, W]
#[allow(clippy::too_many_arguments)]
fn col2im(
    col: &[f32],
    c_in: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let oh = (ih + 2 * padding - kh) / stride + 1;
    let ow = (iw + 2 * padding - kw) / stride + 1;
    let col_w = oh * ow;
    let mut img = vec![0.0f32; c_in * ih * iw];

    for ci in 0..c_in {
        for ky in 0..kh {
            for kx in 0..kw {
                let col_row = ci * kh * kw + ky * kw + kx;
                let row_offset = col_row * col_w;
                for oy in 0..oh {
                    let iy = (oy * stride + ky) as isize - padding as isize;
                    if iy < 0 || iy >= ih as isize {
                        continue;
                    }
                    let iy = iy as usize;
                    let img_row_offset = ci * ih * iw + iy * iw;
                    for ox in 0..ow {
                        let ix = (ox * stride + kx) as isize - padding as isize;
                        if ix >= 0 && ix < iw as isize {
                            img[img_row_offset + ix as usize] += col[row_offset + oy * ow + ox];
                        }
                    }
                }
            }
        }
    }
    img
}

/// 2D matrix multiply using ndarray's optimized dot (BLAS-backed).
/// a: [m, k], b: [k, n] → result: [m, n], all row-major flat.
fn matmul_2d(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let a_mat = ArrayView2::from_shape((m, k), a).unwrap();
    let b_mat = ArrayView2::from_shape((k, n), b).unwrap();
    let c: Array2<f32> = a_mat.dot(&b_mat);
    c.into_raw_vec()
}

/// Trait for backward functions in the computation graph.
///
/// Each differentiable operation implements this trait.
/// `backward()` receives the gradient of the loss w.r.t. the output
/// and returns gradients w.r.t. each input.
pub trait GradFn {
    /// Compute gradients for each input given the output gradient.
    /// Returns one Option<Tensor> per input — None if that input doesn't need gradients.
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>>;

    /// The input variables that this operation depends on.
    fn inputs(&self) -> Vec<Variable>;
}

// ---- Addition: d(a+b)/da = 1, d(a+b)/db = 1 ----

struct AddBackward {
    a: Variable,
    b: Variable,
}

impl GradFn for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let grad_a = if self.a.requires_grad() {
            Some(grad_output.clone())
        } else {
            None
        };
        let grad_b = if self.b.requires_grad() {
            Some(grad_output.clone())
        } else {
            None
        };
        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.a.clone(), self.b.clone()]
    }
}

pub fn add_forward(a: &Variable, b: &Variable) -> Result<Variable> {
    let result = crate::ops::add(&a.tensor(), &b.tensor())?;
    if a.requires_grad() || b.requires_grad() {
        Ok(Variable::from_op(
            result,
            Box::new(AddBackward {
                a: a.clone(),
                b: b.clone(),
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- Subtraction: d(a-b)/da = 1, d(a-b)/db = -1 ----

struct SubBackward {
    a: Variable,
    b: Variable,
}

impl GradFn for SubBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let grad_a = if self.a.requires_grad() {
            Some(grad_output.clone())
        } else {
            None
        };
        let grad_b = if self.b.requires_grad() {
            Some(grad_output.neg())
        } else {
            None
        };
        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.a.clone(), self.b.clone()]
    }
}

pub fn sub_forward(a: &Variable, b: &Variable) -> Result<Variable> {
    let result = crate::ops::sub(&a.tensor(), &b.tensor())?;
    if a.requires_grad() || b.requires_grad() {
        Ok(Variable::from_op(
            result,
            Box::new(SubBackward {
                a: a.clone(),
                b: b.clone(),
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- Multiplication: d(a*b)/da = b, d(a*b)/db = a ----

struct MulBackward {
    a: Variable,
    b: Variable,
    a_saved: Tensor,
    b_saved: Tensor,
}

impl GradFn for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let grad_a = if self.a.requires_grad() {
            Some(crate::ops::mul(grad_output, &self.b_saved)?)
        } else {
            None
        };
        let grad_b = if self.b.requires_grad() {
            Some(crate::ops::mul(grad_output, &self.a_saved)?)
        } else {
            None
        };
        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.a.clone(), self.b.clone()]
    }
}

pub fn mul_forward(a: &Variable, b: &Variable) -> Result<Variable> {
    let a_tensor = a.tensor();
    let b_tensor = b.tensor();
    let result = crate::ops::mul(&a_tensor, &b_tensor)?;
    if a.requires_grad() || b.requires_grad() {
        Ok(Variable::from_op(
            result,
            Box::new(MulBackward {
                a: a.clone(),
                b: b.clone(),
                a_saved: a_tensor,
                b_saved: b_tensor,
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- Division: d(a/b)/da = 1/b, d(a/b)/db = -a/b² ----

struct DivBackward {
    a: Variable,
    b: Variable,
    a_saved: Tensor,
    b_saved: Tensor,
}

impl GradFn for DivBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let grad_a = if self.a.requires_grad() {
            // d(a/b)/da = 1/b → grad_a = grad_output / b
            Some(crate::ops::div(grad_output, &self.b_saved)?)
        } else {
            None
        };
        let grad_b = if self.b.requires_grad() {
            // d(a/b)/db = -a/b² → grad_b = -grad_output * a / b²
            let b_sq = crate::ops::mul(&self.b_saved, &self.b_saved)?;
            let neg_a = self.a_saved.neg();
            let neg_a_over_b_sq = crate::ops::div(&neg_a, &b_sq)?;
            Some(crate::ops::mul(grad_output, &neg_a_over_b_sq)?)
        } else {
            None
        };
        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.a.clone(), self.b.clone()]
    }
}

pub fn div_forward(a: &Variable, b: &Variable) -> Result<Variable> {
    let a_tensor = a.tensor();
    let b_tensor = b.tensor();
    let result = crate::ops::div(&a_tensor, &b_tensor)?;
    if a.requires_grad() || b.requires_grad() {
        Ok(Variable::from_op(
            result,
            Box::new(DivBackward {
                a: a.clone(),
                b: b.clone(),
                a_saved: a_tensor,
                b_saved: b_tensor,
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- Matrix multiplication: d(A@B)/dA = grad @ B^T, d(A@B)/dB = A^T @ grad ----

struct MatmulBackward {
    a: Variable,
    b: Variable,
    a_saved: Tensor,
    b_saved: Tensor,
}

impl GradFn for MatmulBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let grad_a = if self.a.requires_grad() {
            let b_t = crate::ops::matrix::transpose(&self.b_saved);
            Some(
                crate::ops::matrix::matmul(grad_output, &b_t)
                    .map_err(|e| TensorError::Other { message: e })?,
            )
        } else {
            None
        };
        let grad_b = if self.b.requires_grad() {
            let a_t = crate::ops::matrix::transpose(&self.a_saved);
            Some(
                crate::ops::matrix::matmul(&a_t, grad_output)
                    .map_err(|e| TensorError::Other { message: e })?,
            )
        } else {
            None
        };
        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.a.clone(), self.b.clone()]
    }
}

pub fn matmul_forward(a: &Variable, b: &Variable) -> Result<Variable> {
    let a_tensor = a.tensor();
    let b_tensor = b.tensor();
    let result = crate::ops::matrix::matmul(&a_tensor, &b_tensor)
        .map_err(|e| TensorError::Other { message: e })?;
    if a.requires_grad() || b.requires_grad() {
        Ok(Variable::from_op(
            result,
            Box::new(MatmulBackward {
                a: a.clone(),
                b: b.clone(),
                a_saved: a_tensor,
                b_saved: b_tensor,
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- ReLU: d(relu(x))/dx = 1 if x > 0, 0 otherwise ----

struct ReluBackward {
    input: Variable,
    input_saved: Tensor,
}

impl GradFn for ReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Create mask: 1.0 where input > 0, 0.0 where input <= 0
        let mask_data = self.input_saved.to_vec_f32();
        let mask: Vec<f32> = mask_data
            .iter()
            .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let mask_tensor = Tensor::from_vec(mask, self.input_saved.shape());
        let grad = crate::ops::mul(grad_output, &mask_tensor)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.input.clone()]
    }
}

pub fn relu_forward(input: &Variable) -> Variable {
    let input_tensor = input.tensor();
    let result = crate::ops::relu(&input_tensor);
    if input.requires_grad() {
        Variable::from_op(
            result,
            Box::new(ReluBackward {
                input: input.clone(),
                input_saved: input_tensor,
            }),
        )
    } else {
        Variable::detach(result)
    }
}

// ---- Sigmoid: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x)) ----

struct SigmoidBackward {
    input: Variable,
    output_saved: Tensor,
}

impl GradFn for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // grad = grad_output * sigmoid(x) * (1 - sigmoid(x))
        let one = self.output_saved.ones_like();
        let one_minus_sig = crate::ops::sub(&one, &self.output_saved)?;
        let sig_grad = crate::ops::mul(&self.output_saved, &one_minus_sig)?;
        let grad = crate::ops::mul(grad_output, &sig_grad)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.input.clone()]
    }
}

pub fn sigmoid_forward(input: &Variable) -> Result<Variable> {
    let input_tensor = input.tensor();
    let result = crate::ops::sigmoid(&input_tensor)?;
    if input.requires_grad() {
        Ok(Variable::from_op(
            result.clone(),
            Box::new(SigmoidBackward {
                input: input.clone(),
                output_saved: result,
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- Tanh: d(tanh(x))/dx = 1 - tanh(x)² ----

struct TanhBackward {
    input: Variable,
    output_saved: Tensor,
}

impl GradFn for TanhBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // grad = grad_output * (1 - tanh(x)²)
        let tanh_sq = crate::ops::mul(&self.output_saved, &self.output_saved)?;
        let one = self.output_saved.ones_like();
        let one_minus_sq = crate::ops::sub(&one, &tanh_sq)?;
        let grad = crate::ops::mul(grad_output, &one_minus_sq)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.input.clone()]
    }
}

pub fn tanh_forward(input: &Variable) -> Result<Variable> {
    let input_tensor = input.tensor();
    let result = crate::ops::tanh(&input_tensor)?;
    if input.requires_grad() {
        Ok(Variable::from_op(
            result.clone(),
            Box::new(TanhBackward {
                input: input.clone(),
                output_saved: result,
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- Sum: d(sum(x))/dx_i = 1 for all i ----

struct SumBackward {
    input: Variable,
    input_shape: Vec<usize>,
}

impl GradFn for SumBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // grad_output is a scalar. Broadcast to input shape.
        let grad_val = grad_output.item()?;
        let grad = Tensor::full(&self.input_shape, grad_val as f32);
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.input.clone()]
    }
}

pub fn sum_forward(input: &Variable) -> Result<Variable> {
    let input_tensor = input.tensor();
    let sum_val = crate::ops::sum(&input_tensor);
    let result = Tensor::from_vec(vec![sum_val as f32], &[1]);
    if input.requires_grad() {
        Ok(Variable::from_op(
            result,
            Box::new(SumBackward {
                input: input.clone(),
                input_shape: input_tensor.shape().to_vec(),
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- Mean: d(mean(x))/dx_i = 1/n for all i ----

struct MeanBackward {
    input: Variable,
    input_shape: Vec<usize>,
    n: usize,
}

impl GradFn for MeanBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        let grad_val = grad_output.item()?;
        let grad = Tensor::full(&self.input_shape, (grad_val / self.n as f64) as f32);
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.input.clone()]
    }
}

pub fn mean_forward(input: &Variable) -> Result<Variable> {
    let input_tensor = input.tensor();
    let mean_val = crate::ops::mean(&input_tensor)?;
    let result = Tensor::from_vec(vec![mean_val as f32], &[1]);
    let n = input_tensor.numel();
    if input.requires_grad() {
        Ok(Variable::from_op(
            result,
            Box::new(MeanBackward {
                input: input.clone(),
                input_shape: input_tensor.shape().to_vec(),
                n,
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- Scalar multiplication: d(scalar * x)/dx = scalar ----

struct MulScalarBackward {
    input: Variable,
    scalar: f32,
}

impl GradFn for MulScalarBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        let grad = crate::ops::mul_scalar(grad_output, self.scalar);
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.input.clone()]
    }
}

pub fn mul_scalar_forward(input: &Variable, scalar: f32) -> Variable {
    let result = crate::ops::mul_scalar(&input.tensor(), scalar);
    if input.requires_grad() {
        Variable::from_op(
            result,
            Box::new(MulScalarBackward {
                input: input.clone(),
                scalar,
            }),
        )
    } else {
        Variable::detach(result)
    }
}

// ---- Broadcasting add: supports adding tensors of different but broadcastable shapes ----
// Backward: sum gradients along broadcast dimensions to recover original shape.

struct BroadcastAddBackward {
    a: Variable,
    b: Variable,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

/// Reduce (sum) a tensor's gradient back to the original shape by summing along
/// dimensions that were broadcast (size 1 in original, size > 1 in output).
fn reduce_to_shape(grad: &Tensor, original_shape: &[usize], output_shape: &[usize]) -> Tensor {
    if original_shape == output_shape {
        return grad.clone();
    }

    let grad_data = grad.to_vec_f32();
    let orig_numel: usize = original_shape.iter().product();

    if original_shape.len() == output_shape.len() && output_shape.len() == 2 {
        let (out_rows, out_cols) = (output_shape[0], output_shape[1]);
        let (orig_rows, orig_cols) = (original_shape[0], original_shape[1]);

        if orig_rows == 1 && orig_cols == out_cols {
            // Sum along dim 0: [batch, out] -> [1, out]
            let mut result = vec![0.0f32; out_cols];
            for r in 0..out_rows {
                for c in 0..out_cols {
                    result[c] += grad_data[r * out_cols + c];
                }
            }
            return Tensor::from_vec(result, original_shape);
        }
        if orig_cols == 1 && orig_rows == out_rows {
            // Sum along dim 1: [batch, out] -> [batch, 1]
            let mut result = vec![0.0f32; out_rows];
            for r in 0..out_rows {
                for c in 0..out_cols {
                    result[r] += grad_data[r * out_cols + c];
                }
            }
            return Tensor::from_vec(result, original_shape);
        }
    }

    // Fallback: repeat elements if sizes match
    if grad_data.len() == orig_numel {
        return Tensor::from_vec(grad_data, original_shape);
    }

    // Generic case: sum to reduce
    let ratio = grad_data.len() / orig_numel;
    let mut result = vec![0.0f32; orig_numel];
    for (i, &v) in grad_data.iter().enumerate() {
        result[i % orig_numel] += v;
    }
    let _ = ratio;
    Tensor::from_vec(result, original_shape)
}

impl GradFn for BroadcastAddBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let output_shape = grad_output.shape().to_vec();
        let grad_a = if self.a.requires_grad() {
            Some(reduce_to_shape(grad_output, &self.a_shape, &output_shape))
        } else {
            None
        };
        let grad_b = if self.b.requires_grad() {
            Some(reduce_to_shape(grad_output, &self.b_shape, &output_shape))
        } else {
            None
        };
        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// Addition with broadcasting support.
pub fn broadcast_add_forward(a: &Variable, b: &Variable) -> Result<Variable> {
    let a_tensor = a.tensor();
    let b_tensor = b.tensor();

    // If same shape, use regular add (faster path)
    if a_tensor.shape() == b_tensor.shape() {
        return add_forward(a, b);
    }

    // Broadcast add using ndarray
    let a_data = a_tensor.to_vec_f32();
    let b_data = b_tensor.to_vec_f32();
    let a_shape = a_tensor.shape();
    let b_shape = b_tensor.shape();

    let result_shape = crate::ops::broadcast_shape(a_shape, b_shape)?;

    let a_arr = ndarray::Array::from_shape_vec(ndarray::IxDyn(a_shape), a_data).map_err(|e| {
        TensorError::Other {
            message: format!("broadcast add reshape a: {}", e),
        }
    })?;

    let b_arr = ndarray::Array::from_shape_vec(ndarray::IxDyn(b_shape), b_data).map_err(|e| {
        TensorError::Other {
            message: format!("broadcast add reshape b: {}", e),
        }
    })?;

    let a_broadcast = a_arr
        .broadcast(ndarray::IxDyn(&result_shape))
        .ok_or_else(|| TensorError::BroadcastError {
            shape_a: a_shape.to_vec(),
            shape_b: b_shape.to_vec(),
            reason: "ndarray broadcast failed".to_string(),
        })?;

    let b_broadcast = b_arr
        .broadcast(ndarray::IxDyn(&result_shape))
        .ok_or_else(|| TensorError::BroadcastError {
            shape_a: a_shape.to_vec(),
            shape_b: b_shape.to_vec(),
            reason: "ndarray broadcast failed".to_string(),
        })?;

    let result_data: Vec<f32> = a_broadcast
        .iter()
        .zip(b_broadcast.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    let result = Tensor::from_vec(result_data, &result_shape);

    if a.requires_grad() || b.requires_grad() {
        Ok(Variable::from_op(
            result,
            Box::new(BroadcastAddBackward {
                a: a.clone(),
                b: b.clone(),
                a_shape: a_shape.to_vec(),
                b_shape: b_shape.to_vec(),
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- Transpose: d(x^T)/dx = grad^T ----

struct TransposeBackward {
    input: Variable,
}

impl GradFn for TransposeBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        let grad = crate::ops::matrix::transpose(grad_output);
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.input.clone()]
    }
}

// ---- Reshape: d(reshape(x))/dx = reshape(grad, original_shape) ----

struct ReshapeBackward {
    input: Variable,
    original_shape: Vec<usize>,
}

impl GradFn for ReshapeBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        let grad = crate::ops::matrix::reshape(grad_output, &self.original_shape)
            .map_err(|e| TensorError::Other { message: e })?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.input.clone()]
    }
}

/// Reshape a variable while preserving the computation graph.
pub fn reshape_forward(input: &Variable, new_shape: &[usize]) -> Result<Variable> {
    let tensor = input.tensor();
    let original_shape = tensor.shape().to_vec();
    let result = crate::ops::matrix::reshape(&tensor, new_shape)
        .map_err(|e| TensorError::Other { message: e })?;
    if input.requires_grad() {
        Ok(Variable::from_op(
            result,
            Box::new(ReshapeBackward {
                input: input.clone(),
                original_shape,
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- Conv2d: 2D convolution forward + backward ----
// input: [B, C_in, H, W], weight: [C_out, C_in, kH, kW], bias: [C_out]
// output: [B, C_out, oH, oW]

pub struct Conv2dSavedState {
    input: Variable,
    weight: Variable,
    bias: Option<Variable>,
    input_tensor: Tensor,
    weight_tensor: Tensor,
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
    input_h: usize,
    input_w: usize,
    batch_size: usize,
}

impl GradFn for Conv2dSavedState {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let grad_data = grad_output.to_vec_f32();
        let input_data = self.input_tensor.to_vec_f32();
        let weight_data = self.weight_tensor.to_vec_f32();

        let b = self.batch_size;
        let c_in = self.in_channels;
        let c_out = self.out_channels;
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let stride = self.stride;
        let pad = self.padding;
        let ih = self.input_h;
        let iw = self.input_w;
        let oh = (ih + 2 * pad - kh) / stride + 1;
        let ow = (iw + 2 * pad - kw) / stride + 1;

        let k_dim = c_in * kh * kw;
        let spatial = oh * ow;
        let img_size = c_in * ih * iw;
        let grad_batch_size = c_out * spatial;

        let need_weight_grad = self.weight.requires_grad();
        let need_input_grad = self.input.requires_grad();

        // Accumulate grad_weight across batches
        let mut gw = if need_weight_grad {
            vec![0.0f32; c_out * k_dim]
        } else {
            vec![]
        };

        let mut gi = if need_input_grad {
            vec![0.0f32; b * img_size]
        } else {
            vec![]
        };

        // Hoist weight transpose out of batch loop (constant across batches)
        let w_t_data: Vec<f32> = if need_input_grad {
            let w_mat = ArrayView2::from_shape((c_out, k_dim), &weight_data).unwrap();
            let w_t: Array2<f32> = w_mat.t().to_owned();
            // Force C-contiguous layout
            w_t.as_standard_layout().to_owned().into_raw_vec()
        } else {
            vec![]
        };

        for bi in 0..b {
            let img_start = bi * img_size;
            let grad_start = bi * grad_batch_size;
            let grad_batch = &grad_data[grad_start..grad_start + grad_batch_size];

            // Recompute im2col for this batch (same as forward, avoids storing columns)
            let col = if need_weight_grad || need_input_grad {
                im2col(
                    &input_data[img_start..img_start + img_size],
                    c_in,
                    ih,
                    iw,
                    kh,
                    kw,
                    stride,
                    pad,
                )
            } else {
                vec![]
            };

            // grad_weight += grad_out[b] @ col^T
            // grad_out[b]: [C_out, spatial], col: [k_dim, spatial]
            // result: [C_out, k_dim]
            if need_weight_grad {
                let grad_mat = ArrayView2::from_shape((c_out, spatial), grad_batch).unwrap();
                let col_mat = ArrayView2::from_shape((k_dim, spatial), &col).unwrap();
                let gw_batch: Array2<f32> = grad_mat.dot(&col_mat.t());
                let gw_slice = gw_batch.as_slice().unwrap();
                for i in 0..gw.len() {
                    gw[i] += gw_slice[i];
                }
            }

            // grad_input: col_grad = weight^T @ grad_out[b] → col2im
            // weight^T: [k_dim, C_out] (precomputed), grad_out[b]: [C_out, spatial]
            // col_grad: [k_dim, spatial]
            if need_input_grad {
                let col_grad = matmul_2d(&w_t_data, grad_batch, k_dim, c_out, spatial);
                let gi_batch = col2im(&col_grad, c_in, ih, iw, kh, kw, stride, pad);
                let gi_start = bi * img_size;
                for i in 0..img_size {
                    gi[gi_start + i] += gi_batch[i];
                }
            }
        }

        let grad_weight = if need_weight_grad {
            Some(Tensor::from_vec(gw, &[c_out, c_in, kh, kw]))
        } else {
            None
        };

        let grad_input = if need_input_grad {
            Some(Tensor::from_vec(gi, &[b, c_in, ih, iw]))
        } else {
            None
        };

        // grad_bias[co] = sum_{b, oh_, ow_} grad[b, co, oh_, ow_]
        let grad_bias = if let Some(ref bias_var) = self.bias {
            if bias_var.requires_grad() {
                let mut gb = vec![0.0f32; c_out];
                for bi in 0..b {
                    for (co, gb_co) in gb.iter_mut().enumerate() {
                        let start = bi * grad_batch_size + co * spatial;
                        for s in 0..spatial {
                            *gb_co += grad_data[start + s];
                        }
                    }
                }
                Some(Tensor::from_vec(gb, &[c_out]))
            } else {
                None
            }
        } else {
            None
        };

        let mut grads = vec![grad_input, grad_weight];
        if self.bias.is_some() {
            grads.push(grad_bias);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<Variable> {
        let mut inputs = vec![self.input.clone(), self.weight.clone()];
        if let Some(ref bias) = self.bias {
            inputs.push(bias.clone());
        }
        inputs
    }
}

/// Conv2d forward pass using im2col + matrix multiplication.
/// input: [B, C_in, H, W], weight: [C_out, C_in, kH, kW], bias: Option<[C_out]>
///
/// Algorithm: For each batch element:
///   col = im2col(input[b]) → [C_in*kH*kW, OH*OW]
///   out[b] = weight_mat @ col → [C_out, OH*OW] → reshape to [C_out, OH, OW]
pub fn conv2d_forward(
    input: &Variable,
    weight: &Variable,
    bias: Option<&Variable>,
    stride: usize,
    padding: usize,
) -> Result<Variable> {
    let input_tensor = input.tensor();
    let weight_tensor = weight.tensor();
    let input_shape = input_tensor.shape().to_vec();
    let weight_shape = weight_tensor.shape().to_vec();

    if input_shape.len() != 4 {
        return Err(TensorError::InvalidArgument {
            parameter: "input".to_string(),
            reason: format!(
                "conv2d requires 4D input [B,C,H,W], got {}D",
                input_shape.len()
            ),
        });
    }
    if weight_shape.len() != 4 {
        return Err(TensorError::InvalidArgument {
            parameter: "weight".to_string(),
            reason: format!(
                "conv2d requires 4D weight [C_out,C_in,kH,kW], got {}D",
                weight_shape.len()
            ),
        });
    }

    let batch = input_shape[0];
    let c_in = input_shape[1];
    let ih = input_shape[2];
    let iw = input_shape[3];
    let c_out = weight_shape[0];
    let wc_in = weight_shape[1];
    let kh = weight_shape[2];
    let kw = weight_shape[3];

    if c_in != wc_in {
        return Err(TensorError::InvalidArgument {
            parameter: "weight".to_string(),
            reason: format!("input channels {} != weight channels {}", c_in, wc_in),
        });
    }

    let oh = (ih + 2 * padding - kh) / stride + 1;
    let ow = (iw + 2 * padding - kw) / stride + 1;

    let input_data = input_tensor.to_vec_f32();
    let weight_data = weight_tensor.to_vec_f32();
    let bias_data = bias.map(|b| b.tensor().to_vec_f32());

    // weight_mat: [C_out, C_in*kH*kW] — just a reshape, same data
    let k_dim = c_in * kh * kw;
    let spatial = oh * ow;
    let mut output = vec![0.0f32; batch * c_out * spatial];

    let img_size = c_in * ih * iw;
    let out_size = c_out * spatial;

    for bi in 0..batch {
        // im2col: extract patches → [C_in*kH*kW, OH*OW]
        let img_start = bi * img_size;
        let col = im2col(
            &input_data[img_start..img_start + img_size],
            c_in,
            ih,
            iw,
            kh,
            kw,
            stride,
            padding,
        );

        // matmul: weight_mat[C_out, k_dim] @ col[k_dim, spatial] → [C_out, spatial]
        let out_batch = matmul_2d(&weight_data, &col, c_out, k_dim, spatial);

        // Copy to output + add bias
        let out_start = bi * out_size;
        if let Some(ref bd) = bias_data {
            for (co, &bias_val) in bd.iter().enumerate() {
                let row_start = co * spatial;
                for s in 0..spatial {
                    output[out_start + row_start + s] = out_batch[row_start + s] + bias_val;
                }
            }
        } else {
            output[out_start..out_start + out_size].copy_from_slice(&out_batch);
        }
    }

    let result = Tensor::from_vec(output, &[batch, c_out, oh, ow]);
    let needs_grad =
        input.requires_grad() || weight.requires_grad() || bias.is_some_and(|b| b.requires_grad());

    if needs_grad {
        Ok(Variable::from_op(
            result,
            Box::new(Conv2dSavedState {
                input: input.clone(),
                weight: weight.clone(),
                bias: bias.cloned(),
                input_tensor,
                weight_tensor,
                in_channels: c_in,
                out_channels: c_out,
                kernel_h: kh,
                kernel_w: kw,
                stride,
                padding,
                input_h: ih,
                input_w: iw,
                batch_size: batch,
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- MaxPool2d: 2D max pooling forward + backward ----

pub struct MaxPool2dSavedState {
    input: Variable,
    max_indices: Vec<usize>,
    input_shape: Vec<usize>,
    output_h: usize,
    output_w: usize,
}

impl GradFn for MaxPool2dSavedState {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let grad_data = grad_output.to_vec_f32();
        let numel: usize = self.input_shape.iter().product();
        let mut grad_input = vec![0.0f32; numel];

        let b = self.input_shape[0];
        let c = self.input_shape[1];
        let oh = self.output_h;
        let ow = self.output_w;

        for bi in 0..b {
            for ci in 0..c {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let out_idx = bi * c * oh * ow + ci * oh * ow + oy * ow + ox;
                        let max_idx = self.max_indices[out_idx];
                        grad_input[max_idx] += grad_data[out_idx];
                    }
                }
            }
        }

        Ok(vec![Some(Tensor::from_vec(grad_input, &self.input_shape))])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.input.clone()]
    }
}

/// MaxPool2d forward pass.
/// input: [B, C, H, W], output: [B, C, oH, oW]
pub fn max_pool2d_forward(input: &Variable, kernel_size: usize, stride: usize) -> Result<Variable> {
    let tensor = input.tensor();
    let shape = tensor.shape().to_vec();
    if shape.len() != 4 {
        return Err(TensorError::InvalidArgument {
            parameter: "input".to_string(),
            reason: format!(
                "max_pool2d requires 4D input [B,C,H,W], got {}D",
                shape.len()
            ),
        });
    }

    let batch = shape[0];
    let channels = shape[1];
    let ih = shape[2];
    let iw = shape[3];
    let oh = (ih - kernel_size) / stride + 1;
    let ow = (iw - kernel_size) / stride + 1;

    let data = tensor.to_vec_f32();
    let mut output = vec![0.0f32; batch * channels * oh * ow];
    let mut max_indices = vec![0usize; batch * channels * oh * ow];

    for bi in 0..batch {
        for ci in 0..channels {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0usize;
                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let iy = oy * stride + ky;
                            let ix = ox * stride + kx;
                            let idx = bi * channels * ih * iw + ci * ih * iw + iy * iw + ix;
                            if data[idx] > max_val {
                                max_val = data[idx];
                                max_idx = idx;
                            }
                        }
                    }
                    let out_idx = bi * channels * oh * ow + ci * oh * ow + oy * ow + ox;
                    output[out_idx] = max_val;
                    max_indices[out_idx] = max_idx;
                }
            }
        }
    }

    let result = Tensor::from_vec(output, &[batch, channels, oh, ow]);

    if input.requires_grad() {
        Ok(Variable::from_op(
            result,
            Box::new(MaxPool2dSavedState {
                input: input.clone(),
                max_indices,
                input_shape: shape,
                output_h: oh,
                output_w: ow,
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

/// Transpose a 2D variable while preserving the computation graph.
pub fn transpose_forward(input: &Variable) -> Result<Variable> {
    let tensor = input.tensor();
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(TensorError::InvalidArgument {
            parameter: "tensor".to_string(),
            reason: format!("transpose requires 2D tensor, got {}D", shape.len()),
        });
    }
    let result = crate::ops::matrix::transpose(&tensor);
    if input.requires_grad() {
        Ok(Variable::from_op(
            result,
            Box::new(TransposeBackward {
                input: input.clone(),
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- BatchNorm2d: batch normalization for 4D tensors [B, C, H, W] ----
// y = gamma * (x - mean) / sqrt(var + eps) + beta
// During training: compute batch stats, update running stats.
// During eval: use running stats.

struct BatchNorm2dBackward {
    input: Variable,
    weight: Variable,  // gamma
    bias: Variable,    // beta
    x_norm: Tensor,    // normalized values (before affine)
    inv_std: Vec<f32>, // 1/sqrt(var + eps) per channel
    _num_features: usize,
}

impl GradFn for BatchNorm2dBackward {
    #[allow(clippy::needless_range_loop)]
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let grad_data = grad_output.to_vec_f32();
        let x_norm_data = self.x_norm.to_vec_f32();
        let shape = grad_output.shape();
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let spatial = h * w;
        let m = (b * spatial) as f32; // number of elements per channel

        let weight_data = self.weight.tensor().to_vec_f32();

        // grad_weight[c] = sum over (b, h, w) of grad_output[b,c,h,w] * x_norm[b,c,h,w]
        let grad_weight = if self.weight.requires_grad() {
            let mut gw = vec![0.0f32; c];
            for ci in 0..c {
                for bi in 0..b {
                    for si in 0..spatial {
                        let idx = bi * c * spatial + ci * spatial + si;
                        gw[ci] += grad_data[idx] * x_norm_data[idx];
                    }
                }
            }
            Some(Tensor::from_vec(gw, &[c]))
        } else {
            None
        };

        // grad_bias[c] = sum over (b, h, w) of grad_output[b,c,h,w]
        let grad_bias = if self.bias.requires_grad() {
            let mut gb = vec![0.0f32; c];
            for ci in 0..c {
                for bi in 0..b {
                    for si in 0..spatial {
                        let idx = bi * c * spatial + ci * spatial + si;
                        gb[ci] += grad_data[idx];
                    }
                }
            }
            Some(Tensor::from_vec(gb, &[c]))
        } else {
            None
        };

        // grad_input: full batchnorm backward
        // Simplified using x_norm = (x - mean) * inv_std:
        // dx = inv_std/m * (m * dx_norm - sum(dx_norm) - x_norm * sum(dx_norm * x_norm))
        let grad_input = if self.input.requires_grad() {
            let mut gi = vec![0.0f32; b * c * spatial];

            for ci in 0..c {
                let gamma = weight_data[ci];
                let istd = self.inv_std[ci];

                let mut sum_dx_norm = 0.0f32;
                let mut sum_dx_norm_xn = 0.0f32;

                for bi in 0..b {
                    for si in 0..spatial {
                        let idx = bi * c * spatial + ci * spatial + si;
                        let dx_norm = grad_data[idx] * gamma;
                        sum_dx_norm += dx_norm;
                        sum_dx_norm_xn += dx_norm * x_norm_data[idx];
                    }
                }

                for bi in 0..b {
                    for si in 0..spatial {
                        let idx = bi * c * spatial + ci * spatial + si;
                        let dx_norm = grad_data[idx] * gamma;
                        gi[idx] = istd / m
                            * (m * dx_norm - sum_dx_norm - x_norm_data[idx] * sum_dx_norm_xn);
                    }
                }
            }

            Some(Tensor::from_vec(gi, shape))
        } else {
            None
        };

        Ok(vec![grad_input, grad_weight, grad_bias])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.input.clone(), self.weight.clone(), self.bias.clone()]
    }
}

/// BatchNorm2d forward pass.
///
/// During training: normalizes with batch stats, updates running stats.
/// During eval: normalizes with running stats.
#[allow(clippy::too_many_arguments)]
pub fn batchnorm2d_forward(
    input: &Variable,
    weight: &Variable,
    bias: &Variable,
    running_mean: &std::cell::RefCell<Vec<f32>>,
    running_var: &std::cell::RefCell<Vec<f32>>,
    training: bool,
    momentum: f32,
    eps: f32,
) -> Result<Variable> {
    let tensor = input.tensor();
    let shape = tensor.shape();
    if shape.len() != 4 {
        return Err(TensorError::InvalidArgument {
            parameter: "input".to_string(),
            reason: format!(
                "BatchNorm2d expects 4D input [B,C,H,W], got {}D",
                shape.len()
            ),
        });
    }

    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let spatial = h * w;
    let m = (b * spatial) as f32;
    let data = tensor.to_vec_f32();
    let weight_data = weight.tensor().to_vec_f32();
    let bias_data = bias.tensor().to_vec_f32();

    let (mean, var) = if training {
        // Compute batch mean and variance per channel
        let mut mean = vec![0.0f32; c];
        let mut var = vec![0.0f32; c];

        for ci in 0..c {
            let mut sum = 0.0f32;
            for bi in 0..b {
                for si in 0..spatial {
                    let idx = bi * c * spatial + ci * spatial + si;
                    sum += data[idx];
                }
            }
            mean[ci] = sum / m;

            let mut var_sum = 0.0f32;
            for bi in 0..b {
                for si in 0..spatial {
                    let idx = bi * c * spatial + ci * spatial + si;
                    let diff = data[idx] - mean[ci];
                    var_sum += diff * diff;
                }
            }
            var[ci] = var_sum / m;
        }

        // Update running stats: running = (1 - momentum) * running + momentum * batch
        {
            let mut rm = running_mean.borrow_mut();
            let mut rv = running_var.borrow_mut();
            // Use Bessel's correction for running variance (unbiased)
            let correction = if m > 1.0 { m / (m - 1.0) } else { 1.0 };
            for ci in 0..c {
                rm[ci] = (1.0 - momentum) * rm[ci] + momentum * mean[ci];
                rv[ci] = (1.0 - momentum) * rv[ci] + momentum * var[ci] * correction;
            }
        }

        (mean, var)
    } else {
        // Use running stats
        let rm = running_mean.borrow();
        let rv = running_var.borrow();
        (rm.clone(), rv.clone())
    };

    // Normalize: x_norm = (x - mean) / sqrt(var + eps)
    let mut inv_std = vec![0.0f32; c];
    let mut x_norm_data = vec![0.0f32; b * c * spatial];
    let mut output_data = vec![0.0f32; b * c * spatial];

    for ci in 0..c {
        inv_std[ci] = 1.0 / (var[ci] + eps).sqrt();

        for bi in 0..b {
            for si in 0..spatial {
                let idx = bi * c * spatial + ci * spatial + si;
                x_norm_data[idx] = (data[idx] - mean[ci]) * inv_std[ci];
                output_data[idx] = weight_data[ci] * x_norm_data[idx] + bias_data[ci];
            }
        }
    }

    let result = Tensor::from_vec(output_data, shape);
    let x_norm = Tensor::from_vec(x_norm_data, shape);

    let needs_grad = input.requires_grad() || weight.requires_grad() || bias.requires_grad();
    if needs_grad && training {
        Ok(Variable::from_op(
            result,
            Box::new(BatchNorm2dBackward {
                input: input.clone(),
                weight: weight.clone(),
                bias: bias.clone(),
                x_norm,
                inv_std,
                _num_features: c,
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- Dropout: randomly zero elements during training ----
// Forward: mask = Bernoulli(1-p), output = input * mask / (1-p)
// Backward: grad_input = grad_output * mask / (1-p)

struct DropoutBackward {
    input: Variable,
    mask: Vec<f32>, // 0.0 or scale (1/(1-p))
}

impl GradFn for DropoutBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let grad_input = if self.input.requires_grad() {
            let grad_data = grad_output.to_vec_f32();
            let shape = grad_output.shape();
            let result: Vec<f32> = grad_data
                .iter()
                .zip(self.mask.iter())
                .map(|(&g, &m)| g * m)
                .collect();
            Some(Tensor::from_vec(result, shape))
        } else {
            None
        };

        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.input.clone()]
    }
}

/// Dropout forward pass. Only called during training with p > 0.
pub fn dropout_forward(input: &Variable, p: f32) -> Result<Variable> {
    use rand::Rng;

    let tensor = input.tensor();
    let data = tensor.to_vec_f32();
    let shape = tensor.shape();
    let scale = 1.0 / (1.0 - p);

    let mut rng = rand::thread_rng();
    let mask: Vec<f32> = (0..data.len())
        .map(|_| if rng.gen::<f32>() >= p { scale } else { 0.0 })
        .collect();

    let output: Vec<f32> = data.iter().zip(mask.iter()).map(|(&x, &m)| x * m).collect();
    let result = Tensor::from_vec(output, shape);

    if input.requires_grad() {
        Ok(Variable::from_op(
            result,
            Box::new(DropoutBackward {
                input: input.clone(),
                mask,
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- LogSoftmax: numerically stable log(softmax(x)) ----
// log_softmax(x_i) = x_i - max(x) - log(sum(exp(x_j - max(x))))
// Backward: grad_input = grad_output - softmax(x) * sum(grad_output)

struct LogSoftmaxBackward {
    input: Variable,
    softmax_output: Tensor, // softmax(x), saved for backward
    dim: usize,
}

impl GradFn for LogSoftmaxBackward {
    #[allow(clippy::needless_range_loop)]
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let grad_input = if self.input.requires_grad() {
            let grad_data = grad_output.to_vec_f32();
            let softmax_data = self.softmax_output.to_vec_f32();
            let shape = grad_output.shape();

            // For 2D [B, C] with dim=1:
            // grad_input[b,c] = grad_output[b,c] - softmax[b,c] * sum_c(grad_output[b,c])
            if shape.len() == 2 && self.dim == 1 {
                let (b, c) = (shape[0], shape[1]);
                let mut gi = vec![0.0f32; b * c];

                for bi in 0..b {
                    let mut sum_grad = 0.0f32;
                    for ci in 0..c {
                        sum_grad += grad_data[bi * c + ci];
                    }
                    for ci in 0..c {
                        let idx = bi * c + ci;
                        gi[idx] = grad_data[idx] - softmax_data[idx] * sum_grad;
                    }
                }
                Some(Tensor::from_vec(gi, shape))
            } else if shape.len() == 1 {
                // 1D case
                let sum_grad: f32 = grad_data.iter().sum();
                let gi: Vec<f32> = grad_data
                    .iter()
                    .zip(softmax_data.iter())
                    .map(|(&g, &s)| g - s * sum_grad)
                    .collect();
                Some(Tensor::from_vec(gi, shape))
            } else {
                // General fallback: not implemented for >2D
                return Err(TensorError::InvalidArgument {
                    parameter: "dim".to_string(),
                    reason: format!(
                        "LogSoftmax backward only supports 1D and 2D (dim=1), got {}D dim={}",
                        shape.len(),
                        self.dim
                    ),
                });
            }
        } else {
            None
        };

        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.input.clone()]
    }
}

/// LogSoftmax forward: log(softmax(x, dim)) with numerical stability.
///
/// Uses the log-sum-exp trick: log_softmax(x_i) = x_i - max(x) - log(sum(exp(x - max(x))))
pub fn log_softmax_forward(input: &Variable, dim: usize) -> Result<Variable> {
    let tensor = input.tensor();
    let shape = tensor.shape();
    let data = tensor.to_vec_f32();

    if shape.len() == 2 && dim == 1 {
        let (b, c) = (shape[0], shape[1]);
        let mut log_softmax_data = vec![0.0f32; b * c];
        let mut softmax_data = vec![0.0f32; b * c];

        for bi in 0..b {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for ci in 0..c {
                max_val = max_val.max(data[bi * c + ci]);
            }

            // Compute exp(x - max) and sum
            let mut sum_exp = 0.0f32;
            for ci in 0..c {
                let exp_val = (data[bi * c + ci] - max_val).exp();
                softmax_data[bi * c + ci] = exp_val;
                sum_exp += exp_val;
            }

            let log_sum_exp = sum_exp.ln();

            // log_softmax = x - max - log(sum(exp))
            // softmax = exp / sum
            for ci in 0..c {
                let idx = bi * c + ci;
                log_softmax_data[idx] = data[idx] - max_val - log_sum_exp;
                softmax_data[idx] /= sum_exp;
            }
        }

        let result = Tensor::from_vec(log_softmax_data, shape);
        let softmax_output = Tensor::from_vec(softmax_data, shape);

        if input.requires_grad() {
            Ok(Variable::from_op(
                result,
                Box::new(LogSoftmaxBackward {
                    input: input.clone(),
                    softmax_output,
                    dim,
                }),
            ))
        } else {
            Ok(Variable::detach(result))
        }
    } else if shape.len() == 1 {
        let n = shape[0];
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut softmax_data = vec![0.0f32; n];
        let mut sum_exp = 0.0f32;

        for i in 0..n {
            let e = (data[i] - max_val).exp();
            softmax_data[i] = e;
            sum_exp += e;
        }

        let log_sum_exp = sum_exp.ln();
        let log_softmax_data: Vec<f32> = data.iter().map(|&x| x - max_val - log_sum_exp).collect();
        for v in softmax_data.iter_mut() {
            *v /= sum_exp;
        }

        let result = Tensor::from_vec(log_softmax_data, shape);
        let softmax_output = Tensor::from_vec(softmax_data, shape);

        if input.requires_grad() {
            Ok(Variable::from_op(
                result,
                Box::new(LogSoftmaxBackward {
                    input: input.clone(),
                    softmax_output,
                    dim: 0,
                }),
            ))
        } else {
            Ok(Variable::detach(result))
        }
    } else {
        Err(TensorError::InvalidArgument {
            parameter: "input".to_string(),
            reason: format!(
                "log_softmax currently supports 1D and 2D (dim=1), got {}D",
                shape.len()
            ),
        })
    }
}

/// Cross-entropy loss forward: -sum(target * log_softmax(input)) / batch_size
///
/// Input: logits [B, C] (raw, unnormalized scores)
/// Target: one-hot or probability distribution [B, C]
/// Returns: scalar loss Variable
pub fn cross_entropy_loss_forward(input: &Variable, target: &Variable) -> Result<Variable> {
    let log_probs = log_softmax_forward(input, 1)?;
    // -sum(target * log_probs) / batch_size
    let neg_log_probs = log_probs.mul_scalar(-1.0);
    let elementwise = neg_log_probs.mul(target)?;
    elementwise.mean()
}

// ---- Average Pooling 2D ----

struct AvgPool2dSavedState {
    input: Variable,
    input_shape: Vec<usize>,
    kernel_size: usize,
    stride: usize,
    output_h: usize,
    output_w: usize,
}

impl GradFn for AvgPool2dSavedState {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let grad_data = grad_output.to_vec_f32();
        let numel: usize = self.input_shape.iter().product();
        let mut grad_input = vec![0.0f32; numel];

        let b = self.input_shape[0];
        let c = self.input_shape[1];
        let ih = self.input_shape[2];
        let iw = self.input_shape[3];
        let oh = self.output_h;
        let ow = self.output_w;
        let k = self.kernel_size;
        let pool_size = (k * k) as f32;

        // Each output gradient is distributed equally to all input positions in its window
        for bi in 0..b {
            for ci in 0..c {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let out_idx = bi * c * oh * ow + ci * oh * ow + oy * ow + ox;
                        let grad_val = grad_data[out_idx] / pool_size;
                        for ky in 0..k {
                            for kx in 0..k {
                                let iy = oy * self.stride + ky;
                                let ix = ox * self.stride + kx;
                                if iy < ih && ix < iw {
                                    let in_idx = bi * c * ih * iw + ci * ih * iw + iy * iw + ix;
                                    grad_input[in_idx] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(vec![Some(Tensor::from_vec(grad_input, &self.input_shape))])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.input.clone()]
    }
}

/// AvgPool2d forward pass.
/// input: [B, C, H, W], output: [B, C, oH, oW]
pub fn avg_pool2d_forward(input: &Variable, kernel_size: usize, stride: usize) -> Result<Variable> {
    let tensor = input.tensor();
    let shape = tensor.shape().to_vec();
    if shape.len() != 4 {
        return Err(TensorError::InvalidArgument {
            parameter: "input".to_string(),
            reason: format!(
                "avg_pool2d requires 4D input [B,C,H,W], got {}D",
                shape.len()
            ),
        });
    }

    let batch = shape[0];
    let channels = shape[1];
    let ih = shape[2];
    let iw = shape[3];
    let oh = (ih - kernel_size) / stride + 1;
    let ow = (iw - kernel_size) / stride + 1;

    let data = tensor.to_vec_f32();
    let mut output = vec![0.0f32; batch * channels * oh * ow];
    let pool_size = (kernel_size * kernel_size) as f32;

    for bi in 0..batch {
        for ci in 0..channels {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut sum = 0.0f32;
                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let iy = oy * stride + ky;
                            let ix = ox * stride + kx;
                            let idx = bi * channels * ih * iw + ci * ih * iw + iy * iw + ix;
                            sum += data[idx];
                        }
                    }
                    let out_idx = bi * channels * oh * ow + ci * oh * ow + oy * ow + ox;
                    output[out_idx] = sum / pool_size;
                }
            }
        }
    }

    let result = Tensor::from_vec(output, &[batch, channels, oh, ow]);

    if input.requires_grad() {
        Ok(Variable::from_op(
            result,
            Box::new(AvgPool2dSavedState {
                input: input.clone(),
                input_shape: shape,
                kernel_size,
                stride,
                output_h: oh,
                output_w: ow,
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

// ---- Adaptive Average Pooling 2D ----

struct AdaptiveAvgPool2dSavedState {
    input: Variable,
    input_shape: Vec<usize>,
    output_h: usize,
    output_w: usize,
}

impl GradFn for AdaptiveAvgPool2dSavedState {
    #[allow(clippy::needless_range_loop)]
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let grad_data = grad_output.to_vec_f32();
        let numel: usize = self.input_shape.iter().product();
        let mut grad_input = vec![0.0f32; numel];

        let b = self.input_shape[0];
        let c = self.input_shape[1];
        let ih = self.input_shape[2];
        let iw = self.input_shape[3];
        let oh = self.output_h;
        let ow = self.output_w;

        for bi in 0..b {
            for ci in 0..c {
                for oy in 0..oh {
                    let y_start = (oy * ih) / oh;
                    let y_end = ((oy + 1) * ih) / oh;
                    for ox in 0..ow {
                        let x_start = (ox * iw) / ow;
                        let x_end = ((ox + 1) * iw) / ow;
                        let pool_size = ((y_end - y_start) * (x_end - x_start)) as f32;
                        let out_idx = bi * c * oh * ow + ci * oh * ow + oy * ow + ox;
                        let grad_val = grad_data[out_idx] / pool_size;

                        for iy in y_start..y_end {
                            for ix in x_start..x_end {
                                let in_idx = bi * c * ih * iw + ci * ih * iw + iy * iw + ix;
                                grad_input[in_idx] += grad_val;
                            }
                        }
                    }
                }
            }
        }

        Ok(vec![Some(Tensor::from_vec(grad_input, &self.input_shape))])
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![self.input.clone()]
    }
}

/// AdaptiveAvgPool2d forward pass.
/// Maps any spatial size to the target (output_h, output_w).
/// input: [B, C, H, W], output: [B, C, output_h, output_w]
pub fn adaptive_avg_pool2d_forward(
    input: &Variable,
    output_h: usize,
    output_w: usize,
) -> Result<Variable> {
    let tensor = input.tensor();
    let shape = tensor.shape().to_vec();
    if shape.len() != 4 {
        return Err(TensorError::InvalidArgument {
            parameter: "input".to_string(),
            reason: format!(
                "adaptive_avg_pool2d requires 4D input [B,C,H,W], got {}D",
                shape.len()
            ),
        });
    }

    let batch = shape[0];
    let channels = shape[1];
    let ih = shape[2];
    let iw = shape[3];

    let data = tensor.to_vec_f32();
    let mut output = vec![0.0f32; batch * channels * output_h * output_w];

    for bi in 0..batch {
        for ci in 0..channels {
            for oy in 0..output_h {
                let y_start = (oy * ih) / output_h;
                let y_end = ((oy + 1) * ih) / output_h;
                for ox in 0..output_w {
                    let x_start = (ox * iw) / output_w;
                    let x_end = ((ox + 1) * iw) / output_w;
                    let pool_size = ((y_end - y_start) * (x_end - x_start)) as f32;

                    let mut sum = 0.0f32;
                    for iy in y_start..y_end {
                        for ix in x_start..x_end {
                            let idx = bi * channels * ih * iw + ci * ih * iw + iy * iw + ix;
                            sum += data[idx];
                        }
                    }
                    let out_idx = bi * channels * output_h * output_w
                        + ci * output_h * output_w
                        + oy * output_w
                        + ox;
                    output[out_idx] = sum / pool_size;
                }
            }
        }
    }

    let result = Tensor::from_vec(output, &[batch, channels, output_h, output_w]);

    if input.requires_grad() {
        Ok(Variable::from_op(
            result,
            Box::new(AdaptiveAvgPool2dSavedState {
                input: input.clone(),
                input_shape: shape,
                output_h,
                output_w,
            }),
        ))
    } else {
        Ok(Variable::detach(result))
    }
}

#[cfg(test)]
mod im2col_tests {
    use super::*;

    #[test]
    fn test_im2col_no_padding() {
        // 1 channel, 3x3 input, 2x2 kernel, stride 1, no padding
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let col = im2col(&input, 1, 3, 3, 2, 2, 1, 0);
        assert_eq!(col.len(), 16);
        assert_eq!(&col[0..4], &[1.0, 2.0, 4.0, 5.0]);
        assert_eq!(&col[4..8], &[2.0, 3.0, 5.0, 6.0]);
        assert_eq!(&col[8..12], &[4.0, 5.0, 7.0, 8.0]);
        assert_eq!(&col[12..16], &[5.0, 6.0, 8.0, 9.0]);
    }

    #[test]
    fn test_im2col_with_padding() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let col = im2col(&input, 1, 2, 2, 3, 3, 1, 1);
        assert_eq!(col.len(), 36);
        assert_eq!(col[0], 0.0);
        assert_eq!(col[1], 0.0);
    }

    #[test]
    fn test_im2col_stride() {
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let col = im2col(&input, 1, 4, 4, 2, 2, 2, 0);
        assert_eq!(col.len(), 16);
        assert_eq!(&col[0..4], &[1.0, 3.0, 9.0, 11.0]);
    }

    #[test]
    fn test_im2col_multichannel() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let col = im2col(&input, 2, 2, 2, 2, 2, 1, 0);
        assert_eq!(col, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_col2im_no_overlap() {
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let col = im2col(&input, 1, 4, 4, 2, 2, 2, 0);
        let reconstructed = col2im(&col, 1, 4, 4, 2, 2, 2, 0);
        assert_eq!(input, reconstructed);
    }

    #[test]
    fn test_col2im_overlap_accumulates() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let col = im2col(&input, 1, 3, 3, 2, 2, 1, 0);
        let reconstructed = col2im(&col, 1, 3, 3, 2, 2, 1, 0);
        // corners: 1 patch, edges: 2 patches, center: 4 patches
        assert_eq!(reconstructed[0], 1.0);
        assert_eq!(reconstructed[1], 4.0);
        assert_eq!(reconstructed[2], 3.0);
        assert_eq!(reconstructed[4], 20.0);
        assert_eq!(reconstructed[8], 9.0);
    }

    #[test]
    fn test_matmul_2d_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = matmul_2d(&a, &b, 2, 3, 2);
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_im2col_conv_equivalence() {
        // Verify im2col+matmul = direct convolution
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0];

        // Direct conv: sum of diagonal elements in each 2x2 window
        let direct = vec![
            1.0 * 1.0 + 2.0 * 0.0 + 4.0 * 0.0 + 5.0 * 1.0, // = 6
            2.0 * 1.0 + 3.0 * 0.0 + 5.0 * 0.0 + 6.0 * 1.0, // = 8
            4.0 * 1.0 + 5.0 * 0.0 + 7.0 * 0.0 + 8.0 * 1.0, // = 12
            5.0 * 1.0 + 6.0 * 0.0 + 8.0 * 0.0 + 9.0 * 1.0, // = 14
        ];

        let col = im2col(&input, 1, 3, 3, 2, 2, 1, 0);
        let result = matmul_2d(&weight, &col, 1, 4, 4);

        assert_eq!(direct, result);
    }

    #[test]
    fn test_im2col_1x1_kernel() {
        // 1x1 kernel: col should just be the input reshaped
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let col = im2col(&input, 2, 2, 2, 1, 1, 1, 0);
        // [2*1*1, 2*2] = [2, 4]
        assert_eq!(col, input);
    }

    #[test]
    fn test_col2im_with_padding() {
        // Round-trip with padding
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let col = im2col(&input, 1, 2, 2, 2, 2, 1, 1);
        // oh = (2+2-2)/1+1 = 3, ow = 3 → col is [4, 9]
        let restored = col2im(&col, 1, 2, 2, 2, 2, 1, 1);
        // Each input position gets accumulated from multiple patches
        // Just verify shape is correct
        assert_eq!(restored.len(), 4);
    }
}
