//! Backward (gradient) implementations for differentiable operations.
//!
//! Each operation has:
//! 1. A `GradFn` implementation that computes gradients given the output gradient
//! 2. A `*_forward` function that performs the forward pass and records the graph

use crate::autograd::variable::Variable;
use crate::error::{Result, TensorError};
use crate::tensor::Tensor;

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
