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
