//! Activation function modules.
//!
//! These wrap the Variable-level activation operations as Module implementations,
//! so they can be composed in Sequential containers.

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;

/// ReLU activation module: max(0, x)
#[derive(Debug)]
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        Ok(input.relu())
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Sigmoid activation module: 1 / (1 + exp(-x))
#[derive(Debug)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.sigmoid()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Tanh activation module
#[derive(Debug)]
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Tanh
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Tanh {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.tanh_act()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// SiLU / Swish activation: `x * sigmoid(x)`.
///
/// Same shape as the input. This is the activation used inside the
/// SwiGLU FFN block in Llama-style transformers.
#[derive(Debug)]
pub struct SiLU;

impl SiLU {
    pub fn new() -> Self {
        SiLU
    }
}

impl Default for SiLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SiLU {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.silu()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// SwiGLU fused activation: `swiglu(gate, value) = silu(gate) * value`.
///
/// This is the element-wise step inside Llama-style gated FFN blocks:
///
/// ```text
/// gate  = W_gate @ x
/// value = W_value @ x
/// h     = silu(gate) * value
/// out   = W_out @ h
/// ```
///
/// SwiGLU is *not* a `Module` because it needs two inputs; call
/// [`swiglu`] directly from your FFN implementation. Kept here so
/// callers can import it from `nn::activation::swiglu`.
pub fn swiglu(gate: &Variable, value: &Variable) -> Result<Variable> {
    gate.silu()?.mul(value)
}
