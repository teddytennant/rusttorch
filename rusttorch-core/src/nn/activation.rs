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
