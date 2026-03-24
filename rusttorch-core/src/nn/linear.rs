//! Linear (fully connected) layer: y = x @ W^T + b

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;

/// A fully connected linear layer.
///
/// Applies y = x @ W^T + b where:
/// - x has shape [batch, in_features]
/// - W has shape [out_features, in_features]
/// - b has shape [1, out_features] (broadcast across batch)
/// - y has shape [batch, out_features]
pub struct Linear {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    /// Create a new Linear layer with Kaiming uniform weight init.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight =
            Parameter::kaiming_uniform(&[out_features, in_features], in_features, "weight");
        let bound = 1.0 / (in_features as f32).sqrt();
        let bias = Some(Parameter::uniform(&[1, out_features], bound, "bias"));

        Linear {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Create a Linear layer without bias.
    pub fn no_bias(in_features: usize, out_features: usize) -> Self {
        let weight =
            Parameter::kaiming_uniform(&[out_features, in_features], in_features, "weight");

        Linear {
            weight,
            bias: None,
            in_features,
            out_features,
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        // input: [batch, in_features], weight: [out_features, in_features]
        // weight^T: [in_features, out_features]
        // input @ weight^T = [batch, out_features]
        let weight_t = self.weight.var().t()?;
        let out = input.matmul(&weight_t)?;

        match &self.bias {
            Some(bias) => out.broadcast_add(bias.var()),
            None => Ok(out),
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

impl std::fmt::Debug for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Linear(in={}, out={}, bias={})",
            self.in_features,
            self.out_features,
            self.bias.is_some()
        )
    }
}
