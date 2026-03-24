//! Module trait — the base abstraction for all neural network layers.

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::parameter::Parameter;
use crate::nn::state_dict::StateDict;

/// The base trait for all neural network modules (layers, activations, losses, containers).
///
/// Every module must implement `forward()` and `parameters()`.
/// This mirrors PyTorch's `nn.Module`.
pub trait Module {
    /// Forward pass: compute output from input.
    fn forward(&self, input: &Variable) -> Result<Variable>;

    /// Return all learnable parameters in this module.
    fn parameters(&self) -> Vec<Parameter>;

    /// Zero all parameter gradients.
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }

    /// Count total number of trainable parameters.
    fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .map(|p| p.shape().iter().product::<usize>())
            .sum()
    }

    /// Export the module's learnable parameters as a StateDict.
    ///
    /// The default implementation collects all parameters by name.
    /// Override to include non-learnable state (e.g., BatchNorm running stats).
    fn state_dict(&self) -> StateDict {
        let mut sd = StateDict::new();
        for param in self.parameters() {
            sd.insert(param.name(), param.tensor());
        }
        sd
    }

    /// Load parameters from a StateDict.
    ///
    /// The default implementation matches parameters by name.
    /// Override to load non-learnable state (e.g., BatchNorm running stats).
    fn load_state_dict(&self, state_dict: &StateDict) {
        for param in self.parameters() {
            if let Some(tensor) = state_dict.get(param.name()) {
                param.update(tensor.clone());
            }
        }
    }
}
