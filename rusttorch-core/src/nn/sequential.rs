//! Sequential container — chains modules in order.

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;
use crate::nn::state_dict::StateDict;

/// A sequential container that chains modules in order.
///
/// The output of each module is fed as input to the next.
///
/// # Example
/// ```ignore
/// let model = Sequential::new(vec![
///     Box::new(Linear::new(2, 4)),
///     Box::new(ReLU::new()),
///     Box::new(Linear::new(4, 1)),
///     Box::new(Sigmoid::new()),
/// ]);
/// let output = model.forward(&input)?;
/// ```
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// Create a new Sequential from a list of boxed modules.
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
    }

    /// Get the number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    fn state_dict(&self) -> StateDict {
        let mut sd = StateDict::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_sd = layer.state_dict();
            sd.merge_prefixed(&i.to_string(), &layer_sd);
        }
        sd
    }

    fn load_state_dict(&self, state_dict: &StateDict) {
        for (i, layer) in self.layers.iter().enumerate() {
            let sub = state_dict.sub_dict(&i.to_string());
            if !sub.is_empty() {
                layer.load_state_dict(&sub);
            }
        }
    }
}

impl std::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Sequential({} layers)", self.layers.len())
    }
}
