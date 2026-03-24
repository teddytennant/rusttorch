//! Embedding — learnable lookup table for discrete tokens.
//!
//! Maps integer indices to dense vectors. Used as the input layer
//! in NLP models (word embeddings, token embeddings, positional embeddings).

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::parameter::Parameter;
use crate::nn::state_dict::StateDict;

/// Embedding lookup table.
///
/// Weight matrix of shape [num_embeddings, embedding_dim].
/// Forward takes integer indices and returns the corresponding rows.
pub struct Embedding {
    pub weight: Parameter,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
}

impl Embedding {
    /// Create a new embedding table with random normal initialization.
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        // Standard normal init (PyTorch default)
        let weight = Parameter::uniform(&[num_embeddings, embedding_dim], 1.0, "weight");

        Embedding {
            weight,
            num_embeddings,
            embedding_dim,
        }
    }

    /// Forward pass: look up embeddings for the given indices.
    ///
    /// indices: flat slice of token indices
    /// output_shape: shape of the output tensor (e.g., [batch, seq_len, embedding_dim])
    pub fn forward_indices(&self, indices: &[usize], output_shape: &[usize]) -> Result<Variable> {
        crate::autograd::ops::embedding_forward(&self.weight.var, indices, output_shape)
    }

    /// Forward pass for a 2D index tensor [batch, seq_len].
    /// Returns [batch, seq_len, embedding_dim].
    pub fn forward_2d(&self, indices: &[Vec<usize>]) -> Result<Variable> {
        let batch = indices.len();
        let seq_len = if batch > 0 { indices[0].len() } else { 0 };

        let flat_indices: Vec<usize> = indices.iter().flatten().copied().collect();
        let output_shape = [batch, seq_len, self.embedding_dim];

        self.forward_indices(&flat_indices, &output_shape)
    }

    /// Get the parameters (weight matrix).
    pub fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone()]
    }

    /// Export state dict.
    pub fn state_dict(&self) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert("weight", self.weight.tensor());
        sd
    }

    /// Load state dict.
    pub fn load_state_dict(&self, state_dict: &StateDict) {
        if let Some(tensor) = state_dict.get("weight") {
            self.weight.update(tensor.clone());
        }
    }
}

impl std::fmt::Debug for Embedding {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Embedding(num_embeddings={}, embedding_dim={})",
            self.num_embeddings, self.embedding_dim
        )
    }
}
