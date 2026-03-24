//! Transformer building blocks.
//!
//! Implements the standard Transformer encoder layer from
//! "Attention Is All You Need" (Vaswani et al., 2017):
//!
//! TransformerEncoderLayer:
//!   x = x + MultiHeadAttention(LayerNorm(x))   (pre-norm)
//!   x = x + FFN(LayerNorm(x))
//!
//! FFN:
//!   x = Linear(d_model, d_ff) → ReLU → Linear(d_ff, d_model)

use crate::autograd::Variable;
use crate::error::Result;
use crate::nn::attention::MultiHeadAttention;
use crate::nn::layernorm::LayerNorm;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::parameter::Parameter;
use crate::nn::state_dict::StateDict;

/// Transformer encoder layer (pre-norm variant).
///
/// Input: [batch, seq_len, d_model]
/// Output: [batch, seq_len, d_model]
///
/// Architecture:
/// ```text
/// x → LayerNorm → MultiHeadAttention → + → LayerNorm → FFN → + → output
///  └──────────────────────────────────→ ┘ └────────────────→ ┘
/// ```
pub struct TransformerEncoderLayer {
    pub self_attn: MultiHeadAttention,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub ff_linear1: Linear,
    pub ff_linear2: Linear,
    pub d_model: usize,
    pub d_ff: usize,
    pub num_heads: usize,
}

impl TransformerEncoderLayer {
    /// Create a new Transformer encoder layer.
    ///
    /// - d_model: model dimension (embedding size)
    /// - num_heads: number of attention heads
    /// - d_ff: feed-forward inner dimension (typically 4 * d_model)
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        TransformerEncoderLayer {
            self_attn: MultiHeadAttention::new(d_model, num_heads),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
            ff_linear1: Linear::new(d_model, d_ff),
            ff_linear2: Linear::new(d_ff, d_model),
            d_model,
            d_ff,
            num_heads,
        }
    }

    /// Forward pass.
    ///
    /// Input: [batch, seq_len, d_model]
    /// Returns: [batch, seq_len, d_model]
    pub fn forward(&self, input: &Variable) -> Result<Variable> {
        let shape = input.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Pre-norm self-attention block
        // x_norm = LayerNorm(x)
        let x_flat = input.reshape(&[batch * seq_len, self.d_model])?;
        let x_norm1 = self.norm1.forward(&x_flat)?;
        let x_norm1_3d = x_norm1.reshape(&[batch, seq_len, self.d_model])?;

        // attn_out = MultiHeadAttention(x_norm, x_norm, x_norm)
        let attn_out = self.self_attn.forward_self_attn(&x_norm1_3d)?;

        // Residual connection: x = x + attn_out
        let attn_out_flat = attn_out.reshape(&[batch * seq_len, self.d_model])?;
        let x_residual1 = x_flat.add(&attn_out_flat)?;

        // Pre-norm FFN block
        let x_norm2 = self.norm2.forward(&x_residual1)?;

        // FFN: Linear → ReLU → Linear
        let ff_out = self.ff_linear1.forward(&x_norm2)?;
        let ff_out = ff_out.relu();
        let ff_out = self.ff_linear2.forward(&ff_out)?;

        // Residual connection: x = x + ff_out
        let x_residual2 = x_residual1.add(&ff_out)?;

        // Reshape back to 3D
        x_residual2.reshape(&[batch, seq_len, self.d_model])
    }

    /// Get all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.ff_linear1.parameters());
        params.extend(self.ff_linear2.parameters());
        params
    }

    /// Total number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .map(|p| p.shape().iter().product::<usize>())
            .sum()
    }

    /// Export state dict with hierarchical naming.
    pub fn state_dict(&self) -> StateDict {
        let mut sd = StateDict::new();
        sd.merge_prefixed("self_attn", &self.self_attn.state_dict());
        sd.merge_prefixed("norm1", &self.norm1.state_dict());
        sd.merge_prefixed("norm2", &self.norm2.state_dict());
        sd.merge_prefixed("ff_linear1", &self.ff_linear1.state_dict());
        sd.merge_prefixed("ff_linear2", &self.ff_linear2.state_dict());
        sd
    }

    /// Load state dict.
    pub fn load_state_dict(&self, state_dict: &StateDict) {
        self.self_attn
            .load_state_dict(&state_dict.sub_dict("self_attn"));
        self.norm1.load_state_dict(&state_dict.sub_dict("norm1"));
        self.norm2.load_state_dict(&state_dict.sub_dict("norm2"));
        self.ff_linear1
            .load_state_dict(&state_dict.sub_dict("ff_linear1"));
        self.ff_linear2
            .load_state_dict(&state_dict.sub_dict("ff_linear2"));
    }
}

impl std::fmt::Debug for TransformerEncoderLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "TransformerEncoderLayer(d_model={}, num_heads={}, d_ff={})",
            self.d_model, self.num_heads, self.d_ff
        )
    }
}

/// A stack of Transformer encoder layers.
///
/// Input: [batch, seq_len, d_model]
/// Output: [batch, seq_len, d_model]
pub struct TransformerEncoder {
    pub layers: Vec<TransformerEncoderLayer>,
    pub norm: LayerNorm,
    pub d_model: usize,
}

impl TransformerEncoder {
    /// Create a new TransformerEncoder with `num_layers` identical layers.
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, num_layers: usize) -> Self {
        let layers: Vec<_> = (0..num_layers)
            .map(|_| TransformerEncoderLayer::new(d_model, num_heads, d_ff))
            .collect();
        let norm = LayerNorm::new(d_model);

        TransformerEncoder {
            layers,
            norm,
            d_model,
        }
    }

    /// Forward pass through all encoder layers.
    pub fn forward(&self, input: &Variable) -> Result<Variable> {
        let shape = input.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        // Final layer norm
        let x_flat = x.reshape(&[batch * seq_len, self.d_model])?;
        let x_norm = self.norm.forward(&x_flat)?;
        x_norm.reshape(&[batch, seq_len, self.d_model])
    }

    /// Get all parameters from all layers + final norm.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.extend(self.norm.parameters());
        params
    }

    /// Total number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .map(|p| p.shape().iter().product::<usize>())
            .sum()
    }

    /// Export state dict.
    pub fn state_dict(&self) -> StateDict {
        let mut sd = StateDict::new();
        for (i, layer) in self.layers.iter().enumerate() {
            sd.merge_prefixed(&format!("layers.{}", i), &layer.state_dict());
        }
        sd.merge_prefixed("norm", &self.norm.state_dict());
        sd
    }

    /// Load state dict.
    pub fn load_state_dict(&self, state_dict: &StateDict) {
        for (i, layer) in self.layers.iter().enumerate() {
            layer.load_state_dict(&state_dict.sub_dict(&format!("layers.{}", i)));
        }
        self.norm.load_state_dict(&state_dict.sub_dict("norm"));
    }
}

impl std::fmt::Debug for TransformerEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "TransformerEncoder(num_layers={}, d_model={}, params={})",
            self.layers.len(),
            self.d_model,
            self.num_parameters()
        )
    }
}
