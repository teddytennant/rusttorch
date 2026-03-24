//! Multi-Head Attention — the core building block of Transformer models.
//!
//! Implements the attention mechanism from "Attention Is All You Need" (Vaswani et al., 2017):
//! MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
//! where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)

use crate::autograd::Variable;
use crate::error::{Result, TensorError};
use crate::nn::parameter::Parameter;
use crate::nn::state_dict::StateDict;

/// Multi-Head Attention module.
///
/// Input: [batch, seq_len, d_model]
/// Projects Q, K, V through linear layers, splits into heads,
/// applies scaled dot-product attention, concatenates, and projects output.
pub struct MultiHeadAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// Query projection weight [d_model, d_model]
    pub w_q: Parameter,
    /// Key projection weight [d_model, d_model]
    pub w_k: Parameter,
    /// Value projection weight [d_model, d_model]
    pub w_v: Parameter,
    /// Output projection weight [d_model, d_model]
    pub w_o: Parameter,
    /// Query projection bias [d_model]
    pub b_q: Parameter,
    /// Key projection bias [d_model]
    pub b_k: Parameter,
    /// Value projection bias [d_model]
    pub b_v: Parameter,
    /// Output projection bias [d_model]
    pub b_o: Parameter,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention module.
    ///
    /// d_model must be divisible by num_heads.
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert!(
            d_model.is_multiple_of(num_heads),
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );

        let head_dim = d_model / num_heads;

        let w_q = Parameter::kaiming_uniform(&[d_model, d_model], d_model, "q_proj.weight");
        let w_k = Parameter::kaiming_uniform(&[d_model, d_model], d_model, "k_proj.weight");
        let w_v = Parameter::kaiming_uniform(&[d_model, d_model], d_model, "v_proj.weight");
        let w_o = Parameter::kaiming_uniform(&[d_model, d_model], d_model, "out_proj.weight");

        let bound = 1.0 / (d_model as f32).sqrt();
        let b_q = Parameter::uniform(&[1, d_model], bound, "q_proj.bias");
        let b_k = Parameter::uniform(&[1, d_model], bound, "k_proj.bias");
        let b_v = Parameter::uniform(&[1, d_model], bound, "v_proj.bias");
        let b_o = Parameter::uniform(&[1, d_model], bound, "out_proj.bias");

        MultiHeadAttention {
            d_model,
            num_heads,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
            b_q,
            b_k,
            b_v,
            b_o,
        }
    }

    /// Forward pass for self-attention (Q=K=V=input).
    ///
    /// Input: [batch, seq_len, d_model]
    /// Returns: [batch, seq_len, d_model]
    pub fn forward_self_attn(&self, input: &Variable) -> Result<Variable> {
        self.forward_impl(input, input, input, false)
    }

    /// Forward pass for causal self-attention (Q=K=V=input with lower-triangular mask).
    ///
    /// Each position can only attend to itself and previous positions.
    /// Input: [batch, seq_len, d_model]
    /// Returns: [batch, seq_len, d_model]
    pub fn forward_causal(&self, input: &Variable) -> Result<Variable> {
        self.forward_impl(input, input, input, true)
    }

    /// Forward pass with separate Q, K, V inputs (cross-attention).
    ///
    /// Q: [batch, seq_q, d_model]
    /// K: [batch, seq_k, d_model]
    /// V: [batch, seq_k, d_model]
    /// Returns: [batch, seq_q, d_model]
    pub fn forward_qkv(
        &self,
        query: &Variable,
        key: &Variable,
        value: &Variable,
    ) -> Result<Variable> {
        self.forward_impl(query, key, value, false)
    }

    fn forward_impl(
        &self,
        query: &Variable,
        key: &Variable,
        value: &Variable,
        causal: bool,
    ) -> Result<Variable> {
        let q_shape = query.shape();
        let k_shape = key.shape();

        if q_shape.len() != 3 {
            return Err(TensorError::InvalidArgument {
                parameter: "query".to_string(),
                reason: format!("expected 3D [batch, seq, d_model], got {}D", q_shape.len()),
            });
        }

        let batch = q_shape[0];
        let seq_q = q_shape[1];
        let seq_k = k_shape[1];

        // Linear projections: [B, seq, d_model] @ [d_model, d_model]^T + bias
        let q = self.linear_project(query, &self.w_q, &self.b_q)?;
        let k = self.linear_project(key, &self.w_k, &self.b_k)?;
        let v = self.linear_project(value, &self.w_v, &self.b_v)?;

        // Reshape to [B * num_heads, seq, head_dim]
        let q = q.reshape(&[batch * self.num_heads, seq_q, self.head_dim])?;
        let k = k.reshape(&[batch * self.num_heads, seq_k, self.head_dim])?;
        let v = v.reshape(&[batch * self.num_heads, seq_k, self.head_dim])?;

        // Scaled dot-product attention (with optional causal masking)
        let (attn_output, _attn_weights) = if causal {
            crate::autograd::ops::scaled_dot_product_attention_causal_forward(&q, &k, &v)?
        } else {
            crate::autograd::ops::scaled_dot_product_attention_forward(&q, &k, &v)?
        };

        // Reshape back: [B * num_heads, seq_q, head_dim] → [B, seq_q, d_model]
        let attn_output = attn_output.reshape(&[batch, seq_q, self.d_model])?;

        // Output projection
        let output = self.linear_project(&attn_output, &self.w_o, &self.b_o)?;

        Ok(output)
    }

    /// Apply linear projection: x @ W^T + b for 3D input.
    /// x: [B, seq, d_in], W: [d_out, d_in], b: [1, d_out]
    /// Result: [B, seq, d_out]
    fn linear_project(
        &self,
        x: &Variable,
        weight: &Parameter,
        bias: &Parameter,
    ) -> Result<Variable> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let d_in = shape[2];

        // Reshape to [B*seq, d_in]
        let x_flat = x.reshape(&[batch * seq_len, d_in])?;

        // matmul: [B*seq, d_in] @ [d_in, d_out] = [B*seq, d_out]
        let w_t = weight.var().t()?;
        let out = x_flat.matmul(&w_t)?;

        // Add bias via broadcast: [B*seq, d_out] + [1, d_out]
        let out = out.broadcast_add(bias.var())?;

        // Reshape back: [B*seq, d_out] → [B, seq, d_out]
        let d_out = weight.shape()[0];
        out.reshape(&[batch, seq_len, d_out])
    }

    /// Get all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        vec![
            self.w_q.clone(),
            self.b_q.clone(),
            self.w_k.clone(),
            self.b_k.clone(),
            self.w_v.clone(),
            self.b_v.clone(),
            self.w_o.clone(),
            self.b_o.clone(),
        ]
    }

    /// Export state dict.
    pub fn state_dict(&self) -> StateDict {
        let mut sd = StateDict::new();
        for p in self.parameters() {
            sd.insert(p.name(), p.tensor());
        }
        sd
    }

    /// Load state dict.
    pub fn load_state_dict(&self, state_dict: &StateDict) {
        for p in self.parameters() {
            if let Some(tensor) = state_dict.get(p.name()) {
                p.update(tensor.clone());
            }
        }
    }

    /// Total number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .map(|p| p.shape().iter().product::<usize>())
            .sum()
    }
}

impl std::fmt::Debug for MultiHeadAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "MultiHeadAttention(d_model={}, num_heads={}, head_dim={})",
            self.d_model, self.num_heads, self.head_dim
        )
    }
}
