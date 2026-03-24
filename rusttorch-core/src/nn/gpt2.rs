//! GPT-2 model definition for inference
//!
//! Loads pre-trained GPT-2 weights from HuggingFace safetensors format
//! and runs autoregressive text generation.
//!
//! # Architecture (GPT-2 Small)
//! - 12 transformer layers
//! - 768 embedding dimensions
//! - 12 attention heads (head_dim = 64)
//! - 3072 FFN hidden dimensions (4 * d_model)
//! - 50257 vocabulary (byte-level BPE)
//! - 1024 max context length
//!
//! # Weight format
//! GPT-2 uses Conv1D (transposed Linear): y = x @ W + b
//! Weights are stored as [in_features, out_features], which is the
//! transpose of nn.Linear convention [out_features, in_features].

use crate::nn::safetensors::load_safetensors;
use crate::nn::state_dict::StateDict;
use crate::ops::{self, matrix};
use crate::tensor::Tensor;
use std::path::Path;

/// Map any error to String for uniform error handling.
fn err_str<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

/// GPT-2 model configuration.
#[derive(Debug, Clone)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub layer_norm_epsilon: f32,
}

impl Gpt2Config {
    /// GPT-2 Small (124M parameters)
    pub fn gpt2_small() -> Self {
        Gpt2Config {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            layer_norm_epsilon: 1e-5,
        }
    }
}

/// GPT-2 model for inference.
#[allow(missing_debug_implementations)]
pub struct Gpt2Model {
    config: Gpt2Config,
    /// Token embeddings [vocab_size, n_embd]
    wte: Tensor,
    /// Position embeddings [n_positions, n_embd]
    wpe: Tensor,
    /// Transformer layers
    layers: Vec<Gpt2Layer>,
    /// Final layer norm
    ln_f_weight: Tensor,
    ln_f_bias: Tensor,
}

/// A single GPT-2 transformer layer.
struct Gpt2Layer {
    // Pre-attention layer norm
    ln_1_weight: Tensor,
    ln_1_bias: Tensor,
    // Attention (fused QKV)
    c_attn_weight: Tensor, // [n_embd, 3 * n_embd] (Conv1D format)
    c_attn_bias: Tensor,   // [3 * n_embd]
    c_proj_weight: Tensor,  // [n_embd, n_embd]
    c_proj_bias: Tensor,    // [n_embd]
    // Pre-FFN layer norm
    ln_2_weight: Tensor,
    ln_2_bias: Tensor,
    // FFN
    mlp_fc_weight: Tensor,   // [n_embd, 4 * n_embd]
    mlp_fc_bias: Tensor,     // [4 * n_embd]
    mlp_proj_weight: Tensor, // [4 * n_embd, n_embd]
    mlp_proj_bias: Tensor,   // [n_embd]
}

impl Gpt2Model {
    /// Load a GPT-2 model from a safetensors file.
    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        config: Gpt2Config,
    ) -> Result<Self, String> {
        let sd = load_safetensors(path).map_err(|e| format!("Failed to load safetensors: {}", e))?;
        Self::from_state_dict(sd, config)
    }

    /// Load a GPT-2 model from a StateDict.
    pub fn from_state_dict(sd: StateDict, config: Gpt2Config) -> Result<Self, String> {
        let get = |name: &str| -> Result<Tensor, String> {
            sd.get(name)
                .cloned()
                .ok_or_else(|| format!("Missing weight: {}", name))
        };

        let wte = get("wte.weight")?;
        let wpe = get("wpe.weight")?;
        let ln_f_weight = get("ln_f.weight")?;
        let ln_f_bias = get("ln_f.bias")?;

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let prefix = format!("h.{}", i);
            let layer = Gpt2Layer {
                ln_1_weight: get(&format!("{}.ln_1.weight", prefix))?,
                ln_1_bias: get(&format!("{}.ln_1.bias", prefix))?,
                c_attn_weight: get(&format!("{}.attn.c_attn.weight", prefix))?,
                c_attn_bias: get(&format!("{}.attn.c_attn.bias", prefix))?,
                c_proj_weight: get(&format!("{}.attn.c_proj.weight", prefix))?,
                c_proj_bias: get(&format!("{}.attn.c_proj.bias", prefix))?,
                ln_2_weight: get(&format!("{}.ln_2.weight", prefix))?,
                ln_2_bias: get(&format!("{}.ln_2.bias", prefix))?,
                mlp_fc_weight: get(&format!("{}.mlp.c_fc.weight", prefix))?,
                mlp_fc_bias: get(&format!("{}.mlp.c_fc.bias", prefix))?,
                mlp_proj_weight: get(&format!("{}.mlp.c_proj.weight", prefix))?,
                mlp_proj_bias: get(&format!("{}.mlp.c_proj.bias", prefix))?,
            };
            layers.push(layer);
        }

        Ok(Gpt2Model {
            config,
            wte,
            wpe,
            layers,
            ln_f_weight,
            ln_f_bias,
        })
    }

    /// Forward pass: token IDs → logits.
    ///
    /// Input: slice of token IDs (length = sequence length)
    /// Output: logits tensor [seq_len, vocab_size]
    pub fn forward(&self, token_ids: &[u32]) -> Result<Tensor, String> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err("Empty input".to_string());
        }
        if seq_len > self.config.n_positions {
            return Err(format!(
                "Sequence length {} exceeds max position {}",
                seq_len, self.config.n_positions
            ));
        }

        // Token embeddings: look up each token ID
        let token_emb = embedding_lookup(&self.wte, token_ids)?;

        // Position embeddings: positions 0..seq_len
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        let pos_emb = embedding_lookup(&self.wpe, &positions)?;

        // x = token_emb + pos_emb [seq_len, n_embd]
        let mut x = ops::elementwise::add(&token_emb, &pos_emb).map_err(err_str)?;

        // Run through transformer layers
        for layer in &self.layers {
            x = self.forward_layer(layer, &x)?;
        }

        // Final layer norm
        x = layer_norm(&x, &self.ln_f_weight, &self.ln_f_bias, self.config.layer_norm_epsilon)?;

        // LM head: x @ wte.T (tied weights)
        let wte_t = matrix::transpose(&self.wte);
        let logits = matrix::matmul(&x, &wte_t)?;

        Ok(logits)
    }

    /// Generate text autoregressively.
    ///
    /// - `token_ids`: prompt token IDs
    /// - `max_new_tokens`: number of tokens to generate
    /// - `temperature`: sampling temperature (0.0 = greedy, higher = more random)
    pub fn generate(
        &self,
        token_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<u32>, String> {
        let mut tokens = token_ids.to_vec();

        for _ in 0..max_new_tokens {
            // Truncate to max context length
            let start = if tokens.len() > self.config.n_positions {
                tokens.len() - self.config.n_positions
            } else {
                0
            };
            let context = &tokens[start..];

            // Forward pass
            let logits = self.forward(context)?;

            // Get logits for the last position
            let seq_len = context.len();
            let last_logits = extract_row(&logits, seq_len - 1)?;

            // Sample next token
            let next_token = if temperature <= 0.0 {
                argmax_f32(&last_logits)
            } else {
                sample_with_temperature(&last_logits, temperature)
            };

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Number of parameters in the model.
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;
        total += self.wte.numel();
        total += self.wpe.numel();
        total += self.ln_f_weight.numel();
        total += self.ln_f_bias.numel();
        for layer in &self.layers {
            total += layer.ln_1_weight.numel();
            total += layer.ln_1_bias.numel();
            total += layer.c_attn_weight.numel();
            total += layer.c_attn_bias.numel();
            total += layer.c_proj_weight.numel();
            total += layer.c_proj_bias.numel();
            total += layer.ln_2_weight.numel();
            total += layer.ln_2_bias.numel();
            total += layer.mlp_fc_weight.numel();
            total += layer.mlp_fc_bias.numel();
            total += layer.mlp_proj_weight.numel();
            total += layer.mlp_proj_bias.numel();
        }
        total
    }

    /// Forward pass through a single transformer layer.
    fn forward_layer(&self, layer: &Gpt2Layer, x: &Tensor) -> Result<Tensor, String> {
        let n_embd = self.config.n_embd;
        let n_head = self.config.n_head;
        let head_dim = n_embd / n_head;
        let seq_len = x.shape()[0];

        // Pre-attention layer norm
        let ln1 = layer_norm(x, &layer.ln_1_weight, &layer.ln_1_bias, self.config.layer_norm_epsilon)?;

        // Attention: fused QKV projection (Conv1D: x @ W + b)
        let qkv = conv1d_forward(&ln1, &layer.c_attn_weight, &layer.c_attn_bias)?;

        // Split into Q, K, V [seq_len, n_embd] each
        let q = slice_columns(&qkv, 0, n_embd)?;
        let k = slice_columns(&qkv, n_embd, 2 * n_embd)?;
        let v = slice_columns(&qkv, 2 * n_embd, 3 * n_embd)?;

        // Reshape to [n_head, seq_len, head_dim] and compute attention
        let attn_out = multi_head_attention(&q, &k, &v, n_head, head_dim, seq_len)?;

        // Output projection
        let attn_proj = conv1d_forward(&attn_out, &layer.c_proj_weight, &layer.c_proj_bias)?;

        // Residual connection
        let x2 = ops::elementwise::add(x, &attn_proj).map_err(err_str)?;

        // Pre-FFN layer norm
        let ln2 = layer_norm(&x2, &layer.ln_2_weight, &layer.ln_2_bias, self.config.layer_norm_epsilon)?;

        // FFN: fc → gelu → proj
        let fc = conv1d_forward(&ln2, &layer.mlp_fc_weight, &layer.mlp_fc_bias)?;
        let activated = gelu_new(&fc)?;
        let proj = conv1d_forward(&activated, &layer.mlp_proj_weight, &layer.mlp_proj_bias)?;

        // Residual connection
        ops::elementwise::add(&x2, &proj).map_err(err_str)
    }
}

// ---- Helper functions ----

/// Embedding lookup: gather rows from a weight matrix by index.
fn embedding_lookup(weight: &Tensor, indices: &[u32]) -> Result<Tensor, String> {
    let data = weight.to_vec_f32();
    let n_embd = weight.shape()[1];
    let vocab_size = weight.shape()[0];
    let seq_len = indices.len();

    let mut output = vec![0.0f32; seq_len * n_embd];
    for (i, &idx) in indices.iter().enumerate() {
        let idx = idx as usize;
        if idx >= vocab_size {
            return Err(format!("Token ID {} out of range (vocab_size={})", idx, vocab_size));
        }
        let src_start = idx * n_embd;
        let dst_start = i * n_embd;
        output[dst_start..dst_start + n_embd].copy_from_slice(&data[src_start..src_start + n_embd]);
    }

    Ok(Tensor::from_vec(output, &[seq_len, n_embd]))
}

/// Conv1D forward: y = x @ W + b (GPT-2 weight convention).
fn conv1d_forward(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor, String> {
    // x: [seq_len, in_features], weight: [in_features, out_features]
    let y = matrix::matmul(x, weight)?;
    // Add bias: [out_features] broadcast to [seq_len, out_features]
    ops::broadcast::add_broadcast(&y, bias).map_err(err_str)
}

/// Layer normalization: (x - mean) / sqrt(var + eps) * weight + bias.
fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor, String> {
    let data = x.to_vec_f32();
    let shape = x.shape().to_vec();
    let n = *shape.last().unwrap();
    let batch = data.len() / n;

    let w = weight.to_vec_f32();
    let b = bias.to_vec_f32();

    let mut output = vec![0.0f32; data.len()];

    for i in 0..batch {
        let start = i * n;
        let end = start + n;
        let row = &data[start..end];

        // Compute mean
        let mean: f32 = row.iter().sum::<f32>() / n as f32;

        // Compute variance
        let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        // Normalize, scale, shift
        for j in 0..n {
            output[start + j] = (row[j] - mean) * inv_std * w[j] + b[j];
        }
    }

    Ok(Tensor::from_vec(output, &shape))
}

/// Slice columns from a 2D tensor.
fn slice_columns(tensor: &Tensor, start: usize, end: usize) -> Result<Tensor, String> {
    let data = tensor.to_vec_f32();
    let rows = tensor.shape()[0];
    let cols = tensor.shape()[1];
    let out_cols = end - start;

    let mut output = vec![0.0f32; rows * out_cols];
    for i in 0..rows {
        for j in 0..out_cols {
            output[i * out_cols + j] = data[i * cols + start + j];
        }
    }

    Ok(Tensor::from_vec(output, &[rows, out_cols]))
}

/// Multi-head attention with causal masking.
fn multi_head_attention(
    q: &Tensor, k: &Tensor, v: &Tensor,
    n_head: usize, head_dim: usize, seq_len: usize,
) -> Result<Tensor, String> {
    let q_data = q.to_vec_f32();
    let k_data = k.to_vec_f32();
    let v_data = v.to_vec_f32();
    let n_embd = n_head * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut output = vec![0.0f32; seq_len * n_embd];

    for h in 0..n_head {
        // Extract head slices
        let q_head = extract_head(&q_data, seq_len, n_embd, h, head_dim);
        let k_head = extract_head(&k_data, seq_len, n_embd, h, head_dim);
        let v_head = extract_head(&v_data, seq_len, n_embd, h, head_dim);

        // Compute attention scores: Q @ K^T * scale
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    // Causal mask: future positions get -inf
                    scores[i * seq_len + j] = f32::NEG_INFINITY;
                } else {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_head[i * head_dim + d] * k_head[j * head_dim + d];
                    }
                    scores[i * seq_len + j] = dot * scale;
                }
            }
        }

        // Softmax over each row
        for i in 0..seq_len {
            let row_start = i * seq_len;
            let row_end = row_start + seq_len;
            let row = &mut scores[row_start..row_end];

            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }

        // Attention output: scores @ V
        for i in 0..seq_len {
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for j in 0..seq_len {
                    val += scores[i * seq_len + j] * v_head[j * head_dim + d];
                }
                output[i * n_embd + h * head_dim + d] = val;
            }
        }
    }

    Ok(Tensor::from_vec(output, &[seq_len, n_embd]))
}

/// Extract a single attention head's data from the interleaved layout.
fn extract_head(data: &[f32], seq_len: usize, n_embd: usize, head: usize, head_dim: usize) -> Vec<f32> {
    let mut head_data = vec![0.0f32; seq_len * head_dim];
    let offset = head * head_dim;
    for i in 0..seq_len {
        for d in 0..head_dim {
            head_data[i * head_dim + d] = data[i * n_embd + offset + d];
        }
    }
    head_data
}

/// GPT-2's "gelu_new" activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu_new(tensor: &Tensor) -> Result<Tensor, String> {
    let data = tensor.to_vec_f32();
    let sqrt_2_pi = (2.0f32 / std::f32::consts::PI).sqrt();

    let output: Vec<f32> = data.iter().map(|&x| {
        let inner = sqrt_2_pi * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }).collect();

    Ok(Tensor::from_vec(output, tensor.shape()))
}

/// Extract a single row from a 2D tensor.
fn extract_row(tensor: &Tensor, row: usize) -> Result<Vec<f32>, String> {
    let data = tensor.to_vec_f32();
    let cols = tensor.shape()[1];
    let start = row * cols;
    Ok(data[start..start + cols].to_vec())
}

/// Argmax over f32 slice, returns index as u32.
fn argmax_f32(data: &[f32]) -> u32 {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in data.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx as u32
}

/// Sample from logits with temperature.
fn sample_with_temperature(logits: &[f32], temperature: f32) -> u32 {
    use rand::Rng;

    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|&x| x / sum).collect();

    // Sample from categorical distribution
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }

    // Fallback (shouldn't reach here)
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_small_config() -> Gpt2Config {
        Gpt2Config {
            vocab_size: 10,
            n_positions: 16,
            n_embd: 8,
            n_layer: 2,
            n_head: 2,
            layer_norm_epsilon: 1e-5,
        }
    }

    fn make_small_state_dict(config: &Gpt2Config) -> StateDict {
        let mut sd = StateDict::new();
        let n = config.n_embd;
        let ff = 4 * n;

        // Embeddings
        sd.insert("wte.weight".to_string(),
            Tensor::from_vec(vec![0.1; config.vocab_size * n], &[config.vocab_size, n]));
        sd.insert("wpe.weight".to_string(),
            Tensor::from_vec(vec![0.01; config.n_positions * n], &[config.n_positions, n]));

        // Final layer norm
        sd.insert("ln_f.weight".to_string(), Tensor::from_vec(vec![1.0; n], &[n]));
        sd.insert("ln_f.bias".to_string(), Tensor::from_vec(vec![0.0; n], &[n]));

        for i in 0..config.n_layer {
            let p = format!("h.{}", i);

            // LN1
            sd.insert(format!("{}.ln_1.weight", p), Tensor::from_vec(vec![1.0; n], &[n]));
            sd.insert(format!("{}.ln_1.bias", p), Tensor::from_vec(vec![0.0; n], &[n]));

            // Attention
            sd.insert(format!("{}.attn.c_attn.weight", p),
                Tensor::from_vec(vec![0.01; n * 3 * n], &[n, 3 * n]));
            sd.insert(format!("{}.attn.c_attn.bias", p),
                Tensor::from_vec(vec![0.0; 3 * n], &[3 * n]));
            sd.insert(format!("{}.attn.c_proj.weight", p),
                Tensor::from_vec(vec![0.01; n * n], &[n, n]));
            sd.insert(format!("{}.attn.c_proj.bias", p),
                Tensor::from_vec(vec![0.0; n], &[n]));

            // LN2
            sd.insert(format!("{}.ln_2.weight", p), Tensor::from_vec(vec![1.0; n], &[n]));
            sd.insert(format!("{}.ln_2.bias", p), Tensor::from_vec(vec![0.0; n], &[n]));

            // MLP
            sd.insert(format!("{}.mlp.c_fc.weight", p),
                Tensor::from_vec(vec![0.01; n * ff], &[n, ff]));
            sd.insert(format!("{}.mlp.c_fc.bias", p),
                Tensor::from_vec(vec![0.0; ff], &[ff]));
            sd.insert(format!("{}.mlp.c_proj.weight", p),
                Tensor::from_vec(vec![0.01; ff * n], &[ff, n]));
            sd.insert(format!("{}.mlp.c_proj.bias", p),
                Tensor::from_vec(vec![0.0; n], &[n]));
        }

        sd
    }

    #[test]
    fn test_gpt2_config_small() {
        let config = Gpt2Config::gpt2_small();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.n_embd, 768);
        assert_eq!(config.n_layer, 12);
        assert_eq!(config.n_head, 12);
        assert_eq!(config.n_positions, 1024);
    }

    #[test]
    fn test_gpt2_load_from_state_dict() {
        let config = make_small_config();
        let sd = make_small_state_dict(&config);
        let model = Gpt2Model::from_state_dict(sd, config).unwrap();
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_gpt2_forward_shape() {
        let config = make_small_config();
        let sd = make_small_state_dict(&config);
        let model = Gpt2Model::from_state_dict(sd, config.clone()).unwrap();

        let logits = model.forward(&[0, 1, 2]).unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size]);
    }

    #[test]
    fn test_gpt2_forward_single_token() {
        let config = make_small_config();
        let sd = make_small_state_dict(&config);
        let model = Gpt2Model::from_state_dict(sd, config.clone()).unwrap();

        let logits = model.forward(&[0]).unwrap();
        assert_eq!(logits.shape(), &[1, config.vocab_size]);
    }

    #[test]
    fn test_gpt2_forward_empty_fails() {
        let config = make_small_config();
        let sd = make_small_state_dict(&config);
        let model = Gpt2Model::from_state_dict(sd, config).unwrap();

        assert!(model.forward(&[]).is_err());
    }

    #[test]
    fn test_gpt2_forward_too_long_fails() {
        let config = make_small_config();
        let sd = make_small_state_dict(&config);
        let model = Gpt2Model::from_state_dict(sd, config).unwrap();

        let long_input: Vec<u32> = (0..20).collect(); // > n_positions (16)
        assert!(model.forward(&long_input).is_err());
    }

    #[test]
    fn test_gpt2_generate_greedy() {
        let config = make_small_config();
        let sd = make_small_state_dict(&config);
        let model = Gpt2Model::from_state_dict(sd, config).unwrap();

        let result = model.generate(&[0, 1], 3, 0.0).unwrap();
        assert_eq!(result.len(), 5); // 2 prompt + 3 generated
    }

    #[test]
    fn test_gpt2_generate_with_temperature() {
        let config = make_small_config();
        let sd = make_small_state_dict(&config);
        let model = Gpt2Model::from_state_dict(sd, config).unwrap();

        let result = model.generate(&[0], 5, 1.0).unwrap();
        assert_eq!(result.len(), 6); // 1 prompt + 5 generated
        // All tokens should be valid
        for &t in &result {
            assert!(t < 10); // vocab_size = 10
        }
    }

    #[test]
    fn test_gpt2_num_parameters() {
        let config = make_small_config();
        let sd = make_small_state_dict(&config);
        let model = Gpt2Model::from_state_dict(sd, config).unwrap();
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_embedding_lookup() {
        let weight = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
        );
        let result = embedding_lookup(&weight, &[2, 0]).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        let data = result.to_vec_f32();
        assert_eq!(data, vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn test_embedding_lookup_out_of_range() {
        let weight = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]);
        assert!(embedding_lookup(&weight, &[5]).is_err());
    }

    #[test]
    fn test_layer_norm() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let w = Tensor::from_vec(vec![1.0, 1.0], &[2]);
        let b = Tensor::from_vec(vec![0.0, 0.0], &[2]);
        let result = layer_norm(&x, &w, &b, 1e-5).unwrap();
        let data = result.to_vec_f32();
        // Each row should have mean≈0
        assert!((data[0] + data[1]).abs() < 0.01);
        assert!((data[2] + data[3]).abs() < 0.01);
    }

    #[test]
    fn test_slice_columns() {
        let t = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        );
        let sliced = slice_columns(&t, 1, 3).unwrap();
        assert_eq!(sliced.shape(), &[2, 2]);
        assert_eq!(sliced.to_vec_f32(), vec![2.0, 3.0, 5.0, 6.0]);
    }

    #[test]
    fn test_gelu_new_values() {
        let t = Tensor::from_vec(vec![0.0, 1.0, -1.0, 3.0], &[1, 4]);
        let result = gelu_new(&t).unwrap();
        let data = result.to_vec_f32();
        assert!((data[0] - 0.0).abs() < 0.01);
        assert!((data[1] - 0.841).abs() < 0.01);
        assert!((data[2] - (-0.159)).abs() < 0.01);
    }

    #[test]
    fn test_conv1d_forward() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let w = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 3]);
        let b = Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]);
        let y = conv1d_forward(&x, &w, &b).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax_f32(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax_f32(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(argmax_f32(&[1.0, 2.0, 5.0]), 2);
    }

    #[test]
    fn test_sample_with_temperature() {
        let logits = vec![100.0, -100.0, -100.0]; // Overwhelming preference for 0
        let token = sample_with_temperature(&logits, 1.0);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_gpt2_deterministic_greedy() {
        let config = make_small_config();
        let sd = make_small_state_dict(&config);
        let model = Gpt2Model::from_state_dict(sd, config).unwrap();

        // Greedy should be deterministic
        let r1 = model.generate(&[0, 1], 5, 0.0).unwrap();
        let r2 = model.generate(&[0, 1], 5, 0.0).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_missing_weight_error() {
        let config = make_small_config();
        let sd = StateDict::new(); // empty
        let result = Gpt2Model::from_state_dict(sd, config);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.contains("Missing weight"), "Error was: {}", err);
    }

    #[test]
    fn test_multi_head_attention_single_token() {
        // Single token should produce valid output
        let q = Tensor::from_vec(vec![0.1; 8], &[1, 8]);
        let k = q.clone();
        let v = q.clone();
        let result = multi_head_attention(&q, &k, &v, 2, 4, 1).unwrap();
        assert_eq!(result.shape(), &[1, 8]);
    }

    #[test]
    fn test_multi_head_attention_causal() {
        // Two tokens: position 0 should only attend to itself
        let q = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], &[2, 4]);
        let k = q.clone();
        let v = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        let result = multi_head_attention(&q, &k, &v, 2, 2, 2).unwrap();
        assert_eq!(result.shape(), &[2, 4]);
        // Position 0 output should equal position 0 values (only attends to self)
        let data = result.to_vec_f32();
        assert!((data[0] - 1.0).abs() < 0.01);
        assert!((data[1] - 2.0).abs() < 0.01);
    }
}
