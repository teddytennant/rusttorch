//! Character-level GPT — text generation with a Transformer decoder.
//!
//! Demonstrates the full RustTorch Transformer pipeline:
//! - Character-level tokenization
//! - Learnable token + positional embeddings
//! - Causal (masked) self-attention
//! - TransformerEncoder stack with causal masking (= GPT decoder)
//! - Cross-entropy loss for next-token prediction
//! - Temperature-based text generation
//!
//! Run with: cargo run -p rusttorch-core --example gpt_char --release
//!
//! Optional arguments:
//!   cargo run -p rusttorch-core --example gpt_char --release -- [text_file] [epochs] [context_len]
//!
//! Example:
//!   cargo run -p rusttorch-core --example gpt_char --release -- shakespeare.txt 10 64

use rusttorch_core::autograd::Variable;
use rusttorch_core::data::shuffle_indices;
use rusttorch_core::nn::*;
use rusttorch_core::tensor::Tensor;

// ---- Sample text (used when no file is provided) ----
const SAMPLE_TEXT: &str = "\
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry,
And lose the name of action. Soft you now!
The fair Ophelia! Nymph, in thy orisons
Be all my sins remember'd.
";

// ---- Character tokenizer ----

struct CharTokenizer {
    char_to_idx: std::collections::HashMap<char, usize>,
    idx_to_char: Vec<char>,
    vocab_size: usize,
}

impl CharTokenizer {
    fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>().into_iter().collect();
        chars.sort();
        let char_to_idx: std::collections::HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let vocab_size = chars.len();
        CharTokenizer {
            char_to_idx,
            idx_to_char: chars,
            vocab_size,
        }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| self.char_to_idx[&c]).collect()
    }

    fn decode(&self, indices: &[usize]) -> String {
        indices.iter().map(|&i| self.idx_to_char[i]).collect()
    }
}

// ---- GPT model ----

struct CharGPT {
    token_emb: Embedding,
    pos_emb: Embedding,
    transformer: TransformerEncoder,
    head: Linear,
    context_len: usize,
    d_model: usize,
}

impl CharGPT {
    fn new(vocab_size: usize, d_model: usize, num_heads: usize, d_ff: usize, num_layers: usize, context_len: usize) -> Self {
        CharGPT {
            token_emb: Embedding::new(vocab_size, d_model),
            pos_emb: Embedding::new(context_len, d_model),
            transformer: TransformerEncoder::new(d_model, num_heads, d_ff, num_layers),
            head: Linear::new(d_model, vocab_size),
            context_len,
            d_model,
        }
    }

    /// Forward pass: token indices → logits.
    /// tokens: [batch, seq_len] as Vec<Vec<usize>>
    /// Returns: logits [batch * seq_len, vocab_size]
    fn forward(&self, tokens: &[Vec<usize>]) -> rusttorch_core::error::Result<Variable> {
        let batch = tokens.len();
        let seq_len = tokens[0].len();

        // Token embeddings: [batch, seq_len, d_model]
        let tok_emb = self.token_emb.forward_2d(tokens)?;

        // Position indices: [0, 1, 2, ..., seq_len-1] for each batch element
        let pos_indices: Vec<Vec<usize>> = (0..batch)
            .map(|_| (0..seq_len).collect())
            .collect();
        let pos_emb = self.pos_emb.forward_2d(&pos_indices)?;

        // Combined embedding: tok + pos
        let x = tok_emb.add(&pos_emb)?;

        // Transformer with causal masking (GPT-style decoder)
        let x = self.transformer.forward_causal(&x)?;

        // Project to vocab: [batch * seq_len, d_model] → [batch * seq_len, vocab_size]
        let x_flat = x.reshape(&[batch * seq_len, self.d_model])?;
        self.head.forward(&x_flat)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.token_emb.parameters());
        params.extend(self.pos_emb.parameters());
        params.extend(self.transformer.parameters());
        params.extend(self.head.parameters());
        params
    }

    fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .map(|p| p.shape().iter().product::<usize>())
            .sum()
    }

    /// Generate text autoregressively.
    fn generate(&self, tokenizer: &CharTokenizer, prompt: &str, max_tokens: usize, temperature: f32) -> String {
        let mut tokens = tokenizer.encode(prompt);

        for _ in 0..max_tokens {
            // Take the last context_len tokens
            let start = if tokens.len() > self.context_len {
                tokens.len() - self.context_len
            } else {
                0
            };
            let context = tokens[start..].to_vec();
            let batch = vec![context];

            // Forward pass (no gradient needed)
            let logits = self.forward(&batch).expect("forward failed");
            let logits_data = logits.tensor().to_vec_f32();
            let vocab_size = tokenizer.vocab_size;

            // Get logits for the last position
            let seq_len = batch[0].len();
            let last_logits_start = (seq_len - 1) * vocab_size;
            let last_logits = &logits_data[last_logits_start..last_logits_start + vocab_size];

            // Temperature-scaled softmax sampling
            let next_token = sample_from_logits(last_logits, temperature);
            tokens.push(next_token);
        }

        tokenizer.decode(&tokens)
    }
}

/// Sample a token index from logits with temperature.
fn sample_from_logits(logits: &[f32], temperature: f32) -> usize {
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // Softmax
    let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|&x| x / sum).collect();

    // Weighted random sampling
    let r: f32 = rand::random();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Load text
    let text = if args.len() > 1 && !args[1].is_empty() && !args[1].starts_with('-') {
        std::fs::read_to_string(&args[1]).expect("Failed to read text file")
    } else {
        SAMPLE_TEXT.to_string()
    };

    let num_epochs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let context_len: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);

    println!("=== Character-level GPT (RustTorch) ===\n");
    println!("Text length: {} chars", text.len());

    // Tokenize
    let tokenizer = CharTokenizer::from_text(&text);
    let tokens = tokenizer.encode(&text);
    println!("Vocabulary: {} unique characters", tokenizer.vocab_size);
    println!("Tokens: {}", tokens.len());
    println!("Context length: {}", context_len);

    // Hyperparameters — small model for CPU training
    let d_model = 64;
    let num_heads = 4;
    let d_ff = 128;
    let num_layers = 2;
    let batch_size = 16;
    let lr = 3e-4;

    // Build model
    let model = CharGPT::new(
        tokenizer.vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        context_len,
    );
    let params = model.parameters();
    let num_params = model.num_parameters();
    println!(
        "Model: {} layers, d_model={}, heads={}, d_ff={} ({} parameters)\n",
        num_layers, d_model, num_heads, d_ff, num_params
    );

    // Optimizer
    let mut optimizer = Adam::new(params, lr);

    // Create training sequences: sliding window over the token stream
    let num_sequences = if tokens.len() > context_len + 1 {
        tokens.len() - context_len
    } else {
        1
    };
    println!("Training sequences: {}", num_sequences);
    println!("Batch size: {}", batch_size);
    println!("Epochs: {}\n", num_epochs);

    let loss_fn = CrossEntropyLoss::new();

    // Generate before training
    println!("--- Before training ---");
    let sample = model.generate(&tokenizer, "To ", 100, 1.0);
    println!("{}\n", sample);

    // Training loop
    for epoch in 0..num_epochs {
        let indices = shuffle_indices(num_sequences);
        let num_batches = num_sequences / batch_size;
        let mut total_loss = 0.0f32;
        let mut batch_count = 0;

        for batch_idx in 0..num_batches {
            // Build batch: input = tokens[i..i+context_len], target = tokens[i+1..i+context_len+1]
            let mut input_batch: Vec<Vec<usize>> = Vec::with_capacity(batch_size);
            let mut target_batch: Vec<Vec<usize>> = Vec::with_capacity(batch_size);

            for b in 0..batch_size {
                let seq_start = indices[batch_idx * batch_size + b];
                let input_seq: Vec<usize> = tokens[seq_start..seq_start + context_len].to_vec();
                let target_seq: Vec<usize> = tokens[seq_start + 1..seq_start + context_len + 1].to_vec();
                input_batch.push(input_seq);
                target_batch.push(target_seq);
            }

            // Forward: logits [batch * context_len, vocab_size]
            let logits = model.forward(&input_batch).expect("forward failed");

            // Target: one-hot [batch * context_len, vocab_size]
            let flat_targets: Vec<usize> = target_batch.into_iter().flatten().collect();
            let num_targets = flat_targets.len();
            let mut target_one_hot = vec![0.0f32; num_targets * tokenizer.vocab_size];
            for (i, &t) in flat_targets.iter().enumerate() {
                target_one_hot[i * tokenizer.vocab_size + t] = 1.0;
            }
            let target_var = Variable::new(
                Tensor::from_vec(target_one_hot, &[num_targets, tokenizer.vocab_size]),
                false,
            );

            // Loss
            let loss = loss_fn.forward(&logits, &target_var).expect("loss failed");
            let loss_val = loss.tensor().to_vec_f32()[0];
            total_loss += loss_val;
            batch_count += 1;

            // Backward
            optimizer.zero_grad();
            loss.backward().expect("backward failed");
            clip_grad_norm(&model.parameters(), 1.0);
            optimizer.step().expect("optimizer step failed");

            if (batch_idx + 1) % 20 == 0 || batch_idx == 0 {
                println!(
                    "  Epoch {}/{} [batch {}/{}] loss={:.4}",
                    epoch + 1,
                    num_epochs,
                    batch_idx + 1,
                    num_batches,
                    loss_val
                );
            }
        }

        let avg_loss = if batch_count > 0 {
            total_loss / batch_count as f32
        } else {
            0.0
        };
        println!(
            "Epoch {}/{} complete — avg loss: {:.4}",
            epoch + 1,
            num_epochs,
            avg_loss
        );

        // Generate sample every 5 epochs
        if (epoch + 1) % 5 == 0 || epoch == 0 {
            println!("\n--- Sample (epoch {}) ---", epoch + 1);
            let sample = model.generate(&tokenizer, "To ", 150, 0.8);
            println!("{}\n", sample);
        }
    }

    // Final generation
    println!("=== Final generation (temperature=0.8) ===\n");
    for prompt in &["To be", "The ", "And ", "For "] {
        println!("Prompt: \"{}\"", prompt);
        let generated = model.generate(&tokenizer, prompt, 200, 0.8);
        println!("{}\n", generated);
    }

    println!("Done. {} parameters trained over {} epochs.", num_params, num_epochs);
}
