//! Benchmark: KV-cached vs uncached GPT-2 generation.
//!
//! Run with:
//!   cargo run -p rusttorch-core --example gpt2_bench --features datasets --release

use rusttorch_core::data::tokenizer::BpeTokenizer;
use rusttorch_core::nn::gpt2::{Gpt2Config, Gpt2Model};
use std::path::PathBuf;
use std::time::Instant;

fn cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache").join("rusttorch").join("gpt2")
}

fn main() {
    let dir = cache_dir();

    eprintln!("Loading tokenizer...");
    let tokenizer = BpeTokenizer::from_files(
        dir.join("vocab.json"),
        dir.join("merges.txt"),
    ).expect("Failed to load tokenizer");

    eprintln!("Loading model...");
    let config = Gpt2Config::gpt2_small();
    let model = Gpt2Model::from_safetensors(
        dir.join("model.safetensors"),
        config,
    ).expect("Failed to load model");

    let prompt = "The meaning of life is";
    let prompt_ids = tokenizer.encode(prompt);
    let max_tokens = 30;

    println!("=== GPT-2 Generation Benchmark ===");
    println!("Prompt: \"{}\" ({} tokens)", prompt, prompt_ids.len());
    println!("Generating {} tokens (greedy)\n", max_tokens);

    // Benchmark cached generation
    let t0 = Instant::now();
    let cached_ids = model.generate_cached(&prompt_ids, max_tokens, 0.0)
        .expect("Cached generation failed");
    let cached_elapsed = t0.elapsed();
    let cached_text = tokenizer.decode(&cached_ids);
    let cached_tps = max_tokens as f64 / cached_elapsed.as_secs_f64();

    println!("--- KV Cached ---");
    println!("{}", cached_text);
    println!("{:.1}s ({:.1} tokens/sec)\n", cached_elapsed.as_secs_f64(), cached_tps);

    // Benchmark uncached generation
    let t0 = Instant::now();
    let uncached_ids = model.generate(&prompt_ids, max_tokens, 0.0)
        .expect("Uncached generation failed");
    let uncached_elapsed = t0.elapsed();
    let uncached_text = tokenizer.decode(&uncached_ids);
    let uncached_tps = max_tokens as f64 / uncached_elapsed.as_secs_f64();

    println!("--- No Cache ---");
    println!("{}", uncached_text);
    println!("{:.1}s ({:.1} tokens/sec)\n", uncached_elapsed.as_secs_f64(), uncached_tps);

    // Compare
    let speedup = uncached_elapsed.as_secs_f64() / cached_elapsed.as_secs_f64();
    println!("=== Results ===");
    println!("Cached:   {:.1}s ({:.1} tok/s)", cached_elapsed.as_secs_f64(), cached_tps);
    println!("Uncached: {:.1}s ({:.1} tok/s)", uncached_elapsed.as_secs_f64(), uncached_tps);
    println!("Speedup:  {:.1}x", speedup);
    println!("Output match: {}", if cached_ids == uncached_ids { "YES (identical)" } else { "NO (different!)" });
}
