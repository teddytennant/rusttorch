//! GPT-2 text generation — load pre-trained weights and generate text.
//!
//! Downloads GPT-2 Small (124M params) from HuggingFace, loads weights
//! via safetensors, and generates text autoregressively.
//!
//! Run with:
//!   cargo run -p rusttorch-core --example gpt2 --features datasets --release
//!
//! Optional arguments:
//!   cargo run -p rusttorch-core --example gpt2 --features datasets --release -- "Your prompt" 50 0.8
//!   (prompt, max_tokens, temperature)
//!
//! First run downloads ~550MB of model files to ~/.cache/rusttorch/gpt2/

use rusttorch_core::data::tokenizer::BpeTokenizer;
use rusttorch_core::nn::gpt2::{Gpt2Config, Gpt2Model};
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

const MODEL_DIR: &str = "gpt2";
const BASE_URL: &str = "https://huggingface.co/openai-community/gpt2/resolve/main";

fn cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache").join("rusttorch").join(MODEL_DIR)
}

fn download_file(url: &str, path: &std::path::Path) -> Result<(), String> {
    if path.exists() {
        return Ok(());
    }

    let filename = path.file_name().unwrap().to_str().unwrap();
    eprintln!("Downloading {}...", filename);

    let resp = ureq::get(url)
        .call()
        .map_err(|e| format!("Download failed: {}", e))?;

    let mut body = Vec::new();
    resp.into_reader()
        .read_to_end(&mut body)
        .map_err(|e| format!("Read failed: {}", e))?;

    fs::write(path, &body).map_err(|e| format!("Write failed: {}", e))?;
    eprintln!("  Saved {} ({:.1} MB)", filename, body.len() as f64 / 1_000_000.0);

    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prompt = if args.len() > 1 { &args[1] } else { "The meaning of life is" };
    let max_tokens: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);
    let temperature: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.8);

    println!("=== RustTorch GPT-2 Text Generation ===\n");

    // Download model files
    let dir = cache_dir();
    fs::create_dir_all(&dir).expect("Failed to create cache directory");

    let files = [
        ("model.safetensors", format!("{}/model.safetensors", BASE_URL)),
        ("vocab.json", format!("{}/vocab.json", BASE_URL)),
        ("merges.txt", format!("{}/merges.txt", BASE_URL)),
    ];

    for (name, url) in &files {
        download_file(url, &dir.join(name)).expect("Download failed");
    }

    // Load tokenizer
    eprintln!("Loading tokenizer...");
    let t0 = Instant::now();
    let tokenizer = BpeTokenizer::from_files(
        dir.join("vocab.json"),
        dir.join("merges.txt"),
    ).expect("Failed to load tokenizer");
    eprintln!("  Tokenizer loaded in {:.1}ms (vocab_size={})", t0.elapsed().as_millis(), tokenizer.vocab_size());

    // Load model
    eprintln!("Loading model...");
    let t0 = Instant::now();
    let config = Gpt2Config::gpt2_small();
    let model = Gpt2Model::from_safetensors(
        dir.join("model.safetensors"),
        config,
    ).expect("Failed to load model");
    eprintln!("  Model loaded in {:.1}s ({} parameters)", t0.elapsed().as_secs_f64(), model.num_parameters());

    // Tokenize prompt
    let prompt_ids = tokenizer.encode(prompt);
    eprintln!("  Prompt: \"{}\" ({} tokens)\n", prompt, prompt_ids.len());

    // Generate
    println!("Generating {} tokens (temperature={})...\n", max_tokens, temperature);

    let t0 = Instant::now();
    let output_ids = model.generate(&prompt_ids, max_tokens, temperature)
        .expect("Generation failed");
    let elapsed = t0.elapsed();

    let output_text = tokenizer.decode(&output_ids);
    let new_tokens = output_ids.len() - prompt_ids.len();
    let tokens_per_sec = new_tokens as f64 / elapsed.as_secs_f64();

    println!("{}", output_text);
    println!("\n--- Stats ---");
    println!("Generated {} tokens in {:.1}s ({:.1} tokens/sec)", new_tokens, elapsed.as_secs_f64(), tokens_per_sec);
    println!("Prompt tokens: {}", prompt_ids.len());
    println!("Total tokens: {}", output_ids.len());
}
