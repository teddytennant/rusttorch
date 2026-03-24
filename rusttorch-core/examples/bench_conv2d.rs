//! Benchmark for Conv2d im2col performance at CIFAR-10 scale.
//!
//! Usage: cargo run -p rusttorch-core --example bench_conv2d --release

use rusttorch_core::autograd::variable::Variable;
use rusttorch_core::nn::conv2d::Conv2d;
use rusttorch_core::nn::module::Module;
use rusttorch_core::tensor::Tensor;
use std::time::Instant;

fn bench_forward(label: &str, conv: &Conv2d, input: &Variable, iterations: usize) {
    // Warmup
    for _ in 0..3 {
        let _ = conv.forward(input).unwrap();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = conv.forward(input).unwrap();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iterations as u32;
    println!(
        "{}: {:.2}ms avg ({} iters, {:.2}ms total)",
        label,
        per_iter.as_secs_f64() * 1000.0,
        iterations,
        elapsed.as_secs_f64() * 1000.0,
    );
}

fn bench_forward_backward(label: &str, conv: &Conv2d, input: &Variable, iterations: usize) {
    // Warmup
    for _ in 0..3 {
        let out = conv.forward(input).unwrap();
        let loss = out.sum().unwrap();
        loss.backward();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let out = conv.forward(input).unwrap();
        let loss = out.sum().unwrap();
        loss.backward();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iterations as u32;
    println!(
        "{}: {:.2}ms avg ({} iters, {:.2}ms total)",
        label,
        per_iter.as_secs_f64() * 1000.0,
        iterations,
        elapsed.as_secs_f64() * 1000.0,
    );
}

fn main() {
    println!("=== Conv2d im2col Benchmark ===\n");

    // CIFAR-10 scale: 32x32 images
    println!("--- CIFAR-10 scale (32x32) ---");

    // First conv layer: 3 → 64 channels, 3x3 kernel
    let conv1 = Conv2d::with_options(3, 64, 3, 1, 1);
    let input_b1 = Variable::new(Tensor::from_vec(vec![0.5f32; 1 * 3 * 32 * 32], &[1, 3, 32, 32]), true);
    let input_b8 = Variable::new(Tensor::from_vec(vec![0.5f32; 8 * 3 * 32 * 32], &[8, 3, 32, 32]), true);
    let input_b32 = Variable::new(Tensor::from_vec(vec![0.5f32; 32 * 3 * 32 * 32], &[32, 3, 32, 32]), true);

    bench_forward("Conv2d(3→64, 3x3) batch=1  fwd", &conv1, &input_b1, 50);
    bench_forward("Conv2d(3→64, 3x3) batch=8  fwd", &conv1, &input_b8, 20);
    bench_forward("Conv2d(3→64, 3x3) batch=32 fwd", &conv1, &input_b32, 10);

    println!();
    bench_forward_backward("Conv2d(3→64, 3x3) batch=1  fwd+bwd", &conv1, &input_b1, 30);
    bench_forward_backward("Conv2d(3→64, 3x3) batch=8  fwd+bwd", &conv1, &input_b8, 10);

    // Deeper layer: 64 → 128 channels, 3x3 kernel, 16x16 spatial
    println!("\n--- Deeper layer (16x16) ---");
    let conv2 = Conv2d::with_options(64, 128, 3, 1, 1);
    let input_deep_b1 = Variable::new(Tensor::from_vec(vec![0.5f32; 1 * 64 * 16 * 16], &[1, 64, 16, 16]), true);
    let input_deep_b8 = Variable::new(Tensor::from_vec(vec![0.5f32; 8 * 64 * 16 * 16], &[8, 64, 16, 16]), true);

    bench_forward("Conv2d(64→128, 3x3) batch=1 fwd", &conv2, &input_deep_b1, 20);
    bench_forward("Conv2d(64→128, 3x3) batch=8 fwd", &conv2, &input_deep_b8, 5);

    println!();
    bench_forward_backward("Conv2d(64→128, 3x3) batch=1 fwd+bwd", &conv2, &input_deep_b1, 10);

    // MNIST scale: 28x28, 1 channel
    println!("\n--- MNIST scale (28x28) ---");
    let conv_mnist = Conv2d::new(1, 8, 5);
    let mnist_b32 = Variable::new(Tensor::from_vec(vec![0.5f32; 32 * 1 * 28 * 28], &[32, 1, 28, 28]), true);
    bench_forward("Conv2d(1→8, 5x5) batch=32 fwd", &conv_mnist, &mnist_b32, 50);
    bench_forward_backward("Conv2d(1→8, 5x5) batch=32 fwd+bwd", &conv_mnist, &mnist_b32, 20);

    println!("\nDone.");
}
