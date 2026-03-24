//! MNIST handwritten digit classification with a CNN.
//!
//! Demonstrates the full RustTorch training pipeline:
//! - Loading MNIST dataset (auto-download)
//! - Building a convolutional neural network
//! - Training with cross-entropy loss and Adam optimizer
//! - Evaluating accuracy on the test set
//!
//! Run with: cargo run --example mnist --features datasets --release
//!
//! Expected output (3 epochs, ~95%+ test accuracy):
//! ```text
//! Loading MNIST dataset...
//! Train: 60000 images, Test: 10000 images (28x28)
//! Architecture: Conv(1→8,5) → ReLU → Pool(2) → Conv(8→16,5) → ReLU → Pool(2) → FC(256→10)
//! Epoch 1/3 [batch 100/937] loss=0.452
//! ...
//! Epoch 1/3 complete — avg loss: 0.312
//! Test accuracy: 96.42% (9642/10000)
//! ```

use rusttorch_core::autograd::Variable;
use rusttorch_core::data::mnist::MnistDataset;
use rusttorch_core::data::shuffle_indices;
use rusttorch_core::nn::*;

fn main() {
    let data_dir = std::env::args().nth(1).unwrap_or_else(|| "./data/mnist".to_string());
    let num_epochs: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let batch_size: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    println!("Loading MNIST dataset...");
    let dataset = MnistDataset::load(&data_dir).expect("Failed to load MNIST");
    println!(
        "Train: {} images, Test: {} images ({}x{})",
        dataset.train_len(),
        dataset.test_len(),
        dataset.rows,
        dataset.cols
    );

    // Build CNN
    // Conv2d(1, 8, 5) → ReLU → MaxPool(2) → Conv2d(8, 16, 5) → ReLU → MaxPool(2) → Flatten → Linear(256, 10)
    //
    // Shape progression:
    // [B, 1, 28, 28] → Conv(1→8, k=5) → [B, 8, 24, 24] → Pool(2) → [B, 8, 12, 12]
    //                → Conv(8→16, k=5) → [B, 16, 8, 8]  → Pool(2) → [B, 16, 4, 4]
    //                → Flatten → [B, 256] → Linear → [B, 10]
    let conv1 = Conv2d::new(1, 8, 5);
    let pool1 = MaxPool2d::new(2);
    let conv2 = Conv2d::new(8, 16, 5);
    let pool2 = MaxPool2d::new(2);
    let flatten = Flatten::new();
    let fc = Linear::new(256, 10);

    let loss_fn = CrossEntropyLoss::new();

    // Collect all parameters
    let mut all_params = Vec::new();
    all_params.extend(conv1.parameters());
    all_params.extend(conv2.parameters());
    all_params.extend(fc.parameters());
    let mut optimizer = Adam::new(all_params, 0.001);

    println!(
        "Architecture: Conv(1->8,5) -> ReLU -> Pool(2) -> Conv(8->16,5) -> ReLU -> Pool(2) -> FC(256->10)"
    );
    println!(
        "Optimizer: Adam(lr=0.001), Batch size: {}, Epochs: {}",
        batch_size, num_epochs
    );
    println!();

    let num_batches = dataset.train_len() / batch_size;

    for epoch in 0..num_epochs {
        let indices = shuffle_indices(dataset.train_len());
        let mut epoch_loss = 0.0f32;
        let mut batch_count = 0;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let batch_indices = &indices[start..start + batch_size];
            let (images, labels) = dataset.train_batch_by_indices(batch_indices);

            // One-hot encode labels
            let target_tensor = MnistDataset::labels_to_one_hot(&labels, 10);

            // Forward pass
            let input = Variable::new(images, false);
            let target = Variable::new(target_tensor, false);

            optimizer.zero_grad();

            let x = conv1.forward(&input).unwrap();
            let x = x.relu();
            let x = pool1.forward(&x).unwrap();
            let x = conv2.forward(&x).unwrap();
            let x = x.relu();
            let x = pool2.forward(&x).unwrap();
            let x = flatten.forward(&x).unwrap();
            let logits = fc.forward(&x).unwrap();

            let loss = loss_fn.forward(&logits, &target).unwrap();
            let loss_val = loss.tensor().to_vec_f32()[0];
            epoch_loss += loss_val;
            batch_count += 1;

            // Backward + optimize
            loss.backward().unwrap();
            optimizer.step().unwrap();

            if (batch_idx + 1) % 100 == 0 {
                println!(
                    "  Epoch {}/{} [batch {}/{}] loss={:.4}",
                    epoch + 1,
                    num_epochs,
                    batch_idx + 1,
                    num_batches,
                    epoch_loss / batch_count as f32
                );
            }
        }

        let avg_loss = epoch_loss / batch_count as f32;
        println!(
            "Epoch {}/{} complete - avg loss: {:.4}",
            epoch + 1,
            num_epochs,
            avg_loss
        );

        // Evaluate on test set
        let accuracy = evaluate(&dataset, &conv1, &pool1, &conv2, &pool2, &flatten, &fc);
        println!(
            "Test accuracy: {:.2}% ({}/{})\n",
            accuracy * 100.0,
            (accuracy * dataset.test_len() as f64).round() as usize,
            dataset.test_len()
        );
    }
}

fn evaluate(
    dataset: &MnistDataset,
    conv1: &Conv2d,
    pool1: &MaxPool2d,
    conv2: &Conv2d,
    pool2: &MaxPool2d,
    flatten: &Flatten,
    fc: &Linear,
) -> f64 {
    let eval_batch_size = 100;
    let mut correct = 0usize;
    let mut total = 0usize;

    for start in (0..dataset.test_len()).step_by(eval_batch_size) {
        let (images, labels) = dataset.test_batch(start, eval_batch_size);
        let actual_batch = labels.len();

        let input = Variable::detach(images);

        let x = conv1.forward(&input).unwrap();
        let x = x.relu();
        let x = pool1.forward(&x).unwrap();
        let x = conv2.forward(&x).unwrap();
        let x = x.relu();
        let x = pool2.forward(&x).unwrap();
        let x = flatten.forward(&x).unwrap();
        let logits = fc.forward(&x).unwrap();

        // Get predictions (argmax)
        let logit_data = logits.tensor().to_vec_f32();
        let num_classes = 10;
        for i in 0..actual_batch {
            let offset = i * num_classes;
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for c in 0..num_classes {
                if logit_data[offset + c] > max_val {
                    max_val = logit_data[offset + c];
                    max_idx = c;
                }
            }
            if max_idx == labels[i] as usize {
                correct += 1;
            }
            total += 1;
        }
    }

    correct as f64 / total as f64
}
