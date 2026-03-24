//! CIFAR-10 image classification with ResNet-18.
//!
//! Demonstrates production-grade CNN training:
//! - ResNet-18 (CIFAR variant: 3x3 stem, no maxpool)
//! - SGD with momentum + weight decay
//! - Multi-step learning rate decay
//! - Data augmentation (random crop + horizontal flip)
//! - Per-channel normalization
//! - Model saving
//!
//! Run with: cargo run -p rusttorch-core --example cifar10 --features datasets --release
//!
//! Expected output (~10 epochs with augmentation):
//! ```text
//! Loading CIFAR-10 dataset...
//! Train: 50000 images, Test: 10000 images
//! Model: ResNet-18 (11,173,962 parameters)
//! Epoch 1/10 — lr=0.1000, avg loss: 1.42, test accuracy: 55.30%
//! Epoch 2/10 — lr=0.1000, avg loss: 0.98, test accuracy: 66.20%
//! ...
//! ```

use rusttorch_core::autograd::Variable;
use rusttorch_core::data::cifar10::Cifar10Dataset;
use rusttorch_core::data::shuffle_indices;
use rusttorch_core::data::transforms;
use rusttorch_core::nn::*;

fn main() {
    let data_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./data/cifar10".to_string());
    let num_epochs: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let batch_size: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let lr: f32 = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1);

    println!("Loading CIFAR-10 dataset...");
    let dataset = Cifar10Dataset::load(&data_dir).expect("Failed to load CIFAR-10");
    println!(
        "Train: {} images, Test: {} images (3x32x32, 10 classes)",
        dataset.train_len(),
        dataset.test_len(),
    );

    // Build ResNet-18 (CIFAR-10 variant)
    let model = ResNet::resnet18(10);
    let num_params: usize = model.parameters().iter().map(|p| p.tensor().numel()).sum();
    println!(
        "Model: ResNet-18 ({} parameters)",
        format_number(num_params),
    );

    let loss_fn = CrossEntropyLoss::new();

    // SGD with momentum + weight decay — standard ResNet recipe
    let all_params = model.parameters();
    let mut optimizer = SGD::with_momentum_and_weight_decay(all_params, lr, 0.9, 5e-4);

    // Multi-step LR schedule: decay by 10x at 40% and 70% of training
    let milestone1 = (num_epochs as f32 * 0.4).ceil() as usize;
    let milestone2 = (num_epochs as f32 * 0.7).ceil() as usize;
    let scheduler = MultiStepLR::new(lr, vec![milestone1, milestone2], 0.1);

    println!("Optimizer: SGD(lr={}, momentum=0.9, weight_decay=5e-4)", lr);
    println!(
        "Scheduler: MultiStepLR(milestones=[{}, {}], gamma=0.1)",
        milestone1, milestone2
    );
    println!("Augmentation: RandomCrop(32, pad=4) + RandomHFlip(0.5)");
    println!("Batch size: {}, Epochs: {}", batch_size, num_epochs);
    println!();

    let num_batches = dataset.train_len() / batch_size;
    let mut best_accuracy = 0.0f64;

    for epoch in 0..num_epochs {
        // Update learning rate
        let current_lr = scheduler.lr_at(epoch);
        optimizer.set_lr(current_lr);

        model.train();
        let indices = shuffle_indices(dataset.train_len());
        let mut epoch_loss = 0.0f64;
        let mut batch_count = 0;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let batch_indices = &indices[start..start + batch_size];
            let (images, labels) = dataset.train_batch_by_indices(batch_indices);

            // Data augmentation: random crop with padding + random horizontal flip
            let images = transforms::random_crop(&images, 32, 4);
            let images = transforms::random_horizontal_flip(&images, 0.5);

            // Per-channel normalization
            let images = Cifar10Dataset::normalize_batch(&images);

            // One-hot encode labels
            let target_tensor = Cifar10Dataset::labels_to_one_hot(&labels, 10);

            // Forward pass
            let input = Variable::new(images, false);
            let target = Variable::new(target_tensor, false);

            optimizer.zero_grad();

            let logits = model.forward(&input).unwrap();
            let loss = loss_fn.forward(&logits, &target).unwrap();
            let loss_val = loss.tensor().to_vec_f32()[0];
            epoch_loss += loss_val as f64;
            batch_count += 1;

            // Backward + optimize
            loss.backward().unwrap();
            optimizer.step().unwrap();

            if (batch_idx + 1) % 50 == 0 {
                println!(
                    "  [{}/{}] batch {}/{} loss={:.4}",
                    epoch + 1,
                    num_epochs,
                    batch_idx + 1,
                    num_batches,
                    epoch_loss / batch_count as f64
                );
            }
        }

        let avg_loss = epoch_loss / batch_count as f64;

        // Evaluate on test set (no augmentation, just normalize)
        model.eval();
        let accuracy = evaluate(&dataset, &model);
        if accuracy > best_accuracy {
            best_accuracy = accuracy;
        }
        println!(
            "Epoch {}/{} — lr={:.4}, avg loss: {:.4}, test accuracy: {:.2}% (best: {:.2}%)\n",
            epoch + 1,
            num_epochs,
            current_lr,
            avg_loss,
            accuracy * 100.0,
            best_accuracy * 100.0,
        );
    }

    // Save trained model
    let state_dict = model.state_dict();
    let save_path = format!("{}/resnet18_cifar10.rt", data_dir);
    state_dict.save_file(&save_path).unwrap();
    println!(
        "Model saved to {} ({} entries, {} bytes)",
        save_path,
        state_dict.len(),
        std::fs::metadata(&save_path).map(|m| m.len()).unwrap_or(0),
    );
    println!("Best test accuracy: {:.2}%", best_accuracy * 100.0);
}

fn evaluate(dataset: &Cifar10Dataset, model: &ResNet) -> f64 {
    let eval_batch_size = 100;
    let mut correct = 0usize;
    let mut total = 0usize;

    for start in (0..dataset.test_len()).step_by(eval_batch_size) {
        let (images, labels) = dataset.test_batch(start, eval_batch_size);
        let actual_batch = labels.len();

        // Normalize (no augmentation for eval)
        let images = Cifar10Dataset::normalize_batch(&images);
        let input = Variable::detach(images);

        let logits = model.forward(&input).unwrap();

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

fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
