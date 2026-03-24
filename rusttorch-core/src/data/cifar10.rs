//! CIFAR-10 dataset loader.
//!
//! Downloads (if needed) and loads the CIFAR-10 image classification dataset.
//! 60,000 32x32 color images in 10 classes: airplane, automobile, bird, cat, deer,
//! dog, frog, horse, ship, truck.
//!
//! # Usage
//!
//! ```ignore
//! use rusttorch_core::data::cifar10::Cifar10Dataset;
//!
//! let dataset = Cifar10Dataset::load("./data/cifar10")?;
//! println!("Train: {} images, Test: {} images", dataset.train_len(), dataset.test_len());
//!
//! let (images, labels) = dataset.train_batch(0, 64);
//! // images shape: [64, 3, 32, 32]
//! ```
//!
//! Requires the `datasets` feature: `cargo build --features datasets`

use crate::tensor::Tensor;
use std::fs;
use std::path::Path;

#[cfg(feature = "datasets")]
use std::io::Read;

#[cfg(feature = "datasets")]
const CIFAR10_URL: &str = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

const IMAGE_SIZE: usize = 32;
const CHANNELS: usize = 3;
const PIXELS_PER_IMAGE: usize = CHANNELS * IMAGE_SIZE * IMAGE_SIZE; // 3072
const RECORD_SIZE: usize = 1 + PIXELS_PER_IMAGE; // 1 byte label + 3072 bytes image
const IMAGES_PER_BATCH: usize = 10000;
const NUM_TRAIN_BATCHES: usize = 5;
const NUM_CLASSES: usize = 10;

/// Class names for CIFAR-10.
pub const CLASS_NAMES: [&str; NUM_CLASSES] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];

/// The CIFAR-10 dataset: 50,000 training + 10,000 test 32x32 color images in 10 classes.
pub struct Cifar10Dataset {
    /// Training images as f32 [N, 3, 32, 32], normalized to [0, 1]
    pub train_images: Vec<f32>,
    /// Training labels as u8 [0-9]
    pub train_labels: Vec<u8>,
    /// Test images as f32 [N, 3, 32, 32], normalized to [0, 1]
    pub test_images: Vec<f32>,
    /// Test labels as u8 [0-9]
    pub test_labels: Vec<u8>,
    /// Number of training samples (50,000)
    pub train_len: usize,
    /// Number of test samples (10,000)
    pub test_len: usize,
}

impl Cifar10Dataset {
    /// Load CIFAR-10 from the given directory. Downloads and extracts if files are not present.
    #[cfg(feature = "datasets")]
    pub fn load<P: AsRef<Path>>(data_dir: P) -> Result<Self, String> {
        let dir = data_dir.as_ref();
        fs::create_dir_all(dir).map_err(|e| format!("Failed to create data directory: {}", e))?;

        // Check if binary batch files exist
        let first_batch = dir.join("data_batch_1.bin");
        if !first_batch.exists() {
            let tar_gz_path = dir.join("cifar-10-binary.tar.gz");
            if !tar_gz_path.exists() {
                download_cifar10(&tar_gz_path)?;
            }
            extract_tar_gz(&tar_gz_path, dir)?;
        }

        Self::load_from_files(dir)
    }

    /// Load CIFAR-10 from pre-downloaded binary batch files.
    ///
    /// Expects these files in the directory:
    /// - data_batch_1.bin through data_batch_5.bin (training)
    /// - test_batch.bin (test)
    pub fn load_from_files<P: AsRef<Path>>(dir: P) -> Result<Self, String> {
        let dir = dir.as_ref();

        // Load training batches
        let mut train_images =
            Vec::with_capacity(NUM_TRAIN_BATCHES * IMAGES_PER_BATCH * PIXELS_PER_IMAGE);
        let mut train_labels = Vec::with_capacity(NUM_TRAIN_BATCHES * IMAGES_PER_BATCH);

        for i in 1..=NUM_TRAIN_BATCHES {
            let path = dir.join(format!("data_batch_{}.bin", i));
            let (images, labels) = parse_cifar10_batch(&path)?;
            train_images.extend_from_slice(&images);
            train_labels.extend_from_slice(&labels);
        }

        // Load test batch
        let test_path = dir.join("test_batch.bin");
        let (test_images, test_labels) = parse_cifar10_batch(&test_path)?;

        let train_len = train_labels.len();
        let test_len = test_labels.len();

        Ok(Cifar10Dataset {
            train_images,
            train_labels,
            test_images,
            test_labels,
            train_len,
            test_len,
        })
    }

    /// Get a batch of training images as a Tensor [batch_size, 3, 32, 32].
    pub fn train_batch(&self, start: usize, batch_size: usize) -> (Tensor, Vec<u8>) {
        self.get_batch(
            &self.train_images,
            &self.train_labels,
            self.train_len,
            start,
            batch_size,
        )
    }

    /// Get a batch of test images as a Tensor [batch_size, 3, 32, 32].
    pub fn test_batch(&self, start: usize, batch_size: usize) -> (Tensor, Vec<u8>) {
        self.get_batch(
            &self.test_images,
            &self.test_labels,
            self.test_len,
            start,
            batch_size,
        )
    }

    /// Get training images/labels by arbitrary indices (for shuffled batching).
    pub fn train_batch_by_indices(&self, indices: &[usize]) -> (Tensor, Vec<u8>) {
        self.gather(&self.train_images, &self.train_labels, indices)
    }

    /// Get test images/labels by arbitrary indices.
    pub fn test_batch_by_indices(&self, indices: &[usize]) -> (Tensor, Vec<u8>) {
        self.gather(&self.test_images, &self.test_labels, indices)
    }

    /// Convert a batch of labels to one-hot encoded Tensor [batch_size, num_classes].
    pub fn labels_to_one_hot(labels: &[u8], num_classes: usize) -> Tensor {
        let batch_size = labels.len();
        let mut data = vec![0.0f32; batch_size * num_classes];
        for (i, &label) in labels.iter().enumerate() {
            data[i * num_classes + label as usize] = 1.0;
        }
        Tensor::from_vec(data, &[batch_size, num_classes])
    }

    /// Apply per-channel normalization: `(pixel - mean) / std` for each channel.
    ///
    /// Standard CIFAR-10 normalization values:
    /// - R: mean=0.4914, std=0.2470
    /// - G: mean=0.4822, std=0.2435
    /// - B: mean=0.4465, std=0.2616
    pub fn normalize_batch(images: &Tensor) -> Tensor {
        const MEANS: [f32; 3] = [0.4914, 0.4822, 0.4465];
        const STDS: [f32; 3] = [0.2470, 0.2435, 0.2616];

        let shape = images.shape().to_vec();
        let data = images.to_vec_f32();
        let batch_size = shape[0];
        let spatial = IMAGE_SIZE * IMAGE_SIZE; // 1024

        let mut normalized = vec![0.0f32; data.len()];
        for b in 0..batch_size {
            for c in 0..CHANNELS {
                let offset = b * PIXELS_PER_IMAGE + c * spatial;
                for p in 0..spatial {
                    normalized[offset + p] = (data[offset + p] - MEANS[c]) / STDS[c];
                }
            }
        }

        Tensor::from_vec(normalized, &shape)
    }

    /// Get number of training samples.
    pub fn train_len(&self) -> usize {
        self.train_len
    }

    /// Get number of test samples.
    pub fn test_len(&self) -> usize {
        self.test_len
    }

    fn gather(&self, images: &[f32], labels: &[u8], indices: &[usize]) -> (Tensor, Vec<u8>) {
        let batch_size = indices.len();
        let mut batch_data = Vec::with_capacity(batch_size * PIXELS_PER_IMAGE);
        let mut batch_labels = Vec::with_capacity(batch_size);

        for &idx in indices {
            let start = idx * PIXELS_PER_IMAGE;
            batch_data.extend_from_slice(&images[start..start + PIXELS_PER_IMAGE]);
            batch_labels.push(labels[idx]);
        }

        let tensor = Tensor::from_vec(batch_data, &[batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]);
        (tensor, batch_labels)
    }

    fn get_batch(
        &self,
        images: &[f32],
        labels: &[u8],
        total: usize,
        start: usize,
        batch_size: usize,
    ) -> (Tensor, Vec<u8>) {
        let end = (start + batch_size).min(total);
        let actual_batch = end - start;

        let img_start = start * PIXELS_PER_IMAGE;
        let img_end = end * PIXELS_PER_IMAGE;
        let batch_data = images[img_start..img_end].to_vec();
        let batch_labels = labels[start..end].to_vec();

        let tensor = Tensor::from_vec(
            batch_data,
            &[actual_batch, CHANNELS, IMAGE_SIZE, IMAGE_SIZE],
        );
        (tensor, batch_labels)
    }
}

/// Parse a single CIFAR-10 binary batch file.
///
/// Format: 10000 records of (1 byte label + 3072 bytes image).
/// Image bytes are in CHW order: 1024 red, 1024 green, 1024 blue.
fn parse_cifar10_batch<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, Vec<u8>), String> {
    let path = path.as_ref();
    let raw = fs::read(path).map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let num_images = raw.len() / RECORD_SIZE;
    if raw.len() % RECORD_SIZE != 0 {
        return Err(format!(
            "Invalid CIFAR-10 batch file {}: size {} is not a multiple of {} (record size)",
            path.display(),
            raw.len(),
            RECORD_SIZE
        ));
    }

    let mut images = Vec::with_capacity(num_images * PIXELS_PER_IMAGE);
    let mut labels = Vec::with_capacity(num_images);

    for i in 0..num_images {
        let offset = i * RECORD_SIZE;
        let label = raw[offset];
        if label >= NUM_CLASSES as u8 {
            return Err(format!(
                "Invalid label {} at record {} in {}",
                label,
                i,
                path.display()
            ));
        }
        labels.push(label);

        // Convert u8 [0, 255] to f32 [0.0, 1.0]
        for j in 0..PIXELS_PER_IMAGE {
            images.push(raw[offset + 1 + j] as f32 / 255.0);
        }
    }

    Ok((images, labels))
}

/// Download CIFAR-10 tar.gz.
#[cfg(feature = "datasets")]
fn download_cifar10(dest: &Path) -> Result<(), String> {
    eprintln!("Downloading CIFAR-10 from {}...", CIFAR10_URL);

    let resp = ureq::get(CIFAR10_URL)
        .call()
        .map_err(|e| format!("Failed to download CIFAR-10: {}", e))?;

    let mut bytes = Vec::new();
    resp.into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Failed to read CIFAR-10 response: {}", e))?;

    fs::write(dest, &bytes).map_err(|e| format!("Failed to write {}: {}", dest.display(), e))?;

    eprintln!(
        "Downloaded CIFAR-10 ({:.1} MB)",
        bytes.len() as f64 / 1_048_576.0
    );
    Ok(())
}

/// Extract CIFAR-10 tar.gz and flatten the binary batch files into the target directory.
///
/// The archive contains a `cifar-10-batches-bin/` subdirectory with the batch files.
/// We extract just the .bin files directly into the target directory.
#[cfg(feature = "datasets")]
fn extract_tar_gz(tar_gz_path: &Path, dest_dir: &Path) -> Result<(), String> {
    use flate2::read::GzDecoder;

    eprintln!("Extracting CIFAR-10...");

    let file = fs::File::open(tar_gz_path)
        .map_err(|e| format!("Failed to open {}: {}", tar_gz_path.display(), e))?;

    let gz = GzDecoder::new(file);
    let mut archive = tar::Archive::new(gz);

    for entry in archive
        .entries()
        .map_err(|e| format!("Failed to read tar entries: {}", e))?
    {
        let mut entry = entry.map_err(|e| format!("Failed to read tar entry: {}", e))?;

        let path = entry
            .path()
            .map_err(|e| format!("Failed to get entry path: {}", e))?
            .to_path_buf();

        // Only extract .bin files
        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            if filename.ends_with(".bin") {
                let dest_path = dest_dir.join(filename);
                let mut data = Vec::new();
                entry
                    .read_to_end(&mut data)
                    .map_err(|e| format!("Failed to read {}: {}", filename, e))?;
                fs::write(&dest_path, &data)
                    .map_err(|e| format!("Failed to write {}: {}", dest_path.display(), e))?;
                eprintln!("  Extracted {} ({} bytes)", filename, data.len());
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a synthetic CIFAR-10 batch file with N images.
    fn make_batch(n: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(n * RECORD_SIZE);
        for i in 0..n {
            // Label: cycle through 0-9
            data.push((i % NUM_CLASSES) as u8);
            // Image: fill with a pattern based on index
            for j in 0..PIXELS_PER_IMAGE {
                data.push(((i * 7 + j * 3) % 256) as u8);
            }
        }
        data
    }

    #[test]
    fn test_parse_batch_basic() {
        let batch_data = make_batch(10);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        fs::write(tmp.path(), &batch_data).unwrap();

        let (images, labels) = parse_cifar10_batch(tmp.path()).unwrap();
        assert_eq!(labels.len(), 10);
        assert_eq!(images.len(), 10 * PIXELS_PER_IMAGE);
        // Labels should cycle 0-9
        for i in 0..10 {
            assert_eq!(labels[i], i as u8);
        }
        // First pixel of first image should be (0*7 + 0*3) % 256 / 255.0 = 0.0
        assert!((images[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_batch_normalization_range() {
        let batch_data = make_batch(5);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        fs::write(tmp.path(), &batch_data).unwrap();

        let (images, _) = parse_cifar10_batch(tmp.path()).unwrap();
        for &val in &images {
            assert!(val >= 0.0 && val <= 1.0, "Pixel value {} out of [0,1]", val);
        }
    }

    #[test]
    fn test_parse_batch_invalid_size() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        // Write data that's not a multiple of RECORD_SIZE
        fs::write(tmp.path(), &[0u8; 100]).unwrap();

        let result = parse_cifar10_batch(tmp.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not a multiple"));
    }

    #[test]
    fn test_parse_batch_invalid_label() {
        let mut data = vec![0u8; RECORD_SIZE];
        data[0] = 10; // Invalid label (0-9 are valid)

        let tmp = tempfile::NamedTempFile::new().unwrap();
        fs::write(tmp.path(), &data).unwrap();

        let result = parse_cifar10_batch(tmp.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid label"));
    }

    #[test]
    fn test_dataset_from_files() {
        let tmp_dir = tempfile::tempdir().unwrap();

        // Create 5 training batches + 1 test batch
        for i in 1..=5 {
            let path = tmp_dir.path().join(format!("data_batch_{}.bin", i));
            fs::write(&path, make_batch(100)).unwrap();
        }
        let test_path = tmp_dir.path().join("test_batch.bin");
        fs::write(&test_path, make_batch(50)).unwrap();

        let dataset = Cifar10Dataset::load_from_files(tmp_dir.path()).unwrap();
        assert_eq!(dataset.train_len(), 500);
        assert_eq!(dataset.test_len(), 50);
        assert_eq!(dataset.train_images.len(), 500 * PIXELS_PER_IMAGE);
        assert_eq!(dataset.test_images.len(), 50 * PIXELS_PER_IMAGE);
    }

    #[test]
    fn test_train_batch() {
        let tmp_dir = tempfile::tempdir().unwrap();
        for i in 1..=5 {
            let path = tmp_dir.path().join(format!("data_batch_{}.bin", i));
            fs::write(&path, make_batch(10)).unwrap();
        }
        let test_path = tmp_dir.path().join("test_batch.bin");
        fs::write(&test_path, make_batch(10)).unwrap();

        let dataset = Cifar10Dataset::load_from_files(tmp_dir.path()).unwrap();
        let (batch, labels) = dataset.train_batch(0, 8);
        assert_eq!(batch.shape(), &[8, 3, 32, 32]);
        assert_eq!(labels.len(), 8);
    }

    #[test]
    fn test_test_batch() {
        let tmp_dir = tempfile::tempdir().unwrap();
        for i in 1..=5 {
            let path = tmp_dir.path().join(format!("data_batch_{}.bin", i));
            fs::write(&path, make_batch(10)).unwrap();
        }
        let test_path = tmp_dir.path().join("test_batch.bin");
        fs::write(&test_path, make_batch(20)).unwrap();

        let dataset = Cifar10Dataset::load_from_files(tmp_dir.path()).unwrap();
        let (batch, labels) = dataset.test_batch(5, 10);
        assert_eq!(batch.shape(), &[10, 3, 32, 32]);
        assert_eq!(labels.len(), 10);
    }

    #[test]
    fn test_batch_clamps_to_end() {
        let tmp_dir = tempfile::tempdir().unwrap();
        for i in 1..=5 {
            let path = tmp_dir.path().join(format!("data_batch_{}.bin", i));
            fs::write(&path, make_batch(10)).unwrap();
        }
        let test_path = tmp_dir.path().join("test_batch.bin");
        fs::write(&test_path, make_batch(10)).unwrap();

        let dataset = Cifar10Dataset::load_from_files(tmp_dir.path()).unwrap();
        let (batch, labels) = dataset.test_batch(8, 100);
        assert_eq!(batch.shape(), &[2, 3, 32, 32]);
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn test_batch_by_indices() {
        let tmp_dir = tempfile::tempdir().unwrap();
        for i in 1..=5 {
            let path = tmp_dir.path().join(format!("data_batch_{}.bin", i));
            fs::write(&path, make_batch(10)).unwrap();
        }
        let test_path = tmp_dir.path().join("test_batch.bin");
        fs::write(&test_path, make_batch(10)).unwrap();

        let dataset = Cifar10Dataset::load_from_files(tmp_dir.path()).unwrap();
        let (batch, labels) = dataset.train_batch_by_indices(&[0, 5, 10, 15]);
        assert_eq!(batch.shape(), &[4, 3, 32, 32]);
        assert_eq!(labels.len(), 4);
        // Labels should match the cycled pattern
        assert_eq!(labels[0], 0); // index 0 → label 0
        assert_eq!(labels[1], 5); // index 5 → label 5
    }

    #[test]
    fn test_labels_to_one_hot() {
        let labels = vec![0, 3, 7, 9];
        let one_hot = Cifar10Dataset::labels_to_one_hot(&labels, 10);
        assert_eq!(one_hot.shape(), &[4, 10]);

        let data = one_hot.to_vec_f32();
        assert_eq!(data[0], 1.0); // class 0
        assert_eq!(data[13], 1.0); // class 3
        assert_eq!(data[27], 1.0); // class 7
        assert_eq!(data[39], 1.0); // class 9

        let sum: f32 = data.iter().sum();
        assert_eq!(sum, 4.0); // 4 labels, each with exactly one 1.0
    }

    #[test]
    fn test_normalize_batch() {
        // Create a small batch of known values
        let data = vec![0.5f32; 2 * PIXELS_PER_IMAGE]; // 2 images, all 0.5
        let tensor = Tensor::from_vec(data, &[2, 3, 32, 32]);

        let normalized = Cifar10Dataset::normalize_batch(&tensor);
        assert_eq!(normalized.shape(), &[2, 3, 32, 32]);

        let norm_data = normalized.to_vec_f32();
        // R channel: (0.5 - 0.4914) / 0.2470 ≈ 0.0348
        let expected_r = (0.5 - 0.4914) / 0.2470;
        assert!(
            (norm_data[0] - expected_r).abs() < 1e-4,
            "R norm: expected {}, got {}",
            expected_r,
            norm_data[0]
        );
        // G channel: (0.5 - 0.4822) / 0.2435 ≈ 0.0731
        let expected_g = (0.5 - 0.4822) / 0.2435;
        let g_offset = IMAGE_SIZE * IMAGE_SIZE; // 1024
        assert!(
            (norm_data[g_offset] - expected_g).abs() < 1e-4,
            "G norm: expected {}, got {}",
            expected_g,
            norm_data[g_offset]
        );
    }

    #[test]
    fn test_class_names() {
        assert_eq!(CLASS_NAMES.len(), 10);
        assert_eq!(CLASS_NAMES[0], "airplane");
        assert_eq!(CLASS_NAMES[9], "truck");
    }
}
