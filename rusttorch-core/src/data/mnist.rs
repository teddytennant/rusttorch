//! MNIST dataset loader.
//!
//! Downloads (if needed) and loads the MNIST handwritten digit dataset.
//! Returns tensors ready for training.
//!
//! # Usage
//!
//! ```ignore
//! use rusttorch_core::data::mnist::MnistDataset;
//!
//! let dataset = MnistDataset::load("./data")?;
//! println!("Train: {} images, Test: {} images", dataset.train_len(), dataset.test_len());
//!
//! // Get batches for training
//! let (images, labels) = dataset.train_batch(0, 64);
//! ```
//!
//! Requires the `datasets` feature: `cargo build --features datasets`

use crate::data::idx;
use crate::tensor::Tensor;
use std::fs;
use std::path::Path;

#[cfg(feature = "datasets")]
use std::path::PathBuf;

#[cfg(feature = "datasets")]
const MNIST_BASE_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/";

#[cfg(feature = "datasets")]
const TRAIN_IMAGES: &str = "train-images-idx3-ubyte.gz";
#[cfg(feature = "datasets")]
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte.gz";
#[cfg(feature = "datasets")]
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte.gz";
#[cfg(feature = "datasets")]
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte.gz";

/// The MNIST dataset: 60,000 training + 10,000 test images of handwritten digits.
pub struct MnistDataset {
    /// Training images as f32 [N, 1, 28, 28], normalized to [0, 1]
    pub train_images: Vec<f32>,
    /// Training labels as u8 [0-9]
    pub train_labels: Vec<u8>,
    /// Test images as f32 [N, 1, 28, 28], normalized to [0, 1]
    pub test_images: Vec<f32>,
    /// Test labels as u8 [0-9]
    pub test_labels: Vec<u8>,
    /// Number of training samples
    pub train_len: usize,
    /// Number of test samples
    pub test_len: usize,
    /// Image height
    pub rows: usize,
    /// Image width
    pub cols: usize,
}

impl MnistDataset {
    /// Load MNIST from the given directory. Downloads if files are not present.
    ///
    /// # Arguments
    /// * `data_dir` - Directory to store/read MNIST files
    #[cfg(feature = "datasets")]
    pub fn load<P: AsRef<Path>>(data_dir: P) -> Result<Self, String> {
        let dir = data_dir.as_ref();
        fs::create_dir_all(dir).map_err(|e| format!("Failed to create data directory: {}", e))?;

        // Download if needed
        for filename in &[TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS] {
            let gz_path = dir.join(filename);
            let raw_path = dir.join(filename.trim_end_matches(".gz"));

            if !raw_path.exists() {
                if !gz_path.exists() {
                    download_file(filename, &gz_path)?;
                }
                decompress_gz(&gz_path, &raw_path)?;
            }
        }

        Self::load_from_files(dir)
    }

    /// Load MNIST from pre-downloaded IDX files in the given directory.
    ///
    /// Expects these files (uncompressed):
    /// - train-images-idx3-ubyte
    /// - train-labels-idx1-ubyte
    /// - t10k-images-idx3-ubyte
    /// - t10k-labels-idx1-ubyte
    pub fn load_from_files<P: AsRef<Path>>(dir: P) -> Result<Self, String> {
        let dir = dir.as_ref();

        let train_img_path = dir.join("train-images-idx3-ubyte");
        let train_lbl_path = dir.join("train-labels-idx1-ubyte");
        let test_img_path = dir.join("t10k-images-idx3-ubyte");
        let test_lbl_path = dir.join("t10k-labels-idx1-ubyte");

        // Parse training images
        let mut f = fs::File::open(&train_img_path)
            .map_err(|e| format!("Failed to open {}: {}", train_img_path.display(), e))?;
        let (train_images, train_n, rows, cols) = idx::parse_idx_images(&mut f)?;

        // Parse training labels
        let mut f = fs::File::open(&train_lbl_path)
            .map_err(|e| format!("Failed to open {}: {}", train_lbl_path.display(), e))?;
        let (train_labels, train_lbl_n) = idx::parse_idx_labels(&mut f)?;

        if train_n != train_lbl_n {
            return Err(format!(
                "Training image count ({}) != label count ({})",
                train_n, train_lbl_n
            ));
        }

        // Parse test images
        let mut f = fs::File::open(&test_img_path)
            .map_err(|e| format!("Failed to open {}: {}", test_img_path.display(), e))?;
        let (test_images, test_n, test_rows, test_cols) = idx::parse_idx_images(&mut f)?;

        if rows != test_rows || cols != test_cols {
            return Err(format!(
                "Image dimensions mismatch: train={}x{}, test={}x{}",
                rows, cols, test_rows, test_cols
            ));
        }

        // Parse test labels
        let mut f = fs::File::open(&test_lbl_path)
            .map_err(|e| format!("Failed to open {}: {}", test_lbl_path.display(), e))?;
        let (test_labels, test_lbl_n) = idx::parse_idx_labels(&mut f)?;

        if test_n != test_lbl_n {
            return Err(format!(
                "Test image count ({}) != label count ({})",
                test_n, test_lbl_n
            ));
        }

        Ok(MnistDataset {
            train_images,
            train_labels,
            test_images,
            test_labels,
            train_len: train_n,
            test_len: test_n,
            rows,
            cols,
        })
    }

    /// Get a batch of training images as a Tensor [batch_size, 1, 28, 28].
    pub fn train_batch(&self, start: usize, batch_size: usize) -> (Tensor, Vec<u8>) {
        self.get_batch(
            &self.train_images,
            &self.train_labels,
            self.train_len,
            start,
            batch_size,
        )
    }

    /// Get a batch of test images as a Tensor [batch_size, 1, 28, 28].
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

    fn gather(&self, images: &[f32], labels: &[u8], indices: &[usize]) -> (Tensor, Vec<u8>) {
        let pixels_per_image = self.rows * self.cols;
        let batch_size = indices.len();

        let mut batch_data = Vec::with_capacity(batch_size * pixels_per_image);
        let mut batch_labels = Vec::with_capacity(batch_size);

        for &idx in indices {
            let start = idx * pixels_per_image;
            batch_data.extend_from_slice(&images[start..start + pixels_per_image]);
            batch_labels.push(labels[idx]);
        }

        let tensor = Tensor::from_vec(batch_data, &[batch_size, 1, self.rows, self.cols]);
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
        let pixels_per_image = self.rows * self.cols;

        let img_start = start * pixels_per_image;
        let img_end = end * pixels_per_image;
        let batch_data = images[img_start..img_end].to_vec();

        let batch_labels = labels[start..end].to_vec();

        let tensor = Tensor::from_vec(batch_data, &[actual_batch, 1, self.rows, self.cols]);

        (tensor, batch_labels)
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

    /// Get number of training samples.
    pub fn train_len(&self) -> usize {
        self.train_len
    }

    /// Get number of test samples.
    pub fn test_len(&self) -> usize {
        self.test_len
    }
}

/// Download a file from the MNIST mirror.
#[cfg(feature = "datasets")]
fn download_file(filename: &str, dest: &PathBuf) -> Result<(), String> {
    let url = format!("{}{}", MNIST_BASE_URL, filename);
    eprintln!("Downloading {}...", url);

    let resp = ureq::get(&url)
        .call()
        .map_err(|e| format!("Failed to download {}: {}", url, e))?;

    let mut bytes = Vec::new();
    resp.into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Failed to read response for {}: {}", filename, e))?;

    fs::write(dest, &bytes).map_err(|e| format!("Failed to write {}: {}", dest.display(), e))?;

    eprintln!("Downloaded {} ({} bytes)", filename, bytes.len());
    Ok(())
}

/// Decompress a .gz file.
#[cfg(feature = "datasets")]
fn decompress_gz(gz_path: &PathBuf, out_path: &PathBuf) -> Result<(), String> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let file = fs::File::open(gz_path)
        .map_err(|e| format!("Failed to open {}: {}", gz_path.display(), e))?;

    let mut decoder = GzDecoder::new(file);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .map_err(|e| format!("Failed to decompress {}: {}", gz_path.display(), e))?;

    fs::write(out_path, &decompressed)
        .map_err(|e| format!("Failed to write {}: {}", out_path.display(), e))?;

    eprintln!(
        "Decompressed {} -> {} ({} bytes)",
        gz_path.display(),
        out_path.display(),
        decompressed.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_labels_to_one_hot() {
        let labels = vec![0, 3, 7, 9];
        let one_hot = MnistDataset::labels_to_one_hot(&labels, 10);
        assert_eq!(one_hot.shape(), &[4, 10]);

        let data = one_hot.to_vec_f32();
        // Label 0: [1,0,0,0,0,0,0,0,0,0]
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 0.0);
        // Label 3: [0,0,0,1,0,0,0,0,0,0]
        assert_eq!(data[13], 1.0);
        assert_eq!(data[10], 0.0);
        // Label 7: [0,0,0,0,0,0,0,1,0,0]
        assert_eq!(data[27], 1.0);
        // Label 9: [0,0,0,0,0,0,0,0,0,1]
        assert_eq!(data[39], 1.0);
    }

    #[test]
    fn test_labels_to_one_hot_single() {
        let labels = vec![5];
        let one_hot = MnistDataset::labels_to_one_hot(&labels, 10);
        assert_eq!(one_hot.shape(), &[1, 10]);
        let data = one_hot.to_vec_f32();
        assert_eq!(data[5], 1.0);
        let sum: f32 = data.iter().sum();
        assert_eq!(sum, 1.0);
    }

    #[test]
    fn test_get_batch() {
        // Create a mini dataset manually
        let rows = 2;
        let cols = 2;
        let pixels_per = rows * cols;
        let n = 5;
        let images: Vec<f32> = (0..n * pixels_per).map(|i| i as f32 / 255.0).collect();
        let labels: Vec<u8> = vec![0, 1, 2, 3, 4];

        let ds = MnistDataset {
            train_images: images.clone(),
            train_labels: labels.clone(),
            test_images: images,
            test_labels: labels,
            train_len: n,
            test_len: n,
            rows,
            cols,
        };

        let (batch, batch_labels) = ds.train_batch(1, 3);
        assert_eq!(batch.shape(), &[3, 1, 2, 2]);
        assert_eq!(batch_labels, vec![1, 2, 3]);
    }

    #[test]
    fn test_get_batch_clamps_to_end() {
        let rows = 2;
        let cols = 2;
        let n = 3;
        let images: Vec<f32> = vec![0.0; n * rows * cols];
        let labels: Vec<u8> = vec![0, 1, 2];

        let ds = MnistDataset {
            train_images: images.clone(),
            train_labels: labels.clone(),
            test_images: images,
            test_labels: labels,
            train_len: n,
            test_len: n,
            rows,
            cols,
        };

        // Request batch starting at 2 with size 5 — should clamp to 1 sample
        let (batch, batch_labels) = ds.train_batch(2, 5);
        assert_eq!(batch.shape(), &[1, 1, 2, 2]);
        assert_eq!(batch_labels, vec![2]);
    }
}
