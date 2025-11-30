//! Data loading and preprocessing utilities
//!
//! This module provides utilities for loading and preprocessing data:
//! - CSV/TSV file parsing
//! - Batching and shuffling
//! - Data normalization
//! - Train/val/test splitting

use crate::tensor::{Tensor, TensorData};
use crate::DType;
use ndarray::Array;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Load data from a CSV file
///
/// # Arguments
/// * `path` - Path to CSV file
/// * `has_header` - Whether the first row is a header
/// * `delimiter` - Column delimiter (typically ',' or '\t')
///
/// # Returns
/// Tensor containing the data (rows x columns)
pub fn load_csv<P: AsRef<Path>>(
    path: P,
    has_header: bool,
    delimiter: char,
) -> Result<Tensor, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::new(file);

    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut num_cols = 0;

    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("Failed to read line {}: {}", i, e))?;

        // Skip header
        if has_header && i == 0 {
            continue;
        }

        let values: Result<Vec<f32>, _> = line
            .split(delimiter)
            .map(|s| s.trim().parse::<f32>())
            .collect();

        let values = values.map_err(|e| {
            format!("Failed to parse line {} as floats: {}", i, e)
        })?;

        if num_cols == 0 {
            num_cols = values.len();
        } else if values.len() != num_cols {
            return Err(format!(
                "Inconsistent number of columns at line {}: expected {}, got {}",
                i,
                num_cols,
                values.len()
            ));
        }

        rows.push(values);
    }

    if rows.is_empty() {
        return Err("No data rows found in CSV".to_string());
    }

    let num_rows = rows.len();
    let flat_data: Vec<f32> = rows.into_iter().flatten().collect();

    Ok(Tensor::from_vec(flat_data, &[num_rows, num_cols]))
}

/// Normalize tensor data using z-score normalization
///
/// Transforms data to have mean=0 and std=1
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `dim` - Dimension along which to normalize (None for global normalization)
///
/// # Returns
/// (Normalized tensor, mean, std)
pub fn normalize(tensor: &Tensor) -> (Tensor, f64, f64) {
    let mean = crate::ops::mean(tensor);
    let std = compute_std(tensor, mean);

    let normalized = match tensor.data() {
        TensorData::Float32(arr) => {
            let normalized = arr.mapv(|x| ((x as f64 - mean) / std) as f32);
            Tensor::from_data(TensorData::Float32(normalized), DType::Float32)
        }
        TensorData::Float64(arr) => {
            let normalized = arr.mapv(|x| (x - mean) / std);
            Tensor::from_data(TensorData::Float64(normalized), DType::Float64)
        }
        _ => panic!("Normalization only supports floating-point tensors"),
    };

    (normalized, mean, std)
}

/// Compute standard deviation
fn compute_std(tensor: &Tensor, mean: f64) -> f64 {
    let n = tensor.numel() as f64;

    let variance = match tensor.data() {
        TensorData::Float32(arr) => {
            let sum_sq: f32 = arr.iter().map(|&x| (x as f64 - mean).powi(2) as f32).sum();
            sum_sq as f64 / n
        }
        TensorData::Float64(arr) => {
            let sum_sq: f64 = arr.iter().map(|&x| (x - mean).powi(2)).sum();
            sum_sq / n
        }
        _ => panic!("Standard deviation only supports floating-point tensors"),
    };

    variance.sqrt()
}

/// Create batches from a dataset
///
/// # Arguments
/// * `data` - Input data tensor (samples x features)
/// * `batch_size` - Number of samples per batch
/// * `drop_last` - Whether to drop the last incomplete batch
///
/// # Returns
/// Vector of batched tensors
pub fn create_batches(data: &Tensor, batch_size: usize, drop_last: bool) -> Vec<Tensor> {
    let num_samples = data.shape()[0];
    let num_features = data.shape()[1];

    let num_batches = if drop_last {
        num_samples / batch_size
    } else {
        (num_samples + batch_size - 1) / batch_size
    };

    let mut batches = Vec::with_capacity(num_batches);

    match data.data() {
        TensorData::Float32(arr) => {
            for i in 0..num_batches {
                let start = i * batch_size;
                let end = ((i + 1) * batch_size).min(num_samples);
                let actual_batch_size = end - start;

                if drop_last && actual_batch_size < batch_size {
                    break;
                }

                let batch_data: Vec<f32> = (start..end)
                    .flat_map(|row| {
                        (0..num_features).map(move |col| arr[[row, col]])
                    })
                    .collect();

                batches.push(Tensor::from_vec(
                    batch_data,
                    &[actual_batch_size, num_features],
                ));
            }
        }
        TensorData::Float64(arr) => {
            for i in 0..num_batches {
                let start = i * batch_size;
                let end = ((i + 1) * batch_size).min(num_samples);
                let actual_batch_size = end - start;

                if drop_last && actual_batch_size < batch_size {
                    break;
                }

                let batch_data: Vec<f32> = (start..end)
                    .flat_map(|row| {
                        (0..num_features).map(move |col| arr[[row, col]] as f32)
                    })
                    .collect();

                batches.push(Tensor::from_vec(
                    batch_data,
                    &[actual_batch_size, num_features],
                ));
            }
        }
        _ => panic!("Batching only supports floating-point tensors"),
    }

    batches
}

/// Shuffle indices for random batching
///
/// # Arguments
/// * `num_samples` - Total number of samples
///
/// # Returns
/// Vector of shuffled indices
pub fn shuffle_indices(num_samples: usize) -> Vec<usize> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    let mut indices: Vec<usize> = (0..num_samples).collect();
    indices.shuffle(&mut thread_rng());
    indices
}

/// Split data into train/validation/test sets
///
/// # Arguments
/// * `data` - Input data tensor
/// * `train_ratio` - Proportion for training (e.g., 0.7)
/// * `val_ratio` - Proportion for validation (e.g., 0.15)
/// * `shuffle` - Whether to shuffle before splitting
///
/// # Returns
/// (train_data, val_data, test_data)
pub fn train_val_test_split(
    data: &Tensor,
    train_ratio: f64,
    val_ratio: f64,
    shuffle: bool,
) -> (Tensor, Tensor, Tensor) {
    assert!(
        train_ratio > 0.0 && train_ratio < 1.0,
        "train_ratio must be in (0, 1)"
    );
    assert!(
        val_ratio >= 0.0 && val_ratio < 1.0,
        "val_ratio must be in [0, 1)"
    );
    assert!(
        train_ratio + val_ratio < 1.0,
        "train_ratio + val_ratio must be < 1"
    );

    let num_samples = data.shape()[0];
    let num_features = data.shape()[1];

    let num_train = (num_samples as f64 * train_ratio) as usize;
    let num_val = (num_samples as f64 * val_ratio) as usize;
    let num_test = num_samples - num_train - num_val;

    let indices: Vec<usize> = if shuffle {
        shuffle_indices(num_samples)
    } else {
        (0..num_samples).collect()
    };

    let train_indices = &indices[0..num_train];
    let val_indices = &indices[num_train..num_train + num_val];
    let test_indices = &indices[num_train + num_val..];

    let extract_subset = |subset_indices: &[usize]| -> Tensor {
        match data.data() {
            TensorData::Float32(arr) => {
                let subset_data: Vec<f32> = subset_indices
                    .iter()
                    .flat_map(|&idx| (0..num_features).map(move |col| arr[[idx, col]]))
                    .collect();
                Tensor::from_vec(subset_data, &[subset_indices.len(), num_features])
            }
            TensorData::Float64(arr) => {
                let subset_data: Vec<f32> = subset_indices
                    .iter()
                    .flat_map(|&idx| {
                        (0..num_features).map(move |col| arr[[idx, col]] as f32)
                    })
                    .collect();
                Tensor::from_vec(subset_data, &[subset_indices.len(), num_features])
            }
            _ => panic!("Splitting only supports floating-point tensors"),
        }
    };

    (
        extract_subset(train_indices),
        extract_subset(val_indices),
        extract_subset(test_indices),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_csv() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "a,b,c").unwrap();
        writeln!(file, "1.0,2.0,3.0").unwrap();
        writeln!(file, "4.0,5.0,6.0").unwrap();
        file.flush().unwrap();

        let tensor = load_csv(file.path(), true, ',').unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
    }

    #[test]
    fn test_normalize() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5, 1]);
        let (normalized, mean, std) = normalize(&data);

        assert_eq!(normalized.shape(), &[5, 1]);
        assert!((mean - 3.0).abs() < 0.01);
        assert!(std > 0.0);
    }

    #[test]
    fn test_create_batches() {
        let data = Tensor::from_vec((0..20).map(|x| x as f32).collect(), &[10, 2]);
        let batches = create_batches(&data, 3, false);

        assert_eq!(batches.len(), 4); // 3 full + 1 partial
        assert_eq!(batches[0].shape(), &[3, 2]);
        assert_eq!(batches[3].shape(), &[1, 2]); // Last partial batch
    }

    #[test]
    fn test_create_batches_drop_last() {
        let data = Tensor::from_vec((0..20).map(|x| x as f32).collect(), &[10, 2]);
        let batches = create_batches(&data, 3, true);

        assert_eq!(batches.len(), 3); // Drop last incomplete batch
        assert_eq!(batches[0].shape(), &[3, 2]);
    }

    #[test]
    fn test_train_val_test_split() {
        let data = Tensor::from_vec((0..100).map(|x| x as f32).collect(), &[50, 2]);
        let (train, val, test) = train_val_test_split(&data, 0.7, 0.15, false);

        assert_eq!(train.shape()[0], 35); // 70% of 50
        assert_eq!(val.shape()[0], 7); // 15% of 50
        assert_eq!(test.shape()[0], 8); // Remaining
        assert_eq!(train.shape()[1], 2);
    }

    #[test]
    fn test_shuffle_indices() {
        let indices = shuffle_indices(100);
        assert_eq!(indices.len(), 100);

        // Check all indices are present
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
    }
}
