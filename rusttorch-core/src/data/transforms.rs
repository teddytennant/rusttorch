//! Data augmentation transforms for image tensors.
//!
//! Standard transforms for training CNNs on image datasets like CIFAR-10.
//! All transforms operate on Tensor data in NCHW format [B, C, H, W].
//!
//! # Standard CIFAR-10 augmentation
//!
//! ```ignore
//! use rusttorch_core::data::transforms;
//!
//! // Pad 4px, random crop back to 32x32, random horizontal flip
//! let augmented = transforms::random_crop(&images, 32, 4);
//! let augmented = transforms::random_horizontal_flip(&augmented, 0.5);
//! ```

use crate::tensor::Tensor;
use rand::Rng;

/// Randomly flip images horizontally with the given probability.
///
/// Operates on NCHW tensors [B, C, H, W]. Each image in the batch is
/// independently flipped with probability `p`.
///
/// # Arguments
/// * `images` — input tensor in NCHW format
/// * `p` — probability of flipping each image (0.5 is standard)
pub fn random_horizontal_flip(images: &Tensor, p: f32) -> Tensor {
    let shape = images.shape().to_vec();
    assert!(shape.len() == 4, "Expected NCHW tensor, got {:?}", shape);
    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

    let data = images.to_vec_f32();
    let mut out = data.clone();
    let mut rng = rand::thread_rng();

    let spatial = height * width;
    let img_size = channels * spatial;

    for b in 0..batch {
        if rng.gen::<f32>() < p {
            // Flip this image horizontally: reverse each row
            for c in 0..channels {
                for h in 0..height {
                    let row_start = b * img_size + c * spatial + h * width;
                    // Reverse the row in-place
                    let row = &mut out[row_start..row_start + width];
                    row.reverse();
                }
            }
        }
    }

    Tensor::from_vec(out, &shape)
}

/// Random crop with zero-padding.
///
/// Pads each spatial dimension by `padding` pixels (zeros), then crops a
/// `crop_size × crop_size` region at a random position. Each image in the
/// batch gets a different random crop.
///
/// Standard for CIFAR-10: `random_crop(images, 32, 4)` — pad to 40x40, crop back to 32x32.
///
/// # Arguments
/// * `images` — input tensor in NCHW format [B, C, H, W]
/// * `crop_size` — output spatial dimensions (both H and W)
/// * `padding` — number of zero-padding pixels on each side
pub fn random_crop(images: &Tensor, crop_size: usize, padding: usize) -> Tensor {
    let shape = images.shape().to_vec();
    assert!(shape.len() == 4, "Expected NCHW tensor, got {:?}", shape);
    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

    let padded_h = height + 2 * padding;
    let padded_w = width + 2 * padding;
    assert!(
        crop_size <= padded_h && crop_size <= padded_w,
        "Crop size {} exceeds padded dimensions {}x{}",
        crop_size,
        padded_h,
        padded_w
    );

    let data = images.to_vec_f32();
    let spatial = height * width;
    let img_size = channels * spatial;
    let out_spatial = crop_size * crop_size;
    let out_img_size = channels * out_spatial;
    let mut out = vec![0.0f32; batch * out_img_size];
    let mut rng = rand::thread_rng();

    for b in 0..batch {
        // Random crop offset within padded image
        let top = rng.gen_range(0..=padded_h - crop_size);
        let left = rng.gen_range(0..=padded_w - crop_size);

        for c in 0..channels {
            for oh in 0..crop_size {
                for ow in 0..crop_size {
                    // Map output pixel back to padded coordinates
                    let ph = top + oh;
                    let pw = left + ow;

                    // Check if this padded coordinate falls within the original image
                    let val = if ph >= padding
                        && ph < padding + height
                        && pw >= padding
                        && pw < padding + width
                    {
                        let ih = ph - padding;
                        let iw = pw - padding;
                        data[b * img_size + c * spatial + ih * width + iw]
                    } else {
                        0.0 // zero padding
                    };

                    out[b * out_img_size + c * out_spatial + oh * crop_size + ow] = val;
                }
            }
        }
    }

    Tensor::from_vec(out, &[batch, channels, crop_size, crop_size])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_batch(batch: usize, channels: usize, h: usize, w: usize) -> Tensor {
        let size = batch * channels * h * w;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        Tensor::from_vec(data, &[batch, channels, h, w])
    }

    #[test]
    fn test_random_horizontal_flip_p0_no_flip() {
        let images = make_batch(2, 3, 4, 4);
        let result = random_horizontal_flip(&images, 0.0);
        assert_eq!(result.to_vec_f32(), images.to_vec_f32());
    }

    #[test]
    fn test_random_horizontal_flip_p1_always_flip() {
        // Single image, 1 channel, 1x4 — easy to verify
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let images = Tensor::from_vec(data, &[1, 1, 1, 4]);
        let result = random_horizontal_flip(&images, 1.0);
        assert_eq!(result.to_vec_f32(), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_random_horizontal_flip_preserves_shape() {
        let images = make_batch(4, 3, 32, 32);
        let result = random_horizontal_flip(&images, 0.5);
        assert_eq!(result.shape(), &[4, 3, 32, 32]);
    }

    #[test]
    fn test_random_horizontal_flip_multichannel() {
        // 1 image, 2 channels, 2x3
        let data = vec![
            1.0, 2.0, 3.0, // ch0 row0
            4.0, 5.0, 6.0, // ch0 row1
            7.0, 8.0, 9.0, // ch1 row0
            10.0, 11.0, 12.0, // ch1 row1
        ];
        let images = Tensor::from_vec(data, &[1, 2, 2, 3]);
        let result = random_horizontal_flip(&images, 1.0);
        assert_eq!(
            result.to_vec_f32(),
            vec![
                3.0, 2.0, 1.0, // ch0 row0 flipped
                6.0, 5.0, 4.0, // ch0 row1 flipped
                9.0, 8.0, 7.0, // ch1 row0 flipped
                12.0, 11.0, 10.0, // ch1 row1 flipped
            ]
        );
    }

    #[test]
    fn test_random_crop_no_padding() {
        // No padding, crop_size = original size → identity
        let images = make_batch(1, 1, 4, 4);
        let result = random_crop(&images, 4, 0);
        assert_eq!(result.shape(), &[1, 1, 4, 4]);
        assert_eq!(result.to_vec_f32(), images.to_vec_f32());
    }

    #[test]
    fn test_random_crop_preserves_shape() {
        let images = make_batch(8, 3, 32, 32);
        let result = random_crop(&images, 32, 4);
        assert_eq!(result.shape(), &[8, 3, 32, 32]);
    }

    #[test]
    fn test_random_crop_values_in_range() {
        // All input values are in [0, N). Output should be in [0, N) or 0 (padding).
        let images = make_batch(2, 3, 32, 32);
        let max_val = (2 * 3 * 32 * 32) as f32;
        let result = random_crop(&images, 32, 4);
        for &v in result.to_vec_f32().iter() {
            assert!(v >= 0.0 && v < max_val, "Value {} out of range", v);
        }
    }

    #[test]
    fn test_random_crop_small() {
        // 1 image, 1 channel, 2x2, pad=1 → 4x4 padded, crop 2x2
        // Result should be 2x2 with values from original or 0
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let images = Tensor::from_vec(data.clone(), &[1, 1, 2, 2]);
        let result = random_crop(&images, 2, 1);
        assert_eq!(result.shape(), &[1, 1, 2, 2]);
        // All values should be from {0, 1, 2, 3, 4}
        for &v in result.to_vec_f32().iter() {
            assert!(
                v == 0.0 || data.contains(&v),
                "Unexpected value {} in crop",
                v
            );
        }
    }

    #[test]
    fn test_random_crop_deterministic_no_padding() {
        // With padding=0 and crop_size=original, result equals input regardless of randomness
        let images = make_batch(4, 3, 8, 8);
        let result = random_crop(&images, 8, 0);
        assert_eq!(result.to_vec_f32(), images.to_vec_f32());
    }
}
