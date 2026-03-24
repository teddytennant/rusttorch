//! IDX file format parser.
//!
//! The IDX format is a simple binary format for vectors and multidimensional matrices
//! of various numerical types. It is used by MNIST and Fashion-MNIST datasets.
//!
//! Format:
//! - Magic number (4 bytes, big-endian):
//!   - Byte 0-1: always 0x00 0x00
//!   - Byte 2: data type (0x08 = u8, 0x09 = i8, 0x0B = i16, 0x0C = i32, 0x0D = f32, 0x0E = f64)
//!   - Byte 3: number of dimensions
//! - Dimension sizes (4 bytes each, big-endian)
//! - Raw data (big-endian)

use std::io::Read;

/// Parse an IDX file and return the raw bytes and shape.
///
/// Returns (data, shape) where data is the raw u8 values and shape is the dimensions.
pub fn parse_idx<R: Read>(reader: &mut R) -> Result<(Vec<u8>, Vec<usize>), String> {
    // Read magic number (4 bytes)
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .map_err(|e| format!("Failed to read IDX magic number: {}", e))?;

    if magic[0] != 0 || magic[1] != 0 {
        return Err(format!(
            "Invalid IDX magic number: first two bytes must be 0x00, got 0x{:02x} 0x{:02x}",
            magic[0], magic[1]
        ));
    }

    let data_type = magic[2];
    if data_type != 0x08 {
        return Err(format!(
            "Unsupported IDX data type: 0x{:02x} (only unsigned byte 0x08 is supported)",
            data_type
        ));
    }

    let num_dims = magic[3] as usize;
    if num_dims == 0 {
        return Err("IDX file has 0 dimensions".to_string());
    }

    // Read dimension sizes
    let mut shape = Vec::with_capacity(num_dims);
    for i in 0..num_dims {
        let mut dim_bytes = [0u8; 4];
        reader
            .read_exact(&mut dim_bytes)
            .map_err(|e| format!("Failed to read IDX dimension {}: {}", i, e))?;
        let dim = u32::from_be_bytes(dim_bytes) as usize;
        shape.push(dim);
    }

    // Total number of elements
    let total: usize = shape.iter().product();
    if total == 0 {
        return Err("IDX file has zero total elements".to_string());
    }

    // Read all data
    let mut data = vec![0u8; total];
    reader
        .read_exact(&mut data)
        .map_err(|e| format!("Failed to read IDX data ({} bytes): {}", total, e))?;

    Ok((data, shape))
}

/// Parse an IDX file containing images and return as f32 values normalized to [0, 1].
///
/// Returns (data, num_images, rows, cols).
pub fn parse_idx_images<R: Read>(
    reader: &mut R,
) -> Result<(Vec<f32>, usize, usize, usize), String> {
    let (raw, shape) = parse_idx(reader)?;

    if shape.len() != 3 {
        return Err(format!(
            "Expected 3D IDX file for images (n, rows, cols), got {} dimensions",
            shape.len()
        ));
    }

    let num_images = shape[0];
    let rows = shape[1];
    let cols = shape[2];

    // Convert u8 [0, 255] to f32 [0.0, 1.0]
    let data: Vec<f32> = raw.into_iter().map(|b| b as f32 / 255.0).collect();

    Ok((data, num_images, rows, cols))
}

/// Parse an IDX file containing labels and return as u8 values.
///
/// Returns (labels, num_labels).
pub fn parse_idx_labels<R: Read>(reader: &mut R) -> Result<(Vec<u8>, usize), String> {
    let (data, shape) = parse_idx(reader)?;

    if shape.len() != 1 {
        return Err(format!(
            "Expected 1D IDX file for labels, got {} dimensions",
            shape.len()
        ));
    }

    let num_labels = shape[0];
    Ok((data, num_labels))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_idx_1d(data: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        // Magic: 0x00 0x00 0x08 (u8) 0x01 (1 dim)
        buf.extend_from_slice(&[0x00, 0x00, 0x08, 0x01]);
        // Dimension: data.len() as u32 big-endian
        buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
        buf.extend_from_slice(data);
        buf
    }

    fn make_idx_3d(n: u32, rows: u32, cols: u32, data: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        // Magic: 0x00 0x00 0x08 (u8) 0x03 (3 dims)
        buf.extend_from_slice(&[0x00, 0x00, 0x08, 0x03]);
        buf.extend_from_slice(&n.to_be_bytes());
        buf.extend_from_slice(&rows.to_be_bytes());
        buf.extend_from_slice(&cols.to_be_bytes());
        buf.extend_from_slice(data);
        buf
    }

    #[test]
    fn test_parse_idx_labels() {
        let raw = make_idx_1d(&[0, 1, 2, 3, 7, 9]);
        let (labels, count) = parse_idx_labels(&mut &raw[..]).unwrap();
        assert_eq!(count, 6);
        assert_eq!(labels, vec![0, 1, 2, 3, 7, 9]);
    }

    #[test]
    fn test_parse_idx_images() {
        // 2 images, 2x2 pixels each
        let pixels = vec![0, 255, 128, 64, 255, 0, 192, 32];
        let raw = make_idx_3d(2, 2, 2, &pixels);
        let (data, n, rows, cols) = parse_idx_images(&mut &raw[..]).unwrap();
        assert_eq!(n, 2);
        assert_eq!(rows, 2);
        assert_eq!(cols, 2);
        assert_eq!(data.len(), 8);
        assert!((data[0] - 0.0).abs() < 1e-6); // 0/255
        assert!((data[1] - 1.0).abs() < 1e-6); // 255/255
        assert!((data[2] - 128.0 / 255.0).abs() < 1e-4);
    }

    #[test]
    fn test_parse_idx_invalid_magic() {
        let raw = vec![0x01, 0x00, 0x08, 0x01, 0, 0, 0, 1, 42];
        let result = parse_idx(&mut &raw[..]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid IDX magic"));
    }

    #[test]
    fn test_parse_idx_unsupported_type() {
        let raw = vec![0x00, 0x00, 0x0B, 0x01, 0, 0, 0, 1, 0, 42];
        let result = parse_idx(&mut &raw[..]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unsupported IDX data type"));
    }

    #[test]
    fn test_parse_idx_truncated_data() {
        // Claims 4 elements but only provides 2
        let mut raw = Vec::new();
        raw.extend_from_slice(&[0x00, 0x00, 0x08, 0x01]);
        raw.extend_from_slice(&4u32.to_be_bytes());
        raw.extend_from_slice(&[1, 2]); // Only 2 bytes, need 4
        let result = parse_idx(&mut &raw[..]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_idx_labels_wrong_dims() {
        // 3D data passed to label parser
        let raw = make_idx_3d(1, 2, 2, &[0, 1, 2, 3]);
        let result = parse_idx_labels(&mut &raw[..]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Expected 1D"));
    }

    #[test]
    fn test_parse_idx_images_wrong_dims() {
        // 1D data passed to image parser
        let raw = make_idx_1d(&[0, 1, 2]);
        let result = parse_idx_images(&mut &raw[..]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Expected 3D"));
    }

    #[test]
    fn test_parse_idx_single_image() {
        let pixels = vec![0, 128, 255];
        let raw = make_idx_3d(1, 1, 3, &pixels);
        let (data, n, rows, cols) = parse_idx_images(&mut &raw[..]).unwrap();
        assert_eq!(n, 1);
        assert_eq!(rows, 1);
        assert_eq!(cols, 3);
        assert_eq!(data.len(), 3);
    }

    #[test]
    fn test_parse_idx_zero_dims() {
        let raw = vec![0x00, 0x00, 0x08, 0x00]; // 0 dimensions
        let result = parse_idx(&mut &raw[..]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("0 dimensions"));
    }
}
