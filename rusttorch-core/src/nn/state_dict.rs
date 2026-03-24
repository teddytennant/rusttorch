//! StateDict — save and load model parameters.
//!
//! A StateDict is a flat mapping from parameter names to tensors.
//! It can be serialized to/from a compact binary format for saving
//! and loading trained models.
//!
//! # Binary Format
//!
//! ```text
//! [8 bytes magic: "RTTENSOR"]
//! [4 bytes version: u32 LE]
//! [4 bytes num_entries: u32 LE]
//! For each entry:
//!   [4 bytes name_len: u32 LE]
//!   [name_len bytes: UTF-8 name]
//!   [1 byte dtype: 0=f32, 1=f64, 2=i32, 3=i64]
//!   [4 bytes ndims: u32 LE]
//!   [ndims * 8 bytes: shape dimensions as u64 LE]
//!   [4 bytes data_len: u32 LE]
//!   [data_len bytes: raw tensor data]
//! ```

use crate::tensor::{DType, Tensor};
use std::collections::HashMap;
use std::io::{Read, Write};

const MAGIC: &[u8; 8] = b"RTTENSOR";
const VERSION: u32 = 1;

/// A mapping from parameter names to tensors.
///
/// Used for saving and loading model weights, analogous to
/// PyTorch's `state_dict()`.
#[derive(Debug, Clone)]
pub struct StateDict {
    pub tensors: HashMap<String, Tensor>,
}

impl StateDict {
    /// Create an empty StateDict.
    pub fn new() -> Self {
        StateDict {
            tensors: HashMap::new(),
        }
    }

    /// Insert a named tensor.
    pub fn insert(&mut self, name: impl Into<String>, tensor: Tensor) {
        self.tensors.insert(name.into(), tensor);
    }

    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the dict is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// All parameter names, sorted.
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<_> = self.tensors.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Save to a binary writer.
    pub fn save<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Header
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&(self.tensors.len() as u32).to_le_bytes())?;

        // Sort keys for deterministic output
        let mut keys: Vec<_> = self.tensors.keys().collect();
        keys.sort();

        for key in keys {
            let tensor = &self.tensors[key];

            // Name
            let name_bytes = key.as_bytes();
            writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(name_bytes)?;

            // DType
            let dtype_byte: u8 = match tensor.dtype() {
                DType::Float32 => 0,
                DType::Float64 => 1,
                DType::Int32 => 2,
                DType::Int64 => 3,
            };
            writer.write_all(&[dtype_byte])?;

            // Shape
            let shape = tensor.shape();
            writer.write_all(&(shape.len() as u32).to_le_bytes())?;
            for &dim in shape {
                writer.write_all(&(dim as u64).to_le_bytes())?;
            }

            // Data
            let data = tensor_to_bytes(tensor);
            writer.write_all(&(data.len() as u32).to_le_bytes())?;
            writer.write_all(&data)?;
        }

        Ok(())
    }

    /// Save to a file.
    pub fn save_file(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        self.save(&mut file)
    }

    /// Load from a binary reader.
    pub fn load<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        // Magic
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid magic number — not a RustTorch state dict",
            ));
        }

        // Version
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported version: {} (expected {})", version, VERSION),
            ));
        }

        // Num entries
        let mut num_bytes = [0u8; 4];
        reader.read_exact(&mut num_bytes)?;
        let num_entries = u32::from_le_bytes(num_bytes) as usize;

        let mut tensors = HashMap::new();

        for _ in 0..num_entries {
            // Name
            let mut name_len_bytes = [0u8; 4];
            reader.read_exact(&mut name_len_bytes)?;
            let name_len = u32::from_le_bytes(name_len_bytes) as usize;
            let mut name_buf = vec![0u8; name_len];
            reader.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

            // DType
            let mut dtype_byte = [0u8; 1];
            reader.read_exact(&mut dtype_byte)?;

            // Shape
            let mut ndims_bytes = [0u8; 4];
            reader.read_exact(&mut ndims_bytes)?;
            let ndims = u32::from_le_bytes(ndims_bytes) as usize;
            let mut shape = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                let mut dim_bytes = [0u8; 8];
                reader.read_exact(&mut dim_bytes)?;
                shape.push(u64::from_le_bytes(dim_bytes) as usize);
            }

            // Data
            let mut data_len_bytes = [0u8; 4];
            reader.read_exact(&mut data_len_bytes)?;
            let data_len = u32::from_le_bytes(data_len_bytes) as usize;
            let mut data = vec![0u8; data_len];
            reader.read_exact(&mut data)?;

            let tensor = bytes_to_tensor(&data, &shape, dtype_byte[0])?;
            tensors.insert(name, tensor);
        }

        Ok(StateDict { tensors })
    }

    /// Load from a file.
    pub fn load_file(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        Self::load(&mut file)
    }

    /// Merge another StateDict into this one, prefixing all keys.
    pub fn merge_prefixed(&mut self, prefix: &str, other: &StateDict) {
        for (key, tensor) in &other.tensors {
            self.insert(format!("{}.{}", prefix, key), tensor.clone());
        }
    }

    /// Extract a sub-dict with the given prefix, stripping the prefix.
    pub fn sub_dict(&self, prefix: &str) -> StateDict {
        let prefix_dot = format!("{}.", prefix);
        let mut result = StateDict::new();
        for (key, tensor) in &self.tensors {
            if let Some(stripped) = key.strip_prefix(&prefix_dot) {
                result.insert(stripped, tensor.clone());
            }
        }
        result
    }
}

impl Default for StateDict {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a tensor to raw bytes.
fn tensor_to_bytes(tensor: &Tensor) -> Vec<u8> {
    let data = tensor.to_vec_f32();
    match tensor.dtype() {
        DType::Float32 => {
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for val in &data {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
            bytes
        }
        DType::Float64 => {
            // to_vec_f32 already converts f64 to f32, but we want to preserve f64
            // Serialize as f32 for now (framework uses f32 primarily)
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for val in &data {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
            bytes
        }
        _ => {
            // For int types, also serialize as f32 (simplification)
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for val in &data {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
            bytes
        }
    }
}

/// Convert raw bytes back to a tensor.
fn bytes_to_tensor(data: &[u8], shape: &[usize], dtype_byte: u8) -> std::io::Result<Tensor> {
    match dtype_byte {
        0..=3 => {
            // All stored as f32 currently
            if !data.len().is_multiple_of(4) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Data length not aligned to 4 bytes",
                ));
            }
            let values: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            let expected_len: usize = shape.iter().product();
            if values.len() != expected_len {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Data length {} doesn't match shape {:?} (expected {})",
                        values.len(),
                        shape,
                        expected_len
                    ),
                ));
            }
            Ok(Tensor::from_vec(values, shape))
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unknown dtype: {}", dtype_byte),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state_dict() {
        let sd = StateDict::new();
        assert!(sd.is_empty());
        assert_eq!(sd.len(), 0);
        assert!(sd.keys().is_empty());
    }

    #[test]
    fn insert_and_get() {
        let mut sd = StateDict::new();
        sd.insert("weight", Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]));
        assert_eq!(sd.len(), 1);
        assert!(!sd.is_empty());
        let t = sd.get("weight").unwrap();
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn keys_sorted() {
        let mut sd = StateDict::new();
        sd.insert("z_param", Tensor::zeros(&[1], DType::Float32));
        sd.insert("a_param", Tensor::zeros(&[1], DType::Float32));
        sd.insert("m_param", Tensor::zeros(&[1], DType::Float32));
        let keys = sd.keys();
        assert_eq!(keys, vec!["a_param", "m_param", "z_param"]);
    }

    #[test]
    fn save_load_roundtrip_1d() {
        let mut sd = StateDict::new();
        sd.insert("weight", Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]));

        let mut buf = Vec::new();
        sd.save(&mut buf).unwrap();

        let loaded = StateDict::load(&mut &buf[..]).unwrap();
        assert_eq!(loaded.len(), 1);
        let t = loaded.get("weight").unwrap();
        assert_eq!(t.shape(), &[4]);
        assert_eq!(t.to_vec_f32(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn save_load_roundtrip_2d() {
        let mut sd = StateDict::new();
        sd.insert(
            "weight",
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
        );

        let mut buf = Vec::new();
        sd.save(&mut buf).unwrap();

        let loaded = StateDict::load(&mut &buf[..]).unwrap();
        let t = loaded.get("weight").unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn save_load_roundtrip_4d() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let mut sd = StateDict::new();
        sd.insert("conv.weight", Tensor::from_vec(data.clone(), &[2, 3, 2, 2]));

        let mut buf = Vec::new();
        sd.save(&mut buf).unwrap();

        let loaded = StateDict::load(&mut &buf[..]).unwrap();
        let t = loaded.get("conv.weight").unwrap();
        assert_eq!(t.shape(), &[2, 3, 2, 2]);
        assert_eq!(t.to_vec_f32(), data);
    }

    #[test]
    fn save_load_multiple_tensors() {
        let mut sd = StateDict::new();
        sd.insert("layer1.weight", Tensor::from_vec(vec![1.0, 2.0], &[2]));
        sd.insert("layer1.bias", Tensor::from_vec(vec![0.5], &[1]));
        sd.insert(
            "layer2.weight",
            Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], &[2, 2]),
        );

        let mut buf = Vec::new();
        sd.save(&mut buf).unwrap();

        let loaded = StateDict::load(&mut &buf[..]).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(
            loaded.get("layer1.weight").unwrap().to_vec_f32(),
            vec![1.0, 2.0]
        );
        assert_eq!(loaded.get("layer1.bias").unwrap().to_vec_f32(), vec![0.5]);
        assert_eq!(
            loaded.get("layer2.weight").unwrap().to_vec_f32(),
            vec![3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn save_load_file_roundtrip() {
        let mut sd = StateDict::new();
        sd.insert("w", Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]));

        let dir = std::env::temp_dir().join("rusttorch_test_state_dict");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.rt");

        sd.save_file(&path).unwrap();
        let loaded = StateDict::load_file(&path).unwrap();
        assert_eq!(loaded.get("w").unwrap().to_vec_f32(), vec![1.0, 2.0, 3.0]);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn invalid_magic_error() {
        let bad_data = b"BADMAGIC\x01\x00\x00\x00\x00\x00\x00\x00";
        let result = StateDict::load(&mut &bad_data[..]);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_version_error() {
        let mut data = Vec::new();
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&99u32.to_le_bytes()); // bad version
        data.extend_from_slice(&0u32.to_le_bytes());
        let result = StateDict::load(&mut &data[..]);
        assert!(result.is_err());
    }

    #[test]
    fn merge_prefixed() {
        let mut parent = StateDict::new();
        let mut child = StateDict::new();
        child.insert("weight", Tensor::from_vec(vec![1.0], &[1]));
        child.insert("bias", Tensor::from_vec(vec![2.0], &[1]));

        parent.merge_prefixed("layer0", &child);
        assert_eq!(parent.len(), 2);
        assert!(parent.get("layer0.weight").is_some());
        assert!(parent.get("layer0.bias").is_some());
    }

    #[test]
    fn sub_dict() {
        let mut sd = StateDict::new();
        sd.insert("layer0.weight", Tensor::from_vec(vec![1.0], &[1]));
        sd.insert("layer0.bias", Tensor::from_vec(vec![2.0], &[1]));
        sd.insert("layer1.weight", Tensor::from_vec(vec![3.0], &[1]));

        let sub = sd.sub_dict("layer0");
        assert_eq!(sub.len(), 2);
        assert!(sub.get("weight").is_some());
        assert!(sub.get("bias").is_some());
        assert!(sub.get("layer1.weight").is_none());
    }

    #[test]
    fn empty_save_load() {
        let sd = StateDict::new();
        let mut buf = Vec::new();
        sd.save(&mut buf).unwrap();

        let loaded = StateDict::load(&mut &buf[..]).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn scalar_tensor() {
        let mut sd = StateDict::new();
        sd.insert("scalar", Tensor::from_vec(vec![42.0], &[1]));

        let mut buf = Vec::new();
        sd.save(&mut buf).unwrap();

        let loaded = StateDict::load(&mut &buf[..]).unwrap();
        assert_eq!(loaded.get("scalar").unwrap().to_vec_f32(), vec![42.0]);
    }
}
