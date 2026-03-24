//! safetensors format support — load model weights from the HuggingFace ecosystem.
//!
//! The safetensors format is the standard for distributing ML model weights.
//! This module enables loading pre-trained models from HuggingFace into
//! RustTorch's StateDict for inference or fine-tuning.
//!
//! # Format
//!
//! ```text
//! [8 bytes]           header_size: u64 LE
//! [header_size bytes] JSON header: {"tensor_name": {"dtype": "F32", "shape": [d0, d1, ...], "data_offsets": [start, end]}, "__metadata__": {...}}
//! [remaining bytes]   raw tensor data (little-endian, contiguous)
//! ```
//!
//! # Supported dtypes
//!
//! - F32 (float32) — loaded directly
//! - F16 (float16) — converted to f32 on load
//! - BF16 (bfloat16) — converted to f32 on load
//! - F64 (float64) — converted to f32 on load
//!
//! # Example
//!
//! ```ignore
//! use rusttorch_core::nn::safetensors::load_safetensors;
//! use rusttorch_core::nn::StateDict;
//!
//! let state_dict = load_safetensors("model.safetensors").unwrap();
//! println!("Loaded {} tensors", state_dict.len());
//! for key in state_dict.keys() {
//!     let t = state_dict.get(&key).unwrap();
//!     println!("  {}: {:?}", key, t.shape());
//! }
//! ```

use crate::nn::state_dict::StateDict;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::io::Read;

/// Errors that can occur when loading safetensors files.
#[derive(Debug)]
pub enum SafetensorsError {
    Io(std::io::Error),
    Json(String),
    InvalidFormat(String),
    UnsupportedDtype(String),
}

impl std::fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SafetensorsError::Io(e) => write!(f, "I/O error: {}", e),
            SafetensorsError::Json(e) => write!(f, "JSON parse error: {}", e),
            SafetensorsError::InvalidFormat(e) => write!(f, "Invalid safetensors format: {}", e),
            SafetensorsError::UnsupportedDtype(e) => write!(f, "Unsupported dtype: {}", e),
        }
    }
}

impl std::error::Error for SafetensorsError {}

impl From<std::io::Error> for SafetensorsError {
    fn from(e: std::io::Error) -> Self {
        SafetensorsError::Io(e)
    }
}

/// Tensor metadata from the safetensors header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// Parsed safetensors header.
#[derive(Debug)]
pub struct SafetensorsHeader {
    pub tensors: Vec<TensorInfo>,
    pub metadata: HashMap<String, String>,
}

/// Load a safetensors file into a StateDict.
///
/// All tensor types (F16, BF16, F32, F64) are converted to f32.
pub fn load_safetensors(path: impl AsRef<std::path::Path>) -> Result<StateDict, SafetensorsError> {
    let data = std::fs::read(path)?;
    load_safetensors_from_bytes(&data)
}

/// Load safetensors from a byte slice.
pub fn load_safetensors_from_bytes(data: &[u8]) -> Result<StateDict, SafetensorsError> {
    if data.len() < 8 {
        return Err(SafetensorsError::InvalidFormat(
            "File too small — need at least 8 bytes for header size".to_string(),
        ));
    }

    // Read header size
    let header_size = u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]) as usize;

    if header_size == 0 {
        return Err(SafetensorsError::InvalidFormat(
            "Header size is zero".to_string(),
        ));
    }

    let header_end = 8 + header_size;
    if header_end > data.len() {
        return Err(SafetensorsError::InvalidFormat(format!(
            "Header size {} exceeds file size {}",
            header_size,
            data.len()
        )));
    }

    // Parse JSON header
    let header_json = std::str::from_utf8(&data[8..header_end])
        .map_err(|e| SafetensorsError::InvalidFormat(format!("Header is not valid UTF-8: {}", e)))?;

    let header = parse_header(header_json)?;
    let tensor_data = &data[header_end..];

    // Load tensors
    let mut state_dict = StateDict::new();

    for info in &header.tensors {
        let (start, end) = info.data_offsets;
        if end > tensor_data.len() {
            return Err(SafetensorsError::InvalidFormat(format!(
                "Tensor '{}' data_offsets [{}, {}] exceeds data buffer size {}",
                info.name,
                start,
                end,
                tensor_data.len()
            )));
        }

        let raw = &tensor_data[start..end];
        let values = decode_tensor_data(raw, &info.dtype, &info.shape)?;

        let tensor = Tensor::from_vec(values, &info.shape);
        state_dict.insert(&info.name, tensor);
    }

    Ok(state_dict)
}

/// Parse the safetensors JSON header.
///
/// The header is a JSON object where each key is a tensor name mapping to
/// {"dtype": "F32", "shape": [d0, d1, ...], "data_offsets": [start, end]}.
/// The special key "__metadata__" contains string key-value metadata.
fn parse_header(json: &str) -> Result<SafetensorsHeader, SafetensorsError> {
    // Minimal JSON parser — no external dependency needed.
    // safetensors headers are simple flat objects.
    let json = json.trim();
    if !json.starts_with('{') || !json.ends_with('}') {
        return Err(SafetensorsError::Json(
            "Header must be a JSON object".to_string(),
        ));
    }

    let mut tensors = Vec::new();
    let mut metadata = HashMap::new();

    // Parse the JSON object using a simple state machine
    let entries = parse_json_object(json)?;

    for (key, value) in entries {
        if key == "__metadata__" {
            // Parse metadata as string -> string mapping
            if let Ok(meta_entries) = parse_json_object(&value) {
                for (mk, mv) in meta_entries {
                    metadata.insert(mk, strip_json_string(&mv));
                }
            }
            continue;
        }

        // Parse tensor entry: {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, 8]}
        let entry_fields = parse_json_object(&value)?;
        let mut dtype = String::new();
        let mut shape = Vec::new();
        let mut offsets = (0usize, 0usize);

        for (field_key, field_value) in &entry_fields {
            match field_key.as_str() {
                "dtype" => {
                    dtype = strip_json_string(field_value);
                }
                "shape" => {
                    shape = parse_json_usize_array(field_value)?;
                }
                "data_offsets" => {
                    let arr = parse_json_usize_array(field_value)?;
                    if arr.len() != 2 {
                        return Err(SafetensorsError::Json(format!(
                            "data_offsets must have exactly 2 elements, got {}",
                            arr.len()
                        )));
                    }
                    offsets = (arr[0], arr[1]);
                }
                _ => {} // ignore unknown fields
            }
        }

        if dtype.is_empty() {
            return Err(SafetensorsError::Json(format!(
                "Tensor '{}' missing dtype",
                key
            )));
        }

        tensors.push(TensorInfo {
            name: key,
            dtype,
            shape,
            data_offsets: offsets,
        });
    }

    Ok(SafetensorsHeader { tensors, metadata })
}

/// Decode raw tensor data to f32 based on dtype string.
fn decode_tensor_data(
    raw: &[u8],
    dtype: &str,
    shape: &[usize],
) -> Result<Vec<f32>, SafetensorsError> {
    let expected_elements: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };

    match dtype {
        "F32" => {
            if raw.len() != expected_elements * 4 {
                return Err(SafetensorsError::InvalidFormat(format!(
                    "F32 tensor: expected {} bytes, got {}",
                    expected_elements * 4,
                    raw.len()
                )));
            }
            Ok(raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        "F64" => {
            if raw.len() != expected_elements * 8 {
                return Err(SafetensorsError::InvalidFormat(format!(
                    "F64 tensor: expected {} bytes, got {}",
                    expected_elements * 8,
                    raw.len()
                )));
            }
            Ok(raw
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect())
        }
        "F16" => {
            if raw.len() != expected_elements * 2 {
                return Err(SafetensorsError::InvalidFormat(format!(
                    "F16 tensor: expected {} bytes, got {}",
                    expected_elements * 2,
                    raw.len()
                )));
            }
            Ok(raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        "BF16" => {
            if raw.len() != expected_elements * 2 {
                return Err(SafetensorsError::InvalidFormat(format!(
                    "BF16 tensor: expected {} bytes, got {}",
                    expected_elements * 2,
                    raw.len()
                )));
            }
            Ok(raw
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        "I32" => {
            if raw.len() != expected_elements * 4 {
                return Err(SafetensorsError::InvalidFormat(format!(
                    "I32 tensor: expected {} bytes, got {}",
                    expected_elements * 4,
                    raw.len()
                )));
            }
            Ok(raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect())
        }
        "I64" => {
            if raw.len() != expected_elements * 8 {
                return Err(SafetensorsError::InvalidFormat(format!(
                    "I64 tensor: expected {} bytes, got {}",
                    expected_elements * 8,
                    raw.len()
                )));
            }
            Ok(raw
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect())
        }
        "BOOL" => {
            if raw.len() != expected_elements {
                return Err(SafetensorsError::InvalidFormat(format!(
                    "BOOL tensor: expected {} bytes, got {}",
                    expected_elements,
                    raw.len()
                )));
            }
            Ok(raw.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect())
        }
        "U8" => {
            if raw.len() != expected_elements {
                return Err(SafetensorsError::InvalidFormat(format!(
                    "U8 tensor: expected {} bytes, got {}",
                    expected_elements,
                    raw.len()
                )));
            }
            Ok(raw.iter().map(|&b| b as f32).collect())
        }
        _ => Err(SafetensorsError::UnsupportedDtype(dtype.to_string())),
    }
}

/// Convert IEEE 754 half-precision (float16) to float32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: normalize
            let mut e = exp;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e = e.wrapping_sub(1);
            }
            f &= 0x3FF;
            let exp32 = (127u32 - 15).wrapping_add(e).wrapping_add(1);
            f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13))
        }
    } else if exp == 31 {
        if frac == 0 {
            // Infinity
            f32::from_bits((sign << 31) | 0x7F800000)
        } else {
            // NaN
            f32::from_bits((sign << 31) | 0x7F800000 | (frac << 13))
        }
    } else {
        // Normal
        let exp32 = exp + (127 - 15);
        f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
    }
}

/// Convert bfloat16 to float32.
/// bfloat16 is just the upper 16 bits of float32, so conversion is trivial.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ---- Minimal JSON parser (no external deps) ----

/// Parse a JSON object into key-value pairs.
/// Returns (key, raw_value_string) pairs.
fn parse_json_object(json: &str) -> Result<Vec<(String, String)>, SafetensorsError> {
    let json = json.trim();
    if !json.starts_with('{') || !json.ends_with('}') {
        return Err(SafetensorsError::Json(format!(
            "Expected JSON object, got: {}...",
            &json[..json.len().min(50)]
        )));
    }

    let inner = &json[1..json.len() - 1];
    let mut entries = Vec::new();
    let mut pos = 0;
    let chars: Vec<char> = inner.chars().collect();

    while pos < chars.len() {
        // Skip whitespace and commas
        while pos < chars.len() && (chars[pos].is_whitespace() || chars[pos] == ',') {
            pos += 1;
        }
        if pos >= chars.len() {
            break;
        }

        // Parse key (must be a quoted string)
        if chars[pos] != '"' {
            return Err(SafetensorsError::Json(format!(
                "Expected '\"' for key at position {}, got '{}'",
                pos, chars[pos]
            )));
        }
        let (key, new_pos) = parse_json_string(&chars, pos)?;
        pos = new_pos;

        // Skip whitespace and colon
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }
        if pos >= chars.len() || chars[pos] != ':' {
            return Err(SafetensorsError::Json(format!(
                "Expected ':' after key '{}' at position {}",
                key, pos
            )));
        }
        pos += 1; // skip ':'

        // Skip whitespace
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }

        // Parse value (any JSON value — string, number, array, object)
        let (value, new_pos) = parse_json_value(&chars, pos)?;
        pos = new_pos;

        entries.push((key, value));
    }

    Ok(entries)
}

/// Parse a JSON string starting at the given position.
/// Returns (string_content, position_after_closing_quote).
fn parse_json_string(chars: &[char], start: usize) -> Result<(String, usize), SafetensorsError> {
    if chars[start] != '"' {
        return Err(SafetensorsError::Json(format!(
            "Expected '\"' at position {}",
            start
        )));
    }

    let mut result = String::new();
    let mut pos = start + 1;
    while pos < chars.len() {
        if chars[pos] == '\\' && pos + 1 < chars.len() {
            match chars[pos + 1] {
                '"' => {
                    result.push('"');
                    pos += 2;
                }
                '\\' => {
                    result.push('\\');
                    pos += 2;
                }
                '/' => {
                    result.push('/');
                    pos += 2;
                }
                'n' => {
                    result.push('\n');
                    pos += 2;
                }
                't' => {
                    result.push('\t');
                    pos += 2;
                }
                'r' => {
                    result.push('\r');
                    pos += 2;
                }
                _ => {
                    result.push(chars[pos + 1]);
                    pos += 2;
                }
            }
        } else if chars[pos] == '"' {
            return Ok((result, pos + 1));
        } else {
            result.push(chars[pos]);
            pos += 1;
        }
    }

    Err(SafetensorsError::Json(
        "Unterminated string".to_string(),
    ))
}

/// Parse any JSON value starting at the given position.
/// Returns (raw_value_string, position_after_value).
fn parse_json_value(chars: &[char], start: usize) -> Result<(String, usize), SafetensorsError> {
    if start >= chars.len() {
        return Err(SafetensorsError::Json(
            "Unexpected end of input".to_string(),
        ));
    }

    match chars[start] {
        '"' => {
            // String value — return with quotes stripped
            let (s, pos) = parse_json_string(chars, start)?;
            Ok((format!("\"{}\"", s), pos))
        }
        '{' => {
            // Object — find matching brace
            let end = find_matching_bracket(chars, start, '{', '}')?;
            let value: String = chars[start..=end].iter().collect();
            Ok((value, end + 1))
        }
        '[' => {
            // Array — find matching bracket
            let end = find_matching_bracket(chars, start, '[', ']')?;
            let value: String = chars[start..=end].iter().collect();
            Ok((value, end + 1))
        }
        _ => {
            // Number, true, false, null
            let mut pos = start;
            while pos < chars.len() && !matches!(chars[pos], ',' | '}' | ']') {
                pos += 1;
            }
            let value: String = chars[start..pos].iter().collect();
            Ok((value.trim().to_string(), pos))
        }
    }
}

/// Find the position of a matching closing bracket.
fn find_matching_bracket(
    chars: &[char],
    start: usize,
    open: char,
    close: char,
) -> Result<usize, SafetensorsError> {
    let mut depth = 0;
    let mut in_string = false;
    let mut pos = start;

    while pos < chars.len() {
        if in_string {
            if chars[pos] == '\\' {
                pos += 1; // skip escaped char
            } else if chars[pos] == '"' {
                in_string = false;
            }
        } else {
            if chars[pos] == '"' {
                in_string = true;
            } else if chars[pos] == open {
                depth += 1;
            } else if chars[pos] == close {
                depth -= 1;
                if depth == 0 {
                    return Ok(pos);
                }
            }
        }
        pos += 1;
    }

    Err(SafetensorsError::Json(format!(
        "Unmatched '{}' at position {}",
        open, start
    )))
}

/// Strip JSON string quotes.
fn strip_json_string(s: &str) -> String {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

/// Parse a JSON array of unsigned integers.
fn parse_json_usize_array(s: &str) -> Result<Vec<usize>, SafetensorsError> {
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return Err(SafetensorsError::Json(format!(
            "Expected JSON array, got: {}",
            s
        )));
    }
    let inner = s[1..s.len() - 1].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|e| SafetensorsError::Json(format!("Invalid number '{}': {}", part.trim(), e)))
        })
        .collect()
}

// ---- Write support ----

/// Save a StateDict as a safetensors file.
///
/// All tensors are saved as F32. Compatible with the HuggingFace safetensors ecosystem.
pub fn save_safetensors(
    state_dict: &StateDict,
    path: impl AsRef<std::path::Path>,
) -> Result<(), SafetensorsError> {
    let data = save_safetensors_to_bytes(state_dict)?;
    std::fs::write(path, data)?;
    Ok(())
}

/// Serialize a StateDict to safetensors bytes.
pub fn save_safetensors_to_bytes(state_dict: &StateDict) -> Result<Vec<u8>, SafetensorsError> {
    // Sort keys for deterministic output
    let keys = state_dict.keys();

    // Compute data offsets
    let mut tensor_data = Vec::new();
    let mut tensor_infos: Vec<(String, Vec<usize>, usize, usize)> = Vec::new();

    for key in &keys {
        let tensor = state_dict.get(key).unwrap();
        let start = tensor_data.len();
        let values = tensor.to_vec_f32();
        for v in &values {
            tensor_data.extend_from_slice(&v.to_le_bytes());
        }
        let end = tensor_data.len();
        tensor_infos.push((key.clone(), tensor.shape().to_vec(), start, end));
    }

    // Build JSON header
    let mut header_parts = Vec::new();
    for (name, shape, start, end) in &tensor_infos {
        let shape_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        header_parts.push(format!(
            "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            name,
            shape_str.join(","),
            start,
            end
        ));
    }
    let header_json = format!("{{{}}}", header_parts.join(","));
    let header_bytes = header_json.as_bytes();

    // Assemble file
    let mut output = Vec::new();
    output.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    output.extend_from_slice(header_bytes);
    output.extend_from_slice(&tensor_data);

    Ok(output)
}

/// Inspect a safetensors file without loading tensor data.
/// Returns the header with tensor metadata.
pub fn inspect_safetensors(
    path: impl AsRef<std::path::Path>,
) -> Result<SafetensorsHeader, SafetensorsError> {
    let mut file = std::fs::File::open(path)?;

    // Read header size
    let mut size_buf = [0u8; 8];
    file.read_exact(&mut size_buf)?;
    let header_size = u64::from_le_bytes(size_buf) as usize;

    // Read header
    let mut header_buf = vec![0u8; header_size];
    file.read_exact(&mut header_buf)?;

    let header_json = std::str::from_utf8(&header_buf)
        .map_err(|e| SafetensorsError::InvalidFormat(format!("Header is not valid UTF-8: {}", e)))?;

    parse_header(header_json)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- f16/bf16 conversion tests ----

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // Negative zero
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_one() {
        // 1.0 in f16 = 0x3C00
        let val = f16_to_f32(0x3C00);
        assert!((val - 1.0).abs() < 1e-6, "expected 1.0, got {}", val);
    }

    #[test]
    fn test_f16_to_f32_negative() {
        // -1.0 in f16 = 0xBC00
        let val = f16_to_f32(0xBC00);
        assert!((val - (-1.0)).abs() < 1e-6, "expected -1.0, got {}", val);
    }

    #[test]
    fn test_f16_to_f32_half() {
        // 0.5 in f16 = 0x3800
        let val = f16_to_f32(0x3800);
        assert!((val - 0.5).abs() < 1e-6, "expected 0.5, got {}", val);
    }

    #[test]
    fn test_f16_to_f32_inf() {
        let val = f16_to_f32(0x7C00);
        assert!(val.is_infinite() && val.is_sign_positive());
        let val = f16_to_f32(0xFC00);
        assert!(val.is_infinite() && val.is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_nan() {
        let val = f16_to_f32(0x7C01);
        assert!(val.is_nan());
    }

    #[test]
    fn test_bf16_to_f32_one() {
        // 1.0 in bf16 = 0x3F80 (upper 16 bits of f32 1.0 = 0x3F800000)
        let val = bf16_to_f32(0x3F80);
        assert!((val - 1.0).abs() < 1e-6, "expected 1.0, got {}", val);
    }

    #[test]
    fn test_bf16_to_f32_negative_two() {
        // -2.0 in bf16 = 0xC000
        let val = bf16_to_f32(0xC000);
        assert!((val - (-2.0)).abs() < 1e-6, "expected -2.0, got {}", val);
    }

    #[test]
    fn test_bf16_to_f32_zero() {
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    // ---- JSON parser tests ----

    #[test]
    fn test_parse_empty_object() {
        let entries = parse_json_object("{}").unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_parse_simple_object() {
        let json = r#"{"key": "value"}"#;
        let entries = parse_json_object(json).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "key");
    }

    #[test]
    fn test_parse_nested_object() {
        let json = r#"{"tensor": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}"#;
        let entries = parse_json_object(json).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "tensor");
    }

    #[test]
    fn test_parse_usize_array() {
        assert_eq!(parse_json_usize_array("[1, 2, 3]").unwrap(), vec![1, 2, 3]);
        assert_eq!(parse_json_usize_array("[768]").unwrap(), vec![768]);
        assert_eq!(parse_json_usize_array("[]").unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn test_parse_header_single_tensor() {
        let json =
            r#"{"weight": {"dtype": "F32", "shape": [3, 4], "data_offsets": [0, 48]}}"#;
        let header = parse_header(json).unwrap();
        assert_eq!(header.tensors.len(), 1);
        assert_eq!(header.tensors[0].name, "weight");
        assert_eq!(header.tensors[0].dtype, "F32");
        assert_eq!(header.tensors[0].shape, vec![3, 4]);
        assert_eq!(header.tensors[0].data_offsets, (0, 48));
    }

    #[test]
    fn test_parse_header_with_metadata() {
        let json = r#"{"__metadata__": {"format": "pt"}, "bias": {"dtype": "F32", "shape": [10], "data_offsets": [0, 40]}}"#;
        let header = parse_header(json).unwrap();
        assert_eq!(header.tensors.len(), 1);
        assert_eq!(header.tensors[0].name, "bias");
        assert_eq!(header.metadata.get("format"), Some(&"pt".to_string()));
    }

    #[test]
    fn test_parse_header_multiple_tensors() {
        let json = r#"{"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}, "b": {"dtype": "F16", "shape": [3], "data_offsets": [8, 14]}}"#;
        let header = parse_header(json).unwrap();
        assert_eq!(header.tensors.len(), 2);
    }

    // ---- Load/save roundtrip tests ----

    #[test]
    fn test_save_load_roundtrip_f32() {
        let mut sd = StateDict::new();
        sd.insert("weight", Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        sd.insert("bias", Tensor::from_vec(vec![0.5, -0.5], &[2]));

        let bytes = save_safetensors_to_bytes(&sd).unwrap();
        let loaded = load_safetensors_from_bytes(&bytes).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(
            loaded.get("weight").unwrap().to_vec_f32(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(loaded.get("weight").unwrap().shape(), &[2, 2]);
        assert_eq!(
            loaded.get("bias").unwrap().to_vec_f32(),
            vec![0.5, -0.5]
        );
        assert_eq!(loaded.get("bias").unwrap().shape(), &[2]);
    }

    #[test]
    fn test_save_load_roundtrip_empty() {
        let sd = StateDict::new();
        let bytes = save_safetensors_to_bytes(&sd).unwrap();
        let loaded = load_safetensors_from_bytes(&bytes).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_save_load_roundtrip_4d() {
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let mut sd = StateDict::new();
        sd.insert("conv.weight", Tensor::from_vec(data.clone(), &[2, 3, 2, 2]));

        let bytes = save_safetensors_to_bytes(&sd).unwrap();
        let loaded = load_safetensors_from_bytes(&bytes).unwrap();

        let t = loaded.get("conv.weight").unwrap();
        assert_eq!(t.shape(), &[2, 3, 2, 2]);
        let loaded_data = t.to_vec_f32();
        for (a, b) in data.iter().zip(loaded_data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_save_load_file_roundtrip() {
        let mut sd = StateDict::new();
        sd.insert("w", Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]));

        let dir = std::env::temp_dir().join("rusttorch_safetensors_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.safetensors");

        save_safetensors(&sd, &path).unwrap();
        let loaded = load_safetensors(&path).unwrap();
        assert_eq!(loaded.get("w").unwrap().to_vec_f32(), vec![1.0, 2.0, 3.0]);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_load_f16_tensor() {
        // Manually construct a safetensors file with an F16 tensor
        // F16 value for 1.0 = 0x3C00, for 2.0 = 0x4000
        let header = r#"{"x": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}}"#;
        let header_bytes = header.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(header_bytes);
        // 1.0 in f16 LE
        data.extend_from_slice(&0x3C00u16.to_le_bytes());
        // 2.0 in f16 LE
        data.extend_from_slice(&0x4000u16.to_le_bytes());

        let sd = load_safetensors_from_bytes(&data).unwrap();
        let t = sd.get("x").unwrap();
        let vals = t.to_vec_f32();
        assert!((vals[0] - 1.0).abs() < 1e-3, "expected ~1.0, got {}", vals[0]);
        assert!((vals[1] - 2.0).abs() < 1e-3, "expected ~2.0, got {}", vals[1]);
    }

    #[test]
    fn test_load_bf16_tensor() {
        // BF16 value for 1.0 = 0x3F80, for -1.0 = 0xBF80
        let header = r#"{"x": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}"#;
        let header_bytes = header.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(header_bytes);
        data.extend_from_slice(&0x3F80u16.to_le_bytes());
        data.extend_from_slice(&0xBF80u16.to_le_bytes());

        let sd = load_safetensors_from_bytes(&data).unwrap();
        let vals = sd.get("x").unwrap().to_vec_f32();
        assert!((vals[0] - 1.0).abs() < 1e-3);
        assert!((vals[1] - (-1.0)).abs() < 1e-3);
    }

    #[test]
    fn test_load_f64_tensor() {
        let header = r#"{"x": {"dtype": "F64", "shape": [2], "data_offsets": [0, 16]}}"#;
        let header_bytes = header.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(header_bytes);
        data.extend_from_slice(&3.14f64.to_le_bytes());
        data.extend_from_slice(&2.71f64.to_le_bytes());

        let sd = load_safetensors_from_bytes(&data).unwrap();
        let vals = sd.get("x").unwrap().to_vec_f32();
        assert!((vals[0] - 3.14).abs() < 1e-3);
        assert!((vals[1] - 2.71).abs() < 1e-3);
    }

    #[test]
    fn test_load_i32_tensor() {
        let header = r#"{"x": {"dtype": "I32", "shape": [3], "data_offsets": [0, 12]}}"#;
        let header_bytes = header.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(header_bytes);
        data.extend_from_slice(&42i32.to_le_bytes());
        data.extend_from_slice(&(-7i32).to_le_bytes());
        data.extend_from_slice(&0i32.to_le_bytes());

        let sd = load_safetensors_from_bytes(&data).unwrap();
        let vals = sd.get("x").unwrap().to_vec_f32();
        assert_eq!(vals, vec![42.0, -7.0, 0.0]);
    }

    #[test]
    fn test_load_bool_tensor() {
        let header = r#"{"x": {"dtype": "BOOL", "shape": [3], "data_offsets": [0, 3]}}"#;
        let header_bytes = header.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(header_bytes);
        data.push(1);
        data.push(0);
        data.push(1);

        let sd = load_safetensors_from_bytes(&data).unwrap();
        let vals = sd.get("x").unwrap().to_vec_f32();
        assert_eq!(vals, vec![1.0, 0.0, 1.0]);
    }

    // ---- Error handling tests ----

    #[test]
    fn test_error_file_too_small() {
        let result = load_safetensors_from_bytes(&[0, 1, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_header_size_exceeds_file() {
        let mut data = Vec::new();
        data.extend_from_slice(&1000u64.to_le_bytes()); // header claims 1000 bytes
        data.extend_from_slice(b"{}"); // but only 2 bytes of header
        let result = load_safetensors_from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_zero_header() {
        let mut data = Vec::new();
        data.extend_from_slice(&0u64.to_le_bytes());
        let result = load_safetensors_from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_invalid_json() {
        let header = b"not json";
        let mut data = Vec::new();
        data.extend_from_slice(&(header.len() as u64).to_le_bytes());
        data.extend_from_slice(header);
        let result = load_safetensors_from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_data_offset_out_of_bounds() {
        let header = r#"{"x": {"dtype": "F32", "shape": [100], "data_offsets": [0, 400]}}"#;
        let header_bytes = header.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(header_bytes);
        // Only provide 8 bytes of tensor data instead of 400
        data.extend_from_slice(&[0u8; 8]);

        let result = load_safetensors_from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unsupported_dtype() {
        let header = r#"{"x": {"dtype": "COMPLEX128", "shape": [2], "data_offsets": [0, 32]}}"#;
        let header_bytes = header.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(header_bytes);
        data.extend_from_slice(&[0u8; 32]);

        let result = load_safetensors_from_bytes(&data);
        assert!(matches!(result, Err(SafetensorsError::UnsupportedDtype(_))));
    }

    // ---- Inspect test ----

    #[test]
    fn test_inspect_from_bytes() {
        let mut sd = StateDict::new();
        sd.insert("layer.weight", Tensor::from_vec(vec![1.0; 12], &[3, 4]));
        sd.insert("layer.bias", Tensor::from_vec(vec![0.0; 3], &[3]));

        let bytes = save_safetensors_to_bytes(&sd).unwrap();

        // Parse just the header
        let header_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header_json = std::str::from_utf8(&bytes[8..8 + header_size]).unwrap();
        let header = parse_header(header_json).unwrap();

        assert_eq!(header.tensors.len(), 2);
        let names: Vec<&str> = header.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"layer.weight"));
        assert!(names.contains(&"layer.bias"));
    }

    // ---- Cross-format compatibility test ----

    #[test]
    fn test_safetensors_to_statedict_to_rttensor_roundtrip() {
        // Save as safetensors, load, save as RTTENSOR, load, verify
        let mut original = StateDict::new();
        original.insert("fc.weight", Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        original.insert("fc.bias", Tensor::from_vec(vec![0.1, 0.2], &[2]));

        // Save as safetensors
        let st_bytes = save_safetensors_to_bytes(&original).unwrap();
        // Load from safetensors
        let from_st = load_safetensors_from_bytes(&st_bytes).unwrap();

        // Save as RTTENSOR
        let mut rt_bytes = Vec::new();
        from_st.save(&mut rt_bytes).unwrap();
        // Load from RTTENSOR
        let from_rt = StateDict::load(&mut &rt_bytes[..]).unwrap();

        // Verify
        assert_eq!(from_rt.len(), 2);
        assert_eq!(
            from_rt.get("fc.weight").unwrap().to_vec_f32(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(from_rt.get("fc.weight").unwrap().shape(), &[2, 2]);
    }
}
