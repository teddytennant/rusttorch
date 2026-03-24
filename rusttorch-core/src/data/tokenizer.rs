//! Byte-level BPE tokenizer (GPT-2 style)
//!
//! Implements the byte-level Byte Pair Encoding used by GPT-2, GPT-3, and many
//! HuggingFace models. Key features:
//!
//! - Every byte maps to a Unicode character, so any UTF-8 text is tokenizable
//! - Pre-tokenization splits on word boundaries (GPT-2 regex pattern)
//! - BPE merges applied per-word for efficient encoding
//! - Load from HuggingFace `vocab.json` + `merges.txt` format
//!
//! # Example
//!
//! ```no_run
//! use rusttorch_core::data::tokenizer::BpeTokenizer;
//!
//! let tokenizer = BpeTokenizer::from_files("vocab.json", "merges.txt").unwrap();
//! let ids = tokenizer.encode("Hello, world!");
//! let text = tokenizer.decode(&ids);
//! assert_eq!(text, "Hello, world!");
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Byte-level BPE tokenizer compatible with GPT-2 / HuggingFace format.
pub struct BpeTokenizer {
    /// Token string → token ID
    encoder: HashMap<String, u32>,
    /// Token ID → token string
    decoder: Vec<String>,
    /// Ordered BPE merge rules: (token_a, token_b) → priority (lower = higher priority)
    merges: HashMap<(String, String), usize>,
    /// Byte value → Unicode character (GPT-2 byte encoding)
    byte_encoder: [char; 256],
    /// Unicode character → byte value
    byte_decoder: HashMap<char, u8>,
}

impl BpeTokenizer {
    /// Create a tokenizer from HuggingFace vocab.json and merges.txt files.
    ///
    /// - `vocab_json`: maps token strings to integer IDs
    /// - `merges_txt`: one merge rule per line ("token_a token_b"), first line is a header
    pub fn from_files<P: AsRef<Path>>(vocab_json: P, merges_txt: P) -> Result<Self, String> {
        let vocab_str = fs::read_to_string(vocab_json.as_ref())
            .map_err(|e| format!("Failed to read vocab.json: {}", e))?;
        let merges_str = fs::read_to_string(merges_txt.as_ref())
            .map_err(|e| format!("Failed to read merges.txt: {}", e))?;

        Self::from_strings(&vocab_str, &merges_str)
    }

    /// Create a tokenizer from raw vocab JSON and merges text content.
    pub fn from_strings(vocab_json: &str, merges_txt: &str) -> Result<Self, String> {
        let encoder = parse_vocab_json(vocab_json)?;

        // Build decoder (ID → token)
        let max_id = encoder.values().copied().max().unwrap_or(0) as usize;
        let mut decoder = vec![String::new(); max_id + 1];
        for (token, &id) in &encoder {
            decoder[id as usize] = token.clone();
        }

        // Parse merges
        let merges = parse_merges(merges_txt)?;

        // Build byte encoder/decoder
        let byte_encoder = build_byte_encoder();
        let mut byte_decoder = HashMap::new();
        for (byte_val, &ch) in byte_encoder.iter().enumerate() {
            byte_decoder.insert(ch, byte_val as u8);
        }

        Ok(BpeTokenizer {
            encoder,
            decoder,
            merges,
            byte_encoder,
            byte_decoder,
        })
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut token_ids = Vec::new();

        // Pre-tokenize: split text into words using GPT-2 pattern
        let words = pre_tokenize(text);

        for word in words {
            // Convert each byte in the word to its Unicode representation
            let byte_encoded: String = word.bytes().map(|b| self.byte_encoder[b as usize]).collect();

            // Split into individual characters as initial BPE tokens
            let mut tokens: Vec<String> = byte_encoded.chars().map(|c| c.to_string()).collect();

            // Apply BPE merges
            tokens = self.apply_bpe(tokens);

            // Look up token IDs
            for token in &tokens {
                if let Some(&id) = self.encoder.get(token) {
                    token_ids.push(id);
                }
                // Unknown tokens are silently skipped (GPT-2 byte-level BPE
                // should never produce unknown tokens since every byte is covered)
            }
        }

        token_ids
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        // Concatenate token strings
        let token_str: String = ids
            .iter()
            .filter_map(|&id| self.decoder.get(id as usize))
            .cloned()
            .collect();

        // Convert byte-encoded Unicode chars back to bytes
        let bytes: Vec<u8> = token_str
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect();

        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Look up a token's ID, if it exists.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.encoder.get(token).copied()
    }

    /// Look up an ID's token string, if it exists.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.decoder.get(id as usize).map(|s| s.as_str())
    }

    /// Apply BPE merges to a sequence of tokens.
    fn apply_bpe(&self, mut tokens: Vec<String>) -> Vec<String> {
        if tokens.len() < 2 {
            return tokens;
        }

        loop {
            // Find the highest-priority merge (lowest rank) among adjacent pairs
            let mut best_merge: Option<(usize, usize)> = None; // (position, rank)

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    match best_merge {
                        None => best_merge = Some((i, rank)),
                        Some((_, best_rank)) if rank < best_rank => {
                            best_merge = Some((i, rank))
                        }
                        _ => {}
                    }
                }
            }

            match best_merge {
                None => break, // No more merges possible
                Some((pos, _)) => {
                    // Merge the pair at position `pos`
                    let merged = format!("{}{}", tokens[pos], tokens[pos + 1]);
                    tokens[pos] = merged;
                    tokens.remove(pos + 1);
                }
            }
        }

        tokens
    }

    /// Create a simple tokenizer from a list of tokens (for testing).
    /// Tokens are assigned IDs 0, 1, 2, ... in order.
    pub fn from_vocab_and_merges(
        vocab: &[&str],
        merge_rules: &[(&str, &str)],
    ) -> Self {
        let mut encoder = HashMap::new();
        let mut decoder = Vec::new();
        for (id, &token) in vocab.iter().enumerate() {
            encoder.insert(token.to_string(), id as u32);
            decoder.push(token.to_string());
        }

        let mut merges = HashMap::new();
        for (rank, &(a, b)) in merge_rules.iter().enumerate() {
            merges.insert((a.to_string(), b.to_string()), rank);
        }

        let byte_encoder = build_byte_encoder();
        let mut byte_decoder = HashMap::new();
        for (byte_val, &ch) in byte_encoder.iter().enumerate() {
            byte_decoder.insert(ch, byte_val as u8);
        }

        BpeTokenizer {
            encoder,
            decoder,
            merges,
            byte_encoder,
            byte_decoder,
        }
    }
}

/// Build the GPT-2 byte encoder mapping.
///
/// Maps each byte value (0-255) to a Unicode character. Printable ASCII bytes
/// map to themselves. Non-printable and extended bytes are mapped to Unicode
/// characters starting at U+0100 to avoid control characters.
fn build_byte_encoder() -> [char; 256] {
    let mut encoder = ['\0'; 256];
    let mut offset = 0u32;

    for byte_val in 0u16..256 {
        let ch = if is_printable_byte(byte_val as u8) {
            // Printable bytes map to themselves
            char::from(byte_val as u8)
        } else {
            // Non-printable bytes get mapped to Unicode chars starting at U+0100
            let mapped = 256u32 + offset;
            offset += 1;
            char::from_u32(mapped).unwrap()
        };
        encoder[byte_val as usize] = ch;
    }

    encoder
}

/// Check if a byte is "printable" in the GPT-2 sense.
/// This matches the original OpenAI implementation:
/// - '!' (33) through '~' (126)
/// - '¡' (161) through '¬' (172)
/// - '®' (174) through 'ÿ' (255)
fn is_printable_byte(b: u8) -> bool {
    (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b)
}

/// Pre-tokenize text using GPT-2's word-splitting pattern.
///
/// Splits on:
/// - Contractions ('s, 't, 're, 've, 'm, 'll, 'd)
/// - Sequences of letters
/// - Sequences of digits
/// - Sequences of non-letter/non-digit/non-space characters
/// - Individual whitespace characters (preserved, not stripped)
fn pre_tokenize(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Check for contractions: 's, 't, 're, 've, 'm, 'll, 'd
        if chars[i] == '\'' && i + 1 < len {
            if let Some(contraction) = match_contraction(&chars, i) {
                words.push(contraction.0);
                i += contraction.1;
                continue;
            }
        }

        // Whitespace: each whitespace char is its own token (preserving leading space for next word)
        if chars[i].is_whitespace() {
            // A space followed by a letter sequence forms one token: " word"
            if chars[i] == ' ' && i + 1 < len && chars[i + 1].is_alphabetic() {
                let start = i;
                i += 1; // skip space
                while i < len && chars[i].is_alphabetic() {
                    i += 1;
                }
                words.push(chars[start..i].iter().collect());
                continue;
            }
            // A space followed by digits forms one token: " 123"
            if chars[i] == ' ' && i + 1 < len && chars[i + 1].is_ascii_digit() {
                let start = i;
                i += 1;
                while i < len && chars[i].is_ascii_digit() {
                    i += 1;
                }
                words.push(chars[start..i].iter().collect());
                continue;
            }
            // Lone space or other whitespace
            words.push(chars[i].to_string());
            i += 1;
            continue;
        }

        // Letters
        if chars[i].is_alphabetic() {
            let start = i;
            while i < len && chars[i].is_alphabetic() {
                i += 1;
            }
            words.push(chars[start..i].iter().collect());
            continue;
        }

        // Digits
        if chars[i].is_ascii_digit() {
            let start = i;
            while i < len && chars[i].is_ascii_digit() {
                i += 1;
            }
            words.push(chars[start..i].iter().collect());
            continue;
        }

        // Punctuation / symbols: each one is its own token
        words.push(chars[i].to_string());
        i += 1;
    }

    words
}

/// Match a contraction starting at position `i` (where chars[i] == '\'').
/// Returns (contraction_string, length) if matched.
fn match_contraction(chars: &[char], i: usize) -> Option<(String, usize)> {
    let remaining = chars.len() - i;

    if remaining >= 3 {
        let two_char: String = chars[i + 1..i + 3].iter().collect();
        let two_lower = two_char.to_lowercase();
        if two_lower == "re" || two_lower == "ve" || two_lower == "ll" {
            return Some((chars[i..i + 3].iter().collect(), 3));
        }
    }

    if remaining >= 2 {
        let one_char = chars[i + 1].to_lowercase().to_string();
        if one_char == "s" || one_char == "t" || one_char == "m" || one_char == "d" {
            return Some((chars[i..i + 2].iter().collect(), 2));
        }
    }

    None
}

/// Parse a vocab.json file (minimal JSON parser for flat string→int objects).
fn parse_vocab_json(json: &str) -> Result<HashMap<String, u32>, String> {
    let mut map = HashMap::new();
    let json = json.trim();

    if !json.starts_with('{') || !json.ends_with('}') {
        return Err("vocab.json must be a JSON object".to_string());
    }

    let inner = &json[1..json.len() - 1];
    let mut chars = inner.chars().peekable();

    loop {
        // Skip whitespace
        skip_whitespace(&mut chars);

        if chars.peek().is_none() {
            break;
        }

        // Parse key (string)
        let key = parse_json_string(&mut chars)?;

        // Skip whitespace and colon
        skip_whitespace(&mut chars);
        match chars.next() {
            Some(':') => {}
            other => return Err(format!("Expected ':', got {:?}", other)),
        }

        // Skip whitespace
        skip_whitespace(&mut chars);

        // Parse value (integer)
        let value = parse_json_int(&mut chars)?;

        map.insert(key, value);

        // Skip whitespace and optional comma
        skip_whitespace(&mut chars);
        if chars.peek() == Some(&',') {
            chars.next();
        }
    }

    Ok(map)
}

/// Parse a JSON string (handling escape sequences).
fn parse_json_string(chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<String, String> {
    match chars.next() {
        Some('"') => {}
        other => return Err(format!("Expected '\"', got {:?}", other)),
    }

    let mut s = String::new();
    loop {
        match chars.next() {
            Some('"') => return Ok(s),
            Some('\\') => {
                match chars.next() {
                    Some('"') => s.push('"'),
                    Some('\\') => s.push('\\'),
                    Some('/') => s.push('/'),
                    Some('n') => s.push('\n'),
                    Some('r') => s.push('\r'),
                    Some('t') => s.push('\t'),
                    Some('u') => {
                        // Parse 4-digit hex unicode escape
                        let hex: String = chars.take(4).collect();
                        if hex.len() != 4 {
                            return Err("Incomplete unicode escape".to_string());
                        }
                        let code = u32::from_str_radix(&hex, 16)
                            .map_err(|_| format!("Invalid unicode escape: \\u{}", hex))?;
                        if let Some(c) = char::from_u32(code) {
                            s.push(c);
                        } else {
                            return Err(format!("Invalid unicode code point: {}", code));
                        }
                    }
                    Some(c) => {
                        s.push('\\');
                        s.push(c);
                    }
                    None => return Err("Unexpected end of string".to_string()),
                }
            }
            Some(c) => s.push(c),
            None => return Err("Unexpected end of string".to_string()),
        }
    }
}

/// Parse a JSON integer.
fn parse_json_int(chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<u32, String> {
    let mut num_str = String::new();

    // Handle negative sign (shouldn't appear in vocab but handle gracefully)
    if chars.peek() == Some(&'-') {
        num_str.push(chars.next().unwrap());
    }

    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            num_str.push(c);
            chars.next();
        } else {
            break;
        }
    }

    num_str
        .parse::<u32>()
        .map_err(|_| format!("Failed to parse integer: '{}'", num_str))
}

/// Skip whitespace characters in an iterator.
fn skip_whitespace(chars: &mut std::iter::Peekable<std::str::Chars>) {
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else {
            break;
        }
    }
}

/// Parse a merges.txt file.
/// First line is typically a header ("#version: 0.2"), skip it.
/// Each subsequent line is "token_a token_b".
fn parse_merges(merges_txt: &str) -> Result<HashMap<(String, String), usize>, String> {
    let mut merges = HashMap::new();

    for (rank, line) in merges_txt.lines().enumerate() {
        // Skip header line
        if line.starts_with('#') || line.starts_with("version") {
            continue;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Split on the FIRST space only (tokens themselves might... actually no,
        // BPE tokens in merges.txt are always space-separated pairs)
        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() != 2 {
            continue; // Skip malformed lines
        }

        merges.insert((parts[0].to_string(), parts[1].to_string()), rank);
    }

    Ok(merges)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Byte encoder tests ----

    #[test]
    fn test_byte_encoder_covers_all_256_bytes() {
        let enc = build_byte_encoder();
        // All 256 entries should be unique characters
        let mut seen = std::collections::HashSet::new();
        for &ch in enc.iter() {
            assert!(ch != '\0', "Byte encoder has null character");
            assert!(seen.insert(ch), "Duplicate character in byte encoder: {:?}", ch);
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn test_byte_encoder_printable_identity() {
        let enc = build_byte_encoder();
        // Printable ASCII should map to themselves
        assert_eq!(enc[b'A' as usize], 'A');
        assert_eq!(enc[b'z' as usize], 'z');
        assert_eq!(enc[b'0' as usize], '0');
        assert_eq!(enc[b'!' as usize], '!');
        assert_eq!(enc[b'~' as usize], '~');
    }

    #[test]
    fn test_byte_encoder_nonprintable_mapped() {
        let enc = build_byte_encoder();
        // Space (32), null (0), tab (9) are non-printable in GPT-2 sense
        // They should be mapped to Unicode chars >= U+0100
        assert!(enc[0] as u32 >= 256, "Null byte should map to high Unicode");
        assert!(enc[32] as u32 >= 256, "Space should map to high Unicode");
        assert!(enc[9] as u32 >= 256, "Tab should map to high Unicode");
    }

    #[test]
    fn test_byte_encoder_roundtrip() {
        let enc = build_byte_encoder();
        let mut dec = HashMap::new();
        for (i, &ch) in enc.iter().enumerate() {
            dec.insert(ch, i as u8);
        }

        // Roundtrip every byte
        for byte_val in 0u8..=255 {
            let encoded = enc[byte_val as usize];
            let decoded = dec[&encoded];
            assert_eq!(decoded, byte_val, "Roundtrip failed for byte {}", byte_val);
        }
    }

    // ---- Pre-tokenization tests ----

    #[test]
    fn test_pre_tokenize_simple_words() {
        let words = pre_tokenize("Hello world");
        assert_eq!(words, vec!["Hello", " world"]);
    }

    #[test]
    fn test_pre_tokenize_punctuation() {
        let words = pre_tokenize("Hello, world!");
        assert_eq!(words, vec!["Hello", ",", " world", "!"]);
    }

    #[test]
    fn test_pre_tokenize_numbers() {
        let words = pre_tokenize("I have 42 cats");
        assert_eq!(words, vec!["I", " have", " 42", " cats"]);
    }

    #[test]
    fn test_pre_tokenize_contractions() {
        let words = pre_tokenize("I'm don't we'll they've");
        assert_eq!(words, vec!["I", "'m", " don", "'t", " we", "'ll", " they", "'ve"]);
    }

    #[test]
    fn test_pre_tokenize_mixed() {
        let words = pre_tokenize("Hello! It's 2024.");
        assert_eq!(words, vec!["Hello", "!", " It", "'s", " 2024", "."]);
    }

    #[test]
    fn test_pre_tokenize_empty() {
        let words = pre_tokenize("");
        assert!(words.is_empty());
    }

    #[test]
    fn test_pre_tokenize_whitespace_only() {
        let words = pre_tokenize("   ");
        assert_eq!(words, vec![" ", " ", " "]);
    }

    #[test]
    fn test_pre_tokenize_single_char() {
        let words = pre_tokenize("A");
        assert_eq!(words, vec!["A"]);
    }

    // ---- BPE merge tests ----

    #[test]
    fn test_apply_bpe_no_merges() {
        let tokenizer = BpeTokenizer::from_vocab_and_merges(
            &["a", "b", "c"],
            &[],
        );
        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let result = tokenizer.apply_bpe(tokens.clone());
        assert_eq!(result, tokens);
    }

    #[test]
    fn test_apply_bpe_single_merge() {
        let tokenizer = BpeTokenizer::from_vocab_and_merges(
            &["a", "b", "c", "ab"],
            &[("a", "b")],
        );
        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let result = tokenizer.apply_bpe(tokens);
        assert_eq!(result, vec!["ab", "c"]);
    }

    #[test]
    fn test_apply_bpe_cascading_merges() {
        let tokenizer = BpeTokenizer::from_vocab_and_merges(
            &["a", "b", "c", "ab", "abc"],
            &[("a", "b"), ("ab", "c")],
        );
        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let result = tokenizer.apply_bpe(tokens);
        assert_eq!(result, vec!["abc"]);
    }

    #[test]
    fn test_apply_bpe_priority_order() {
        // If both (a,b) and (b,c) are merges, the one with lower rank wins
        let tokenizer = BpeTokenizer::from_vocab_and_merges(
            &["a", "b", "c", "ab", "bc"],
            &[("b", "c"), ("a", "b")], // b+c has rank 0 (higher priority)
        );
        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let result = tokenizer.apply_bpe(tokens);
        // b+c merges first (rank 0), then a + bc has no merge rule → ["a", "bc"]
        assert_eq!(result, vec!["a", "bc"]);
    }

    #[test]
    fn test_apply_bpe_single_token() {
        let tokenizer = BpeTokenizer::from_vocab_and_merges(
            &["a"],
            &[("a", "b")],
        );
        let tokens = vec!["a".to_string()];
        let result = tokenizer.apply_bpe(tokens);
        assert_eq!(result, vec!["a"]);
    }

    // ---- JSON parsing tests ----

    #[test]
    fn test_parse_vocab_json_simple() {
        let json = r#"{"hello": 0, "world": 1, "foo": 2}"#;
        let vocab = parse_vocab_json(json).unwrap();
        assert_eq!(vocab.get("hello"), Some(&0));
        assert_eq!(vocab.get("world"), Some(&1));
        assert_eq!(vocab.get("foo"), Some(&2));
        assert_eq!(vocab.len(), 3);
    }

    #[test]
    fn test_parse_vocab_json_escaped_strings() {
        let json = r#"{"\"quoted\"": 0, "back\\slash": 1, "new\nline": 2}"#;
        let vocab = parse_vocab_json(json).unwrap();
        assert_eq!(vocab.get("\"quoted\""), Some(&0));
        assert_eq!(vocab.get("back\\slash"), Some(&1));
        assert_eq!(vocab.get("new\nline"), Some(&2));
    }

    #[test]
    fn test_parse_vocab_json_unicode_escape() {
        let json = r#"{"\\u0120": 0}"#;
        // This should parse as the literal string "\u0120" (escaped backslash + u0120)
        // In GPT-2 vocab, keys like "Ġ" (U+0120) are common
        let vocab = parse_vocab_json(json).unwrap();
        assert_eq!(vocab.len(), 1);
    }

    #[test]
    fn test_parse_vocab_json_empty() {
        let json = "{}";
        let vocab = parse_vocab_json(json).unwrap();
        assert!(vocab.is_empty());
    }

    #[test]
    fn test_parse_vocab_json_large_ids() {
        let json = r#"{"a": 50256, "b": 50257}"#;
        let vocab = parse_vocab_json(json).unwrap();
        assert_eq!(vocab.get("a"), Some(&50256));
        assert_eq!(vocab.get("b"), Some(&50257));
    }

    // ---- Merges parsing tests ----

    #[test]
    fn test_parse_merges_basic() {
        let merges = "#version: 0.2\na b\nab c\n";
        let result = parse_merges(merges).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains_key(&("a".to_string(), "b".to_string())));
        assert!(result.contains_key(&("ab".to_string(), "c".to_string())));
    }

    #[test]
    fn test_parse_merges_preserves_rank_order() {
        let merges = "#version: 0.2\na b\nc d\ne f\n";
        let result = parse_merges(merges).unwrap();
        // First merge (a b) should have lowest rank
        let rank_ab = result[&("a".to_string(), "b".to_string())];
        let rank_cd = result[&("c".to_string(), "d".to_string())];
        let rank_ef = result[&("e".to_string(), "f".to_string())];
        assert!(rank_ab < rank_cd);
        assert!(rank_cd < rank_ef);
    }

    #[test]
    fn test_parse_merges_skip_empty_lines() {
        let merges = "#version: 0.2\n\na b\n\nc d\n\n";
        let result = parse_merges(merges).unwrap();
        assert_eq!(result.len(), 2);
    }

    // ---- Encode/decode integration tests ----

    #[test]
    fn test_encode_decode_roundtrip_simple() {
        // Build a minimal tokenizer with byte-level encoding
        let byte_enc = build_byte_encoder();

        // Create vocab entries for individual byte-encoded characters
        let mut vocab_entries: Vec<String> = Vec::new();
        for &ch in byte_enc.iter() {
            vocab_entries.push(ch.to_string());
        }

        // Build vocab JSON
        let mut json = String::from("{");
        for (id, entry) in vocab_entries.iter().enumerate() {
            if id > 0 {
                json.push_str(", ");
            }
            // Escape the token for JSON
            let escaped = entry
                .replace('\\', "\\\\")
                .replace('"', "\\\"");
            json.push_str(&format!("\"{}\": {}", escaped, id));
        }
        json.push('}');

        // No merges
        let merges = "#version: 0.2\n";

        let tokenizer = BpeTokenizer::from_strings(&json, merges).unwrap();

        // Each byte should get its own token
        let text = "Hi";
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_vocab_size() {
        let tokenizer = BpeTokenizer::from_vocab_and_merges(
            &["a", "b", "c", "ab"],
            &[("a", "b")],
        );
        assert_eq!(tokenizer.vocab_size(), 4);
    }

    #[test]
    fn test_token_to_id_and_back() {
        let tokenizer = BpeTokenizer::from_vocab_and_merges(
            &["hello", "world"],
            &[],
        );
        assert_eq!(tokenizer.token_to_id("hello"), Some(0));
        assert_eq!(tokenizer.token_to_id("world"), Some(1));
        assert_eq!(tokenizer.token_to_id("missing"), None);
        assert_eq!(tokenizer.id_to_token(0), Some("hello"));
        assert_eq!(tokenizer.id_to_token(1), Some("world"));
        assert_eq!(tokenizer.id_to_token(99), None);
    }

    #[test]
    fn test_encode_empty_string() {
        let tokenizer = BpeTokenizer::from_vocab_and_merges(&["a"], &[]);
        let encoded = tokenizer.encode("");
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_decode_empty() {
        let tokenizer = BpeTokenizer::from_vocab_and_merges(&["a"], &[]);
        let decoded = tokenizer.decode(&[]);
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_bpe_with_repeated_patterns() {
        // "aababab" with merge (a, b) → ab
        let tokenizer = BpeTokenizer::from_vocab_and_merges(
            &["a", "b", "ab"],
            &[("a", "b")],
        );
        let tokens = vec![
            "a".to_string(), "a".to_string(), "b".to_string(),
            "a".to_string(), "b".to_string(), "a".to_string(), "b".to_string(),
        ];
        let result = tokenizer.apply_bpe(tokens);
        assert_eq!(result, vec!["a", "ab", "ab", "ab"]);
    }

    #[test]
    fn test_pre_tokenize_preserves_leading_space() {
        let words = pre_tokenize(" hello");
        // Space followed by word forms one token
        assert_eq!(words, vec![" hello"]);
    }

    #[test]
    fn test_pre_tokenize_multiple_spaces() {
        let words = pre_tokenize("a  b");
        assert_eq!(words, vec!["a", " ", " b"]);
    }

    #[test]
    fn test_is_printable_byte() {
        // Printable range: 33-126, 161-172, 174-255
        assert!(is_printable_byte(b'A'));
        assert!(is_printable_byte(b'~'));
        assert!(is_printable_byte(b'!'));
        assert!(!is_printable_byte(b' ')); // space is not printable in GPT-2
        assert!(!is_printable_byte(0));     // null
        assert!(!is_printable_byte(9));     // tab
        assert!(!is_printable_byte(10));    // newline
        assert!(!is_printable_byte(173));   // soft hyphen (gap in printable range)
    }

    #[test]
    fn test_contraction_detection() {
        let chars: Vec<char> = "'s test".chars().collect();
        let result = match_contraction(&chars, 0);
        assert!(result.is_some());
        let (s, len) = result.unwrap();
        assert_eq!(s, "'s");
        assert_eq!(len, 2);
    }

    #[test]
    fn test_no_contraction_at_end() {
        let chars: Vec<char> = "'".chars().collect();
        let result = match_contraction(&chars, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_vocab_json_invalid() {
        let result = parse_vocab_json("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_bpe_merge_rounds() {
        // Build: a+b→ab, ab+c→abc, abc+d→abcd
        let tokenizer = BpeTokenizer::from_vocab_and_merges(
            &["a", "b", "c", "d", "ab", "abc", "abcd"],
            &[("a", "b"), ("ab", "c"), ("abc", "d")],
        );
        let tokens = vec![
            "a".to_string(), "b".to_string(),
            "c".to_string(), "d".to_string(),
        ];
        let result = tokenizer.apply_bpe(tokens);
        assert_eq!(result, vec!["abcd"]);
    }

    #[test]
    fn test_bpe_non_adjacent_merges() {
        // "a b a b" with merge (a, b) → both pairs should merge
        let tokenizer = BpeTokenizer::from_vocab_and_merges(
            &["a", "b", "ab", "x"],
            &[("a", "b")],
        );
        let tokens = vec![
            "a".to_string(), "b".to_string(),
            "x".to_string(),
            "a".to_string(), "b".to_string(),
        ];
        let result = tokenizer.apply_bpe(tokens);
        assert_eq!(result, vec!["ab", "x", "ab"]);
    }
}
