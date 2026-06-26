// MIT License
//
// Copyright (c) 2026 Cedric Gegout
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use std::collections::HashMap;

/// Representation of a parsed Markdown document structured by its headings.
#[derive(Debug, Clone)]
pub struct MarkdownDocument {
    /// Map of lowercased and trimmed heading titles to their respective section text.
    pub sections: HashMap<String, String>,
}

impl MarkdownDocument {
    /// Parses a raw Markdown string into structured sections based on ATX and Setext headings.
    pub fn parse(raw: &str) -> Self {
        let lines: Vec<&str> = raw.lines().map(|l| l.trim_end()).collect();
        let mut i = 0;
        let mut current_section = "introduction".to_string();
        let mut section_contents: HashMap<String, Vec<String>> = HashMap::new();

        while i < lines.len() {
            let line = lines[i];
            let trimmed = line.trim();

            let is_setext = if i + 1 < lines.len() && !trimmed.is_empty() {
                let next_line = lines[i + 1].trim();
                next_line.len() >= 3
                    && (next_line.chars().all(|c| c == '=') || next_line.chars().all(|c| c == '-'))
            } else {
                false
            };

            if trimmed.starts_with('#') {
                let hashes = trimmed.chars().take_while(|&c| c == '#').count();
                if hashes > 0 && hashes <= 6 {
                    let rest = &trimmed[hashes..];
                    if rest.starts_with(char::is_whitespace) || rest.is_empty() {
                        current_section = rest.trim().to_lowercase();
                        i += 1;
                        continue;
                    }
                }
            }

            if is_setext {
                current_section = trimmed.to_lowercase();
                i += 2;
                continue;
            }

            section_contents
                .entry(current_section.clone())
                .or_default()
                .push(line.to_string());

            i += 1;
        }

        let mut sections = HashMap::new();
        for (k, v) in section_contents {
            sections.insert(k, v.join("\n").trim().to_string());
        }

        Self { sections }
    }

    /// Gets the raw content of the first section matching any of the provided aliases.
    pub fn get_section(&self, aliases: &[String]) -> Option<&str> {
        for alias in aliases {
            let lower = alias.trim().to_lowercase();
            if let Some(content) = self.sections.get(&lower) {
                return Some(content);
            }
        }
        None
    }

    /// Extracts a list of strings from a section's content.
    /// Supports bullet points (*, -, +) and numbered lists (e.g., 1., 2.).
    /// If no list formatting prefix is found, it returns the non-empty lines directly.
    pub fn get_list(&self, aliases: &[String]) -> Vec<String> {
        let content = match self.get_section(aliases) {
            Some(c) => c,
            None => return Vec::new(),
        };

        let mut items = Vec::new();
        let lines = content.lines().map(|l| l.trim());

        for line in lines {
            if line.is_empty() {
                continue;
            }

            let mut item = None;
            if line.starts_with("- ") || line.starts_with("* ") || line.starts_with("+ ") {
                item = Some(line[2..].trim());
            } else if line.starts_with('-') || line.starts_with('*') || line.starts_with('+') {
                item = Some(line[1..].trim());
            } else if let Some(dot_idx) = line.find('.') {
                let prefix = &line[..dot_idx];
                if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()) {
                    item = Some(line[dot_idx + 1..].trim());
                }
            }

            if let Some(it) = item {
                if !it.is_empty() {
                    items.push(it.to_string());
                }
            } else {
                items.push(line.to_string());
            }
        }

        items
    }

    /// Extracts the first integer value found in the section's content.
    pub fn get_integer(&self, aliases: &[String]) -> Option<i32> {
        let content = self.get_section(aliases)?;
        let mut start_idx = None;

        for (i, c) in content.char_indices() {
            if c.is_ascii_digit() {
                if start_idx.is_none() {
                    start_idx = Some(i);
                }
            } else if let Some(start) = start_idx {
                let num_str = &content[start..i];
                if let Ok(val) = num_str.parse::<i32>() {
                    return Some(val);
                }
                start_idx = None;
            }
        }

        if let Some(start) = start_idx {
            let num_str = &content[start..];
            if let Ok(val) = num_str.parse::<i32>() {
                return Some(val);
            }
        }

        None
    }
}

/// Safely truncates a response to max_chars.
/// Attempts to truncate at a Markdown heading if possible, falling back to a hard character boundary.
pub fn truncate_response(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }
    let truncated_raw = &s[..max_chars];
    if let Some(last_hash) = truncated_raw.rfind("\n#") {
        if last_hash > 0 {
            return s[..last_hash].trim_end().to_string();
        }
    }
    if let Some(last_hash) = truncated_raw.rfind('#') {
        if last_hash == 0 || (last_hash > 0 && s.as_bytes()[last_hash - 1] == b'\n') {
            return s[..last_hash].trim_end().to_string();
        }
    }
    // Hard fallback: find char boundary
    let mut end = max_chars;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}
