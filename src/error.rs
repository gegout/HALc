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

use thiserror::Error;

/// Custom error types representing failure modes within HALc.
#[derive(Error, Debug)]
pub enum HalcError {
    /// Wrapper for standard input/output operation errors.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Wrapper for TOML parsing errors in configuration loading.
    #[error("TOML parsing error: {0}")]
    Toml(#[from] toml::de::Error),

    /// Wrapper for JSON parsing or serialization errors.
    #[error("JSON parsing/serialization error: {0}")]
    Json(#[from] serde_json::Error),

    /// Wrapper for API network query failures.
    #[error("HTTP request error: {0}")]
    Http(#[from] reqwest::Error),

    /// Configuration errors (missing elements, invalid inputs).
    #[error("Configuration error: {0}")]
    Config(String),

    /// Agent-level validation or connectivity checks failure.
    #[error("Agent validation failed: {0}")]
    AgentValidation(String),

    /// Errors returned directly by the OpenAI or LLM compatible server.
    #[error("OpenAI API error: {0}")]
    OpenAiApi(String),

    /// Failure occurring inside the constructive confrontation stages.
    #[error("Confrontation algorithm failure: {0}")]
    Algorithm(String),

    /// Fallback error catcher wrapping anyhow.
    #[error("Unknown/other error: {0}")]
    Other(#[from] anyhow::Error),
}

/// Generic Result type alias for HALc operations.
pub type Result<T> = std::result::Result<T, HalcError>;
