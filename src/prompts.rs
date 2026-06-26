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

//! Prompt template loading and rendering system.
//!
//! Each prompt lives in a standalone Markdown file referenced from `agent.toml`.
//! Template variables use the `{{variable_name}}` syntax and are replaced at
//! call time via [`render`].  Changing a prompt never requires recompilation.

use crate::config::PromptsConfig;
use crate::error::{HalcError, Result};
use std::path::PathBuf;

/// The marker string that separates the main pairwise comparison prompt from
/// the retry addendum in `pairwise_comparison.md`.
pub const RETRY_MARKER: &str = "<!-- RETRY_PROMPT -->";

/// Loaded text of every agent-facing prompt template.
#[derive(Debug, Clone)]
pub struct PromptTemplates {
    /// Prompt 1 — Startup context handshake (`context_init.md`).
    pub context_init: String,
    /// Prompt 2 — Connectivity ping (`connectivity_test.md`).
    pub connectivity_test: String,
    /// Prompt 3 — Stage 1 agent synthesis (`synthesis.md`).
    pub synthesis: String,
    /// Prompt 4 — Stage 3 pairwise comparison, including retry section (`pairwise_comparison.md`).
    pub pairwise_comparison: String,
    /// Prompt 5 — Stage 4 tie-breaker for the judge (`tie_breaker.md`).
    pub tie_breaker: String,
    /// Prompt 6 — Stage 6 shared context generation for the judge (`context_generation.md`).
    pub context_generation: String,
}

impl PromptTemplates {
    /// Read all prompt templates from the paths declared in `PromptsConfig`.
    ///
    /// Returns an error if any file cannot be read.
    pub fn load(config: &PromptsConfig) -> Result<Self> {
        Ok(Self {
            context_init: read_template(&config.context_init)?,
            connectivity_test: read_template(&config.connectivity_test)?,
            synthesis: read_template(&config.synthesis)?,
            pairwise_comparison: read_template(&config.pairwise_comparison)?,
            tie_breaker: read_template(&config.tie_breaker)?,
            context_generation: read_template(&config.context_generation)?,
        })
    }

    /// Return the main body of the pairwise comparison prompt (before the
    /// `<!-- RETRY_PROMPT -->` marker).
    pub fn pairwise_main(&self) -> &str {
        match self.pairwise_comparison.split_once(RETRY_MARKER) {
            Some((main, _)) => main.trim_end(),
            None => self.pairwise_comparison.as_str(),
        }
    }

    /// Return the retry addendum (after the `<!-- RETRY_PROMPT -->` marker),
    /// or a safe built-in fallback if the marker is absent.
    pub fn pairwise_retry(&self) -> &str {
        match self.pairwise_comparison.split_once(RETRY_MARKER) {
            Some((_, retry)) => retry.trim_start(),
            None => "Your previous comparison was invalid. Please output a valid markdown comparison with a clear # Winner and # Loser.",
        }
    }
}

/// Replace every `{{key}}` occurrence in `template` with the matching value.
///
/// `vars` is a slice of `(key, value)` pairs.  Unknown keys are left as-is.
pub fn render(template: &str, vars: &[(&str, &str)]) -> String {
    let mut out = template.to_string();
    for (key, value) in vars {
        out = out.replace(&format!("{{{{{}}}}}", key), value);
    }
    out
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Expand `~/` prefix and read the file at `path`.
fn read_template(path: &str) -> Result<String> {
    let full = expand_home(path);
    std::fs::read_to_string(&full).map_err(|e| {
        HalcError::Config(format!(
            "Cannot read prompt template '{}': {}",
            full.display(),
            e
        ))
    })
}

/// Resolve `~/…` paths relative to `$HOME`.
fn expand_home(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(stripped);
        }
    }
    PathBuf::from(path)
}
