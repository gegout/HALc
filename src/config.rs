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

use crate::error::{HalcError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Verbosity level of the logging and CLI output.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Deserialize, serde::Serialize,
)]
pub enum Verbosity {
    Normal,
    Verbose,
    Debug,
}

/// Configured parameters for a single LLM agent.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AgentConfig {
    /// Unique identifier for the agent (e.g. "architect", "security").
    pub name: String,

    /// Target API server URL (OpenAI-compatible).
    pub endpoint_url: String,

    /// Target model (e.g. "gpt-4", "llama-3").
    pub model: String,

    /// Authorization bearer token key.
    pub api_key: String,

    /// Instructions shaping agent behavior and perspective.
    pub system_prompt: String,

    /// If true, this agent acts as a tie-breaker judge in confrontations.
    #[serde(default)]
    pub judge: bool,

    /// If true, this agent is a proposer. Defaults to true if judge is false, and false if judge is true.
    #[serde(default)]
    pub proposer: Option<bool>,

    /// Optional additional fields sent to the endpoint (e.g. temperature, presence_penalty).
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Semantic alias mappings to normalize variations in Markdown headings.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct NormalizationConfig {
    #[serde(default = "default_alias_problem_statement")]
    pub problem_statement: Vec<String>,
    #[serde(default = "default_alias_constraints")]
    pub constraints: Vec<String>,
    #[serde(default = "default_alias_success_metrics")]
    pub success_metrics: Vec<String>,
    #[serde(default = "default_alias_deadline")]
    pub deadline: Vec<String>,
    #[serde(default = "default_alias_clarifying_assumptions")]
    pub clarifying_assumptions: Vec<String>,
    #[serde(default = "default_alias_proposal")]
    pub proposal: Vec<String>,
    #[serde(default = "default_alias_rationale")]
    pub rationale: Vec<String>,
    #[serde(default = "default_alias_expected_benefits")]
    pub expected_benefits: Vec<String>,
    #[serde(default = "default_alias_known_risks")]
    pub known_risks: Vec<String>,
    #[serde(default = "default_alias_expected_outcomes")]
    pub expected_outcomes: Vec<String>,
    #[serde(default = "default_alias_claims")]
    pub claims: Vec<String>,
    #[serde(default = "default_alias_assumptions")]
    pub assumptions: Vec<String>,
    #[serde(default = "default_alias_risks")]
    pub risks: Vec<String>,
    #[serde(default = "default_alias_evidence")]
    pub evidence: Vec<String>,
    #[serde(default = "default_alias_challenge")]
    pub challenge: Vec<String>,
    #[serde(default = "default_alias_strengths")]
    pub strengths: Vec<String>,
    #[serde(default = "default_alias_weaknesses")]
    pub weaknesses: Vec<String>,
    #[serde(default = "default_alias_missing_evidence")]
    pub missing_evidence: Vec<String>,
    #[serde(default = "default_alias_risk_assessment")]
    pub risk_assessment: Vec<String>,
    #[serde(default = "default_alias_score")]
    pub score: Vec<String>,
    #[serde(default = "default_alias_new_context")]
    pub new_context: Vec<String>,
    #[serde(default = "default_alias_revised_position")]
    pub revised_position: Vec<String>,
    #[serde(default = "default_alias_lessons_learned")]
    pub lessons_learned: Vec<String>,
    #[serde(default = "default_alias_future_considerations")]
    pub future_considerations: Vec<String>,
    #[serde(default = "default_alias_winning_agent")]
    pub winning_agent: Vec<String>,
    #[serde(default = "default_alias_priorities_used")]
    pub priorities_used: Vec<String>,
    #[serde(default = "default_alias_loser")]
    pub loser: Vec<String>,
    #[serde(default = "default_alias_confidence")]
    pub confidence: Vec<String>,
    #[serde(default = "default_alias_priority_alignment")]
    pub priority_alignment: Vec<String>,
    #[serde(default = "default_alias_risks_in_loser")]
    pub risks_in_loser: Vec<String>,
    #[serde(default = "default_alias_risks_in_winner")]
    pub risks_in_winner: Vec<String>,
    #[serde(default = "default_alias_reasoning")]
    pub reasoning: Vec<String>,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            problem_statement: default_alias_problem_statement(),
            constraints: default_alias_constraints(),
            success_metrics: default_alias_success_metrics(),
            deadline: default_alias_deadline(),
            clarifying_assumptions: default_alias_clarifying_assumptions(),
            proposal: default_alias_proposal(),
            rationale: default_alias_rationale(),
            expected_benefits: default_alias_expected_benefits(),
            known_risks: default_alias_known_risks(),
            expected_outcomes: default_alias_expected_outcomes(),
            claims: default_alias_claims(),
            assumptions: default_alias_assumptions(),
            risks: default_alias_risks(),
            evidence: default_alias_evidence(),
            challenge: default_alias_challenge(),
            strengths: default_alias_strengths(),
            weaknesses: default_alias_weaknesses(),
            missing_evidence: default_alias_missing_evidence(),
            risk_assessment: default_alias_risk_assessment(),
            score: default_alias_score(),
            new_context: default_alias_new_context(),
            revised_position: default_alias_revised_position(),
            lessons_learned: default_alias_lessons_learned(),
            future_considerations: default_alias_future_considerations(),
            winning_agent: default_alias_winning_agent(),
            priorities_used: default_alias_priorities_used(),
            loser: default_alias_loser(),
            confidence: default_alias_confidence(),
            priority_alignment: default_alias_priority_alignment(),
            risks_in_loser: default_alias_risks_in_loser(),
            risks_in_winner: default_alias_risks_in_winner(),
            reasoning: default_alias_reasoning(),
        }
    }
}

fn default_alias_problem_statement() -> Vec<String> {
    vec![
        "problem statement".to_string(),
        "problem".to_string(),
        "framing".to_string(),
        "goal".to_string(),
    ]
}
fn default_alias_constraints() -> Vec<String> {
    vec![
        "constraints".to_string(),
        "limitations".to_string(),
        "rules".to_string(),
    ]
}
fn default_alias_success_metrics() -> Vec<String> {
    vec![
        "success metrics".to_string(),
        "metrics".to_string(),
        "kpis".to_string(),
    ]
}
fn default_alias_deadline() -> Vec<String> {
    vec![
        "deadline".to_string(),
        "timeline".to_string(),
        "schedule".to_string(),
    ]
}
fn default_alias_clarifying_assumptions() -> Vec<String> {
    vec![
        "clarifying assumptions".to_string(),
        "assumptions".to_string(),
        "premises".to_string(),
    ]
}
fn default_alias_proposal() -> Vec<String> {
    vec![
        "proposal".to_string(),
        "solution".to_string(),
        "architecture".to_string(),
        "proposed solution".to_string(),
        "proposal overview".to_string(),
        "recommended solution".to_string(),
    ]
}
fn default_alias_rationale() -> Vec<String> {
    vec![
        "rationale".to_string(),
        "reasoning".to_string(),
        "justification".to_string(),
    ]
}
fn default_alias_expected_benefits() -> Vec<String> {
    vec![
        "expected benefits".to_string(),
        "benefits".to_string(),
        "advantages".to_string(),
        "pros".to_string(),
    ]
}
fn default_alias_known_risks() -> Vec<String> {
    vec![
        "known risks".to_string(),
        "risks".to_string(),
        "concerns".to_string(),
        "potential issues".to_string(),
        "potential risks".to_string(),
    ]
}
fn default_alias_expected_outcomes() -> Vec<String> {
    vec![
        "expected outcomes".to_string(),
        "outcomes".to_string(),
        "goals".to_string(),
    ]
}
fn default_alias_claims() -> Vec<String> {
    vec!["claims".to_string(), "assertions".to_string()]
}
fn default_alias_assumptions() -> Vec<String> {
    vec!["assumptions".to_string(), "premises".to_string()]
}
fn default_alias_risks() -> Vec<String> {
    vec![
        "risks".to_string(),
        "known risks".to_string(),
        "concerns".to_string(),
        "potential issues".to_string(),
    ]
}
fn default_alias_evidence() -> Vec<String> {
    vec![
        "evidence".to_string(),
        "proof".to_string(),
        "support".to_string(),
    ]
}
fn default_alias_challenge() -> Vec<String> {
    vec![
        "review".to_string(),
        "critique".to_string(),
        "challenge".to_string(),
    ]
}
fn default_alias_strengths() -> Vec<String> {
    vec![
        "strengths".to_string(),
        "advantages".to_string(),
        "pros".to_string(),
    ]
}
fn default_alias_weaknesses() -> Vec<String> {
    vec![
        "weaknesses".to_string(),
        "flaws".to_string(),
        "cons".to_string(),
    ]
}
fn default_alias_missing_evidence() -> Vec<String> {
    vec!["missing evidence".to_string(), "gaps".to_string()]
}
fn default_alias_risk_assessment() -> Vec<String> {
    vec![
        "risk assessment".to_string(),
        "risks".to_string(),
        "concerns".to_string(),
    ]
}
fn default_alias_score() -> Vec<String> {
    vec!["score".to_string(), "rating".to_string()]
}
fn default_alias_new_context() -> Vec<String> {
    vec![
        "new context".to_string(),
        "updated context".to_string(),
        "context".to_string(),
    ]
}
fn default_alias_revised_position() -> Vec<String> {
    vec!["revised position".to_string(), "position".to_string()]
}
fn default_alias_lessons_learned() -> Vec<String> {
    vec!["lessons learned".to_string(), "lessons".to_string()]
}
fn default_alias_future_considerations() -> Vec<String> {
    vec!["future considerations".to_string(), "future".to_string()]
}
fn default_alias_winning_agent() -> Vec<String> {
    vec!["winning agent".to_string(), "winner".to_string()]
}
fn default_alias_priorities_used() -> Vec<String> {
    vec![
        "priorities used".to_string(),
        "priorities".to_string(),
        "priorities_used".to_string(),
    ]
}
fn default_alias_loser() -> Vec<String> {
    vec!["loser".to_string(), "losing agent".to_string()]
}
fn default_alias_confidence() -> Vec<String> {
    vec![
        "confidence".to_string(),
        "score".to_string(),
        "rating".to_string(),
    ]
}
fn default_alias_priority_alignment() -> Vec<String> {
    vec![
        "priority alignment".to_string(),
        "priority_alignment".to_string(),
    ]
}
fn default_alias_risks_in_loser() -> Vec<String> {
    vec!["risks in loser".to_string(), "risks_in_loser".to_string()]
}
fn default_alias_risks_in_winner() -> Vec<String> {
    vec!["risks in winner".to_string(), "risks_in_winner".to_string()]
}
fn default_alias_reasoning() -> Vec<String> {
    vec![
        "reasoning".to_string(),
        "rationale".to_string(),
        "justification".to_string(),
    ]
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ConfrontationConfig {
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default = "default_max_response_chars")]
    pub max_response_chars: usize,
}

impl Default for ConfrontationConfig {
    fn default() -> Self {
        Self {
            mode: default_mode(),
            max_response_chars: default_max_response_chars(),
        }
    }
}

fn default_mode() -> String {
    "compact_pairwise".to_string()
}

fn default_max_response_chars() -> usize {
    1000
}

/// Paths to the six external Markdown prompt template files.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PromptsConfig {
    /// Prompt 1 — Startup context handshake.
    #[serde(default = "default_prompt_context_init")]
    pub context_init: String,
    /// Prompt 2 — Connectivity ping.
    #[serde(default = "default_prompt_connectivity_test")]
    pub connectivity_test: String,
    /// Prompt 3 — Stage 1 agent synthesis.
    #[serde(default = "default_prompt_synthesis")]
    pub synthesis: String,
    /// Prompt 4 — Stage 3 pairwise comparison (includes retry section).
    #[serde(default = "default_prompt_pairwise_comparison")]
    pub pairwise_comparison: String,
    /// Prompt 5 — Stage 4 tie-breaker judge call.
    #[serde(default = "default_prompt_tie_breaker")]
    pub tie_breaker: String,
    /// Prompt 6 — Stage 6 shared context generation judge call.
    #[serde(default = "default_prompt_context_generation")]
    pub context_generation: String,
}

impl Default for PromptsConfig {
    fn default() -> Self {
        Self {
            context_init: default_prompt_context_init(),
            connectivity_test: default_prompt_connectivity_test(),
            synthesis: default_prompt_synthesis(),
            pairwise_comparison: default_prompt_pairwise_comparison(),
            tie_breaker: default_prompt_tie_breaker(),
            context_generation: default_prompt_context_generation(),
        }
    }
}

fn default_prompt_context_init() -> String {
    "~/.config/HALc/prompts/context_init.md".to_string()
}
fn default_prompt_connectivity_test() -> String {
    "~/.config/HALc/prompts/connectivity_test.md".to_string()
}
fn default_prompt_synthesis() -> String {
    "~/.config/HALc/prompts/synthesis.md".to_string()
}
fn default_prompt_pairwise_comparison() -> String {
    "~/.config/HALc/prompts/pairwise_comparison.md".to_string()
}
fn default_prompt_tie_breaker() -> String {
    "~/.config/HALc/prompts/tie_breaker.md".to_string()
}
fn default_prompt_context_generation() -> String {
    "~/.config/HALc/prompts/context_generation.md".to_string()
}

/// Core application configuration container containing all configured agents.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AppConfig {
    /// Vector of participant agents.
    pub agents: Vec<AgentConfig>,

    /// The API key required to access the gateway server's OpenAI endpoints.
    #[serde(default = "default_gateway_api_key")]
    pub gateway_api_key: String,

    /// Directory where the execution ledger logs are saved.
    #[serde(default = "default_ledger_directory")]
    pub ledger_directory: String,

    /// Configurable semantic alias normalization mapping for headings.
    #[serde(default)]
    pub normalization: NormalizationConfig,

    /// Confrontation mode and length limits.
    #[serde(default)]
    pub confrontation: ConfrontationConfig,

    /// Paths to external Markdown prompt template files.
    #[serde(default)]
    pub prompts: PromptsConfig,
}

fn default_gateway_api_key() -> String {
    "1234567890".to_string()
}

fn default_ledger_directory() -> String {
    "~/.config/HALc/ledger".to_string()
}

impl AppConfig {
    /// Attempts to read and deserialize configuration files from the local filesystem.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: AppConfig = toml::from_str(&content)?;
        if config.agents.is_empty() {
            return Err(HalcError::Config(
                "At least one agent must be configured in agent.toml".to_string(),
            ));
        }
        Ok(config)
    }
}
