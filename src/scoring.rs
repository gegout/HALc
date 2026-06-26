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

use crate::agent::{AgentRuntime, Challenge, Proposal, ProposalAnalysis};
use crate::config::{NormalizationConfig, Verbosity};
use crate::openai::ScoreTableEntry;
use std::collections::HashMap;
use tracing::{info, warn};

/// Aggregates review scores assigned by other agents, filtering out self-scores and failed agents.
pub fn calculate_scores(
    agents: &[AgentRuntime],
    challenges: &HashMap<(String, String), Challenge>,
    failed_agents: &[String],
) -> Vec<ScoreTableEntry> {
    let mut score_table = Vec::new();
    for agent in agents {
        let name = &agent.config.name;
        // A proposal from a failed agent must not be selected or scored.
        if failed_agents.contains(name) {
            continue;
        }
        // Sum scores from all other active agents reviewing this agent
        let total_score: i32 = challenges
            .values()
            .filter(|c| {
                &c.target_agent == name
                    && &c.reviewer_agent != name
                    && !failed_agents.contains(&c.reviewer_agent)
            })
            .map(|c| c.score)
            .sum();
        score_table.push(ScoreTableEntry {
            agent: name.clone(),
            value: total_score,
        });
    }
    // Sort descending by value (highest scoring proposals first)
    score_table.sort_by_key(|b| std::cmp::Reverse(b.value));
    score_table
}

/// Selects the winning agent's proposal using three levels of tie-breaking:
/// 1. Lowest average risk count (Stage 3 analysis).
/// 2. Configure judge agent choice.
/// 3. Deterministic first choice fallback.
pub async fn select_winner(
    user_request: &str,
    proposals: &HashMap<String, Proposal>,
    analyses: &HashMap<String, ProposalAnalysis>,
    score_table: &[ScoreTableEntry],
    judge_agent: Option<&AgentRuntime>,
    norm_config: &NormalizationConfig,
    verbosity: Verbosity,
    tie_breaker_template: &str,
) -> (String, i32) {
    if score_table.is_empty() {
        return (String::new(), 0);
    }

    // Find highest score value
    let max_score = score_table
        .iter()
        .map(|entry| entry.value)
        .max()
        .unwrap_or(0);
    let tied_entries: Vec<&ScoreTableEntry> = score_table
        .iter()
        .filter(|entry| entry.value == max_score)
        .collect();

    // If no tie, return the winner
    if tied_entries.len() == 1 {
        return (tied_entries[0].agent.clone(), max_score);
    }

    info!(
        "Tie detected between: {:?} with score {}",
        tied_entries, max_score
    );

    // Tie-break 1: Prefer proposal with lowest average risk count.
    let mut min_risk = usize::MAX;
    let mut tied_after_risk = Vec::new();

    for entry in &tied_entries {
        let risk_count = analyses
            .get(&entry.agent)
            .map(|a| a.risks.len())
            .unwrap_or(0);
        if risk_count < min_risk {
            min_risk = risk_count;
            tied_after_risk = vec![entry];
        } else if risk_count == min_risk {
            tied_after_risk.push(entry);
        }
    }

    if tied_after_risk.len() == 1 {
        info!(
            "Tie broken by lowest average risk count ({} risks) for agent: {}",
            min_risk, tied_after_risk[0].agent
        );
        return (tied_after_risk[0].agent.clone(), max_score);
    }

    info!(
        "Tie still exists after risk assessment between: {:?}",
        tied_after_risk
    );

    // Tie-break 2: Ask configured judge agent if present.
    if let Some(judge) = judge_agent {
        info!(
            "Invoking judge agent '{}' to break the tie.",
            judge.config.name
        );

        let tied_proposals_info: Vec<String> = tied_after_risk
            .iter()
            .map(|entry| {
                let prop_text = proposals
                    .get(&entry.agent)
                    .map(|p| p.proposal.as_str())
                    .unwrap_or("");
                format!("Agent '{}':\nProposal: {}\n", entry.agent, prop_text)
            })
            .collect();

        let prompt = crate::prompts::render(
            tie_breaker_template,
            &[
                ("user_request", user_request),
                ("tied_proposals", &tied_proposals_info.join("\n---\n")),
            ],
        );

        let messages = vec![
            crate::openai::ChatCompletionMessage {
                role: "system".to_string(),
                content: judge.config.system_prompt.clone(),
            },
            crate::openai::ChatCompletionMessage {
                role: "user".to_string(),
                content: prompt,
            },
        ];

        match judge.query_markdown(messages, verbosity).await {
            Ok(res) => {
                let doc = crate::markdown::MarkdownDocument::parse(&res);
                if let Some(selected_content) = doc.get_section(&norm_config.winning_agent) {
                    let selected = selected_content.trim();
                    let cleaned_selected = selected.trim_matches(|c: char| {
                        c.is_whitespace() || c == '`' || c == '*' || c == '_' || c == '\n'
                    });
                    if tied_after_risk.iter().any(|e| e.agent == cleaned_selected) {
                        info!("Judge selected winner: {}", cleaned_selected);
                        return (cleaned_selected.to_string(), max_score);
                    } else {
                        warn!("Judge selected agent '{}', which was not in the tie list. Defaulting to first.", cleaned_selected);
                    }
                } else {
                    warn!(
                        "Judge response did not contain the 'Winner' heading. Defaulting to first."
                    );
                }
            }
            Err(e) => {
                warn!(
                    "Judge agent failed to make a decision: {:?}. Defaulting to first.",
                    e
                );
            }
        }
    }

    // Tie-break 3: Select first highest-scoring proposal deterministically.
    let fallback = tied_after_risk[0].agent.clone();
    info!("Defaulting to first deterministic agent: {}", fallback);
    (fallback, max_score)
}
