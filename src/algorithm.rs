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

use crate::agent::{AgentRuntime, PairwiseComparison, Synthesis};
use crate::config::{NormalizationConfig, Verbosity};
use crate::error::{HalcError, Result};
use crate::markdown::MarkdownDocument;
use crate::openai::Decision;
use chrono::Utc;
use colored::Colorize;
use futures::StreamExt;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tracing::{error, info, warn};

// --- Ledger Implementation ---

/// Thread-safe logger that appends confrontation events to a JSON Lines (JSONL) file.
#[derive(Debug, Clone)]
pub struct Ledger {
    file_path: Option<String>,
    lock: Arc<tokio::sync::Mutex<()>>,
}

impl Ledger {
    /// Creates a new Ledger writing to the optional file path.
    pub fn new(file_path: Option<String>) -> Self {
        Self {
            file_path,
            lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// Appends a structured log entry containing timestamps, stage name, agent, and payload.
    pub async fn log(
        &self,
        request_id: &str,
        stage: &str,
        agent: Option<&str>,
        payload: serde_json::Value,
    ) {
        let path = match &self.file_path {
            Some(p) => p,
            None => return,
        };

        let entry = json!({
            "timestamp": Utc::now().to_rfc3339(),
            "request_id": request_id,
            "stage": stage,
            "agent": agent.unwrap_or("system"),
            "payload": payload,
        });

        let line = match serde_json::to_string(&entry) {
            Ok(l) => l + "\n",
            Err(e) => {
                error!("Failed to serialize ledger entry: {:?}", e);
                return;
            }
        };

        let _guard = self.lock.lock().await;
        if let Err(e) = Self::append_to_file(path, &line).await {
            error!("Failed to write to ledger file {}: {:?}", path, e);
        }
    }

    async fn append_to_file(path: &str, data: &str) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        file.write_all(data.as_bytes()).await?;
        file.flush().await?;
        Ok(())
    }

    /// Writes a confrontation summary markdown file to the ledger directory.
    pub async fn write_summary(&self, filename: &str, content: &str) -> std::io::Result<()> {
        if let Some(ref path_str) = self.file_path {
            let path = std::path::Path::new(path_str);
            if let Some(parent) = path.parent() {
                let dest = parent.join(filename);
                tokio::fs::write(&dest, content).await?;
                info!(
                    "Confrontation summary stored to: {}",
                    dest.to_string_lossy()
                );
            }
        }
        Ok(())
    }

    /// Writes a shared context markdown file to the ledger directory and returns its absolute path.
    pub async fn write_context_file(
        &self,
        filename: &str,
        content: &str,
    ) -> std::io::Result<String> {
        let dest = if let Some(ref path_str) = self.file_path {
            let path = std::path::Path::new(path_str);
            if let Some(parent) = path.parent() {
                let dest = parent.join(filename);
                tokio::fs::write(&dest, content).await?;
                dest.to_string_lossy().to_string()
            } else {
                tokio::fs::write(filename, content).await?;
                filename.to_string()
            }
        } else {
            let default_dir = std::env::var("HOME")
                .map(|h| {
                    std::path::PathBuf::from(h)
                        .join(".config")
                        .join("HALc")
                        .join("ledger")
                })
                .unwrap_or_else(|_| std::path::PathBuf::from("."));
            let _ = std::fs::create_dir_all(&default_dir);
            let dest = default_dir.join(filename);
            tokio::fs::write(&dest, content).await?;
            dest.to_string_lossy().to_string()
        };
        Ok(dest)
    }
}

// --- Confrontation Loop ---

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RankingEntry {
    pub agent: String,
    pub weighted_copeland_score: i32,
    pub wins: usize,
    pub losses: usize,
    pub total_win_weight: i32,
    pub total_loss_weight: i32,
}

async fn query_with_limit(
    agent: &AgentRuntime,
    messages: Vec<crate::openai::ChatCompletionMessage>,
    max_response_chars: usize,
    verbosity: Verbosity,
) -> Result<String> {
    let response = agent.query_markdown(messages, verbosity).await?;
    if response.len() > max_response_chars {
        warn!(
            "Agent '{}' response exceeded max limit of {} characters (actual length: {}).",
            agent.config.name,
            max_response_chars,
            response.len()
        );
    }
    Ok(response)
}

pub struct ConfrontationParams<'a> {
    pub normalization: &'a NormalizationConfig,
    pub confrontation: &'a crate::config::ConfrontationConfig,
    pub templates: &'a crate::prompts::PromptTemplates,
}

async fn perform_comparison(
    reviewer: &AgentRuntime,
    agent_a: &Synthesis,
    agent_b: &Synthesis,
    user_request: &str,
    max_chars: usize,
    norm: &NormalizationConfig,
    verbosity: Verbosity,
    templates: &crate::prompts::PromptTemplates,
) -> Result<PairwiseComparison> {
    let reviewer_context = reviewer
        .context_store
        .get(&reviewer.config.name)
        .await
        .unwrap_or_default();
    let max_chars_str = max_chars.to_string();
    let get_messages = || {
        let prompt = crate::prompts::render(
            templates.pairwise_main(),
            &[
                ("reviewer_name", &reviewer.config.name),
                ("shared_context", &reviewer_context),
                ("user_request", user_request),
                ("agent_a_name", &agent_a.agent),
                ("agent_a_synthesis", &agent_a.raw_markdown),
                ("agent_b_name", &agent_b.agent),
                ("agent_b_synthesis", &agent_b.raw_markdown),
                ("max_chars", &max_chars_str),
            ],
        );
        vec![
            crate::openai::ChatCompletionMessage {
                role: "system".to_string(),
                content: reviewer.config.system_prompt.clone(),
            },
            crate::openai::ChatCompletionMessage {
                role: "user".to_string(),
                content: prompt,
            },
        ]
    };

    let mut response = query_with_limit(reviewer, get_messages(), max_chars, verbosity).await?;
    let mut doc = MarkdownDocument::parse(&response);
    let mut comp =
        PairwiseComparison::parse_markdown(&doc, norm, &reviewer.config.name, response.clone());

    let is_valid = |c: &PairwiseComparison| {
        let w = c.winner.trim().to_lowercase();
        let l = c.loser.trim().to_lowercase();
        let a = agent_a.agent.trim().to_lowercase();
        let b = agent_b.agent.trim().to_lowercase();
        (w == a && l == b) || (w == b && l == a)
    };

    if !is_valid(&comp) {
        warn!(
            "Agent '{}' produced an invalid pairwise comparison (Winner: '{}', Loser: '{}'). Retrying once.",
            reviewer.config.name, comp.winner, comp.loser
        );
        let mut retry_messages = get_messages();
        retry_messages.push(crate::openai::ChatCompletionMessage {
            role: "assistant".to_string(),
            content: response,
        });
        let retry_text = crate::prompts::render(
            templates.pairwise_retry(),
            &[
                ("agent_a_name", &agent_a.agent),
                ("agent_b_name", &agent_b.agent),
            ],
        );
        retry_messages.push(crate::openai::ChatCompletionMessage {
            role: "user".to_string(),
            content: retry_text,
        });
        response = query_with_limit(reviewer, retry_messages, max_chars, verbosity).await?;
        doc = MarkdownDocument::parse(&response);
        comp = PairwiseComparison::parse_markdown(&doc, norm, &reviewer.config.name, response);
        if !is_valid(&comp) {
            return Err(HalcError::Algorithm(format!(
                "Agent '{}' failed validation twice. Comparison was invalid.",
                reviewer.config.name
            )));
        }
    }

    Ok(comp)
}

/// Runs the Compact Distributed Pairwise Confrontation algorithm between configured agents.
pub async fn run_confrontation(
    request_id: &str,
    user_request: &str,
    agents: &[AgentRuntime],
    ledger: &Ledger,
    progress_tx: Option<tokio::sync::mpsc::Sender<String>>,
    verbosity: Verbosity,
    params: &ConfrontationParams<'_>,
) -> Result<Decision> {
    let max_chars = params.confrontation.max_response_chars;

    if verbosity >= Verbosity::Normal {
        println!(
            "{}",
            "HALc — Compact Distributed Pairwise Confrontation"
                .bold()
                .cyan()
        );
        println!("\nRequest");
        println!("└── {}", user_request.yellow());
    }

    // Helper closure to push logs to progress channel
    let log_progress = |msg: &str| {
        if let Some(tx) = &progress_tx {
            let tx_clone = tx.clone();
            let msg_clone = msg.to_string();
            tokio::spawn(async move {
                let _ = tx_clone.send(msg_clone).await;
            });
        }
    };

    // Filter proposer agents: judge agent is ignored as proposer unless explicitly marked
    let proposer_agents: Vec<&AgentRuntime> = agents
        .iter()
        .filter(|a| {
            let is_proposer_default = !a.config.judge;
            a.config.proposer.unwrap_or(is_proposer_default)
        })
        .collect();

    if proposer_agents.len() < 3 {
        return Err(HalcError::Algorithm(
            "Compact distributed pairwise confrontation requires at least 3 proposer agents."
                .to_string(),
        ));
    }

    // ==========================================
    // Stage 1: Agent synthesis
    // ==========================================
    log_progress("[Stage 1/6] Agent synthesis...");
    if verbosity >= Verbosity::Normal {
        println!("\nStage 1 — Agent synthesis");
    }

    let mut syntheses = Vec::new();
    let mut stage1_futures = futures::stream::FuturesUnordered::new();

    for (idx, agent) in proposer_agents.iter().enumerate() {
        let u_req = user_request.to_string();
        let agent_name = agent.config.name.clone();
        let system_prompt = agent.config.system_prompt.clone();
        let norm_clone = params.normalization.clone();
        let agent_clone = (*agent).clone();
        let synthesis_template = params.templates.synthesis.clone();
        let max_chars_str = max_chars.to_string();

        stage1_futures.push(async move {
            let start_time = Instant::now();
            let shared_context = agent_clone
                .context_store
                .get(&agent_name)
                .await
                .unwrap_or_default();
            let prompt = crate::prompts::render(
                &synthesis_template,
                &[
                    ("agent_name", &agent_name),
                    ("system_prompt", &system_prompt),
                    ("shared_context", &shared_context),
                    ("user_request", &u_req),
                    ("max_chars", &max_chars_str),
                ],
            );

            let messages = vec![
                crate::openai::ChatCompletionMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                },
                crate::openai::ChatCompletionMessage {
                    role: "user".to_string(),
                    content: prompt,
                },
            ];

            let res = query_with_limit(&agent_clone, messages, max_chars, verbosity).await;
            let duration = start_time.elapsed();
            (agent_name, idx, res, duration, norm_clone)
        });
    }

    let mut completed_synthesis_count = 0;
    let proposer_count = proposer_agents.len();

    while let Some((name, _idx, res, duration, norm)) = stage1_futures.next().await {
        completed_synthesis_count += 1;
        let is_last = completed_synthesis_count == proposer_count;
        let prefix = if is_last { "└──" } else { "├──" };
        let indent = if is_last { "    " } else { "│   " };

        match res {
            Ok(markdown_raw) => {
                let doc = MarkdownDocument::parse(&markdown_raw);
                let synth = Synthesis::parse_markdown(&doc, &norm, &name, markdown_raw);

                ledger
                    .log(
                        request_id,
                        "compact_synthesis",
                        Some(&name),
                        serde_json::to_value(&synth)?,
                    )
                    .await;

                if verbosity >= Verbosity::Normal {
                    println!("{} {}", prefix, name.cyan());
                    println!(
                        "{}├── Priorities: {}",
                        indent,
                        synth.priorities_used.join(", ").yellow()
                    );
                    println!(
                        "{}├── Risks: {}",
                        indent,
                        synth.risks.len().to_string().red()
                    );
                    let time_str = if verbosity >= Verbosity::Verbose {
                        format!(" ({:.1?}s)", duration.as_secs_f32())
                    } else {
                        "".to_string()
                    };
                    println!("{}└── ✓ completed{}", indent, time_str);
                }
                syntheses.push(synth);
            }
            Err(e) => {
                if verbosity >= Verbosity::Normal {
                    println!("{} {} - ✘ Failed: {:?}", prefix, name.red(), e);
                }
            }
        }
    }

    // Filter proposer agents that successfully generated synthesis
    let successful_proposers: Vec<&AgentRuntime> = proposer_agents
        .iter()
        .filter(|pa| syntheses.iter().any(|s| s.agent == pa.config.name))
        .cloned()
        .collect();

    let n = successful_proposers.len();
    if n < 3 {
        return Err(HalcError::Algorithm(
            "Compact distributed pairwise confrontation requires at least 3 proposer agents."
                .to_string(),
        ));
    }

    // ==========================================
    // Stage 2: Deterministic pairwise assignment
    // ==========================================
    log_progress("[Stage 2/6] Deterministic pairwise assignment...");
    if verbosity >= Verbosity::Normal {
        println!("\nStage 2 — Pairwise assignment");
    }

    let mut assignments = Vec::new();
    for i in 0..n {
        let reviewer = successful_proposers[i];
        let agent_a = successful_proposers[(i + 1) % n];
        let agent_b = successful_proposers[(i + 2) % n];
        assignments.push((
            reviewer.config.name.clone(),
            agent_a.config.name.clone(),
            agent_b.config.name.clone(),
        ));

        if verbosity >= Verbosity::Normal {
            let is_last = i == n - 1;
            let prefix = if is_last { "└──" } else { "├──" };
            println!(
                "{} {} reviews {} vs {}",
                prefix,
                reviewer.config.name.cyan(),
                agent_a.config.name.cyan(),
                agent_b.config.name.cyan()
            );
        }
    }

    ledger
        .log(
            request_id,
            "pairwise_assignment",
            None,
            serde_json::json!({ "assignments": assignments }),
        )
        .await;

    // ==========================================
    // Stage 3: Pairwise comparison
    // ==========================================
    log_progress("[Stage 3/6] Pairwise comparison...");
    if verbosity >= Verbosity::Normal {
        println!("\nStage 3 — Pairwise comparison");
    }

    let mut comparisons = Vec::new();
    let mut stage3_futures = futures::stream::FuturesUnordered::new();

    for i in 0..n {
        let reviewer = successful_proposers[i];
        let agent_a_name = &successful_proposers[(i + 1) % n].config.name;
        let agent_b_name = &successful_proposers[(i + 2) % n].config.name;

        let agent_a_synth = syntheses
            .iter()
            .find(|s| &s.agent == agent_a_name)
            .unwrap()
            .clone();
        let agent_b_synth = syntheses
            .iter()
            .find(|s| &s.agent == agent_b_name)
            .unwrap()
            .clone();

        let reviewer_clone = reviewer.clone();
        let norm_clone = params.normalization.clone();
        let u_req = user_request.to_string();
        let templates_clone = params.templates.clone();

        stage3_futures.push(async move {
            let start_time = Instant::now();
            let res = perform_comparison(
                &reviewer_clone,
                &agent_a_synth,
                &agent_b_synth,
                &u_req,
                max_chars,
                &norm_clone,
                verbosity,
                &templates_clone,
            )
            .await;
            let duration = start_time.elapsed();
            (reviewer_clone.config.name.clone(), res, duration)
        });
    }

    let mut completed_comparisons = 0;
    while let Some((reviewer_name, res, duration)) = stage3_futures.next().await {
        completed_comparisons += 1;
        let is_last = completed_comparisons == n;
        let prefix = if is_last { "└──" } else { "├──" };

        match res {
            Ok(comp) => {
                if verbosity >= Verbosity::Normal {
                    let time_str = if verbosity >= Verbosity::Verbose {
                        format!(" ({:.1?}s)", duration.as_secs_f32())
                    } else {
                        "".to_string()
                    };
                    println!(
                        "{} {}: {} > {}, confidence {}{}",
                        prefix,
                        comp.reviewer.cyan(),
                        comp.winner.cyan(),
                        comp.loser.cyan(),
                        comp.confidence,
                        time_str
                    );
                }

                ledger
                    .log(
                        request_id,
                        "pairwise_comparison",
                        Some(&reviewer_name),
                        serde_json::to_value(&comp)?,
                    )
                    .await;

                comparisons.push(comp);
            }
            Err(e) => {
                if verbosity >= Verbosity::Normal {
                    println!(
                        "{} reviewer {} reviews - ✘ Failed: {:?}",
                        prefix,
                        reviewer_name.cyan(),
                        e
                    );
                }
            }
        }
    }

    // ==========================================
    // Stage 4: Weighted Copeland ranking
    // ==========================================
    log_progress("[Stage 4/6] Weighted Copeland ranking...");

    let mut wins_count = HashMap::new();
    let mut losses_count = HashMap::new();
    let mut total_win_weight = HashMap::new();
    let mut total_loss_weight = HashMap::new();
    let mut copeland_score = HashMap::new();

    for pa in &successful_proposers {
        let name = pa.config.name.clone();
        wins_count.insert(name.clone(), 0);
        losses_count.insert(name.clone(), 0);
        total_win_weight.insert(name.clone(), 0);
        total_loss_weight.insert(name.clone(), 0);
        copeland_score.insert(name, 0);
    }

    for comp in &comparisons {
        let w = &comp.winner;
        let l = &comp.loser;
        let conf = comp.confidence;

        *wins_count.entry(w.clone()).or_insert(0) += 1;
        *losses_count.entry(l.clone()).or_insert(0) += 1;
        *total_win_weight.entry(w.clone()).or_insert(0) += conf;
        *total_loss_weight.entry(l.clone()).or_insert(0) += conf;
    }

    for pa in &successful_proposers {
        let name = &pa.config.name;
        let win_w = total_win_weight.get(name).copied().unwrap_or(0);
        let loss_w = total_loss_weight.get(name).copied().unwrap_or(0);
        copeland_score.insert(name.clone(), win_w - loss_w);
    }

    let mut ranking: Vec<RankingEntry> = successful_proposers
        .iter()
        .map(|pa| {
            let name = pa.config.name.clone();
            RankingEntry {
                agent: name.clone(),
                weighted_copeland_score: copeland_score.get(&name).copied().unwrap_or(0),
                wins: wins_count.get(&name).copied().unwrap_or(0),
                losses: losses_count.get(&name).copied().unwrap_or(0),
                total_win_weight: total_win_weight.get(&name).copied().unwrap_or(0),
                total_loss_weight: total_loss_weight.get(&name).copied().unwrap_or(0),
            }
        })
        .collect();

    let agent_to_index: HashMap<String, usize> = successful_proposers
        .iter()
        .enumerate()
        .map(|(idx, pa)| (pa.config.name.clone(), idx))
        .collect();

    let agent_to_risks: HashMap<String, usize> = syntheses
        .iter()
        .map(|s| (s.agent.clone(), s.risks.len()))
        .collect();

    ranking.sort_by(|a, b| {
        let c = b.weighted_copeland_score.cmp(&a.weighted_copeland_score);
        if c != std::cmp::Ordering::Equal {
            return c;
        }

        let c = b.total_win_weight.cmp(&a.total_win_weight);
        if c != std::cmp::Ordering::Equal {
            return c;
        }

        let c = a.losses.cmp(&b.losses);
        if c != std::cmp::Ordering::Equal {
            return c;
        }

        let a_risks = agent_to_risks.get(&a.agent).copied().unwrap_or(0);
        let b_risks = agent_to_risks.get(&b.agent).copied().unwrap_or(0);
        let c = a_risks.cmp(&b_risks);
        if c != std::cmp::Ordering::Equal {
            return c;
        }

        let a_idx = agent_to_index.get(&a.agent).copied().unwrap_or(usize::MAX);
        let b_idx = agent_to_index.get(&b.agent).copied().unwrap_or(usize::MAX);
        a_idx.cmp(&b_idx)
    });

    if verbosity >= Verbosity::Normal {
        println!("\nStage 4 — Weighted Copeland ranking");
        for (idx, r) in ranking.iter().enumerate() {
            let is_last = idx == ranking.len() - 1;
            let prefix = if is_last { "└──" } else { "├──" };
            println!(
                "{} {}: +{} -{} = {}",
                prefix,
                r.agent.cyan(),
                r.total_win_weight,
                r.total_loss_weight,
                r.weighted_copeland_score
            );
        }
    }

    ledger
        .log(
            request_id,
            "weighted_copeland_ranking",
            None,
            serde_json::json!({ "ranking": ranking }),
        )
        .await;

    // ==========================================
    // Stage 5: Final response
    // ==========================================
    log_progress("[Stage 5/6] Final decision...");

    let winner_name = &ranking[0].agent;
    let winning_score = ranking[0].weighted_copeland_score;
    let winning_synthesis = syntheses.iter().find(|s| &s.agent == winner_name).unwrap();

    if verbosity >= Verbosity::Normal {
        println!("\nStage 5 — Final decision");
        println!("└── {} wins", winner_name.cyan().bold());
    }

    let mut why_it_won_lines = Vec::new();
    for comp in &comparisons {
        if &comp.winner == winner_name {
            why_it_won_lines.push(format!(
                "* Reviewer '{}' chose '{}' over '{}' with confidence {}: {}",
                comp.reviewer, comp.winner, comp.loser, comp.confidence, comp.reasoning
            ));
        }
    }
    let why_it_won = if why_it_won_lines.is_empty() {
        "Determined by Copeland score.".to_string()
    } else {
        why_it_won_lines.join("\n")
    };

    let mut ranking_lines = Vec::new();
    for (idx, entry) in ranking.iter().enumerate() {
        ranking_lines.push(format!(
            "{}. {} — {}",
            idx + 1,
            entry.agent,
            entry.weighted_copeland_score
        ));
    }
    let ranking_str = ranking_lines.join("\n");

    let main_risks = if winning_synthesis.risks.is_empty() {
        "None identified.".to_string()
    } else {
        winning_synthesis
            .risks
            .iter()
            .map(|r| format!("* {}", r))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let mut summary_lines = Vec::new();
    for comp in &comparisons {
        summary_lines.push(format!(
            "* Reviewer '{}': Winner '{}' vs Loser '{}' (Confidence: {})\n  - Alignment: {}\n  - Reasoning: {}",
            comp.reviewer, comp.winner, comp.loser, comp.confidence, comp.priority_alignment, comp.reasoning
        ));
    }
    let confrontation_summary = summary_lines.join("\n\n");

    let final_markdown = format!(
        "# Final Decision\n\n\
         ## Winner\n\
         {}\n\n\
         ## Winning Proposal\n\
         {}\n\n\
         ## Why It Won\n\
         {}\n\n\
         ## Ranking\n\
         {}\n\n\
         ## Main Risks\n\
         {}\n\n\
         ## Confrontation Summary\n\
         {}",
        winner_name,
        winning_synthesis.proposal,
        why_it_won,
        ranking_str,
        main_risks,
        confrontation_summary
    );

    let score_table: Vec<crate::openai::ScoreTableEntry> = ranking
        .iter()
        .map(|r| crate::openai::ScoreTableEntry {
            agent: r.agent.clone(),
            value: r.weighted_copeland_score,
        })
        .collect();

    let decision = Decision {
        winning_agent: winner_name.clone(),
        winning_score,
        proposal: winning_synthesis.proposal.clone(),
        rationale: winning_synthesis.rationale.clone(),
        risks: winning_synthesis.risks.clone(),
        score_table,
        markdown: final_markdown.clone(),
    };

    ledger
        .log(
            request_id,
            "compact_final_decision",
            None,
            serde_json::to_value(&decision)?,
        )
        .await;

    // ==========================================
    // Stage 6: Shared context generation
    // ==========================================
    log_progress("[Stage 6/6] Generating shared context...");

    // Get original context (e.g. from the first agent)
    let original_context = if let Some(first_agent) = agents.first() {
        first_agent
            .context_store
            .get(&first_agent.config.name)
            .await
            .unwrap_or_default()
    } else {
        "".to_string()
    };

    // Generate the new context generated by the algorithm (deterministic new context)
    let mut lessons = Vec::new();
    for s in &syntheses {
        lessons.push(format!("* {}: {}", s.agent, s.rationale));
    }
    let lessons_str = lessons.join("\n");

    let new_context = format!(
        "# Shared Knowledge\n\
         Consolidated knowledge from the confrontation run.\n\n\
         # User Request\n\
         {}\n\n\
         # Winning Proposal\n\
         {}\n\n\
         # Final Ranking\n\
         {}\n\n\
         # Pairwise Comparisons\n\
         {}\n\n\
         # Lessons Learned\n\
         {}\n\n\
         # Remaining Risks\n\
         {}\n\n\
         # Open Disagreements\n\
         None.\n\n\
         # Next Context For All Agents\n\
         This consolidated context serves as the basis for future confrontations.\n",
        user_request,
        winning_synthesis.proposal,
        ranking_str,
        confrontation_summary,
        lessons_str,
        main_risks
    );

    let mut shared_context_content = String::new();
    let judge_agent = agents.iter().find(|a| a.config.judge);

    if let Some(judge) = judge_agent {
        let max_chars_str = max_chars.to_string();
        let prompt_content = crate::prompts::render(
            &params.templates.context_generation,
            &[
                ("original_context", &original_context),
                ("new_context", &new_context),
                ("max_chars", &max_chars_str),
            ],
        );

        let messages = vec![
            crate::openai::ChatCompletionMessage {
                role: "system".to_string(),
                content: judge.config.system_prompt.clone(),
            },
            crate::openai::ChatCompletionMessage {
                role: "user".to_string(),
                content: prompt_content,
            },
        ];

        let judge_clone = judge.clone();
        let res = query_with_limit(&judge_clone, messages, max_chars, verbosity).await;
        match res {
            Ok(summary_md) => {
                shared_context_content = summary_md;
            }
            Err(e) => {
                warn!("Failed to query judge agent for shared context: {:?}. Falling back to deterministic context.", e);
            }
        }
    }

    if shared_context_content.trim().is_empty() {
        shared_context_content = new_context;
    }

    // Write context file context_<timestamp>.md
    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let context_filename = format!("context_{}.md", timestamp);
    let context_file_path = ledger
        .write_context_file(&context_filename, &shared_context_content)
        .await?;

    if verbosity >= Verbosity::Normal {
        println!("\nStage 6 — Shared context");
        println!("└── {}", context_file_path.cyan());
    }

    // Update in-memory shared context for all agents
    for agent in agents {
        agent
            .context_store
            .set(agent.config.name.clone(), shared_context_content.clone())
            .await;
    }

    ledger
        .log(
            request_id,
            "shared_context_generated",
            None,
            serde_json::json!({ "path": context_file_path }),
        )
        .await;

    Ok(decision)
}
