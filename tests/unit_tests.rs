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
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

use halc::agent::{
    AgentRuntime, Challenge, PairwiseComparison, Proposal, ProposalAnalysis, Synthesis,
};
use halc::algorithm::RankingEntry;
use halc::config::{AgentConfig, AppConfig, ConfrontationConfig, NormalizationConfig, Verbosity};
use halc::context::AgentContextStore;
use halc::markdown::{truncate_response, MarkdownDocument};
use halc::openai::ScoreTableEntry;
use halc::scoring::{calculate_scores, select_winner};

#[test]
fn test_load_agent_toml() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("agent.toml");
    let mut file = File::create(&file_path).unwrap();

    let toml_content = r#"
[[agents]]
name = "architect"
endpoint_url = "https://api.openai.com/v1"
model = "gpt-4.1"
api_key = "sk-12345"
system_prompt = "You are an architecture agent."

[agents.parameters]
temperature = 0.3
max_tokens = 2000
custom_param = "hello"

[confrontation]
mode = "compact_pairwise"
max_response_chars = 1500
    "#;

    file.write_all(toml_content.as_bytes()).unwrap();

    let config = AppConfig::load_from_file(&file_path).unwrap();
    assert_eq!(config.agents.len(), 1);
    assert_eq!(config.agents[0].name, "architect");
    assert_eq!(config.confrontation.mode, "compact_pairwise");
    assert_eq!(config.confrontation.max_response_chars, 1500);
}

#[test]
fn test_max_response_chars_defaulting() {
    let config = ConfrontationConfig::default();
    assert_eq!(config.mode, "compact_pairwise");
    assert_eq!(config.max_response_chars, 1000);
}

#[test]
fn test_active_proposer_filtering() {
    let a1 = AgentConfig {
        name: "a1".to_string(),
        endpoint_url: "".to_string(),
        model: "".to_string(),
        api_key: "".to_string(),
        system_prompt: "".to_string(),
        judge: false,
        proposer: None,
        parameters: HashMap::new(),
    };
    let a2 = AgentConfig {
        name: "a2".to_string(),
        endpoint_url: "".to_string(),
        model: "".to_string(),
        api_key: "".to_string(),
        system_prompt: "".to_string(),
        judge: true,
        proposer: None,
        parameters: HashMap::new(),
    };
    let a3 = AgentConfig {
        name: "a3".to_string(),
        endpoint_url: "".to_string(),
        model: "".to_string(),
        api_key: "".to_string(),
        system_prompt: "".to_string(),
        judge: true,
        proposer: Some(true),
        parameters: HashMap::new(),
    };

    assert!(a1.proposer.unwrap_or(!a1.judge));
    assert!(!a2.proposer.unwrap_or(!a2.judge));
    assert!(a3.proposer.unwrap_or(!a3.judge));
}

#[test]
fn test_pairwise_assignment_logic() {
    // 3 proposer agents
    let n = 3;
    let mut assignments = Vec::new();
    for i in 0..n {
        let reviewer = i;
        let first = (i + 1) % n;
        let second = (i + 2) % n;
        assignments.push((reviewer, first, second));
    }
    // A0 reviews A1 vs A2
    // A1 reviews A2 vs A0
    // A2 reviews A0 vs A1
    assert_eq!(assignments[0], (0, 1, 2));
    assert_eq!(assignments[1], (1, 2, 0));
    assert_eq!(assignments[2], (2, 0, 1));

    // Verify reviewer never compares itself
    for (r, f, s) in &assignments {
        assert_ne!(r, f);
        assert_ne!(r, s);
        assert_ne!(f, s);
    }

    // Verify each agent appears exactly twice as proposal
    let mut counts = vec![0; n];
    for (_, f, s) in &assignments {
        counts[*f] += 1;
        counts[*s] += 1;
    }
    assert_eq!(counts, vec![2, 2, 2]);
}

#[test]
fn test_synthesis_markdown_parsing() {
    let md = r#"
# Problem Statement
Need a system.

# Priorities Used
- speed
- safety

# Constraints
- low cost

# Success Metrics
- 99.9% uptime

# Proposal
Use Kafka.

# Rationale
It is fast.

# Claims
- scalable

# Assumptions
- cluster is stable

# Risks
- network split
- data loss

# Evidence
- benchmark docs

# Expected Outcomes
- success
"#;
    let doc = MarkdownDocument::parse(md);
    let norm = NormalizationConfig::default();
    let synth = Synthesis::parse_markdown(&doc, &norm, "agent-1", md.to_string());

    assert_eq!(synth.agent, "agent-1");
    assert_eq!(synth.problem_statement, "Need a system.");
    assert_eq!(synth.priorities_used, vec!["speed", "safety"]);
    assert_eq!(synth.constraints, vec!["low cost"]);
    assert_eq!(synth.success_metrics, vec!["99.9% uptime"]);
    assert_eq!(synth.proposal, "Use Kafka.");
    assert_eq!(synth.rationale, "It is fast.");
    assert_eq!(synth.claims, vec!["scalable"]);
    assert_eq!(synth.assumptions, vec!["cluster is stable"]);
    assert_eq!(synth.risks, vec!["network split", "data loss"]);
    assert_eq!(synth.evidence, vec!["benchmark docs"]);
    assert_eq!(synth.expected_outcomes, vec!["success"]);
}

#[test]
fn test_pairwise_comparison_markdown_parsing_and_clamping() {
    let md = r#"
# Winner
agent-a

# Loser
agent-b

# Confidence
12

# Priority Alignment
matches priorities perfectly.

# Reasoning
better design.

# Risks In Loser
too slow.

# Risks In Winner
complex setup.
"#;
    let doc = MarkdownDocument::parse(md);
    let norm = NormalizationConfig::default();
    let comp = PairwiseComparison::parse_markdown(&doc, &norm, "reviewer-1", md.to_string());

    assert_eq!(comp.reviewer, "reviewer-1");
    assert_eq!(comp.winner, "agent-a");
    assert_eq!(comp.loser, "agent-b");
    assert_eq!(comp.confidence, 10); // clamped to 10
    assert_eq!(comp.priority_alignment, "matches priorities perfectly.");
    assert_eq!(comp.reasoning, "better design.");
    assert_eq!(comp.risks_in_loser, vec!["too slow."]);
    assert_eq!(comp.risks_in_winner, vec!["complex setup."]);
}

#[test]
fn test_response_truncation() {
    let s = "Hello world\n# Heading 2\nSome content";
    let truncated = truncate_response(s, 20);
    // Heading 2 starts at index 12, so it should truncate before Heading 2
    assert_eq!(truncated, "Hello world");

    let s2 = "Hello world without headings";
    let truncated2 = truncate_response(s2, 10);
    assert_eq!(truncated2, "Hello worl");
}

#[test]
fn test_weighted_copeland_sorting() {
    let mut ranking = [
        RankingEntry {
            agent: "agent-a".to_string(),
            weighted_copeland_score: 5,
            wins: 1,
            losses: 0,
            total_win_weight: 5,
            total_loss_weight: 0,
        },
        RankingEntry {
            agent: "agent-b".to_string(),
            weighted_copeland_score: 5,
            wins: 1,
            losses: 1,
            total_win_weight: 5,
            total_loss_weight: 5,
        },
    ];

    let agent_to_index: HashMap<String, usize> =
        vec![("agent-a".to_string(), 0), ("agent-b".to_string(), 1)]
            .into_iter()
            .collect();

    let agent_to_risks: HashMap<String, usize> =
        vec![("agent-a".to_string(), 2), ("agent-b".to_string(), 1)]
            .into_iter()
            .collect();

    // Sort descending by:
    // 1. score
    // 2. total win weight
    // 3. fewer losses
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

    // agent-a has 0 losses, agent-b has 1 loss. agent-a should sort first.
    assert_eq!(ranking[0].agent, "agent-a");
}

#[test]
fn test_parse_context_md() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("context.md");
    let mut file = File::create(&file_path).unwrap();

    let context_content = "This is the system context.\nIt spans multiple lines.";
    file.write_all(context_content.as_bytes()).unwrap();

    let read_content = std::fs::read_to_string(&file_path).unwrap();
    assert_eq!(read_content, context_content);
}

#[test]
fn test_scoring_proposals() {
    let context_store = AgentContextStore::new();
    let agent_a = AgentRuntime::new(
        halc::config::AgentConfig {
            name: "agent-a".to_string(),
            endpoint_url: "".to_string(),
            model: "".to_string(),
            api_key: "".to_string(),
            system_prompt: "".to_string(),
            judge: false,
            proposer: None,
            parameters: HashMap::new(),
        },
        context_store.clone(),
    );

    let agent_b = AgentRuntime::new(
        halc::config::AgentConfig {
            name: "agent-b".to_string(),
            endpoint_url: "".to_string(),
            model: "".to_string(),
            api_key: "".to_string(),
            system_prompt: "".to_string(),
            judge: false,
            proposer: None,
            parameters: HashMap::new(),
        },
        context_store.clone(),
    );

    let agents = vec![agent_a, agent_b];

    let mut challenges = HashMap::new();
    challenges.insert(
        ("agent-a".to_string(), "agent-b".to_string()),
        Challenge {
            reviewer_agent: "agent-a".to_string(),
            target_agent: "agent-b".to_string(),
            challenge: "weakness".to_string(),
            strengths: vec![],
            weaknesses: vec![],
            missing_evidence: vec![],
            risk_assessment: "".to_string(),
            score: 8,
        },
    );
    challenges.insert(
        ("agent-b".to_string(), "agent-a".to_string()),
        Challenge {
            reviewer_agent: "agent-b".to_string(),
            target_agent: "agent-a".to_string(),
            challenge: "ok".to_string(),
            strengths: vec![],
            weaknesses: vec![],
            missing_evidence: vec![],
            risk_assessment: "".to_string(),
            score: 5,
        },
    );

    let scores = calculate_scores(&agents, &challenges, &[]);
    assert_eq!(scores.len(), 2);

    let score_a = scores.iter().find(|e| e.agent == "agent-a").unwrap();
    let score_b = scores.iter().find(|e| e.agent == "agent-b").unwrap();
    assert_eq!(score_a.value, 5);
    assert_eq!(score_b.value, 8);
}

#[tokio::test]
async fn test_tie_breaking_logic() {
    let _context_store = AgentContextStore::new();

    let mut proposals = HashMap::new();
    proposals.insert(
        "agent-a".to_string(),
        Proposal {
            agent: "agent-a".to_string(),
            proposal: "prop-a".to_string(),
            rationale: "rat".to_string(),
            expected_benefits: vec![],
            known_risks: vec![],
        },
    );
    proposals.insert(
        "agent-b".to_string(),
        Proposal {
            agent: "agent-b".to_string(),
            proposal: "prop-b".to_string(),
            rationale: "rat".to_string(),
            expected_benefits: vec![],
            known_risks: vec![],
        },
    );

    let mut analyses = HashMap::new();
    analyses.insert(
        "agent-a".to_string(),
        ProposalAnalysis {
            agent: "agent-a".to_string(),
            claims: vec![],
            assumptions: vec![],
            risks: vec!["risk1".to_string(), "risk2".to_string()],
            evidence: vec![],
            expected_outcomes: vec![],
        },
    );
    analyses.insert(
        "agent-b".to_string(),
        ProposalAnalysis {
            agent: "agent-b".to_string(),
            claims: vec![],
            assumptions: vec![],
            risks: vec!["risk1".to_string()],
            evidence: vec![],
            expected_outcomes: vec![],
        },
    );

    let score_table = vec![
        ScoreTableEntry {
            agent: "agent-a".to_string(),
            value: 10,
        },
        ScoreTableEntry {
            agent: "agent-b".to_string(),
            value: 10,
        },
    ];

    let (winner, score) = select_winner(
        "request",
        &proposals,
        &analyses,
        &score_table,
        None,
        &NormalizationConfig::default(),
        Verbosity::Normal,
        "",
    )
    .await;

    assert_eq!(winner, "agent-b");
    assert_eq!(score, 10);
}

#[test]
fn test_json_cleaner() {
    let raw_md = "```json\n{\n  \"key\": \"value\"\n}\n```";
    let cleaned = AgentRuntime::extract_json(raw_md);
    assert_eq!(cleaned, "{\n  \"key\": \"value\"\n}");

    let prefix_suffix = "Here is the response: {\"key\": \"value\"} Hope this helps!";
    let cleaned_text = AgentRuntime::extract_json(prefix_suffix);
    assert_eq!(cleaned_text, "{\"key\": \"value\"}");
}

#[test]
fn test_parameter_remapping() {
    let msg1 = "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.";
    assert_eq!(
        AgentRuntime::find_suggested_parameter(msg1),
        Some("max_completion_tokens".to_string())
    );
}
