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

use axum::{routing::post, Json, Router};
use serde_json::json;
use tokio::net::TcpListener;

use halc::agent::AgentRuntime;
use halc::algorithm::{run_confrontation, ConfrontationParams, Ledger};
use halc::config::{AgentConfig, ConfrontationConfig, NormalizationConfig, Verbosity};
use halc::context::AgentContextStore;

// A mock server route for /chat/completions
async fn mock_chat_completions(Json(body): Json<serde_json::Value>) -> Json<serde_json::Value> {
    let messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .expect("Missing messages");

    let last_user_message = messages
        .last()
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .expect("Missing content in user message");

    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");

    let agent_name = if model.contains("agent-a") {
        "agent-a"
    } else if model.contains("agent-b") {
        "agent-b"
    } else if model.contains("agent-c") {
        "agent-c"
    } else {
        "judge"
    };

    let reply = if last_user_message.contains("Store this as your private context") {
        "Context received and stored.".to_string()
    } else if last_user_message.contains("Create a complete proposal") || last_user_message.contains("Synthesis") {
        format!(
            "# Problem Statement\nframe for {}\n# Priorities Used\n- throughput\n# Constraints\n# Success Metrics\n# Proposal\nproposal from {}\n# Rationale\nrationale for {}\n# Claims\n# Assumptions\n# Risks\n- risk\n# Evidence\n# Expected Outcomes\n",
            agent_name, agent_name, agent_name
        )
    } else if last_user_message.contains("Pairwise") || last_user_message.contains("compare two proposals") || last_user_message.contains("Retry") || last_user_message.contains("invalid") {
        // Dynamically find which two agents are being compared from the prompt message.
        // We look for agent-a, agent-b, agent-c.
        let has_a = last_user_message.contains("agent-a");
        let has_b = last_user_message.contains("agent-b");
        let has_c = last_user_message.contains("agent-c");
        
        let (w, l) = if has_a && has_b {
            ("agent-a", "agent-b")
        } else if has_b && has_c {
            ("agent-b", "agent-c")
        } else {
            ("agent-a", "agent-c")
        };
        format!(
            "# Winner\n{}\n# Loser\n{}\n# Confidence\n8\n# Priority Alignment\nok\n# Reasoning\ngood\n# Risks In Loser\n# Risks In Winner\n",
            w, l
        )
    } else if last_user_message.contains("generate the updated shared context") {
        "# Shared Knowledge\nThis is a mock summary of the confrontation.\n# User Request\n# Winning Proposal\n# Final Ranking\n# Pairwise Comparisons\n# Lessons Learned\n# Remaining Risks\n# Open Disagreements\n# Next Context For All Agents".to_string()
    } else {
        r#"{"error": "unknown state"}"#.to_string()
    };

    Json(json!({
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply
                },
                "finish_reason": "stop"
            }
        ]
    }))
}

#[tokio::test]
async fn test_end_to_end_confrontation() {
    // 1. Start mock server
    let app = Router::new().route("/chat/completions", post(mock_chat_completions));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // 2. Prepare Config (Compact distributed pairwise confrontation requires at least 3 proposers)
    let mock_endpoint = format!("http://{}", addr);
    let agent_a_cfg = AgentConfig {
        name: "agent-a".to_string(),
        endpoint_url: mock_endpoint.clone(),
        model: "model-agent-a".to_string(),
        api_key: "sk-a".to_string(),
        system_prompt: "You are agent A".to_string(),
        judge: false,
        proposer: None,
        parameters: std::collections::HashMap::new(),
    };

    let agent_b_cfg = AgentConfig {
        name: "agent-b".to_string(),
        endpoint_url: mock_endpoint.clone(),
        model: "model-agent-b".to_string(),
        api_key: "sk-b".to_string(),
        system_prompt: "You are agent B".to_string(),
        judge: false,
        proposer: None,
        parameters: std::collections::HashMap::new(),
    };

    let agent_c_cfg = AgentConfig {
        name: "agent-c".to_string(),
        endpoint_url: mock_endpoint.clone(),
        model: "model-agent-c".to_string(),
        api_key: "sk-c".to_string(),
        system_prompt: "You are agent C".to_string(),
        judge: false,
        proposer: None,
        parameters: std::collections::HashMap::new(),
    };

    let context_store = AgentContextStore::new();
    let runtime_a = AgentRuntime::new(agent_a_cfg, context_store.clone());
    let runtime_b = AgentRuntime::new(agent_b_cfg, context_store.clone());
    let runtime_c = AgentRuntime::new(agent_c_cfg, context_store.clone());

    // Initialize contexts
    runtime_a
        .initialize_context("Initial Context", "{{initial_context}}", Verbosity::Normal)
        .await
        .unwrap();
    runtime_b
        .initialize_context("Initial Context", "{{initial_context}}", Verbosity::Normal)
        .await
        .unwrap();
    runtime_c
        .initialize_context("Initial Context", "{{initial_context}}", Verbosity::Normal)
        .await
        .unwrap();

    let runtimes = vec![runtime_a, runtime_b, runtime_c];
    let ledger = Ledger::new(None);

    let dummy_templates = halc::prompts::PromptTemplates {
        context_init: "{{initial_context}}".to_string(),
        connectivity_test: "READY".to_string(),
        synthesis: "Synthesis".to_string(),
        pairwise_comparison: "Pairwise: {{agent_a_name}} vs {{agent_b_name}}<!-- RETRY_PROMPT -->Retry: {{agent_a_name}} vs {{agent_b_name}}".to_string(),
        tie_breaker: "Tiebreaker".to_string(),
        context_generation: "ContextGen".to_string(),
    };
    let params = ConfrontationParams {
        normalization: &NormalizationConfig::default(),
        confrontation: &ConfrontationConfig::default(),
        templates: &dummy_templates,
    };
    let decision = run_confrontation(
        "test-req-123",
        "What is the plan?",
        &runtimes,
        &ledger,
        None,
        Verbosity::Normal,
        &params,
    )
    .await
    .unwrap();

    assert_eq!(decision.winning_agent, "agent-a");
    assert!(decision.proposal.contains("proposal from agent-a"));
    assert_eq!(decision.score_table.len(), 3);

    // Clean up server
    server_handle.abort();
}

#[tokio::test]
async fn test_end_to_end_confrontation_with_judge() {
    // 1. Start mock server
    let app = Router::new().route("/chat/completions", post(mock_chat_completions));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // 2. Prepare Config
    let mock_endpoint = format!("http://{}", addr);
    let agent_a_cfg = AgentConfig {
        name: "agent-a".to_string(),
        endpoint_url: mock_endpoint.clone(),
        model: "model-agent-a".to_string(),
        api_key: "sk-a".to_string(),
        system_prompt: "You are agent A".to_string(),
        judge: false,
        proposer: None,
        parameters: std::collections::HashMap::new(),
    };

    let agent_b_cfg = AgentConfig {
        name: "agent-b".to_string(),
        endpoint_url: mock_endpoint.clone(),
        model: "model-agent-b".to_string(),
        api_key: "sk-b".to_string(),
        system_prompt: "You are agent B".to_string(),
        judge: false,
        proposer: None,
        parameters: std::collections::HashMap::new(),
    };

    let agent_c_cfg = AgentConfig {
        name: "agent-c".to_string(),
        endpoint_url: mock_endpoint.clone(),
        model: "model-agent-c".to_string(),
        api_key: "sk-c".to_string(),
        system_prompt: "You are agent C".to_string(),
        judge: false,
        proposer: None,
        parameters: std::collections::HashMap::new(),
    };

    let judge_cfg = AgentConfig {
        name: "judge".to_string(),
        endpoint_url: mock_endpoint.clone(),
        model: "model-judge".to_string(),
        api_key: "sk-j".to_string(),
        system_prompt: "You are the judge".to_string(),
        judge: true,
        proposer: None,
        parameters: std::collections::HashMap::new(),
    };

    let context_store = AgentContextStore::new();
    let runtime_a = AgentRuntime::new(agent_a_cfg, context_store.clone());
    let runtime_b = AgentRuntime::new(agent_b_cfg, context_store.clone());
    let runtime_c = AgentRuntime::new(agent_c_cfg, context_store.clone());
    let runtime_j = AgentRuntime::new(judge_cfg, context_store.clone());

    // Initialize contexts
    runtime_a
        .initialize_context("Initial Context", "{{initial_context}}", Verbosity::Normal)
        .await
        .unwrap();
    runtime_b
        .initialize_context("Initial Context", "{{initial_context}}", Verbosity::Normal)
        .await
        .unwrap();
    runtime_c
        .initialize_context("Initial Context", "{{initial_context}}", Verbosity::Normal)
        .await
        .unwrap();
    runtime_j
        .initialize_context("Initial Context", "{{initial_context}}", Verbosity::Normal)
        .await
        .unwrap();

    let runtimes = vec![runtime_a, runtime_b, runtime_c, runtime_j];

    // Set up a temp ledger directory
    let dir = tempfile::tempdir().unwrap();
    let ledger_file = dir.path().join("ledger.jsonl");
    let ledger_path = ledger_file.to_string_lossy().to_string();
    let ledger = Ledger::new(Some(ledger_path));

    let dummy_templates = halc::prompts::PromptTemplates {
        context_init: "{{initial_context}}".to_string(),
        connectivity_test: "READY".to_string(),
        synthesis: "Synthesis".to_string(),
        pairwise_comparison: "Pairwise: {{agent_a_name}} vs {{agent_b_name}}<!-- RETRY_PROMPT -->Retry: {{agent_a_name}} vs {{agent_b_name}}".to_string(),
        tie_breaker: "Tiebreaker".to_string(),
        context_generation: "ContextGen".to_string(),
     };
    let params = ConfrontationParams {
        normalization: &NormalizationConfig::default(),
        confrontation: &ConfrontationConfig::default(),
        templates: &dummy_templates,
    };
    let decision = run_confrontation(
        "test-req-judge-123",
        "What is the plan?",
        &runtimes,
        &ledger,
        None,
        Verbosity::Normal,
        &params,
    )
    .await
    .unwrap();

    // Verify decision
    assert_eq!(decision.winning_agent, "agent-a");

    // Verify that the ledger file was written
    assert!(ledger_file.exists());

    // Clean up server
    server_handle.abort();
}

#[tokio::test]
async fn test_end_to_end_confrontation_verbose() {
    // 1. Start mock server
    let app = Router::new().route("/chat/completions", post(mock_chat_completions));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // 2. Prepare Config
    let mock_endpoint = format!("http://{}", addr);
    let agent_a_cfg = AgentConfig {
        name: "agent-a".to_string(),
        endpoint_url: mock_endpoint.clone(),
        model: "model-agent-a".to_string(),
        api_key: "sk-a".to_string(),
        system_prompt: "You are agent A".to_string(),
        judge: false,
        proposer: None,
        parameters: std::collections::HashMap::new(),
    };

    let agent_b_cfg = AgentConfig {
        name: "agent-b".to_string(),
        endpoint_url: mock_endpoint.clone(),
        model: "model-agent-b".to_string(),
        api_key: "sk-b".to_string(),
        system_prompt: "You are agent B".to_string(),
        judge: false,
        proposer: None,
        parameters: std::collections::HashMap::new(),
    };

    let agent_c_cfg = AgentConfig {
        name: "agent-c".to_string(),
        endpoint_url: mock_endpoint.clone(),
        model: "model-agent-c".to_string(),
        api_key: "sk-c".to_string(),
        system_prompt: "You are agent C".to_string(),
        judge: false,
        proposer: None,
        parameters: std::collections::HashMap::new(),
    };

    let context_store = AgentContextStore::new();
    let runtime_a = AgentRuntime::new(agent_a_cfg, context_store.clone());
    let runtime_b = AgentRuntime::new(agent_b_cfg, context_store.clone());
    let runtime_c = AgentRuntime::new(agent_c_cfg, context_store.clone());

    // Initialize contexts
    runtime_a
        .initialize_context("Initial Context", "{{initial_context}}", Verbosity::Verbose)
        .await
        .unwrap();
    runtime_b
        .initialize_context("Initial Context", "{{initial_context}}", Verbosity::Verbose)
        .await
        .unwrap();
    runtime_c
        .initialize_context("Initial Context", "{{initial_context}}", Verbosity::Verbose)
        .await
        .unwrap();

    let runtimes = vec![runtime_a, runtime_b, runtime_c];
    let ledger = Ledger::new(None);

    let dummy_templates = halc::prompts::PromptTemplates {
        context_init: "{{initial_context}}".to_string(),
        connectivity_test: "READY".to_string(),
        synthesis: "Synthesis".to_string(),
        pairwise_comparison: "Pairwise: {{agent_a_name}} vs {{agent_b_name}}<!-- RETRY_PROMPT -->Retry: {{agent_a_name}} vs {{agent_b_name}}".to_string(),
        tie_breaker: "Tiebreaker".to_string(),
        context_generation: "ContextGen".to_string(),
    };
    let params = ConfrontationParams {
        normalization: &NormalizationConfig::default(),
        confrontation: &ConfrontationConfig::default(),
        templates: &dummy_templates,
    };
    let decision = run_confrontation(
        "test-req-verbose-123",
        "What is the plan?",
        &runtimes,
        &ledger,
        None,
        Verbosity::Verbose,
        &params,
    )
    .await
    .unwrap();

    assert_eq!(decision.winning_agent, "agent-a");

    // Clean up server
    server_handle.abort();
}

#[tokio::test]
async fn test_openrouter_endpoint_headers() {
    use axum::http::HeaderMap;

    let app = Router::new().route(
        "/openrouter.ai/chat/completions",
        post(
            |headers: HeaderMap, Json(_body): Json<serde_json::Value>| async move {
                let referer = headers
                    .get("HTTP-Referer")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("");
                let title = headers
                    .get("X-Title")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("");
                let auth = headers
                    .get("Authorization")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("");

                assert_eq!(referer, "https://github.com/cgegout/HALc");
                assert_eq!(title, "HALc");
                assert_eq!(auth, "Bearer sk-openrouter-key");

                Json(json!({
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "READY"
                            },
                            "finish_reason": "stop"
                        }
                    ]
                }))
            },
        ),
    );

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let mock_endpoint = format!("http://{}/openrouter.ai", addr);
    let agent_cfg = AgentConfig {
        name: "test-agent".to_string(),
        endpoint_url: mock_endpoint,
        model: "google/gemini-2.5-flash-lite".to_string(),
        api_key: "sk-openrouter-key".to_string(),
        system_prompt: "You are a test agent".to_string(),
        judge: false,
        proposer: None,
        parameters: std::collections::HashMap::new(),
    };

    let context_store = AgentContextStore::new();
    let runtime = AgentRuntime::new(agent_cfg, context_store);

    let messages = vec![halc::openai::ChatCompletionMessage {
        role: "user".to_string(),
        content: "test message".to_string(),
    }];

    let response = runtime
        .call_endpoint(messages, Verbosity::Normal)
        .await
        .unwrap();
    assert_eq!(response, "READY");

    server_handle.abort();
}
