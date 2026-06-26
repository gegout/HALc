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

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use std::convert::Infallible;
use std::sync::Arc;
use tower_http::services::ServeDir;
use tracing::{error, info};
use uuid::Uuid;

use crate::agent::AgentRuntime;
use crate::algorithm::{run_confrontation, Ledger};
use crate::config::AppConfig;
use crate::context::AgentContextStore;
use crate::openai::{
    ChatCompletionChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionUsage, ModelListResponse, ModelResponse,
};

/// Response payload returned by the GET /v1/status endpoint.
#[derive(serde::Serialize)]
struct StatusResponse {
    openai_endpoint: String,
    agents_path: String,
    context_path: String,
    config: Option<AppConfig>,
    context: Option<String>,
    app_executable_path: Option<String>,
}

/// Represents a single stage in the application workflow.
#[derive(serde::Serialize)]
struct WorkflowStage {
    id: String,
    name: String,
    title: String,
    trigger: String,
}

/// Response payload returned by the GET /v1/description endpoint.
#[derive(serde::Serialize)]
struct DescriptionResponse {
    name: String,
    version: String,
    description: String,
    stages: Vec<WorkflowStage>,
}

/// Request payload sent to the POST /v1/onboard endpoint.
#[derive(serde::Deserialize)]
struct OnboardRequest {
    agents: Vec<crate::config::AgentConfig>,
    context: String,
}

/// Verification outcome details for a single agent.
#[derive(serde::Serialize)]
struct ValidationCheckResult {
    agent: String,
    status: String,
    response: Option<String>,
    error: Option<String>,
}

/// Response payload returned by the POST /v1/onboard endpoint.
#[derive(serde::Serialize)]
struct OnboardResponse {
    status: String,
    validation_results: Vec<ValidationCheckResult>,
}

/// Shares global server resources (runtimes and file ledger logger).
pub struct AppState {
    /// Active agent runtimes, wrapped in a read-write lock to enable reloading.
    pub agents: tokio::sync::RwLock<Vec<AgentRuntime>>,

    /// Server confrontation transaction ledger log.
    pub ledger: Ledger,

    /// The API key required to access the gateway server's OpenAI endpoints.
    pub gateway_api_key: tokio::sync::RwLock<String>,

    /// Configurable semantic alias normalization mapping for headings.
    pub normalization: crate::config::NormalizationConfig,

    /// Confrontation configuration.
    pub confrontation: crate::config::ConfrontationConfig,

    /// Loaded prompt templates (reloaded on agent reload via /v1/onboard).
    pub templates: tokio::sync::RwLock<crate::prompts::PromptTemplates>,
}

/// Sets up Axum router routes, binders, and context middleware.
pub fn make_router(state: Arc<AppState>) -> Router {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let ui_dir = std::path::PathBuf::from(home)
        .join("Documents")
        .join("Antigravity")
        .join("SimpleWebAI");

    let cors = tower_http::cors::CorsLayer::permissive();

    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/status", get(status_handler))
        .route("/v1/onboard", post(onboard_handler))
        .route("/v1/description", get(description_handler))
        .route("/v1/browse", get(browse_handler))
        .route("/v1/resolve", get(resolve_handler))
        .fallback_service(ServeDir::new(ui_dir))
        .layer(cors)
        .with_state(state)
}

/// GET /health
/// Returns a simple health check payload.
async fn health_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({ "status": "healthy" })),
    )
}

/// GET /v1/models
/// Lists all available models including the main confrontation gateway model ('halc')
/// and individual configured agents.
fn is_authorized(headers: &axum::http::HeaderMap, expected_key: &str) -> bool {
    if let Some(auth_val) = headers.get(axum::http::header::AUTHORIZATION) {
        if let Ok(auth_str) = auth_val.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                return token == expected_key;
            }
        }
    }
    false
}

async fn models_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> impl IntoResponse {
    let expected_key = state.gateway_api_key.read().await;
    if !is_authorized(&headers, &expected_key) {
        let err_body = serde_json::json!({
            "error": {
                "message": "Incorrect API key provided.",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        });
        return (StatusCode::UNAUTHORIZED, Json(err_body)).into_response();
    }
    let mut data = vec![ModelResponse {
        id: "halc".to_string(),
        object: "model".to_string(),
        created: 1677652288,
        owned_by: "halc".to_string(),
    }];

    let agents = state.agents.read().await;
    for agent in agents.iter() {
        data.push(ModelResponse {
            id: agent.config.name.clone(),
            object: "model".to_string(),
            created: 1677652288,
            owned_by: agent.config.name.clone(),
        });
    }

    Json(ModelListResponse {
        object: "list".to_string(),
        data,
    })
    .into_response()
}

/// GET /v1/status
/// Exposes parameters and paths.
async fn status_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let expected_key = state.gateway_api_key.read().await;
    if !is_authorized(&headers, &expected_key) {
        let err_body = serde_json::json!({
            "error": {
                "message": "Incorrect API key provided.",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        });
        return (StatusCode::UNAUTHORIZED, Json(err_body)).into_response();
    }
    let app_path = params.get("app_path").cloned().unwrap_or_default();

    if !app_path.is_empty() {
        match run_cli_status(&app_path) {
            Ok(resp) => return (StatusCode::OK, Json(resp)).into_response(),
            Err(e) => {
                let err_body =
                    serde_json::json!({ "error": format!("Failed to run status: {}", e) });
                return (StatusCode::BAD_REQUEST, Json(err_body)).into_response();
            }
        }
    }

    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let config_dir = std::path::PathBuf::from(home).join(".config").join("HALc");
    let agent_toml_path = config_dir.join("agent.toml");
    let context_md_path = config_dir.join("context.md");

    let openai_endpoint = "http://127.0.0.1:8330/v1/chat/completions".to_string();

    let config = AppConfig::load_from_file(&agent_toml_path).ok();
    let context = std::fs::read_to_string(&context_md_path).ok();

    let app_executable_path = std::env::current_exe()
        .ok()
        .map(|p| p.to_string_lossy().to_string());

    let resp = StatusResponse {
        openai_endpoint,
        agents_path: agent_toml_path.to_string_lossy().to_string(),
        context_path: context_md_path.to_string_lossy().to_string(),
        config,
        context,
        app_executable_path,
    };

    (StatusCode::OK, Json(resp)).into_response()
}

/// GET /v1/description
/// Returns a JSON describing the app metadata and its pipeline stages.
async fn description_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let expected_key = state.gateway_api_key.read().await;
    if !is_authorized(&headers, &expected_key) {
        let err_body = serde_json::json!({
            "error": {
                "message": "Incorrect API key provided.",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        });
        return (StatusCode::UNAUTHORIZED, Json(err_body)).into_response();
    }
    let app_path = params.get("app_path").cloned().unwrap_or_default();

    if !app_path.is_empty() {
        match run_cli_description(&app_path) {
            Ok(resp) => return (StatusCode::OK, Json(resp)).into_response(),
            Err(e) => {
                let err_body =
                    serde_json::json!({ "error": format!("Failed to run description: {}", e) });
                return (StatusCode::BAD_REQUEST, Json(err_body)).into_response();
            }
        }
    }

    let stages = vec![
        WorkflowStage {
            id: "stage-1".to_string(),
            name: "Framing".to_string(),
            title: "Frame the Problem".to_string(),
            trigger: "[Stage 1/7]".to_string(),
        },
        WorkflowStage {
            id: "stage-2".to_string(),
            name: "Proposals".to_string(),
            title: "Generate Proposals".to_string(),
            trigger: "[Stage 2/7]".to_string(),
        },
        WorkflowStage {
            id: "stage-3".to_string(),
            name: "Assumptions".to_string(),
            title: "Extract Assumptions".to_string(),
            trigger: "[Stage 3/7]".to_string(),
        },
        WorkflowStage {
            id: "stage-4".to_string(),
            name: "Critiques".to_string(),
            title: "Challenge Phase".to_string(),
            trigger: "[Stage 4/7]".to_string(),
        },
        WorkflowStage {
            id: "stage-5".to_string(),
            name: "Scoring".to_string(),
            title: "Scoring Phase".to_string(),
            trigger: "[Stage 5/7]".to_string(),
        },
        WorkflowStage {
            id: "stage-6".to_string(),
            name: "Decision".to_string(),
            title: "Decision Phase".to_string(),
            trigger: "Winner determined".to_string(),
        },
        WorkflowStage {
            id: "stage-7".to_string(),
            name: "Steelman".to_string(),
            title: "Steelman Phase".to_string(),
            trigger: "[Stage 7/7]".to_string(),
        },
    ];

    let resp = DescriptionResponse {
        name: "HALc".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: "Heuristic Agent Ledger by confrontation".to_string(),
        stages,
    };

    (StatusCode::OK, Json(resp)).into_response()
}

/// POST /v1/onboard
/// Sets parameters, writes files, and verifies agent endpoints dynamically.
async fn onboard_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
    Json(payload): Json<OnboardRequest>,
) -> impl IntoResponse {
    let expected_key = state.gateway_api_key.read().await;
    if !is_authorized(&headers, &expected_key) {
        let err_body = serde_json::json!({
            "error": {
                "message": "Incorrect API key provided.",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        });
        return (StatusCode::UNAUTHORIZED, Json(err_body)).into_response();
    }
    let app_path = params.get("app_path").cloned().unwrap_or_default();

    let (agent_toml_path, context_md_path) = if !app_path.is_empty() {
        match run_cli_status(&app_path) {
            Ok(status) => (
                std::path::PathBuf::from(status.agents_path),
                std::path::PathBuf::from(status.context_path),
            ),
            Err(e) => {
                let err_body = serde_json::json!({ "error": format!("Failed to find config folder from status: {}", e) });
                return (StatusCode::BAD_REQUEST, Json(err_body)).into_response();
            }
        }
    } else {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let config_dir = std::path::PathBuf::from(home).join(".config").join("HALc");
        (config_dir.join("agent.toml"), config_dir.join("context.md"))
    };

    // Ensure parent directories exist
    if let Some(parent) = agent_toml_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Some(parent) = context_md_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Write to agent.toml
    let gateway_key = state.gateway_api_key.read().await.clone();
    let (ledger_dir, prompts_config) = AppConfig::load_from_file(&agent_toml_path)
        .map(|cfg| (cfg.ledger_directory, cfg.prompts))
        .unwrap_or_else(|_| ("~/.config/HALc/ledger".to_string(), crate::config::PromptsConfig::default()));
    let app_config = AppConfig {
        agents: payload.agents,
        gateway_api_key: gateway_key,
        ledger_directory: ledger_dir,
        normalization: state.normalization.clone(),
        confrontation: state.confrontation.clone(),
        prompts: prompts_config,
    };
    let toml_str = match toml::to_string_pretty(&app_config) {
        Ok(s) => s,
        Err(e) => {
            let err_body =
                serde_json::json!({ "error": format!("TOML serialization error: {}", e) });
            return (StatusCode::BAD_REQUEST, Json(err_body)).into_response();
        }
    };
    if let Err(e) = std::fs::write(&agent_toml_path, toml_str) {
        let err_body = serde_json::json!({ "error": format!("Failed to write agent.toml: {}", e) });
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(err_body)).into_response();
    }

    // Write to context.md
    if let Err(e) = std::fs::write(&context_md_path, &payload.context) {
        let err_body = serde_json::json!({ "error": format!("Failed to write context.md: {}", e) });
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(err_body)).into_response();
    }

    // Reinitialize runtimes
    let context_store = AgentContextStore::new();
    let mut new_runtimes = Vec::new();
    for agent_cfg in &app_config.agents {
        new_runtimes.push(AgentRuntime::new(agent_cfg.clone(), context_store.clone()));
    }

    let mut validation_results = Vec::new();
    let mut test_futures = Vec::new();

    for runtime in &new_runtimes {
        let name = runtime.config.name.clone();
        let rt = runtime.clone();
        test_futures.push(async move {
            let test_messages = vec![
                crate::openai::ChatCompletionMessage {
                    role: "system".to_string(),
                    content: rt.config.system_prompt.clone(),
                },
                crate::openai::ChatCompletionMessage {
                    role: "user".to_string(),
                    content: "Respond ONLY with the word \"READY\" if you can hear me.".to_string(),
                },
            ];

            let result = tokio::time::timeout(
                std::time::Duration::from_secs(10),
                rt.call_endpoint(test_messages, crate::config::Verbosity::Normal),
            )
            .await;

            (name, result)
        });
    }

    let test_results = futures::future::join_all(test_futures).await;
    for (name, result) in test_results {
        match result {
            Ok(Ok(response)) => {
                validation_results.push(ValidationCheckResult {
                    agent: name,
                    status: "operational".to_string(),
                    response: Some(response.trim().to_string()),
                    error: None,
                });
            }
            Ok(Err(e)) => {
                validation_results.push(ValidationCheckResult {
                    agent: name,
                    status: "failed".to_string(),
                    response: None,
                    error: Some(format!("{:?}", e)),
                });
            }
            Err(_) => {
                validation_results.push(ValidationCheckResult {
                    agent: name,
                    status: "failed".to_string(),
                    response: None,
                    error: Some("Timeout (10s exceeded)".to_string()),
                });
            }
        }
    }

    // Reload the config in memory
    {
        let mut guard = state.agents.write().await;
        *guard = new_runtimes;
    }

    let response_body = OnboardResponse {
        status: "success".to_string(),
        validation_results,
    };

    (StatusCode::OK, Json(response_body)).into_response()
}

/// POST /v1/chat/completions
/// Main endpoint receiving user instructions. Executes the multi-agent confrontation loop
/// and returns the winning proposal as the final assistant completion response, either in stream or non-stream format.
async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let expected_key = state.gateway_api_key.read().await;
    if !is_authorized(&headers, &expected_key) {
        let err_body = serde_json::json!({
            "error": {
                "message": "Incorrect API key provided.",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        });
        return (StatusCode::UNAUTHORIZED, Json(err_body)).into_response();
    }
    let user_message = payload
        .messages
        .iter()
        .rev()
        .find(|msg| msg.role == "user")
        .map(|msg| msg.content.clone());

    let user_request = match user_message {
        Some(msg) => msg,
        None => {
            let err_body = serde_json::json!({
                "error": {
                    "message": "Missing user message in request",
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": null
                }
            });
            return (StatusCode::BAD_REQUEST, Json(err_body)).into_response();
        }
    };

    let request_id = Uuid::new_v4().to_string();
    info!(
        "Received chat completion request (ID: {}). Content: '{}'",
        request_id, user_request
    );

    if payload.stream {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(100);
        let state_clone = state.clone();
        let request_id_clone = request_id.clone();
        let user_request_clone = user_request.clone();

        tokio::spawn(async move {
            let agents_lock = state_clone.agents.read().await;
            let templates_lock = state_clone.templates.read().await;
            let params = crate::algorithm::ConfrontationParams {
                normalization: &state_clone.normalization,
                confrontation: &state_clone.confrontation,
                templates: &templates_lock,
            };
            match run_confrontation(
                &request_id_clone,
                &user_request_clone,
                &agents_lock,
                &state_clone.ledger,
                Some(tx.clone()),
                crate::config::Verbosity::Normal,
                &params,
            )
            .await
            {
                Ok(decision) => {
                    let _ = tx
                        .send(format!("__DECISION_START__{}", decision.markdown))
                        .await;
                }
                Err(e) => {
                    let _ = tx.send(format!("__ERROR__{}", e)).await;
                }
            }
        });

        let event_stream = async_stream::stream! {
             while let Some(msg) = rx.recv().await {
                if let Some(json_data) = msg.strip_prefix("__DECISION_START__") {
                    for line in json_data.lines() {
                        let chunk = serde_json::json!({
                            "choices": [{
                                "delta": {
                                    "content": format!("{}\n", line)
                                }
                            }]
                        });
                        yield Ok::<_, Infallible>(Event::default().json_data(chunk).unwrap());
                        tokio::time::sleep(std::time::Duration::from_millis(15)).await;
                    }
                } else if let Some(err_text) = msg.strip_prefix("__ERROR__") {
                    let chunk = serde_json::json!({
                        "choices": [{
                            "delta": {
                                "content": format!("\nError occurred: {}\n", err_text)
                            }
                        }]
                    });
                    yield Ok::<_, Infallible>(Event::default().json_data(chunk).unwrap());
                } else {
                    let chunk = serde_json::json!({
                        "choices": [{
                            "delta": {
                                "content": format!("{}\n", msg)
                            }
                        }]
                    });
                    yield Ok::<_, Infallible>(Event::default().json_data(chunk).unwrap());
                }
            }
        };

        Sse::new(event_stream).into_response()
    } else {
        let agents = state.agents.read().await;
        let templates_lock = state.templates.read().await;
        let params = crate::algorithm::ConfrontationParams {
            normalization: &state.normalization,
            confrontation: &state.confrontation,
            templates: &templates_lock,
        };
        match run_confrontation(
            &request_id,
            &user_request,
            &agents,
            &state.ledger,
            None,
            crate::config::Verbosity::Normal,
            &params,
        )
        .await
        {
            Ok(decision) => {
                let response = ChatCompletionResponse {
                    id: format!("chatcmpl-{}", request_id),
                    object: "chat.completion".to_string(),
                    created: Utc::now().timestamp(),
                    model: "halc".to_string(),
                    choices: vec![ChatCompletionChoice {
                        index: 0,
                        message: ChatCompletionMessage {
                            role: "assistant".to_string(),
                            content: decision.markdown.clone(),
                        },
                        finish_reason: "stop".to_string(),
                    }],
                    usage: ChatCompletionUsage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                };

                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => {
                error!("Confrontation execution failed: {:?}", e);
                let err_body = serde_json::json!({
                    "error": {
                        "message": format!("Confrontation loop error: {}", e),
                        "type": "api_error",
                        "param": null,
                        "code": null
                    }
                });
                (StatusCode::INTERNAL_SERVER_ERROR, Json(err_body)).into_response()
            }
        }
    }
}

fn run_cli_status(app_path: &str) -> Result<StatusResponse, Box<dyn std::error::Error>> {
    let output = std::process::Command::new(app_path)
        .arg("status")
        .output()?;

    if !output.status.success() {
        return Err(format!("CLI status returned exit status: {}", output.status).into());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut config_dir = std::path::PathBuf::new();
    let mut openai_endpoint = String::new();

    for line in stdout.lines() {
        let line = line.trim();
        if line.starts_with("Configuration Folder:") {
            let path_str = line.replace("Configuration Folder:", "").trim().to_string();
            config_dir = std::path::PathBuf::from(path_str);
        } else if line.starts_with("OpenAI Gateway Endpoint:") {
            openai_endpoint = line
                .replace("OpenAI Gateway Endpoint:", "")
                .trim()
                .to_string();
        }
    }

    if config_dir.as_os_str().is_empty() {
        return Err("Could not find Configuration Folder in status output".into());
    }

    let agent_toml_path = config_dir.join("agent.toml");
    let context_md_path = config_dir.join("context.md");

    let config = AppConfig::load_from_file(&agent_toml_path).ok();
    let context = std::fs::read_to_string(&context_md_path).ok();

    Ok(StatusResponse {
        openai_endpoint,
        agents_path: agent_toml_path.to_string_lossy().to_string(),
        context_path: context_md_path.to_string_lossy().to_string(),
        config,
        context,
        app_executable_path: Some(app_path.to_string()),
    })
}

fn run_cli_description(app_path: &str) -> Result<DescriptionResponse, Box<dyn std::error::Error>> {
    let output = std::process::Command::new(app_path)
        .arg("description")
        .output()?;

    if !output.status.success() {
        return Err(format!("CLI description returned exit status: {}", output.status).into());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut name = String::new();
    let mut description = String::new();
    let mut stages = Vec::new();

    for line in stdout.lines() {
        let line = line.trim();
        if line.starts_with("Application:") {
            name = line.replace("Application:", "").trim().to_string();
        } else if line.starts_with("Description:") {
            description = line.replace("Description:", "").trim().to_string();
        } else if line.contains(" - [Trigger:") {
            if let Some(dot_idx) = line.find('.') {
                if let Some(dash_idx) = line.find(" - [Trigger:") {
                    let id_str = line[..dot_idx].trim().to_string();
                    let stage_name = line[dot_idx + 1..dash_idx].trim().to_string();

                    let trigger_part = &line[dash_idx..];
                    if let Some(start_quote) = trigger_part.find('"') {
                        if let Some(end_quote) = trigger_part[start_quote + 1..].find('"') {
                            let trigger = trigger_part
                                [start_quote + 1..start_quote + 1 + end_quote]
                                .to_string();

                            let remaining = &trigger_part[start_quote + 1 + end_quote + 1..];
                            let title = remaining.trim().trim_start_matches(']').trim().to_string();

                            stages.push(WorkflowStage {
                                id: format!("stage-{}", id_str),
                                name: stage_name,
                                title,
                                trigger,
                            });
                        }
                    }
                }
            }
        }
    }

    Ok(DescriptionResponse {
        name,
        version: "dynamic".to_string(),
        description,
        stages,
    })
}

async fn browse_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> impl IntoResponse {
    let expected_key = state.gateway_api_key.read().await;
    if !is_authorized(&headers, &expected_key) {
        let err_body = serde_json::json!({
            "error": {
                "message": "Incorrect API key provided.",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        });
        return (StatusCode::UNAUTHORIZED, Json(err_body)).into_response();
    }

    match run_zenity_file_selection() {
        Ok(path) => (StatusCode::OK, Json(serde_json::json!({ "path": path }))).into_response(),
        Err(e) => {
            let err_body = serde_json::json!({ "error": format!("Failed to select file: {}", e) });
            (StatusCode::BAD_REQUEST, Json(err_body)).into_response()
        }
    }
}

fn run_zenity_file_selection() -> Result<String, Box<dyn std::error::Error>> {
    let output = std::process::Command::new("zenity")
        .arg("--file-selection")
        .arg("--title=Select Application Executable")
        .output()?;

    if !output.status.success() {
        return Err("File selection cancelled or failed".into());
    }

    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(path)
}

async fn resolve_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let expected_key = state.gateway_api_key.read().await;
    if !is_authorized(&headers, &expected_key) {
        let err_body = serde_json::json!({
            "error": {
                "message": "Incorrect API key provided.",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        });
        return (StatusCode::UNAUTHORIZED, Json(err_body)).into_response();
    }
    let filename = params.get("filename").cloned().unwrap_or_default();
    if filename.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "Missing filename parameter" })),
        )
            .into_response();
    }

    match find_executable_in_workspace(&filename) {
        Some(path) => (StatusCode::OK, Json(serde_json::json!({ "path": path }))).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "Executable file not found in workspace" })),
        )
            .into_response(),
    }
}

fn find_executable_in_workspace(filename: &str) -> Option<String> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let workspace = std::path::PathBuf::from(home.clone())
        .join("Documents")
        .join("Antigravity")
        .join("HALc");

    let release_path = workspace.join("target").join("release").join(filename);
    if release_path.exists() {
        return Some(release_path.to_string_lossy().to_string());
    }

    let debug_path = workspace.join("target").join("debug").join(filename);
    if debug_path.exists() {
        return Some(debug_path.to_string_lossy().to_string());
    }

    if let Ok(output) = std::process::Command::new("which").arg(filename).output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }
    }

    let project_root = std::path::PathBuf::from(home)
        .join("Documents")
        .join("Antigravity");

    let mut dirs_to_visit = vec![project_root];
    while let Some(dir) = dirs_to_visit.pop() {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Some(name) = path.file_name() {
                        let name_str = name.to_string_lossy();
                        if name_str.starts_with('.')
                            || name_str == "target"
                            || name_str == "node_modules"
                        {
                            continue;
                        }
                    }
                    dirs_to_visit.push(path);
                } else if path.is_file() {
                    if let Some(name) = path.file_name() {
                        if name == filename {
                            return Some(path.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
    }

    None
}
