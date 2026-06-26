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

use clap::Parser;
use std::io::Write;
use std::sync::Arc;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use colored::Colorize;
use halc::agent::AgentRuntime;
use halc::algorithm::Ledger;
use halc::config::AppConfig;
use halc::context::AgentContextStore;
use halc::server::{make_router, AppState};
use indicatif::{ProgressBar, ProgressStyle};

/// Command line arguments model for HALc server daemon.
#[derive(Parser, Debug)]
#[command(name = "halc", about = "Heuristic Agent Ledger by confrontation")]
struct Args {
    /// Subcommand wrapper (e.g. "onboard", "status").
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to agent configuration. Default is ~/.config/HALc/agent.toml.
    #[arg(long, short, global = true, help = "Path to agent.toml")]
    agents: Option<String>,

    /// Path to context markdown. Default is ~/.config/HALc/context.md.
    #[arg(long, short, global = true, help = "Path to context.md")]
    context: Option<String>,

    /// Network host to bind HTTP server onto.
    #[arg(
        long,
        global = true,
        default_value = "127.0.0.1",
        help = "HTTP server host"
    )]
    host: String,

    /// Network port to bind HTTP server onto.
    #[arg(long, global = true, default_value_t = 8330, help = "HTTP server port")]
    port: u16,

    /// Optional file path to output confrontation transaction records.
    #[arg(long, global = true, help = "Path to JSONL ledger file")]
    ledger: Option<String>,

    /// Enable verbose mode to show detailed agent interactions, scores, and convergence.
    #[arg(
        long,
        short,
        global = true,
        help = "Verbose mode showing detailed confrontation logs"
    )]
    verbose: bool,

    /// Enable debug mode to show full raw HTTP payloads and responses.
    #[arg(
        long,
        short,
        global = true,
        help = "Debug mode showing raw HTTP requests and responses"
    )]
    debug: bool,
}

/// Commands supported by the CLI interface.
#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Sets up default agent configurations and context files.
    #[command(about = "Onboard the workspace by generating default configuration files")]
    Onboard,

    /// Shows the current configuration status and details.
    #[command(about = "Shows the current configuration status and details")]
    Status,

    /// Shows the application workflow description and stages.
    #[command(about = "Shows the application workflow description and stages")]
    Description,

    /// Starts the gateway server daemon.
    #[command(about = "Starts the gateway server daemon")]
    Run,

    /// Directly input requests and run confrontation between agents
    #[command(
        name = "runandchat",
        about = "Directly input requests and run agent confrontation"
    )]
    RunAndChat {
        /// The user request. If not specified, you will be prompted.
        request: Option<String>,
    },
}

/// Utility function that locates and returns the home-based configuration directory.
fn get_config_dir() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let home = std::env::var("HOME")
        .map_err(|_| "HOME environment variable not set. Cannot locate configuration directory.")?;
    let path = std::path::PathBuf::from(home).join(".config").join("HALc");
    Ok(path)
}

fn resolve_path(p: &str) -> std::path::PathBuf {
    if let Some(stripped) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return std::path::PathBuf::from(home).join(stripped);
        }
    }
    std::path::PathBuf::from(p)
}

/// Helper function that prints a prompt message to the console and reads standard input,
/// falling back to a default value if input is empty.
fn read_input(prompt: &str, default: &str) -> String {
    print!("{} [default: {}]: ", prompt, default);
    std::io::stdout().flush().unwrap();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let trimmed = input.trim();
    if trimmed.is_empty() {
        default.to_string()
    } else {
        trimmed.to_string()
    }
}

/// Outputs the status parameters, endpoints, and configs loaded from the default paths.
fn run_status_command() -> Result<(), Box<dyn std::error::Error>> {
    let config_dir = get_config_dir()?;
    let agent_toml_path = config_dir.join("agent.toml");
    let context_md_path = config_dir.join("context.md");

    println!("================================================================================");
    println!(" HALc Configuration & Operations Status");
    println!("================================================================================\n");
    println!("Configuration Folder: {}", config_dir.to_string_lossy());
    println!("OpenAI Gateway Endpoint: http://localhost:8330/v1/chat/completions");
    println!("\n--------------------------------------------------------------------------------");
    println!(" File Status:");
    println!("--------------------------------------------------------------------------------");

    if agent_toml_path.exists() {
        println!(
            "[+] agent.toml: Found ({} bytes)",
            agent_toml_path.metadata()?.len()
        );
        match AppConfig::load_from_file(&agent_toml_path) {
            Ok(cfg) => {
                println!("    Agents Configured: {}", cfg.agents.len());
                for (idx, agent) in cfg.agents.iter().enumerate() {
                    let judge_label = if agent.judge { " (JUDGE)" } else { "" };
                    println!(
                        "    {}. '{}' - model: '{}', endpoint: '{}'{}",
                        idx + 1,
                        agent.name,
                        agent.model,
                        agent.endpoint_url,
                        judge_label
                    );
                }
            }
            Err(e) => {
                println!("    [!] Error parsing agent.toml: {:?}", e);
            }
        }
    } else {
        println!("[-] agent.toml: Not Found");
    }

    if context_md_path.exists() {
        println!(
            "[+] context.md: Found ({} bytes)",
            context_md_path.metadata()?.len()
        );
        let context = std::fs::read_to_string(&context_md_path)?;
        println!("    Content:\n    ---");
        for line in context.lines() {
            println!("    {}", line);
        }
        println!("    ---");
    } else {
        println!("[-] context.md: Not Found");
    }
    println!("\n================================================================================");

    Ok(())
}

/// Outputs the workflow description, stages, and stream match triggers.
fn run_description_command() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!(" HALc Workflow & Stages Description");
    println!("================================================================================\n");
    println!("Application: HALc");
    println!("Description: Heuristic Agent Ledger by confrontation");
    println!("\nWorkflow Stages:");
    println!("--------------------------------------------------------------------------------");
    println!("1. Framing     - [Trigger: \"[Stage 1/7]\"]  Frame the Problem");
    println!("2. Proposals   - [Trigger: \"[Stage 2/7]\"]  Generate Proposals");
    println!("3. Assumptions - [Trigger: \"[Stage 3/7]\"]  Extract Assumptions");
    println!("4. Critiques   - [Trigger: \"[Stage 4/7]\"]  Challenge Phase");
    println!("5. Scoring     - [Trigger: \"[Stage 5/7]\"]  Scoring Phase");
    println!("6. Decision    - [Trigger: \"Winner determined\"]  Decision Phase");
    println!("7. Steelman    - [Trigger: \"[Stage 7/7]\"]  Steelman Phase");
    println!("================================================================================");
    Ok(())
}

/// Orchestrates the interactive configuration questionnaire and runs connection sanity tests.
async fn run_onboarding() -> Result<(), Box<dyn std::error::Error>> {
    let config_dir = get_config_dir()?;
    if !config_dir.exists() {
        std::fs::create_dir_all(&config_dir)?;
    }

    let agent_toml_path = config_dir.join("agent.toml");
    let context_md_path = config_dir.join("context.md");

    // Load existing items if available
    let existing_config = if agent_toml_path.exists() {
        AppConfig::load_from_file(&agent_toml_path).ok()
    } else {
        None
    };

    let existing_context = if context_md_path.exists() {
        std::fs::read_to_string(&context_md_path).ok()
    } else {
        None
    };

    println!("================================================================================");
    println!(" Welcome to the HALc Setup Onboarding wizard");
    println!(" This script generates default configuration settings in ~/.config/HALc/");
    println!("================================================================================\n");

    let default_count = existing_config
        .as_ref()
        .map(|c| c.agents.len())
        .unwrap_or(4);
    let count_str = read_input(
        "Enter number of agents to configure",
        &default_count.to_string(),
    );
    let count: usize = count_str.parse().unwrap_or(default_count);

    let mut new_agents = Vec::new();

    for i in 0..count {
        println!("\n--- Agent #{} Configuration ---", i + 1);
        let existing_agent = existing_config.as_ref().and_then(|c| c.agents.get(i));

        let def_name = existing_agent
            .map(|a| a.name.as_str())
            .unwrap_or_else(|| match i {
                0 => "architect",
                1 => "product",
                2 => "security",
                _ => "judge",
            });

        let def_prompt = existing_agent
            .map(|a| a.system_prompt.as_str())
            .unwrap_or_else(|| match i {
                0 => "You are an architecture-focused agent. Challenge assumptions.",
                1 => "You are a product management-focused agent. Prioritize user value.",
                2 => "You are a security-focused agent. Evaluate vulnerability risks.",
                _ => "You are the confrontation judge resolving ties.",
            });

        let def_judge = existing_agent
            .map(|a| if a.judge { "y" } else { "n" })
            .unwrap_or_else(|| if i >= 3 { "y" } else { "n" });

        let def_temp = existing_agent
            .and_then(|a| a.parameters.get("temperature").and_then(|v| v.as_f64()))
            .unwrap_or(0.3);
        let def_tokens = existing_agent
            .and_then(|a| a.parameters.get("max_tokens").and_then(|v| v.as_u64()))
            .unwrap_or(2000);

        let name = read_input("Agent Name", def_name);

        let default_provider = existing_agent
            .map(|a| {
                if a.endpoint_url.contains("openrouter.ai") {
                    "openrouter"
                } else {
                    "openai"
                }
            })
            .unwrap_or("openai");
        let provider = read_input("API Provider (openai/openrouter)", default_provider);
        let is_openrouter = provider.trim().to_lowercase() == "openrouter";

        let fallback_url = if is_openrouter {
            "https://openrouter.ai/api/v1"
        } else {
            "https://api.openai.com/v1"
        };
        let fallback_model = if is_openrouter {
            "google/gemini-2.5-flash-lite"
        } else {
            "gpt-4.1"
        };

        let def_url = existing_agent
            .map(|a| a.endpoint_url.as_str())
            .unwrap_or(fallback_url);
        let def_model = existing_agent
            .map(|a| a.model.as_str())
            .unwrap_or(fallback_model);
        let def_key = existing_agent
            .map(|a| a.api_key.as_str())
            .unwrap_or("sk-placeholder");

        let url = read_input("Endpoint URL", def_url);
        let model = read_input("Model Name", def_model);
        let key = read_input("API Key", def_key);

        println!(
            "System Prompt Default: \"{}\"",
            def_prompt.replace('\n', " ")
        );
        let prompt = read_input("System Prompt (or press Enter for default)", def_prompt);

        let judge_str = read_input("Is this agent a tie-breaker judge? (y/n)", def_judge);
        let judge = judge_str.trim().to_lowercase().starts_with('y');

        let temp_str = read_input("Temperature", &def_temp.to_string());
        let temp: f64 = temp_str.parse().unwrap_or(def_temp);

        let tokens_str = read_input("Max Tokens", &def_tokens.to_string());
        let tokens: u64 = tokens_str.parse().unwrap_or(def_tokens);

        let mut parameters = std::collections::HashMap::new();
        parameters.insert("temperature".to_string(), serde_json::Value::from(temp));
        parameters.insert("max_tokens".to_string(), serde_json::Value::from(tokens));

        new_agents.push(halc::config::AgentConfig {
            name,
            endpoint_url: url,
            model,
            api_key: key,
            system_prompt: prompt,
            judge,
            proposer: None,
            parameters,
        });
    }

    println!("\n--- Shared Context Markdown Configuration ---");
    let mut context_content = String::new();
    let edit_context = read_input(
        "Do you want to enter a custom context markdown file? (y/n)",
        "n",
    );
    if edit_context.trim().to_lowercase().starts_with('y') {
        println!("Enter your custom context markdown content. Type 'EOF' on a new line and press Enter to save:");
        use std::io::BufRead;
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let l = line.unwrap();
            if l.trim() == "EOF" {
                break;
            }
            context_content.push_str(&l);
            context_content.push('\n');
        }
    } else {
        context_content = existing_context.unwrap_or_else(|| {
            r#"# Shared Architecture Context - Project Pegasus

We are designing a new high-throughput event processing platform. 

## Requirements
- Support up to 100,000 events/second ingest rate.
- End-to-end latency must be less than 50ms.
- High availability with a multi-region setup (primary/secondary).
- Storage must persist events for at least 7 days.

## Constraints
- Team size is small (4 engineers).
- Budget is capped at $5,000/month for infrastructure.
- Deliver initial beta version in 6 weeks.
"#
            .to_string()
        });
    }

    // Write Config Files
    let gateway_key = existing_config
        .as_ref()
        .map(|c| c.gateway_api_key.clone())
        .unwrap_or_else(|| "1234567890".to_string());

    let ledger_dir = existing_config
        .as_ref()
        .map(|c| c.ledger_directory.clone())
        .unwrap_or_else(|| "~/.config/HALc/ledger".to_string());

    let app_config = AppConfig {
        agents: new_agents,
        gateway_api_key: gateway_key,
        ledger_directory: ledger_dir,
        normalization: existing_config
            .as_ref()
            .map(|c| c.normalization.clone())
            .unwrap_or_default(),
        confrontation: existing_config
            .as_ref()
            .map(|c| c.confrontation.clone())
            .unwrap_or_default(),
        prompts: existing_config
            .as_ref()
            .map(|c| c.prompts.clone())
            .unwrap_or_default(),
    };
    let toml_str = toml::to_string_pretty(&app_config)?;
    std::fs::write(&agent_toml_path, toml_str)?;
    println!(
        "\n[+] Configuration file written to: {}",
        agent_toml_path.to_string_lossy()
    );

    std::fs::write(&context_md_path, &context_content)?;
    println!(
        "[+] Context file written to: {}",
        context_md_path.to_string_lossy()
    );

    // Execute Operational Verification Checks
    println!("\n================================================================================");
    println!(" Running Connection Operational Sanity Tests for Configured Agents...");
    println!("================================================================================\n");

    let context_store = AgentContextStore::new();
    let mut test_futures = Vec::new();
    for agent in &app_config.agents {
        let runtime = AgentRuntime::new(agent.clone(), context_store.clone());
        test_futures.push(async move {
            let test_messages = vec![
                halc::openai::ChatCompletionMessage {
                    role: "system".to_string(),
                    content: runtime.config.system_prompt.clone(),
                },
                halc::openai::ChatCompletionMessage {
                    role: "user".to_string(),
                    content: "Respond ONLY with the word \"READY\" if you can hear me.".to_string(),
                },
            ];

            let result = tokio::time::timeout(
                std::time::Duration::from_secs(10),
                runtime.call_endpoint(test_messages, halc::config::Verbosity::Normal),
            )
            .await;

            (runtime.config.name.clone(), result)
        });
    }

    let test_results = futures::future::join_all(test_futures).await;
    for (name, result) in test_results {
        match result {
            Ok(Ok(response)) => {
                println!(
                    "[+] Agent '{}' is operational! Response: {:?}",
                    name,
                    response.trim()
                );
            }
            Ok(Err(e)) => {
                println!("[!] Agent '{}' connection check FAILED: {:?}", name, e);
            }
            Err(_) => {
                println!(
                    "[!] Agent '{}' connection check FAILED: Timeout (10 seconds exceeded)",
                    name
                );
            }
        }
    }

    println!("\n================================================================================");
    println!(" Setup Complete.");
    println!("================================================================================\n");
    println!("To start the HALc gateway:");
    println!("  $ halc run");
    println!("\nTo request a confrontation run:");
    println!("  $ curl http://localhost:8330/v1/chat/completions \\");
    println!("      -H \"Content-Type: application/json\" \\");
    println!("      -d '{{");
    println!("        \"model\": \"halc\",");
    println!("        \"messages\": [");
    println!("          {{");
    println!("            \"role\": \"user\",");
    println!("            \"content\": \"What is our architecture choice for the event ingestion pipeline?\"");
    println!("          }}");
    println!("        ]");
    println!("      }}'");
    println!("\n================================================================================");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing/logging with default INFO level
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    // Check subcommands
    if let Some(Commands::Onboard) = args.command {
        run_onboarding().await?;
        return Ok(());
    }

    if let Some(Commands::Status) = args.command {
        run_status_command()?;
        return Ok(());
    }

    if let Some(Commands::Description) = args.command {
        run_description_command()?;
        return Ok(());
    }

    let config_dir = get_config_dir()?;

    let agents_path = args
        .agents
        .unwrap_or_else(|| config_dir.join("agent.toml").to_string_lossy().to_string());

    let context_path = args
        .context
        .unwrap_or_else(|| config_dir.join("context.md").to_string_lossy().to_string());

    // Check if configuration exists
    let verbosity = if args.debug {
        halc::config::Verbosity::Debug
    } else if args.verbose {
        halc::config::Verbosity::Verbose
    } else {
        halc::config::Verbosity::Normal
    };

    // Check if configuration exists
    if !std::path::Path::new(&agents_path).exists() || !std::path::Path::new(&context_path).exists()
    {
        error!(
            "Error: Configuration files not found at '{}' or '{}'.",
            agents_path, context_path
        );
        error!("Please run 'halc onboard' first to initialize configurations.");
        std::process::exit(1);
    }

    if let Some(Commands::RunAndChat { request }) = args.command {
        run_runandchat_command(
            &agents_path,
            &context_path,
            args.ledger.clone(),
            request,
            verbosity,
        )
        .await?;
        return Ok(());
    }

    info!("Starting HALc...");

    // 1. Load agent config
    info!("Loading agents config from '{}'...", agents_path);
    let app_config = AppConfig::load_from_file(&agents_path)?;
    info!("Successfully loaded {} agents.", app_config.agents.len());

    // 1b. Load prompt templates
    let templates = halc::prompts::PromptTemplates::load(&app_config.prompts)
        .map_err(|e| format!("Failed to load prompt templates: {}", e))?;

    // 2. Load context
    info!("Loading context from '{}'...", context_path);
    let context_content = std::fs::read_to_string(&context_path)?;

    // 3. Initialize Context Store & Agent Runtimes
    let context_store = AgentContextStore::new();
    let mut runtimes = Vec::new();

    for agent_cfg in app_config.agents {
        let runtime = AgentRuntime::new(agent_cfg, context_store.clone());
        runtimes.push(runtime);
    }

    // 4. Initialize agent contexts (Startup handshake)
    info!("Initializing agent contexts...");
    let mut init_futures = Vec::new();
    for runtime in &runtimes {
        let ctx = context_content.clone();
        let tmpl = templates.context_init.clone();
        init_futures.push(async move {
            let res = runtime.initialize_context(&ctx, &tmpl, verbosity).await;
            (runtime.config.name.clone(), res)
        });
    }

    let init_results = futures::future::join_all(init_futures).await;
    for (name, result) in init_results {
        match result {
            Ok(_) => info!("Agent '{}' successfully initialized.", name),
            Err(e) => warn!("Agent '{}' context initialization check returned error (endpoint might be mock or offline): {:?}", name, e),
        }
    }

    // 5. Initialize ledger
    let ledger_path = match args.ledger.clone() {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let resolved_dir = resolve_path(&app_config.ledger_directory);
            let _ = std::fs::create_dir_all(&resolved_dir);
            resolved_dir.join("ledger.jsonl")
        }
    };
    let ledger_path_str = ledger_path.to_string_lossy().to_string();
    let ledger = Ledger::new(Some(ledger_path_str.clone()));
    info!("Ledger logging enabled. File path: {}", ledger_path_str);

    // 6. Start Axum server
    let state = Arc::new(AppState {
        agents: tokio::sync::RwLock::new(runtimes),
        ledger,
        gateway_api_key: tokio::sync::RwLock::new(app_config.gateway_api_key.clone()),
        normalization: app_config.normalization,
        confrontation: app_config.confrontation,
        templates: tokio::sync::RwLock::new(templates),
    });

    let app = make_router(state);
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("HALc listening on http://{}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}

async fn run_runandchat_command(
    agents_path: &str,
    context_path: &str,
    ledger_path_opt: Option<String>,
    request: Option<String>,
    verbosity: halc::config::Verbosity,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load agent config
    let app_config = AppConfig::load_from_file(agents_path)?;

    // 1b. Load prompt templates
    let templates = halc::prompts::PromptTemplates::load(&app_config.prompts)
        .map_err(|e| format!("Failed to load prompt templates: {}", e))?;

    // 2. Load context
    let context_content = std::fs::read_to_string(context_path)?;

    // 3. Prompt/read request if not provided
    let request_str = match request {
        Some(r) => r,
        None => {
            println!(
                "{}",
                "=== HALc Interactive Confrontation Mode ===".bold().cyan()
            );
            println!("Enter your request/problem statement:");
            std::io::stdout().flush().unwrap();
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let trimmed = input.trim().to_string();
            if trimmed.is_empty() {
                return Err("Request cannot be empty.".into());
            }
            trimmed
        }
    };

    println!("\nRequest: {}\n", request_str.bold().yellow());

    // 4. Initialize Context Store & Agent Runtimes
    let context_store = AgentContextStore::new();
    let mut runtimes = Vec::new();

    for agent_cfg in app_config.agents {
        let runtime = AgentRuntime::new(agent_cfg, context_store.clone());
        runtimes.push(runtime);
    }

    // 5. Initialize agent contexts (Startup handshake)
    let spinner = if verbosity >= halc::config::Verbosity::Verbose {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        pb.set_message("Initializing agent contexts...");
        pb.enable_steady_tick(std::time::Duration::from_millis(80));
        Some(pb)
    } else {
        None
    };

    let mut init_futures = Vec::new();
    for runtime in &runtimes {
        let ctx = context_content.clone();
        let spinner_clone = spinner.clone();
        let tmpl = templates.context_init.clone();
        init_futures.push(async move {
            let res = runtime.initialize_context(&ctx, &tmpl, verbosity).await;
            if let Some(ref pb) = spinner_clone {
                match &res {
                    Ok(_) => pb.println(format!(
                        "{} Agent '{}' successfully initialized.",
                        "✔".green(),
                        runtime.config.name.cyan()
                    )),
                    Err(e) => pb.println(format!(
                        "{} Agent '{}' context initialization check returned error: {:?}",
                        "✘".red(),
                        runtime.config.name.red(),
                        e
                    )),
                }
            }
            (runtime.config.name.clone(), res)
        });
    }

    let init_results = futures::future::join_all(init_futures).await;
    if let Some(ref pb) = spinner {
        pb.finish_and_clear();
    }
    for (name, result) in init_results {
        if result.is_err() {
            warn!("Agent '{}' context initialization check failed (endpoint might be mock or offline).", name);
        }
    }

    // 6. Initialize ledger
    let ledger_path = match ledger_path_opt {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let resolved_dir = resolve_path(&app_config.ledger_directory);
            let _ = std::fs::create_dir_all(&resolved_dir);
            resolved_dir.join("ledger.jsonl")
        }
    };
    let ledger_path_str = ledger_path.to_string_lossy().to_string();
    let ledger = Ledger::new(Some(ledger_path_str.clone()));

    // 7. Run Confrontation
    let request_id = uuid::Uuid::new_v4().to_string();
    let params = halc::algorithm::ConfrontationParams {
        normalization: &app_config.normalization,
        confrontation: &app_config.confrontation,
        templates: &templates,
    };
    let decision = halc::algorithm::run_confrontation(
        &request_id,
        &request_str,
        &runtimes,
        &ledger,
        None,
        verbosity,
        &params,
    )
    .await?;

    println!(
        "{}",
        "================ Final Decision ================"
            .bold()
            .cyan()
    );
    println!("Winning Agent: {}", decision.winning_agent.bold().yellow());
    println!(
        "Winning Score: {}",
        decision.winning_score.to_string().green()
    );
    println!("\nWinning Proposal:\n{}", decision.proposal.green());
    println!("\nRationale:\n{}", decision.rationale.italic().green());
    if !decision.risks.is_empty() {
        println!("\nKey Risks:\n{}", decision.risks.join("\n").red());
    }
    println!(
        "{}",
        "================================================"
            .bold()
            .cyan()
    );

    Ok(())
}
