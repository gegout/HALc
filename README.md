# HALc: Heuristic Agent Ledger by confrontation

HALc is a Rust application implementing a constructive confrontation algorithm between multiple Large Language Model (LLM) agents. Operating as an OpenAI-compatible gateway, HALc receives user queries, coordinates multi-agent framing, proposals, critiques, scores, and context updates, and returns the winning proposal.

---

## Academic Context & Literature Comparison

For a detailed theoretical background, academic context, and comparison with existing literature on multi-agent consensus and negotiation frameworks, please refer to the following documents:
* **Markdown Version**: [Heuristic Agent Ledger by confrontation (Markdown)](file:///home/cgegout/Documents/Antigravity/HALc/Heuristic%20Agent%20Ledger%20by%20confrontation/Heuristic%20Agent%20Ledger%20by%20confrontation.md)
* **PDF Version**: [Heuristic Agent Ledger by confrontation (PDF)](file:///home/cgegout/Documents/Antigravity/HALc/Heuristic%20Agent%20Ledger%20by%20confrontation/Heuristic%20Agent%20Ledger%20by%20confrontation_v1_2026_06_28.pdf)

---

## Motivation

Building robust architectures or solving complex problems using single LLM prompts often overlooks edge cases, security hazards, and operational risks. Constructive confrontation introduces a dialectic process where multiple distinct agent personas (e.g. `architect`, `product`, `security`) evaluate, challenge, and score each other's proposals. By aggregating these critiques and performing structured Copeland ranking, HALc filters out fragile concepts and presents the most robust solution as an OpenAI-compatible API response.

---

## Core Components & Module Structure

The Rust application is structured into the following modules under `src/`:

* **`config.rs`**: Manages configuration parsing of `/home/cgegout/.config/HALc/agent.toml`. Defines parameters for LLM endpoints (`AgentConfig`), normalized semantic aliases (`NormalizationConfig`), confrontation settings (`ConfrontationConfig` with mode and character limits), and prompt paths (`PromptsConfig`).
* **`prompts.rs`**: Dynamically loads and renders externalized markdown template files (`context_init.md`, `connectivity_test.md`, `synthesis.md`, `pairwise_comparison.md`, `tie_breaker.md`, and `context_generation.md`). Recompilation is never needed when changing agent templates.
* **`markdown.rs`**: Parses raw Markdown texts into segments keyed by headings. It dynamically normalizes custom headings using user-defined aliases (e.g. mapping "risks" or "concerns" to the `risks` field) and extracts lists and numeric ratings.
* **`agent.rs`**: Contains `AgentRuntime`, representing individual agent workers. It handles OpenAI-compatible JSON requests, endpoint communication, private context stores, and prompts rendering.
* **`scoring.rs`**: Contains tie-breaker algorithms. In case of a Copeland score draw, ties are resolved dynamically:
  1. Primary Tie-Breaker: Selecting the agent with the lowest average risk count across Stage 3 assessments.
  2. Secondary Tie-Breaker: Querying the configured judge agent using `tie_breaker.md`.
  3. Fallback: Reverting to the deterministic order of appearance in `agent.toml`.
* **`algorithm.rs`**: Orchestrates the multi-stage confrontation loop, executes comparisons, computes Copeland ranks, and outputs the beautiful Unicode stdio trace.
* **`server.rs`**: Integrates an Axum web server providing endpoints for health, model listing, client onboarding, status, and streaming chat completions.
* **`main.rs`**: Entrypoint handling CLI options, command invocation (onboard, status, runandchat), and initializing server structures.

---

## Detailed Confrontation Stages Specification

HALc coordinates execution across 6 distinct stages using structured Markdown interfaces between agents:

### Stage 1: Agent Synthesis
* **Input**: User Request ($R_u$), shared `context.md`, current shared context state, and Agent system prompt/persona.
* **Prompt**:
  Uses `/home/cgegout/.config/HALc/prompts/synthesis.md`.
* **Parsing**: Establishes a `Synthesis` object by reading Markdown sections and translating headings through the configured aliases:
  - `agent`: Submitting agent name.
  - `problem_statement`: Text under `# Problem Statement` (or matching aliases).
  - `priorities_used`: List items under `# Priorities Used`.
  - `constraints`: List items under `# Constraints`.
  - `success_metrics`: List items under `# Success Metrics`.
  - `proposal`: Text under `# Proposal`.
  - `rationale`: Text under `# Rationale`.
  - `claims`: List items under `# Claims`.
  - `assumptions`: List items under `# Assumptions`.
  - `risks`: List items under `# Risks`.
  - `evidence`: List items under `# Evidence`.
  - `expected_outcomes`: List items under `# Expected Outcomes`.
  - `raw_markdown`: Full unparsed Markdown text.

---

### Stage 2: Deterministic Pairwise Assignment
* **Logic**:
  1. Order proposer agents exactly as they appear in `agent.toml`: $A_0, A_1, A_2, \dots, A_{n-1}$.
  2. For each reviewer agent $A_i$, assign:
     $$A_i \text{ compares } A_{(i + 1) \bmod n} \text{ versus } A_{(i + 2) \bmod n}$$
* **Rules**:
  - Requires at least 3 proposer agents.
  - Reviewer never compares itself.
  - Each reviewer performs exactly one comparison.
  - Each proposal appears in exactly two comparisons when $n \geq 3$.
  - The graph is fully connected.

---

### Stage 3: Pairwise Comparison
* **Input**: Reviewer agent $A_i$ receives syntheses from $A_{(i + 1) \bmod n}$ and $A_{(i + 2) \bmod n}$.
* **Prompt**:
  Uses `/home/cgegout/.config/HALc/prompts/pairwise_comparison.md`.
* **Validation**:
  - Winner must match one of the compared agents.
  - Loser must match the other compared agent.
  - Confidence (0..10) is clamped to `0..10`.
  - Invalid comparison is retried once.
  - Failed comparison is ignored for ranking, but visible in stdio and ledger.

---

### Stage 4: Weighted Copeland Ranking
* **Logic**:
  - Weighted Copeland is the only ranking method.
  - For each valid comparison: `winner -> loser` with `weight = confidence`.
  - Compute:
    $$\text{score}(A) = \sum \text{confidence weights of wins} - \sum \text{confidence weights of losses}$$
  - Sort descending by:
    1. Weighted Copeland score
    2. Total win weight
    3. Fewer losses
    4. Fewer risks in synthesis
    5. Deterministic `agent.toml` order
  - The first ranked agent is the winner.
  - Tie-breaking rules in `scoring.rs` are executed dynamically if scores are equal.

---

### Stage 5: Final Response
* **Format**:
  HALc returns a structured Markdown document through the OpenAI-compatible API:
  ```markdown
  # Final Decision

  ## Winner
  [Agent Name]

  ## Winning Proposal
  ...

  ## Why It Won
  Summarize the main reasons from pairwise comparisons.

  ## Ranking
  1. Agent — score
  2. Agent — score
  3. Agent — score

  ## Main Risks
  ...

  ## Confrontation Summary
  Summarize the pairwise comparisons.
  ```

---

### Stage 6: Shared Context Generation
* **Logic**:
  1. HALc generates a **deterministic new context** from the algorithm results (ranking, comparisons, risks, lessons learned).
  2. The **original context** (`context.md`) already held by each agent is retrieved from the in-memory context store.
  3. If a **judge agent** is configured, it is called once with the concatenation of the original context and the new context, and asked to produce a single unified summarized context document using `context_generation.md`.
  4. If **no judge** is configured (or the judge call fails), the deterministic new context is used directly.
  5. A new Markdown file is written to the ledger folder: `context_<timestamp>.md` (format: `context_YYYY-MM-DD_HH-MM-SS.md`).
  6. The in-memory shared context for all agents is updated with the new content.
  - The shared context uses these headings:
    ```markdown
    # Shared Knowledge
    # User Request
    # Winning Proposal
    # Final Ranking
    # Pairwise Comparisons
    # Lessons Learned
    # Remaining Risks
    # Open Disagreements
    # Next Context For All Agents
    ```

---

## Terminal Output

Stdio output shows the new compact flow in a streaming hierarchy:

```text
HALc — Compact Distributed Pairwise Confrontation

Request
└── I want a new high-throughput event processing platform integrated with Kafka.

Stage 1 — Agent synthesis
├── architect
│   ├── Priorities: throughput, latency, budget
│   ├── Risks: 4
│   └── ✓ completed
├── product
│   └── ✓ completed
└── security
    └── ✓ completed

Stage 2 — Pairwise assignment
├── architect reviews product vs security
├── product reviews security vs architect
└── security reviews architect vs product

Stage 3 — Pairwise comparison
├── architect: product > security, confidence 7
├── product: architect > security, confidence 8
└── security: architect > product, confidence 6

Stage 4 — Weighted Copeland ranking
├── architect: +14 -0 = 14
├── product: +7 -6 = 1
└── security: +0 -15 = -15

Stage 5 — Final decision
└── architect wins

Stage 6 — Shared context
└── ~/.config/HALc/ledger/context_2026-06-26_14-35-22.md
```

---

## Installation & Compilation

Build cleanly:
```bash
cargo build --release
```

To run tests:
```bash
cargo test
```

## License

This project is licensed under the MIT License - see the [LICENSE](file:///home/cgegout/Documents/Antigravity/HALc/LICENSE) file for details. Copyright 2026 Cedric Gegout. All rights reserved.
