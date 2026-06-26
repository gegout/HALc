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
use std::sync::Arc;
use tokio::sync::RwLock;

/// Representation of an agent's context details.
#[derive(Debug, Clone)]
pub struct AgentContext {
    /// Agent identifier.
    pub agent_name: String,

    /// Context text payload.
    pub content: String,
}

/// Thread-safe in-memory cache/store mapping agent name to its current private context.
#[derive(Debug, Clone)]
pub struct AgentContextStore {
    contexts: Arc<RwLock<HashMap<String, String>>>,
}

impl Default for AgentContextStore {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentContextStore {
    /// Creates a new empty context storage container.
    pub fn new() -> Self {
        Self {
            contexts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Fetches the context content associated with a given agent name.
    pub async fn get(&self, agent_name: &str) -> Option<String> {
        let guard = self.contexts.read().await;
        guard.get(agent_name).cloned()
    }

    /// Associates context content with an agent name, overwriting previous values.
    pub async fn set(&self, agent_name: String, context: String) {
        let mut guard = self.contexts.write().await;
        guard.insert(agent_name, context);
    }
}
