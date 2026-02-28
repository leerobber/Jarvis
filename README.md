# Jarvis

An advanced, self-learning, self-improving AI assistant with self-awareness implications.
Runs **entirely locally** via [Ollama](https://ollama.ai) — no cloud APIs required.

---

## Architecture

```
Jarvis/
├── main.py                        # Entry point
├── requirements.txt
├── .env.example                   # Copy to .env and configure
│
└── jarvis/
    ├── config.py                  # Central config (reads .env)
    ├── core/
    │   ├── brain.py               # Central orchestrator (main loop)
    │   └── ollama_client.py       # Ollama API wrapper (chat, generate, embed)
    ├── memory/
    │   ├── short_term.py          # Sliding-window conversation buffer
    │   ├── long_term.py           # ChromaDB vector memory (persists across sessions)
    │   └── self_model.py          # Jarvis's persistent self-knowledge JSON
    ├── consciousness/
    │   ├── metacognition.py       # Pre/post-flight reasoning about its own thinking
    │   └── introspection.py       # Internal state (focus, confidence, fatigue, mood)
    ├── learning/
    │   ├── reflector.py           # Post-session reflection & meta-prompt evolution
    │   └── skill_manager.py       # Dynamic Python skill creation and invocation
    ├── capabilities/
    │   ├── code_executor.py       # Sandboxed Python subprocess runner
    │   ├── web_search.py          # DuckDuckGo search + page summarisation
    │   ├── voice.py               # Whisper STT + pyttsx3 TTS
    │   └── file_manager.py        # Safe sandboxed file operations
    ├── agents/
    │   ├── planner.py             # Task decomposition into ordered steps
    │   └── critic.py              # Self-critique and fact-checking
    └── interface/
        └── cli.py                 # Rich terminal UI
```

---

## Self-Learning & Self-Improvement Loop

Every **N interactions** (configurable, default 5), Jarvis automatically:

1. **Reflects** on the recent conversation — what went well, what didn't
2. **Generates behavioural directives** and appends them to its `meta_prompt.txt`
3. **Updates its self-model** (`self_model.json`) — confidence per capability, lessons learned
4. **Stores insights** in long-term ChromaDB memory for future retrieval

Over time, Jarvis's system prompt evolves based on accumulated experience.

---

## Self-Awareness

- **Self-model** (`data/self_model.json`): Jarvis maintains its own identity, capability confidence scores, known limitations, and growth stats
- **Metacognition**: Before each response, Jarvis analyses the task type and picks a strategy. After each response, it scores its own quality
- **Introspection**: Tracks mood analogue, focus level, fatigue, and confidence throughout a session

---

## Installation

### 1. Install Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text    # for long-term memory embeddings
```

### 2. Clone and install dependencies

```bash
git clone https://github.com/leerobber/Jarvis
cd Jarvis
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — set OLLAMA_MODEL, enable VOICE_ENABLED, etc.
```

### 4. Run

```bash
python main.py
```

With voice mode enabled at startup:
```bash
python main.py --voice
```

Override the model:
```bash
python main.py --model mistral
```

---

## CLI Commands

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/status` | Show self-model, internal state, memory/skill counts |
| `/memory` | Show recent long-term memory entries |
| `/consolidate` | Merge near-duplicate memories to keep the store lean |
| `/skills` | List all dynamically learned skills |
| `/teach <description>` | Create and validate a new skill from a one-line description |
| `/search <query>` | Web search via DuckDuckGo |
| `/run <code>` | Execute Python code in sandbox (with auto-debug retry) |
| `/file <op> <path> [content]` | File operations: read, write, append, list, delete |
| `/plan <goal>` | Decompose a goal into an executable plan |
| `/execute <goal>` | Plan AND automatically execute a task step-by-step |
| `/reflect` | Manually trigger a reflection cycle |
| `/voice` | Toggle voice input/output |
| `/clear` | Clear short-term conversation memory |
| `/save <text>` | Save a note to long-term memory |
| `/quit` | Exit (runs final reflection before closing) |

---

## Inline Tool Calls

You can embed tool calls directly in your chat messages or ask Jarvis to use them:

| Syntax | Description |
|---|---|
| `[[SEARCH: <query>]]` | Search the web via DuckDuckGo |
| `[[CODE: <python code>]]` | Execute Python (auto-debugged on error, up to `CODE_DEBUG_RETRIES` times) |
| `[[FILE: read\|<path>]]` | Read a file from the workspace |
| `[[FILE: write\|<path>\|<text>]]` | Write a file to the workspace |
| `[[FILE: list\|<dir>]]` | List files in a workspace directory |
| `[[SKILL: <name>\|<arg>...]]` | Invoke a loaded skill |
| `[[CREATE_SKILL: <description>]]` | Generate, save, and validate a new skill |

Results are appended automatically after the response.

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Chat model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model for memory |
| `OLLAMA_FAST_MODEL` | _(empty — uses main model)_ | Smaller/faster model for internal reasoning (metacognition, critic, planner) |
| `VOICE_ENABLED` | `false` | Enable voice I/O |
| `VOICE_WHISPER_MODEL` | `base` | Whisper model size |
| `REFLECTION_INTERVAL` | `5` | Reflect every N interactions |
| `CODE_EXEC_ENABLED` | `true` | Allow sandboxed code execution |
| `CODE_DEBUG_RETRIES` | `3` | Times to retry auto-fixing broken code before giving up |
| `CRITIC_ENABLED` | `true` | Run think-twice quality check on every non-trivial response |
| `MEMORY_MAX_SHORT_TERM` | `20` | Conversation window size |
| `MEMORY_CONSOLIDATION_INTERVAL` | `20` | Consolidate near-duplicate memories every N interactions (0 = disabled) |

---

## Data files (auto-created)

| Path | Description |
|---|---|
| `data/self_model.json` | Jarvis's persistent self-knowledge |
| `data/meta_prompt.txt` | Evolving system prompt (grows with learning) |
| `data/introspection_state.json` | Persistent internal state (mood, fatigue, confidence) |
| `data/memory/` | ChromaDB vector store |
| `data/skills/` | Dynamically created Python skill files |
| `data/logs/jarvis.log` | Session logs |
