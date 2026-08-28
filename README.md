# Madison AI Workstation

**Madison** is a self-hosted, lightweight AI workstation and development studio engineered on top of LM Studio's local engine. It combines an asynchronous multi-session chat assistant, an autonomous local coding environment, and a Hugging Face model manager into a single interface.

---

## Key Features

### 💬 Grounded Chat Bot
* **Full Markdown & Code Highlighting:** Render GitHub-flavored markdown, structured data tables, blockquotes, and language-tagged code blocks via Highlight.js.
* **Live Web Search (RAG):** Toggle real-time search context injection powered by DuckDuckGo without external API keys.
* **Temporal Grounding:** Anchors offline models with real-world temporal awareness to prevent hallucinated search claims.
* **Copy & Export Controls:** Message-level one-click clipboard copying and full conversation exports to Markdown (`.md`) or JSON (`.json`).

### 💻 Autonomous Code Studio
* **Workspace Management:** Clone repositories across multiple GitHub accounts, switch branches, create feature branches, and delete local workspaces with one click.
* **Multi-Threaded Workflows:** Manage multiple isolated conversation threads per repository to track separate features or refactoring tasks.
* **Disk Staging & Review:** Review unified Git diffs directly inside chat bubbles before committing.
* **Auto-Commit & Changelog:** Automatically generate Conventional Commit messages from staged diffs and append summaries directly to `CHANGELOG.md`.

### ⚡ Background Agent Queue & Model Optimization
* **Asynchronous Task Processing:** Continue working across different workspaces or chat sessions while tasks execute in the background.
* **Model Grouping Optimization:** Minimizes VRAM reloads by prioritizing queued tasks that share the currently loaded model.
* **Activity & Unread Indicators:** Live SVG spinners on tabs and sub-threads show active background jobs, and pulsing notification dots flag unread results.
* **Automatic Fallback Loading:** Automatically detects and loads available local models when prompts are submitted with no model active in VRAM.

### 🎭 Modular Persona Vault (`core/personas.py`)
* **Domain-Specific Templates:** Pre-configured system prompts for general assistance, systems architecture, autonomous engineering, and security review.
* **Runtime Overrides:** Live-edit prompts directly from the UI or restore defaults with a single click.

### 📦 Model Management & Dynamic Tuner
* **GGUF Catalog Explorer:** Search, filter, and inspect top quantization variants directly from Hugging Face with creator verification and trust scores.
* **Interactive Model Tuner:** Dynamically configure context lengths (from 2K up to 131K tokens), GPU offload layers, and keep-alive (TTL) timers.
* **Hardware Telemetry:** Live GPU VRAM allocation, system RAM, and storage utilization readouts.

### 🖥️ Responsive IDE Layout
* **Elastic Textarea:** Inputs auto-expand up to ~12 lines before scrolling to preserve visibility on long prompts.
* **Resizable Panels:** Horizontal and vertical splitters let you customize panel dimensions.
* **Centered Viewport:** Prompts and responses are constrained to a readable column on ultra-wide displays.

---

## Architecture Overview

```text
/storage/lmstudio/
├── model_manager.py          # FastAPI server, REST API router & static UI host
├── core/
│   ├── hardware.py          # Hardware detection (VRAM, RAM, Disk) & LMS CLI wrapper
│   ├── models.py            # Hugging Face GGUF indexer & download engine
│   ├── github_vault.py      # Git workspace controller & multi-account token manager
│   ├── agent_engine.py      # Multi-file patch parser & commit message generator
│   ├── personas.py          # Persona templates and prompt construction
│   └── task_queue.py        # Asynchronous job queue with model-batching optimization
└── web/
    └── index.html           # Single-page interface (Tailwind CSS, Marked, Highlight.js)

```

---

## Installation & Deployment

### Prerequisites

* Ubuntu / Debian Linux
* Storage volume mounted at `/storage`
* Python 3.10+ with `pip`

### Quick Install / Update

Run the automated installation script with cache-busting enabled:

```bash
curl -fsSL "https://raw.githubusercontent.com/gpocali/lmstudio-server-install/main/install.sh?t=$(date +%s)" | sudo bash

```

### Manual Dependency Installation

If configuring manually, install the required packages:

```bash
sudo apt-get update -y
sudo apt-get install -y curl jq git python3 python3-pip python3-uvicorn python3-fastapi python3-requests
sudo pip3 install -U duckduckgo_search --break-system-packages

```

---

## Service Management

Madison runs as two systemd services:

| Service | Description | Port |
| --- | --- | --- |
| `lmstudio.service` | Headless LM Studio Engine & Link Daemon | `1234` |
| `modelmanager.service` | Madison Workstation Web Application | `8080` |

### Common Commands

```bash
# Restart the Madison UI service
sudo systemctl restart modelmanager.service

# View live service logs
sudo journalctl -u modelmanager.service -f

# Check LM Studio daemon status
sudo systemctl status lmstudio.service

```

---

## Accessing the Interface

Open your browser and navigate to:

```text
http://<SERVER_IP>:8080

```
