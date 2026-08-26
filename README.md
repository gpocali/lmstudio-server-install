# LM Studio Headless Server & Remote Model Manager

An automated deployment script and standalone web manager for running [LM Studio](https://lmstudio.ai/) headlessly on Ubuntu Server 24.04 LTS (and compatible versions). 

This setup runs LM Studio on a dedicated mount (`/storage`), allows remote access and management across your LAN, provides a browser-based Hugging Face GGUF search and downloader, and optionally deploys Open WebUI with live web search.

---

## Quickstart (One-Liner Installation)

You do not need to clone the repository manually. Run this command directly on your Ubuntu server:

```bash
wget -qO- https://raw.githubusercontent.com/gpocali/lmstudio-server-install/main/install.sh | sudo bash

```

---

## Features

* **Dedicated Service User:** Runs under an isolated system account (`lmstudio`) rather than a physical user profile.
* **Dedicated Storage Mount:** Isolates application binaries, cache, and downloaded GGUF weights to `/storage/lmstudio`.
* **Systemd Background Daemons:** Automatically starts `lmstudio.service` and `modelmanager.service` at boot with automatic recovery.
* **Standalone Web Model Manager:** Web UI on port `8080` to search Hugging Face GGUF repositories, view quantizations (`Q4_K_M`, `Q8_0`, etc.), and execute asynchronous background downloads.
* **Native LM Link Integration:** Pairs with the LM Studio desktop client across your network via encrypted mesh networking.
* **Optional Open WebUI Stack:** Docker-based web chat interface on port `3000` with DuckDuckGo search and URL scraping pre-configured.
* **Idempotent / Upgrade Safe:** Re-run the command at any time to pull updates, refresh dependencies, or reconfigure network parameters.

---

## Port Allocation

| Service | Port | Endpoint / Description |
| --- | --- | --- |
| **LM Studio API** | `1234` | `http://<SERVER_IP>:1234/v1` (OpenAI-compatible REST API) |
| **Model Manager UI** | `8080` | `http://<SERVER_IP>:8080` (Hugging Face GGUF browser & downloader) |
| **Open WebUI (Optional)** | `3000` | `http://<SERVER_IP>:3000` (Multi-user web chat + RAG search) |

---

## Prerequisites

* **OS:** Ubuntu Server 22.04 LTS / 24.04 LTS / 26.04 LTS
* **Mount Point:** A mounted storage drive at `/storage` (or modify `BASE_STORAGE` in `install.sh`)
* **Privileges:** `sudo` / root administrative access

---

## Interactive Installation Prompts

When running the installer, you will be prompted for:

1. **LM Link Authentication:** Provides a browser URL to pair the server with your LM Studio account if not already authenticated.
2. **Device Display Name:** Sets the name for this node in the LM Studio desktop client (defaults to hostname).
3. **Open WebUI Deployment:** Asks whether to deploy the Dockerized Open WebUI container (`y/n`).

---

## Usage & Management

### 1. Web Model Manager

Navigate to **`http://<SERVER_IP>:8080`** in any browser:

1. Search for a model name or organization (e.g., `Meta-Llama-3.1-8B`, `Qwen2.5-Coder`, `bartowski`).
2. Click **View Quantizations** to expand the list of available `.gguf` files.
3. Click **Download** to stream the model to `/storage/lmstudio/models` in the background.

### 2. Service Management

Both components run as systemd units:

```bash
# LM Studio Core Daemon
sudo systemctl status lmstudio.service
sudo systemctl restart lmstudio.service
journalctl -u lmstudio.service -f

# Model Manager Web UI
sudo systemctl status modelmanager.service
sudo systemctl restart modelmanager.service
journalctl -u modelmanager.service -f

```

### 3. Command Line Interface (`lms`)

The `lms` command is symlinked globally to `/usr/local/bin/lms`:

```bash
# View active server state and loaded models
lms status

# Load a downloaded model into memory
lms load

# Unload all active models from memory
lms unload --all

```

---

## Repository Structure

```text
lmstudio-server-install/
├── install.sh             # Master installer and update script
├── model_manager.py       # FastAPI backend + Hugging Face downloader UI
└── README.md              # Documentation

```
