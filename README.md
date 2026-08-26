# LM Studio Headless Server & Remote Model Manager

An automated deployment script and standalone web manager for running [LM Studio](https://lmstudio.ai/) headlessly on Ubuntu Server 24.04 LTS. 

This repository allows you to host local LLMs on a dedicated server mount (`/storage`), control the instance remotely across your LAN, search and download Hugging Face GGUF models via a web interface, and optionally deploy Open WebUI with integrated web search.

---

## Features

- **Dedicated Service User:** Runs under an isolated, unprivileged system account (`lmstudio`) rather than a personal user profile.
- **Dedicated Storage Mount:** Isolates application caches, runtimes, and downloaded GGUF weights to `/storage/lmstudio`.
- **Systemd Background Services:** Automatically starts `lmstudio.service` and `modelmanager.service` at boot with auto-restart on failure.
- **Standalone Web Model Manager:** Web UI on port `8080` to search Hugging Face GGUF repositories, inspect available quantizations (`Q4_K_M`, `Q8_0`, etc.), and execute asynchronous background downloads directly to the server.
- **Native LM Link Support:** Pairs with the LM Studio desktop client across your network using end-to-end encrypted mesh networking.
- **Optional Open WebUI Integration:** Automated containerized chat interface on port `3000` with DuckDuckGo web search and URL scraping pre-configured.
- **Idempotent / Upgrade Safe:** Run the script repeatedly to update LM Studio CLI binaries, refresh dependencies, or reconfigure network parameters without losing data.

---

## Port Allocation

| Service | Port | Endpoint / Description |
| :--- | :--- | :--- |
| **LM Studio API** | `1234` | `http://<SERVER_IP>:1234/v1` (OpenAI-compatible endpoint) |
| **Model Manager UI** | `8080` | `http://<SERVER_IP>:8080` (Hugging Face model browser & downloader) |
| **Open WebUI (Optional)** | `3000` | `http://<SERVER_IP>:3000` (Multi-user web chat + RAG search) |

---

## Prerequisites

- **Operating System:** Ubuntu Server 22.04 LTS or 24.04 LTS
- **Mount Point:** A mounted storage drive available at `/storage` (or edit `BASE_STORAGE` in `install.sh`)
- **Privileges:** `sudo` / root administrative access

---

## Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/gpocali/lmstudio-server-install.git](https://github.com/gpocali/lmstudio-server-install.git)
cd lmstudio-server-install

```

### 2. Make the Installer Executable & Run

```bash
chmod +x install.sh
sudo ./install.sh

```

### 3. Interactive Prompts

During installation, the script will guide you through:

1. **LM Link Authentication:** Prompts for authentication if the machine has not yet been linked to an LM Studio account.
2. **Device Display Name:** Sets the name for this node as it appears in the LM Studio desktop client.
3. **Open WebUI Installation:** Asks whether to deploy the Dockerized Open WebUI container (`y/n`).

---

## Usage

### 1. Remote Model Management (Web UI)

Open a browser to **`http://<SERVER_IP>:8080`**:

1. Type a search query (e.g., `Llama-3.1`, `Qwen2.5-Coder`, `bartowski`).
2. Click **View Quantizations** to expand the list of `.gguf` variants.
3. Click **Download** to stream the file directly into `/storage/lmstudio/models` in the background.

### 2. Service Management

Both services are managed via `systemctl`:

```bash
# Check LM Studio daemon status
sudo systemctl status lmstudio.service

# Check Model Manager web service status
sudo systemctl status modelmanager.service

# View live application logs
journalctl -u lmstudio.service -f
journalctl -u modelmanager.service -f

```

### 3. CLI Management (`lms`)

The `lms` command-line utility is available globally from any terminal session:

```bash
# View active server state and loaded models
lms status

# Load a specific model into VRAM
lms load

# Unload active models
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
