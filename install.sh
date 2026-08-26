#!/usr/bin/env bash
set -euo pipefail

if [ "$EUID" -ne 0 ]; then
  echo "[-] Please run this script with sudo: wget -qO- ... | sudo bash"
  exit 1
fi

SERVICE_USER="lmstudio"
SERVICE_GROUP="lmstudio"
BASE_STORAGE="/storage"
APP_DIR="${BASE_STORAGE}/lmstudio"
MODELS_DIR="${APP_DIR}/models"
WORKSPACE_DIR="${APP_DIR}/workspace"
CONFIG_FILE="${APP_DIR}/.install_config"
LM_PORT="1234"
WEBUI_PORT="3000"
MANAGER_PORT="8080"
REPO_RAW_URL="https://raw.githubusercontent.com/gpocali/lmstudio-server-install/main"

IS_UPDATE=false
if [ -f "$CONFIG_FILE" ] && [ -f "${APP_DIR}/model_manager.py" ]; then
  IS_UPDATE=true
fi

echo "================================================================="
if [ "$IS_UPDATE" = true ]; then
  echo "        LM Studio Stack Updater (Non-Interactive Mode)          "
else
  echo "        LM Studio Stack Initial Installer                        "
fi
echo "   Target Directory: ${APP_DIR}                                  "
echo "================================================================="

# 1. Check Mount Point
if [ ! -d "$BASE_STORAGE" ]; then
  echo "[-] Base storage mount '${BASE_STORAGE}' not found. Please ensure the drive is mounted."
  exit 1
fi

# 2. Setup Dedicated User & Directories
echo "[+] Ensuring dedicated user '${SERVICE_USER}' and workspace structure..."
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  useradd -r -m -d "$APP_DIR" -s /bin/bash -c "LM Studio Service Account" "$SERVICE_USER"
else
  usermod -d "$APP_DIR" -s /bin/bash "$SERVICE_USER"
fi

mkdir -p "$APP_DIR" "$MODELS_DIR" "$WORKSPACE_DIR" "${APP_DIR}/.cache/lm-studio/models" "${APP_DIR}/.lmstudio/models"
chmod 755 "$APP_DIR"
chmod -R u+rwX,go+rX "$APP_DIR"
chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$APP_DIR"

# 3. Install System & Python Dependencies
echo "[+] Installing system packages and Python dependencies..."
apt-get update -y
apt-get install -y curl ca-certificates jq gnupg git pipx python3 python3-pip python3-venv python3-uvicorn python3-fastapi python3-requests

# Install Aider using pipx or dedicated isolated venv
echo "[+] Configuring Aider AI Agent in isolated environment..."
if ! command -v aider >/dev/null 2>&1; then
  PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin pipx install --force aider-chat || {
    echo "[*] Falling back to dedicated virtualenv for Aider..."
    python3 -m venv "${APP_DIR}/venv_aider"
    "${APP_DIR}/venv_aider/bin/pip" install --upgrade pip
    "${APP_DIR}/venv_aider/bin/pip" install aider-chat || true
    ln -sf "${APP_DIR}/venv_aider/bin/aider" /usr/local/bin/aider
  }
fi

# 4. Install / Update LM Studio CLI & llmster Daemon
echo ""
echo "--- [ LM Studio Engine Check ] ---"
LMS_BIN=""
if [ -f "${APP_DIR}/.cache/lm-studio/bin/lms" ]; then
  LMS_BIN="${APP_DIR}/.cache/lm-studio/bin/lms"
elif [ -f "${APP_DIR}/.lmstudio/bin/lms" ]; then
  LMS_BIN="${APP_DIR}/.lmstudio/bin/lms"
elif command -v lms >/dev/null 2>&1; then
  LMS_BIN="$(command -v lms)"
fi

if [ "$IS_UPDATE" = true ] && [ -n "$LMS_BIN" ] && [ -x "$LMS_BIN" ]; then
  echo "[✓] Existing LM Studio installation found."
  sudo -u "$SERVICE_USER" HOME="$APP_DIR" "$LMS_BIN" daemon update 2>/dev/null || true
  CURRENT_VER=$(sudo -u "$SERVICE_USER" HOME="$APP_DIR" "$LMS_BIN" --version 2>/dev/null || echo "Installed")
  echo "[✓] Engine Version: $CURRENT_VER"
else
  echo "[+] Fetching LM Studio installer..."
  sudo -u "$SERVICE_USER" HOME="$APP_DIR" bash -c "curl -fsSL https://lmstudio.ai/install.sh | bash"
  
  if [ -f "${APP_DIR}/.cache/lm-studio/bin/lms" ]; then
    LMS_BIN="${APP_DIR}/.cache/lm-studio/bin/lms"
  elif [ -f "${APP_DIR}/.lmstudio/bin/lms" ]; then
    LMS_BIN="${APP_DIR}/.lmstudio/bin/lms"
  else
    LMS_BIN=$(find "$APP_DIR" -type f -name "lms" -perm /111 2>/dev/null | head -n 1)
  fi
fi

if [ -z "$LMS_BIN" ] || [ ! -x "$LMS_BIN" ]; then
  echo "[-] Error: Failed to locate or execute 'lms' binary."
  exit 1
fi

chmod +x "$LMS_BIN"
ln -sf "$LMS_BIN" /usr/local/bin/lms

# 5. Model Library Linking & Indexing
echo ""
echo "--- [ Model Library Linking & Indexing ] ---"
for model_dir in "$MODELS_DIR"/*; do
  if [ -d "$model_dir" ]; then
    folder_name=$(basename "$model_dir")
    sudo -u "$SERVICE_USER" ln -sfn "$model_dir" "${APP_DIR}/.cache/lm-studio/models/${folder_name}"
    sudo -u "$SERVICE_USER" ln -sfn "$model_dir" "${APP_DIR}/.lmstudio/models/${folder_name}"
  fi
done

find "$MODELS_DIR" -type f -name "*.gguf" | while read -r gguf_path; do
  if ! echo "$gguf_path" | grep -Eq -- "-0000[2-9]-of-"; then
    sudo -u "$SERVICE_USER" HOME="$APP_DIR" "$LMS_BIN" import --yes --symbolic-link "$gguf_path" >/dev/null 2>&1 || true
  fi
done
echo "[✓] Model indexing complete."

# 6. Fetch Model Manager UI
echo ""
echo "--- [ Updating Model Manager Script ] ---"
curl -fsSL "${REPO_RAW_URL}/model_manager.py" -o "${APP_DIR}/model_manager.py"
chmod 755 "${APP_DIR}/model_manager.py"
chown "${SERVICE_USER}:${SERVICE_GROUP}" "${APP_DIR}/model_manager.py"

# 7. Configure AI Agent Workspace & Guardrails
echo ""
echo "--- [ Configuring AI Agent Workspace ] ---"
mkdir -p "$WORKSPACE_DIR"

if [ ! -d "${WORKSPACE_DIR}/.git" ]; then
  (
    cd "$WORKSPACE_DIR"
    git init
    git config user.name "AI Coding Agent"
    git config user.email "agent@lmstudio.local"
  )
fi

# Mirror codebase into workspace
cp -f "${APP_DIR}/model_manager.py" "${WORKSPACE_DIR}/" 2>/dev/null || true
curl -fsSL "${REPO_RAW_URL}/install.sh" -o "${WORKSPACE_DIR}/install.sh" 2>/dev/null || true

# Aider Configuration for local LM Studio API
cat > "${WORKSPACE_DIR}/.aider.conf.yml" <<EOF
openai-api-base: http://127.0.0.1:${LM_PORT}/v1
openai-api-key: lm-studio
model: openai/gemma-4-26b-a4b-it-ud
edit-format: diff
auto-commits: true
attribute-author: false
attribute-committer: false
show-diffs: true
EOF

# Pre-Push Syntax Validator Script
cat > "${WORKSPACE_DIR}/validate-and-push.sh" << 'EOF'
#!/usr/bin/env bash
set -e

echo "[*] Validating Python syntax..."
find . -maxdepth 2 -name "*.py" -exec python3 -m py_compile {} +

echo "[*] Validating Shell script syntax..."
find . -maxdepth 2 -name "*.sh" -exec bash -n {} +

echo "[✓] All syntax tests passed!"

TARGET_BRANCH="${1:-dev}"
echo "[*] Pushing verified changes to branch '${TARGET_BRANCH}'..."
git push origin "$TARGET_BRANCH"
EOF
chmod +x "${WORKSPACE_DIR}/validate-and-push.sh"

# Global CLI launcher for the workspace
cat > /usr/local/bin/lms-agent << EOF
#!/usr/bin/env bash
cd "${WORKSPACE_DIR}"
echo "====================================================="
echo "   LM Studio AI Agent Workspace: ${WORKSPACE_DIR}    "
echo "   Connecting to: http://127.0.0.1:${LM_PORT}/v1     "
echo "====================================================="
aider model_manager.py install.sh "\$@"
EOF
chmod +x /usr/local/bin/lms-agent
chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$WORKSPACE_DIR"

# 8. Write Systemd Services
echo ""
echo "--- [ Systemd Service Configuration ] ---"

tee /etc/systemd/system/lmstudio.service > /dev/null <<EOF
[Unit]
Description=LM Studio Headless Server & Link Daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=forking
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${APP_DIR}
Environment=HOME=${APP_DIR}
Environment=PATH=/usr/local/bin:${APP_DIR}/.cache/lm-studio/bin:${APP_DIR}/.lmstudio/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=${LMS_BIN} server start --port ${LM_PORT} --bind 0.0.0.0 --cors
ExecStop=${LMS_BIN} server stop
RemainAfterExit=yes
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

tee /etc/systemd/system/modelmanager.service > /dev/null <<EOF
[Unit]
Description=LM Studio Custom Model Manager Web UI
After=network.target lmstudio.service
Wants=lmstudio.service

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${APP_DIR}
Environment=HOME=${APP_DIR}
Environment=PATH=/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/usr/bin/python3 ${APP_DIR}/model_manager.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now lmstudio.service modelmanager.service
systemctl restart lmstudio.service modelmanager.service

# 9. Open WebUI Container Management
echo ""
echo "--- [ Open WebUI Container Management ] ---"
INSTALL_WEBUI="y"
if [ "$IS_UPDATE" = true ]; then
  if [ -f "$CONFIG_FILE" ]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
  fi
  INSTALL_WEBUI="${ENABLE_OPEN_WEBUI:-y}"
fi

if [[ "$INSTALL_WEBUI" =~ ^[Yy]$ ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  fi

  WEBUI_DATA_DIR="${APP_DIR}/webui_data"
  mkdir -p "$WEBUI_DATA_DIR"
  chown -R 1000:1000 "$WEBUI_DATA_DIR"

  if docker ps -a --format '{{.Names}}' | grep -Eq "^open-webui$"; then
    docker pull ghcr.io/open-webui/open-webui:main >/dev/null
    docker rm -f open-webui >/dev/null
  fi

  docker run -d \
    --name open-webui \
    -p "${WEBUI_PORT}:8080" \
    --add-host=host.docker.internal:host-gateway \
    -e OPENAI_API_BASE_URL="http://host.docker.internal:${LM_PORT}/v1" \
    -e OPENAI_API_KEY="lm-studio" \
    -e ENABLE_WEB_SEARCH=True \
    -e WEB_SEARCH_ENGINE=duckduckgo \
    -e ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION=True \
    -e RAG_WEB_SEARCH_RESULT_COUNT=5 \
    -v "${WEBUI_DATA_DIR}:/app/backend/data" \
    --restart always \
    ghcr.io/open-webui/open-webui:main

  ENABLE_WEBUI_CONF="true"
else
  ENABLE_WEBUI_CONF="false"
fi

# 10. Save Configuration
cat > "$CONFIG_FILE" <<EOF
INSTALLED_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
ENABLE_OPEN_WEBUI="${ENABLE_WEBUI_CONF}"
BASE_STORAGE="${BASE_STORAGE}"
APP_DIR="${APP_DIR}"
EOF
chown "${SERVICE_USER}:${SERVICE_GROUP}" "$CONFIG_FILE"
chmod 600 "$CONFIG_FILE"

SERVER_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "================================================================="
echo "                   SETUP & AGENT READY                           "
echo "================================================================="
echo "• LM Studio API:          http://${SERVER_IP}:${LM_PORT}/v1"
echo "• Model Manager UI:       http://${SERVER_IP}:${MANAGER_PORT}"
echo "• Open WebUI:             http://${SERVER_IP}:${WEBUI_PORT}"
echo "• AI Agent Workspace:     ${WORKSPACE_DIR}"
echo ""
echo "To start coding with the local AI agent, run:"
echo "   lms-agent"
echo "================================================================="