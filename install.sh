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
CORE_DIR="${APP_DIR}/core"
WEB_DIR="${APP_DIR}/web"
MODELS_DIR="${APP_DIR}/models"
WORKSPACES_DIR="${APP_DIR}/workspaces"
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
  echo "        LM Studio Modular Stack Updater (Non-Interactive)        "
else
  echo "        LM Studio Modular Stack Initial Installer                "
fi
echo "   Target Directory: ${APP_DIR}                                  "
echo "================================================================="

# 1. Check Mount Point
if [ ! -d "$BASE_STORAGE" ]; then
  echo "[-] Base storage mount '${BASE_STORAGE}' not found. Please ensure the storage volume is mounted."
  exit 1
fi

# 2. Setup Dedicated User & Directory Hierarchy
echo "[+] Ensuring service user '${SERVICE_USER}' and modular directory layout..."
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  useradd -r -m -d "$APP_DIR" -s /bin/bash -c "LM Studio Service Account" "$SERVICE_USER"
else
  usermod -d "$APP_DIR" -s /bin/bash "$SERVICE_USER"
fi

mkdir -p "$APP_DIR" "$CORE_DIR" "$WEB_DIR" "$MODELS_DIR" "$WORKSPACES_DIR" \
         "${APP_DIR}/.cache/lm-studio/models" "${APP_DIR}/.lmstudio/models"

chmod 755 "$APP_DIR" "$CORE_DIR" "$WEB_DIR" "$MODELS_DIR" "$WORKSPACES_DIR"
chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$APP_DIR"

# Configure Git safe directory globally across user accounts
git config --global --add safe.directory "*" 2>/dev/null || true
sudo -u "$SERVICE_USER" git config --global --add safe.directory "*" 2>/dev/null || true

# 3. Install System & Python Dependencies
echo "[+] Installing system packages and runtime dependencies..."
apt-get update -y
apt-get install -y curl ca-certificates jq gnupg git python3 python3-pip python3-venv python3-uvicorn python3-fastapi python3-requests

# 4. Install / Verify Docker Engine (for Open WebUI)
if ! command -v docker >/dev/null 2>&1; then
  echo "[+] Installing Docker Engine..."
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

usermod -aG docker "$SERVICE_USER" 2>/dev/null || true
if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
  usermod -aG docker "$SUDO_USER" 2>/dev/null || true
fi

# 5. Install / Update LM Studio CLI & Daemon
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

# 6. Model Library Linking & Indexing
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

# 7. Fetch & Deploy Modular Application Files
echo ""
echo "--- [ Deploying Modular Studio Files ] ---"

# Root entrypoint
curl -fsSL "${REPO_RAW_URL}/model_manager.py" -o "${APP_DIR}/model_manager.py"

# Core backend modules
touch "${CORE_DIR}/__init__.py"
curl -fsSL "${REPO_RAW_URL}/core/hardware.py" -o "${CORE_DIR}/hardware.py"
curl -fsSL "${REPO_RAW_URL}/core/models.py" -o "${CORE_DIR}/models.py"
curl -fsSL "${REPO_RAW_URL}/core/github_vault.py" -o "${CORE_DIR}/github_vault.py"
curl -fsSL "${REPO_RAW_URL}/core/agent_engine.py" -o "${CORE_DIR}/agent_engine.py"

# Frontend HTML/JS Studio UI
curl -fsSL "${REPO_RAW_URL}/web/index.html" -o "${WEB_DIR}/index.html"

# Verify python syntax before activating
echo "[*] Validating Python module compilation..."
python3 -m py_compile "${APP_DIR}/model_manager.py"
python3 -m py_compile "${CORE_DIR}/hardware.py"
python3 -m py_compile "${CORE_DIR}/models.py"
python3 -m py_compile "${CORE_DIR}/github_vault.py"
python3 -m py_compile "${CORE_DIR}/agent_engine.py"

chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$APP_DIR"
echo "[✓] All modular application files deployed and verified."

# 8. Configure & Start Systemd Services
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
Environment=PYTHONPATH=${APP_DIR}
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
echo "                  MODULAR STACK READY                            "
echo "================================================================="
echo "• LM Studio API:          http://${SERVER_IP}:${LM_PORT}/v1"
echo "• Code & Model Studio:    http://${SERVER_IP}:${MANAGER_PORT}"
echo "• Open WebUI:             http://${SERVER_IP}:${WEBUI_PORT}"
echo "• Workspaces Root:        ${WORKSPACES_DIR}"
echo "================================================================="