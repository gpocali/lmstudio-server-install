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
MANAGER_PORT="8080"
REPO_RAW_URL="https://raw.githubusercontent.com/gpocali/lmstudio-server-install/main"

IS_UPDATE=false
if [ -f "$CONFIG_FILE" ] && [ -f "${APP_DIR}/model_manager.py" ]; then
  IS_UPDATE=true
fi

echo "================================================================="
if [ "$IS_UPDATE" = true ]; then
  echo "        Madison AI Workstation Updater (Non-Interactive)         "
else
  echo "        Madison AI Workstation Initial Installer                 "
fi
echo "   Target Directory: ${APP_DIR}                                  "
echo "================================================================="

# 1. Check Mount Point
if [ ! -d "$BASE_STORAGE" ]; then
  echo "[-] Base storage mount '${BASE_STORAGE}' not found. Please ensure the storage volume is mounted."
  exit 1
fi

# 2. Setup Dedicated User & Directory Layout
echo "[+] Ensuring service user '${SERVICE_USER}' and modular directory layout..."
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  useradd -r -m -d "$APP_DIR" -s /bin/bash -c "Madison Service Account" "$SERVICE_USER"
else
  usermod -d "$APP_DIR" -s /bin/bash "$SERVICE_USER"
fi

mkdir -p "$APP_DIR" "$CORE_DIR" "$WEB_DIR" "$MODELS_DIR" "$WORKSPACES_DIR" \
         "${APP_DIR}/.cache/lm-studio/models" "${APP_DIR}/.lmstudio/models"

chmod 755 "$APP_DIR" "$CORE_DIR" "$WEB_DIR" "$MODELS_DIR" "$WORKSPACES_DIR"
chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$APP_DIR"

git config --global --add safe.directory "*" 2>/dev/null || true
sudo -u "$SERVICE_USER" git config --global --add safe.directory "*" 2>/dev/null || true

# 3. Install System & Python Dependencies
echo "[+] Installing system packages and runtime dependencies..."
apt-get update -y
apt-get install -y curl ca-certificates jq gnupg git python3 python3-pip python3-venv python3-uvicorn python3-fastapi python3-requests

pip3 install -U duckduckgo_search --break-system-packages 2>/dev/null || pip3 install -U duckduckgo_search || true

# 4. Remove Legacy Open WebUI Containers if Present
if command -v docker >/dev/null 2>&1; then
  if docker ps -a --format '{{.Names}}' | grep -Eq "^open-webui$"; then
    echo "[+] Removing legacy open-webui container..."
    docker rm -f open-webui >/dev/null 2>&1 || true
  fi
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

# 7. Fetch & Deploy Madison Studio Application Files
echo ""
echo "--- [ Deploying Madison Application Files ] ---"
CACHE_BUST="?t=$(date +%s)"

curl -fsSL "${REPO_RAW_URL}/model_manager.py${CACHE_BUST}" -o "${APP_DIR}/model_manager.py"
touch "${CORE_DIR}/__init__.py"
curl -fsSL "${REPO_RAW_URL}/core/hardware.py${CACHE_BUST}" -o "${CORE_DIR}/hardware.py"
curl -fsSL "${REPO_RAW_URL}/core/models.py${CACHE_BUST}" -o "${CORE_DIR}/models.py"
curl -fsSL "${REPO_RAW_URL}/core/github_vault.py${CACHE_BUST}" -o "${CORE_DIR}/github_vault.py"
curl -fsSL "${REPO_RAW_URL}/core/agent_engine.py${CACHE_BUST}" -o "${CORE_DIR}/agent_engine.py"
curl -fsSL "${REPO_RAW_URL}/core/personas.py${CACHE_BUST}" -o "${CORE_DIR}/personas.py"
curl -fsSL "${REPO_RAW_URL}/core/task_queue.py${CACHE_BUST}" -o "${CORE_DIR}/task_queue.py"
curl -fsSL "${REPO_RAW_URL}/web/index.html${CACHE_BUST}" -o "${WEB_DIR}/index.html"

echo "[*] Validating Python module compilation..."
python3 -m py_compile "${APP_DIR}/model_manager.py"
python3 -m py_compile "${CORE_DIR}/hardware.py"
python3 -m py_compile "${CORE_DIR}/models.py"
python3 -m py_compile "${CORE_DIR}/github_vault.py"
python3 -m py_compile "${CORE_DIR}/agent_engine.py"
python3 -m py_compile "${CORE_DIR}/personas.py"
python3 -m py_compile "${CORE_DIR}/task_queue.py"

chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$APP_DIR"
echo "[✓] All files deployed and verified."

# 8. Configure & Start Systemd Services
echo ""
echo "--- [ Systemd Service Configuration ] ---"

tee /etc/systemd/system/lmstudio.service > /dev/null <<EOF
[Unit]
Description=LM Studio Backend Link Daemon
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
Description=Madison AI Workstation Web UI
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

# 9. Save Configuration
cat > "$CONFIG_FILE" <<EOF
INSTALLED_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
BASE_STORAGE="${BASE_STORAGE}"
APP_DIR="${APP_DIR}"
EOF
chown "${SERVICE_USER}:${SERVICE_GROUP}" "$CONFIG_FILE"
chmod 600 "$CONFIG_FILE"

SERVER_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "================================================================="
echo "                  MADISON WORKSTATION READY                      "
echo "================================================================="
echo "• Madison Web UI:         http://${SERVER_IP}:${MANAGER_PORT}"
echo "• LM Studio Backend API:  http://${SERVER_IP}:${LM_PORT}/v1"
echo "• Workspaces Root:        ${WORKSPACES_DIR}"
echo "================================================================="