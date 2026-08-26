#!/usr/bin/env bash
set -euo pipefail

# Ensure script is executed with root/sudo privileges
if [ "$EUID" -ne 0 ]; then
  echo "[-] Please run this script with sudo: wget -qO- ... | sudo bash"
  exit 1
fi

SERVICE_USER="lmstudio"
SERVICE_GROUP="lmstudio"
BASE_STORAGE="/storage"
APP_DIR="${BASE_STORAGE}/lmstudio"
MODELS_DIR="${APP_DIR}/models"
CONFIG_FILE="${APP_DIR}/.install_config"
LM_PORT="1234"
WEBUI_PORT="3000"
MANAGER_PORT="8080"
REPO_RAW_URL="https://raw.githubusercontent.com/gpocali/lmstudio-server-install/main"

# Detect if this is a fresh install or an update
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
echo "[+] Ensuring dedicated user '${SERVICE_USER}' and directory permissions..."
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  useradd -r -m -d "$APP_DIR" -s /bin/bash -c "LM Studio Service Account" "$SERVICE_USER"
else
  usermod -d "$APP_DIR" -s /bin/bash "$SERVICE_USER"
fi

mkdir -p "$APP_DIR" "$MODELS_DIR" "${APP_DIR}/.cache" "${APP_DIR}/.lmstudio"
chmod 755 "$APP_DIR"
chmod -R u+rwX,go+rX "$APP_DIR"
chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$APP_DIR"

# 3. Install System & Python Dependencies
echo "[+] Updating system packages and Python dependencies..."
apt-get update -y
apt-get install -y curl ca-certificates jq gnupg python3 python3-pip python3-uvicorn python3-fastapi python3-requests

# 4. Install / Update LM Studio CLI
echo "[+] Fetching latest LM Studio CLI binary..."
sudo -u "$SERVICE_USER" HOME="$APP_DIR" bash -c "curl -fsSL https://lmstudio.ai/install.sh | bash"

LMS_BIN=""
if [ -f "${APP_DIR}/.cache/lm-studio/bin/lms" ]; then
  LMS_BIN="${APP_DIR}/.cache/lm-studio/bin/lms"
elif [ -f "${APP_DIR}/.lmstudio/bin/lms" ]; then
  LMS_BIN="${APP_DIR}/.lmstudio/bin/lms"
else
  LMS_BIN=$(find "$APP_DIR" -type f -name "lms" -perm /111 2>/dev/null | head -n 1)
fi

if [ -z "$LMS_BIN" ] || [ ! -x "$LMS_BIN" ]; then
  echo "[-] Error: Failed to locate 'lms' executable."
  exit 1
fi

chmod +x "$LMS_BIN"
ln -sf "$LMS_BIN" /usr/local/bin/lms
echo "[✓] LMS CLI ready at /usr/local/bin/lms"

# 5. LM Link Setup (Interactive only on initial install)
echo ""
echo "--- [ LM Link & Node Identity ] ---"
if [ "$IS_UPDATE" = false ]; then
  if ! sudo -u "$SERVICE_USER" HOME="$APP_DIR" "$LMS_BIN" link status >/dev/null 2>&1; then
    echo "[*] Authenticating service account with LM Studio Hub..."
    sudo -u "$SERVICE_USER" HOME="$APP_DIR" "$LMS_BIN" login || true
  fi

  sudo -u "$SERVICE_USER" HOME="$APP_DIR" "$LMS_BIN" link enable || true

  CURRENT_HOST=$(hostname)
  INPUT_DEVICE_NAME=""
  if [ -t 0 ]; then
    read -rp "[?] Enter node name for LM Link [default: $CURRENT_HOST]: " INPUT_DEVICE_NAME
  elif [ -e /dev/tty ]; then
    read -rp "[?] Enter node name for LM Link [default: $CURRENT_HOST]: " INPUT_DEVICE_NAME < /dev/tty || true
  fi
  DEVICE_NAME="${INPUT_DEVICE_NAME:-$CURRENT_HOST}"
  sudo -u "$SERVICE_USER" HOME="$APP_DIR" "$LMS_BIN" link set-device-name "$DEVICE_NAME" || true
else
  echo "[✓] Preserving existing LM Link configuration and device pairing."
  sudo -u "$SERVICE_USER" HOME="$APP_DIR" "$LMS_BIN" link enable >/dev/null 2>&1 || true
fi

# 6. Fetch / Reload Latest model_manager.py
echo ""
echo "--- [ Updating Model Manager Script ] ---"
echo "[+] Pulling latest model_manager.py from GitHub..."
curl -fsSL "${REPO_RAW_URL}/model_manager.py" -o "${APP_DIR}/model_manager.py"

chmod 755 "${APP_DIR}/model_manager.py"
chown "${SERVICE_USER}:${SERVICE_GROUP}" "${APP_DIR}/model_manager.py"
echo "[✓] model_manager.py successfully updated."

# 7. Write / Update Systemd Services
echo ""
echo "--- [ Systemd Service Configuration ] ---"

# LM Studio Service
tee /etc/systemd/system/lmstudio.service > /dev/null <<EOF
[Unit]
Description=LM Studio Headless Server & Link Daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${APP_DIR}
Environment=HOME=${APP_DIR}
Environment=PATH=/usr/local/bin:${APP_DIR}/.cache/lm-studio/bin:${APP_DIR}/.lmstudio/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=${LMS_BIN} server start --host 0.0.0.0 --port ${LM_PORT} --cors
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Model Manager Web UI Service
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
echo "[✓] Background daemons restarted with latest configurations."

# 8. Open WebUI Handling
echo ""
echo "--- [ Open WebUI Container Management ] ---"

INSTALL_WEBUI="y"
if [ "$IS_UPDATE" = true ]; then
  # In update mode, read prior selection
  if [ -f "$CONFIG_FILE" ]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
  fi
  if [ "${ENABLE_OPEN_WEBUI:-false}" = "true" ]; then
    echo "[*] Existing Open WebUI installation detected. Updating container image..."
    INSTALL_WEBUI="y"
  else
    INSTALL_WEBUI="n"
  fi
else
  # Fresh install: prompt user
  if [ -t 0 ]; then
    read -rp "[?] Deploy Open WebUI container with Web Search? (y/n) [default: y]: " INSTALL_WEBUI
  elif [ -e /dev/tty ]; then
    read -rp "[?] Deploy Open WebUI container with Web Search? (y/n) [default: y]: " INSTALL_WEBUI < /dev/tty || true
  fi
  INSTALL_WEBUI="${INSTALL_WEBUI:-y}"
fi

if [[ "$INSTALL_WEBUI" =~ ^[Yy]$ ]]; then
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

  WEBUI_DATA_DIR="${APP_DIR}/webui_data"
  mkdir -p "$WEBUI_DATA_DIR"
  chown -R 1000:1000 "$WEBUI_DATA_DIR"

  if docker ps -a --format '{{.Names}}' | grep -Eq "^open-webui$"; then
    echo "[*] Refreshing Open WebUI container instance..."
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

# 9. Save State to Config File
cat > "$CONFIG_FILE" <<EOF
INSTALLED_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
ENABLE_OPEN_WEBUI="${ENABLE_WEBUI_CONF}"
BASE_STORAGE="${BASE_STORAGE}"
APP_DIR="${APP_DIR}"
EOF
chown "${SERVICE_USER}:${SERVICE_GROUP}" "$CONFIG_FILE"
chmod 600 "$CONFIG_FILE"

# 10. Final Permissions Enforcement
chmod 755 "$APP_DIR"
chmod -R u+rwX,go+rX "$APP_DIR"
chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$APP_DIR"

SERVER_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "================================================================="
if [ "$IS_UPDATE" = true ]; then
  echo "                        UPDATE COMPLETE                          "
else
  echo "                      INSTALLATION COMPLETE                      "
fi
echo "================================================================="
echo "• LM Studio API:          http://${SERVER_IP}:${LM_PORT}/v1"
echo "• Model Manager UI:       http://${SERVER_IP}:${MANAGER_PORT}"
if [ "${ENABLE_WEBUI_CONF}" = "true" ]; then
  echo "• Open WebUI Chat & Search: http://${SERVER_IP}:${WEBUI_PORT}"
fi
echo "• Base Storage Directory: ${APP_DIR}"
echo "================================================================="