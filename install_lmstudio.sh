#!/usr/bin/env bash
set -euo pipefail

# Ensure script is executed with sudo
if [ "$EUID" -ne 0 ]; then
  echo "[-] Please run this script with sudo: sudo ./setup_lmstudio_storage.sh"
  exit 1
fi

SERVICE_USER="lmstudio"
SERVICE_GROUP="lmstudio"
BASE_STORAGE="/storage"
APP_DIR="${BASE_STORAGE}/lmstudio"
MODELS_DIR="${APP_DIR}/models"
LM_PORT="1234"
WEBUI_PORT="3000"
SERVICE_NAME="lmstudio.service"

echo "================================================================="
echo "   LM Studio Dedicated Service & Storage Setup (/storage)        "
echo "================================================================="

# 1. Validate Base Storage Directory
if [ ! -d "$BASE_STORAGE" ]; then
  echo "[-] Target mount path $BASE_STORAGE does not exist. Please mount your drive first."
  exit 1
fi

# 2. Create Dedicated Service User & Group if not present
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  echo "[+] Creating dedicated system user: $SERVICE_USER..."
  useradd -r -m -d "$APP_DIR" -s /bin/bash -c "LM Studio Service Account" "$SERVICE_USER"
else
  echo "[✓] System user $SERVICE_USER already exists."
  usermod -d "$APP_DIR" -s /bin/bash "$SERVICE_USER"
fi

# Create required folder hierarchy
mkdir -p "$APP_DIR" "$MODELS_DIR" "${APP_DIR}/.cache" "${APP_DIR}/.lmstudio"
chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$APP_DIR"

# 3. Install System Dependencies
echo "[+] Installing system prerequisites..."
apt-get update -y
apt-get install -y curl ca-certificates jq gnupg

# 4. Install / Update LM Studio CLI inside the dedicated user environment
echo "[+] Fetching and installing LM Studio CLI for $SERVICE_USER..."
sudo -u "$SERVICE_USER" HOME="$APP_DIR" bash -c "curl -fsSL https://lmstudio.ai/install.sh | bash"

# Locate the installed binary
LMS_BIN=""
if [ -f "${APP_DIR}/.cache/lm-studio/bin/lms" ]; then
  LMS_BIN="${APP_DIR}/.cache/lm-studio/bin/lms"
elif [ -f "${APP_DIR}/.lmstudio/bin/lms" ]; then
  LMS_BIN="${APP_DIR}/.lmstudio/bin/lms"
else
  LMS_BIN=$(find "$APP_DIR" -type f -name "lms" -perm /111 2>/dev/null | head -n 1)
fi

if [ -z "$LMS_BIN" ] || [ ! -x "$LMS_BIN" ]; then
  echo "[-] ERROR: Unable to locate the lms binary in $APP_DIR."
  exit 1
fi

# Symlink to global path so any admin can run 'lms' directly
ln -sf "$LMS_BIN" /usr/local/bin/lms
echo "[✓] Binary available at /usr/local/bin/lms"

# 5. LM Link & Account Authentication
echo ""
echo "--- [ Step 1: LM Studio Account & Link Setup ] ---"
echo "[*] Checking link daemon state for user: $SERVICE_USER..."

# Run login as the dedicated service user if unauthenticated
if ! sudo -u "$SERVICE_USER" HOME="$APP_DIR" "$LMS_BIN" link status >/dev/null 2>&1; then
  echo "[!] Authentication required for the dedicated service account."
  echo "    Please complete the browser authentication prompt below:"
  sudo -u "$SERVICE_USER" HOME="$APP_DIR" "$LMS_BIN" login || true
fi

sudo -u "$SERVICE_USER" HOME="$APP_DIR" "$LMS_BIN" link enable || true

CURRENT_HOST=$(hostname)
read -rp "[?] Enter a display name for this node in LM Link [default: $CURRENT_HOST]: " INPUT_DEVICE_NAME
DEVICE_NAME="${INPUT_DEVICE_NAME:-$CURRENT_HOST}"
sudo -u "$SERVICE_USER" HOME="$APP_DIR" "$LMS_BIN" link set-device-name "$DEVICE_NAME" || true

# 6. Configure Systemd Service
echo ""
echo "--- [ Step 2: Systemd Unit Configuration ] ---"
tee /etc/systemd/system/"${SERVICE_NAME}" > /dev/null <<EOF
[Unit]
Description=LM Studio Headless Daemon
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

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"

# 7. Open WebUI Deployment (Optional)
echo ""
echo "--- [ Step 3: Open WebUI / Web Search Tools ] ---"
read -rp "[?] Deploy/Update Open WebUI container connected to /storage? (y/n) [default: y]: " INSTALL_WEBUI
INSTALL_WEBUI="${INSTALL_WEBUI:-y}"

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

  # Store Open WebUI persistent database under /storage/lmstudio/webui_data
  WEBUI_DATA_DIR="${APP_DIR}/webui_data"
  mkdir -p "$WEBUI_DATA_DIR"
  chown -R 1000:1000 "$WEBUI_DATA_DIR"

  if docker ps -a --format '{{.Names}}' | grep -Eq "^open-webui$"; then
    echo "[*] Refreshing Open WebUI container..."
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

  echo "[✓] Open WebUI deployed on port ${WEBUI_PORT}."
fi

# 8. Final Report
SERVER_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "================================================================="
echo "                      SETUP COMPLETE                             "
echo "================================================================="
echo "• Service User:           $SERVICE_USER (Home: $APP_DIR)"
echo "• Storage Location:       $APP_DIR"
echo "• LM Studio API:          http://${SERVER_IP}:${LM_PORT}/v1"
echo "• Service Status:         sudo systemctl status $SERVICE_NAME"
if [[ "$INSTALL_WEBUI" =~ ^[Yy]$ ]]; then
  echo "• Open WebUI Portal:      http://${SERVER_IP}:${WEBUI_PORT}"
fi
echo "================================================================="