# 1. Install python runtime dependencies
sudo apt-get install -y python3-pip python3-uvicorn python3-fastapi python3-requests

# 2. Set file permissions
sudo chown -R lmstudio:lmstudio /storage/lmstudio/model_manager.py

# 3. Create the Systemd service unit
sudo tee /etc/systemd/system/modelmanager.service > /dev/null <<EOF
[Unit]
Description=LM Studio Custom Model Manager Web UI
After=network.target lmstudio.service
Wants=lmstudio.service

[Service]
Type=simple
User=lmstudio
Group=lmstudio
WorkingDirectory=/storage/lmstudio
Environment=HOME=/storage/lmstudio
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/bin/python3 /storage/lmstudio/model_manager.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 4. Start and enable the service
sudo systemctl daemon-reload
sudo systemctl enable --now modelmanager.service