import os
import json
import shutil
import requests
import subprocess
from core.hardware import STORAGE_PATH

WORKSPACES_ROOT = os.path.join(STORAGE_PATH, "workspaces")
ACCOUNTS_FILE = os.path.join(STORAGE_PATH, ".github_accounts.json")
os.makedirs(WORKSPACES_ROOT, exist_ok=True)

def load_accounts_data():
    if os.path.exists(ACCOUNTS_FILE):
        try:
            with open(ACCOUNTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"accounts": []}
    return {"accounts": []}

def save_accounts_data(data):
    with open(ACCOUNTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(ACCOUNTS_FILE, 0o600)

def get_token_for_user(username: str):
    data = load_accounts_data()
    for acc in data.get("accounts", []):
        if acc.get("username", "").lower() == username.lower():
            return acc.get("token", "")
    return ""

def get_active_workspaces():
    workspaces = []
    if os.path.exists(WORKSPACES_ROOT):
        for d in os.listdir(WORKSPACES_ROOT):
            w_path = os.path.join(WORKSPACES_ROOT, d)
            if os.path.isdir(w_path) and os.path.exists(os.path.join(w_path, ".git")):
                branch_res = subprocess.run(["git", "-C", w_path, "branch", "--show-current"], capture_output=True, text=True)
                branch = branch_res.stdout.strip() or "main"
                display_name = d.replace("_", "/", 1) if "_" in d else d
                workspaces.append({
                    "dir_name": d,
                    "display_name": display_name,
                    "branch": branch,
                    "path": w_path
                })
    return workspaces