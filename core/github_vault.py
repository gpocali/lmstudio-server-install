import os
import re
import json
import shutil
import datetime
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

def get_workspace_history_file(repo_dir_name: str):
    return os.path.join(WORKSPACES_ROOT, repo_dir_name, ".lmstudio_history.json")

def load_workspace_history(repo_dir_name: str):
    hist_file = get_workspace_history_file(repo_dir_name)
    if os.path.exists(hist_file):
        try:
            with open(hist_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Default initial schema
    return {
        "active_thread_id": "thread-default",
        "threads": [
            {
                "id": "thread-default",
                "title": "General Task Thread",
                "created_at": datetime.datetime.utcnow().isoformat(),
                "messages": []
            }
        ]
    }

def save_workspace_history(repo_dir_name: str, data: dict):
    hist_file = get_workspace_history_file(repo_dir_name)
    try:
        with open(hist_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def get_workspace_git_status(repo_dir_name: str):
    w_path = os.path.join(WORKSPACES_ROOT, repo_dir_name)
    if not os.path.exists(w_path):
        return {"has_pending_changes": False, "unpushed_commits": 0, "status_text": "", "diff": ""}
    
    # Check unstaged / uncommitted changes
    status_res = subprocess.run(["git", "-C", w_path, "status", "--porcelain"], capture_output=True, text=True)
    status_lines = [l.strip() for l in status_res.stdout.splitlines() if l.strip() and not l.strip().endswith(".lmstudio_history.json")]
    has_pending = len(status_lines) > 0

    diff_res = subprocess.run(["git", "-C", w_path, "diff"], capture_output=True, text=True)
    diff_text = diff_res.stdout or ""
    if not diff_text and has_pending:
        diff_text = "Staged / Untracked files:\n" + "\n".join(status_lines)

    # Check unpushed commits ahead of upstream
    unpushed_count = 0
    try:
        branch_res = subprocess.run(["git", "-C", w_path, "branch", "--show-current"], capture_output=True, text=True)
        current_b = branch_res.stdout.strip() or "main"
        cherry_res = subprocess.run(["git", "-C", w_path, "cherry", "-v", f"origin/{current_b}"], capture_output=True, text=True, timeout=5)
        if cherry_res.returncode == 0:
            unpushed_count = len([c for c in cherry_res.stdout.splitlines() if c.startswith("+")])
    except Exception:
        unpushed_count = 0

    return {
        "has_pending_changes": has_pending,
        "pending_files": status_lines,
        "unpushed_commits": unpushed_count,
        "diff": diff_text[:5000]
    }

def append_to_changelog(workspace_path: str, branch: str, commit_msg: str, modified_files: list[str], prompt: str):
    changelog_path = os.path.join(workspace_path, "CHANGELOG.md")
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    
    entry = f"\n### [{timestamp}] - {commit_msg}\n"
    entry += f"- **Branch:** `{branch}`\n"
    if prompt:
        entry += f"- **Task Prompt:** {prompt.strip()}\n"
    if modified_files:
        files_str = ", ".join([f"`{f}`" for f in modified_files])
        entry += f"- **Modified Files:** {files_str}\n"
    
    try:
        if not os.path.exists(changelog_path):
            with open(changelog_path, "w", encoding="utf-8") as f:
                f.write("# Project Workspace Changelog\n\nAll automated AI commits and code modifications are documented below.\n")
        
        with open(changelog_path, "a", encoding="utf-8") as f:
            f.write(entry)
        
        subprocess.run(["git", "-C", workspace_path, "add", "CHANGELOG.md"], capture_output=True)
    except Exception:
        pass

def get_active_workspaces():
    workspaces = []
    if os.path.exists(WORKSPACES_ROOT):
        for d in os.listdir(WORKSPACES_ROOT):
            w_path = os.path.join(WORKSPACES_ROOT, d)
            if os.path.isdir(w_path) and os.path.exists(os.path.join(w_path, ".git")):
                branch_res = subprocess.run(["git", "-C", w_path, "branch", "--show-current"], capture_output=True, text=True)
                branch = branch_res.stdout.strip() or "main"
                display_name = d.replace("_", "/", 1) if "_" in d else d
                git_status = get_workspace_git_status(d)
                
                workspaces.append({
                    "dir_name": d,
                    "display_name": display_name,
                    "branch": branch,
                    "path": w_path,
                    "has_pending_changes": git_status["has_pending_changes"],
                    "unpushed_commits": git_status["unpushed_commits"]
                })
    return workspaces

def get_workspace_branches(repo_dir_name: str):
    w_path = os.path.join(WORKSPACES_ROOT, repo_dir_name)
    if not os.path.exists(w_path):
        return {"current": "main", "branches": ["main"]}
    
    current_res = subprocess.run(["git", "-C", w_path, "branch", "--show-current"], capture_output=True, text=True)
    current = current_res.stdout.strip() or "main"

    branches = set()
    branches.add(current)

    all_res = subprocess.run(["git", "-C", w_path, "branch", "-a"], capture_output=True, text=True)
    if all_res.returncode == 0:
        for line in all_res.stdout.splitlines():
            b = line.strip().replace("*", "").strip()
            if b and "->" not in b:
                if b.startswith("remotes/origin/"):
                    b = b.replace("remotes/origin/", "")
                branches.add(b)
    
    return {"current": current, "branches": sorted(list(branches))}