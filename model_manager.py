"""
LM Studio Server Entrypoint Router
Dispatches API requests with General Chat Bot, Model Tuning, and Workspace Timeline.
"""

import os
import re
import glob
import json
import datetime
import requests
import subprocess
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from core.hardware import (
    get_lms_bin,
    get_system_hardware_info,
    get_storage_usage,
    get_loaded_models,
    STORAGE_PATH,
    MODELS_PATH,
    LMS_ENV
)
from core.models import (
    DOWNLOAD_JOBS,
    calculate_trust_score,
    get_quant_description,
    parse_model_metadata,
    run_download_job,
    VERIFIED_CREATORS
)
from core.github_vault import (
    WORKSPACES_ROOT,
    load_accounts_data,
    save_accounts_data,
    get_token_for_user,
    get_active_workspaces,
    get_workspace_branches,
    get_workspace_git_status,
    load_workspace_history,
    save_workspace_history,
    append_to_changelog
)
from core.agent_engine import process_agent_task, generate_commit_msg_from_diff

app = FastAPI(title="LM Studio Code, Chat & Model Studio")

CHAT_SESSIONS_FILE = os.path.join(STORAGE_PATH, ".chat_sessions.json")

# ---------------- Pydantic Request Models ----------------

class DownloadRequest(BaseModel):
    repo_id: str
    group_name: str
    files: list[str]

class DeleteRequest(BaseModel):
    filename: str

class LoadRequest(BaseModel):
    model_path: str
    identifier: str = ""
    gpu_offload: str = "max"
    context_length: int = 32768
    ttl: int = 3600

class ChatCompletionRequest(BaseModel):
    messages: list[dict]
    model_identifier: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95

class SaveChatSessionRequest(BaseModel):
    session_id: str
    title: str = "Chat Conversation"
    messages: list[dict] = []

class DeleteChatSessionRequest(BaseModel):
    session_id: str

class AddAccountRequest(BaseModel):
    token: str
    label: str = ""

class RemoveAccountRequest(BaseModel):
    username: str

class GithubCloneRequest(BaseModel):
    account_username: str
    repo_full_name: str
    branch: str = "main"

class GithubBranchSwitchRequest(BaseModel):
    repo_dir_name: str
    branch: str

class CreateBranchRequest(BaseModel):
    repo_dir_name: str
    branch_name: str

class CreateFileRequest(BaseModel):
    repo_dir_name: str
    file_path: str
    initial_content: str = ""

class CreateThreadRequest(BaseModel):
    repo_dir_name: str
    title: str = "New Chat Thread"

class SwitchThreadRequest(BaseModel):
    repo_dir_name: str
    thread_id: str

class AgentTaskRequest(BaseModel):
    repo_dir_name: str
    target_files: list[str] = []
    instruction: str
    thread_id: str = ""
    model_identifier: str = ""

class GenCommitMsgRequest(BaseModel):
    repo_dir_name: str
    target_files: list[str] = []
    model_identifier: str = ""

class CommitRequest(BaseModel):
    repo_dir_name: str
    commit_message: str
    target_files: list[str] = []
    instruction_summary: str = ""

class DiscardRequest(BaseModel):
    repo_dir_name: str
    target_files: list[str] = []

class WorkspaceActionRequest(BaseModel):
    repo_dir_name: str
    branch: str = "main"

# ---------------- Frontend Route ----------------

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_file = os.path.join(STORAGE_PATH, "web", "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h2>Error: /storage/lmstudio/web/index.html not found</h2>", status_code=404)

# ---------------- General Chat Bot API ----------------

def load_chat_sessions():
    if os.path.exists(CHAT_SESSIONS_FILE):
        try:
            with open(CHAT_SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"sessions": []}
    return {"sessions": []}

def save_chat_sessions(data):
    try:
        with open(CHAT_SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

@app.get("/api/chat/sessions")
def api_get_chat_sessions():
    return load_chat_sessions()

@app.post("/api/chat/sessions/save")
def api_save_chat_session(req: SaveChatSessionRequest):
    data = load_chat_sessions()
    sessions = data.get("sessions", [])
    
    found = False
    for s in sessions:
        if s["id"] == req.session_id:
            s["title"] = req.title
            s["messages"] = req.messages
            s["updated_at"] = datetime.datetime.utcnow().isoformat()
            found = True
            break
    
    if not found:
        sessions.insert(0, {
            "id": req.session_id,
            "title": req.title,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "updated_at": datetime.datetime.utcnow().isoformat(),
            "messages": req.messages
        })
    
    data["sessions"] = sessions
    save_chat_sessions(data)
    return {"status": "success", "session_id": req.session_id}

@app.post("/api/chat/sessions/delete")
def api_delete_chat_session(req: DeleteChatSessionRequest):
    data = load_chat_sessions()
    data["sessions"] = [s for s in data.get("sessions", []) if s.get("id") != req.session_id]
    save_chat_sessions(data)
    return {"status": "success"}

@app.post("/api/chat/completions")
def api_chat_completion(req: ChatCompletionRequest):
    model_id = req.model_identifier
    if not model_id:
        loaded = get_loaded_models()
        model_id = loaded[0] if loaded else "default"

    try:
        payload = {
            "model": model_id,
            "messages": req.messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "top_p": req.top_p
        }
        resp = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=180)
        if resp.status_code != 200:
            return JSONResponse(status_code=500, content={"status": "error", "message": f"LM Studio API Error: {resp.text}"})
        
        return resp.json()
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# ---------------- Hardware & Model Endpoints ----------------

@app.get("/api/system_info")
def api_sys_info():
    return get_system_hardware_info()

@app.get("/api/search")
def api_search_hf(q: str = "", sort_by: str = "downloads", verified_only: bool = False):
    hf_sort = "likes" if sort_by == "likes" else ("lastModified" if sort_by == "lastModified" else "downloads")
    params = {"filter": "gguf", "sort": hf_sort, "direction": "-1", "limit": 60}
    if q.strip(): params["search"] = q.strip()
    try:
        resp = requests.get("https://huggingface.co/api/models", params=params, timeout=10)
        res = resp.json() if resp.status_code == 200 else []
    except Exception: res = []

    results = []
    if isinstance(res, list):
        for m in res:
            repo_id = m.get("id", "")
            if not repo_id: continue
            maker, model_name = repo_id.split('/', 1) if '/' in repo_id else ("Community", repo_id)
            is_verified = maker in VERIFIED_CREATORS
            if verified_only and not is_verified: continue
            dl = m.get("downloads", 0) or 0
            likes = m.get("likes", 0) or 0
            results.append({
                "id": repo_id, "maker": maker, "model_name": model_name,
                "downloads": dl, "likes": likes, "lastModified": (m.get("lastModified") or "")[:10],
                "is_verified": is_verified, "trust_score": calculate_trust_score(dl, likes, is_verified)
            })

    if sort_by == "alphabetical": results.sort(key=lambda x: x["model_name"].lower())
    elif sort_by == "trust": results.sort(key=lambda x: x["trust_score"], reverse=True)
    return results

@app.get("/api/model_files")
def api_model_files(repo_id: str):
    raw_files = {}
    try:
        resp = requests.get(f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true", timeout=8)
        if resp.status_code == 200:
            for item in resp.json():
                path = item.get("path", "")
                if path.endswith(".gguf"):
                    sz = item.get("size", 0) or (item.get("lfs", {}).get("size", 0) if isinstance(item.get("lfs"), dict) else 0)
                    raw_files[path] = sz
    except Exception: pass

    if not raw_files:
        try:
            res = requests.get(f"https://huggingface.co/api/models/{repo_id}", timeout=8).json()
            for s in res.get("siblings", []):
                fname = s.get("rfilename", "")
                if fname.endswith(".gguf"): raw_files[fname] = s.get("size", 0)
        except Exception: pass

    grouped = {}
    for rel_path, size_bytes in raw_files.items():
        fname = os.path.basename(rel_path)
        shard_match = re.search(r'(-\d{5}-of-\d{5})', fname)
        clean_name = fname.replace(shard_match.group(1), "") if shard_match else fname
        if clean_name not in grouped: grouped[clean_name] = {"group_name": clean_name, "paths": [], "total_bytes": 0}
        grouped[clean_name]["paths"].append(rel_path)
        grouped[clean_name]["total_bytes"] += size_bytes

    for g in grouped.values(): g["paths"].sort()

    hw = get_system_hardware_info()
    vram_total = hw["gpu"]["total_vram_gb"]
    ram_total = hw["system_ram"]["total_gb"]

    local_files = set()
    if os.path.exists(MODELS_PATH):
        for root, _, filenames in os.walk(MODELS_PATH, followlinks=True):
            for f in filenames:
                if f.endswith(".gguf"): local_files.add(f)

    parsed = []
    for gname, gdata in grouped.items():
        weight, variant = parse_model_metadata(gname, repo_id)
        size_gb = round(gdata["total_bytes"] / (1024**3), 2) if gdata["total_bytes"] > 0 else 0.0
        est_mem = round(size_gb * 1.2, 2) if size_gb > 0 else 0.0

        fit = "unknown"
        if size_gb > 0:
            if vram_total > 0: fit = "fits_gpu" if est_mem <= vram_total else ("split_gpu_ram" if est_mem <= (vram_total + ram_total * 0.75) else "exceeds")
            else: fit = "fits_ram" if est_mem <= ram_total * 0.85 else "exceeds"

        shard_basenames = [os.path.basename(p) for p in gdata["paths"]]
        is_dl = all(sb in local_files for sb in shard_basenames)
        is_dling = gname in DOWNLOAD_JOBS and DOWNLOAD_JOBS[gname].get("status") == "downloading"
        shard_info = f" ({len(gdata['paths'])} Shards)" if len(gdata['paths']) > 1 else ""

        # Model max context capacity estimation based on architecture name
        max_cap_ctx = 131072 if ("llama-3" in gname.lower() or "qwen" in gname.lower() or "nemotron" in gname.lower()) else 32768

        parsed.append({
            "group_name": gname, "display_name": gname + shard_info, "paths": gdata["paths"],
            "is_sharded": len(gdata["paths"]) > 1, "shard_count": len(gdata["paths"]),
            "weight": weight, "variant": variant, "description": get_quant_description(variant),
            "size_gb": f"{size_gb} GB" if size_gb > 0 else "Pending...", "raw_size_gb": size_gb,
            "est_vram": f"~{est_mem} GB" if est_mem > 0 else "N/A",
            "max_context": max_cap_ctx,
            "fit_status": fit, "is_downloaded": is_dl, "is_downloading": is_dling
        })

    parsed.sort(key=lambda x: x["raw_size_gb"] if x["raw_size_gb"] > 0 else 999)
    return {"repo_id": repo_id, "hardware": hw, "files": parsed}

@app.post("/api/load_model")
def api_load_model(req: LoadRequest):
    lms = get_lms_bin()
    abs_path = os.path.abspath(req.model_path)
    fname = os.path.basename(abs_path)
    clean_fname = re.sub(r'(-0000\d-of-\d{5})?\.gguf$', '', fname).lower()

    try: subprocess.run([lms, "unload", "--all"], env=LMS_ENV, capture_output=True, timeout=5)
    except Exception: pass

    registered_keys = []
    try:
        ls_res = subprocess.run([lms, "ls"], env=LMS_ENV, capture_output=True, text=True, timeout=5)
        if ls_res.returncode == 0:
            in_llm_section = False
            for line in ls_res.stdout.splitlines():
                l_str = line.strip()
                if "LLM" in l_str: in_llm_section = True; continue
                if "EMBEDDING" in l_str or l_str.startswith("---"): in_llm_section = False; continue
                if in_llm_section and l_str:
                    parts = l_str.split()
                    if parts: registered_keys.append(parts[0])
    except Exception: pass

    matched_target = None
    f_strip = re.sub(r'[^a-z0-9]', '', clean_fname)
    for key in registered_keys:
        k_strip = re.sub(r'[^a-z0-9]', '', key.lower())
        if k_strip in f_strip or f_strip in k_strip:
            matched_target = key
            break

    targets_to_try = []
    if matched_target: targets_to_try.append(matched_target)
    targets_to_try.extend(registered_keys)
    targets_to_try.extend([clean_fname, fname])

    seen = set()
    dedup_targets = [t for t in targets_to_try if t and not (t in seen or seen.add(t))]

    last_error = ""
    for target in dedup_targets:
        try:
            cmd = [
                lms, "load", target,
                f"--gpu={req.gpu_offload}",
                f"--context-length={req.context_length}",
                f"--ttl={req.ttl}",
                "--yes"
            ]
            res = subprocess.run(cmd, env=LMS_ENV, capture_output=True, text=True, timeout=30)
            if res.returncode == 0:
                return {"status": "success", "loaded_target": target, "context_length": req.context_length, "output": res.stdout or f"Loaded {target} into GPU VRAM."}
            else:
                last_error = (res.stderr or res.stdout or "").strip()
        except Exception as err:
            last_error = str(err)

    return JSONResponse(status_code=400, content={"status": "error", "message": last_error or f"Could not load model for '{fname}'.", "attempted_candidates": dedup_targets})

@app.post("/api/unload_model")
def api_unload_model():
    lms = get_lms_bin()
    try:
        res = subprocess.run([lms, "unload", "--all"], env=LMS_ENV, capture_output=True, text=True, timeout=5)
        return {"status": "success", "output": res.stdout}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/download")
def api_start_download(req: DownloadRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_download_job, req.repo_id, req.group_name, req.files)
    return {"status": "started", "group_name": req.group_name}

@app.post("/api/delete")
def api_delete_model(req: DeleteRequest):
    deleted = False
    if os.path.exists(MODELS_PATH):
        prefix = req.filename.replace(".gguf", "")
        for root, _, filenames in os.walk(MODELS_PATH, followlinks=True):
            for f in filenames:
                if f == req.filename or (prefix in f and f.endswith(".gguf")):
                    try:
                        os.remove(os.path.join(root, f))
                        deleted = True
                    except Exception: pass
            if not os.listdir(root):
                try: os.rmdir(root)
                except Exception: pass
    if req.filename in DOWNLOAD_JOBS: del DOWNLOAD_JOBS[req.filename]
    return {"status": "deleted" if deleted else "not_found", "storage": get_storage_usage()}

@app.get("/api/tasks")
def api_get_tasks():
    return DOWNLOAD_JOBS

@app.get("/api/local_models")
def api_local_models():
    files = []
    if os.path.exists(MODELS_PATH):
        for root, _, filenames in os.walk(MODELS_PATH, followlinks=True):
            for f in filenames:
                if f.endswith(".gguf") and not re.search(r'-0000[2-9]-of-', f):
                    path = os.path.join(root, f)
                    try: size_gb = round(os.path.getsize(path) / (1024**3), 2)
                    except Exception: size_gb = 0.0
                    weight, variant = parse_model_metadata(f, root)
                    max_cap_ctx = 131072 if ("llama-3" in f.lower() or "qwen" in f.lower() or "nemotron" in f.lower()) else 32768
                    files.append({
                        "filename": f,
                        "weight": weight,
                        "variant": variant,
                        "size_gb": f"{size_gb} GB",
                        "raw_size_gb": size_gb,
                        "max_context": max_cap_ctx,
                        "path": path
                    })
    return {"files": files, "storage": get_storage_usage(), "loaded_models": get_loaded_models()}

# ---------------- GitHub Multi-Account Endpoints ----------------

@app.get("/api/github/accounts")
def api_list_accounts():
    data = load_accounts_data()
    safe = []
    for a in data.get("accounts", []):
        t = a.get("token", "")
        masked = f"ghp_...{t[-4:]}" if len(t) > 4 else "ghp_****"
        safe.append({"username": a.get("username", ""), "label": a.get("label", a.get("username", "")), "masked_token": masked, "avatar_url": a.get("avatar_url", "")})
    return safe

@app.post("/api/github/accounts/add")
def api_add_account(req: AddAccountRequest):
    token = req.token.strip()
    if not token: return JSONResponse(status_code=400, content={"status": "error", "message": "Token cannot be empty"})
    try:
        r = requests.get("https://api.github.com/user", headers={"Authorization": f"Bearer {token}"}, timeout=6)
        if r.status_code != 200: return JSONResponse(status_code=401, content={"status": "error", "message": "Invalid GitHub Token"})
        u = r.json()
        data = load_accounts_data()
        data["accounts"] = [a for a in data.get("accounts", []) if a.get("username", "").lower() != u.get("login", "").lower()]
        data["accounts"].append({"username": u.get("login", ""), "label": req.label.strip() or u.get("login", ""), "token": token, "avatar_url": u.get("avatar_url", "")})
        save_accounts_data(data)
        return {"status": "success", "username": u.get("login", "")}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/github/accounts/remove")
def api_remove_account(req: RemoveAccountRequest):
    data = load_accounts_data()
    orig = len(data.get("accounts", []))
    data["accounts"] = [a for a in data.get("accounts", []) if a.get("username", "").lower() != req.username.lower()]
    if len(data["accounts"]) < orig:
        save_accounts_data(data)
        return {"status": "success"}
    return JSONResponse(status_code=404, content={"status": "error", "message": "Account not found"})

@app.get("/api/github/repos")
def api_list_repos(account_username: str):
    token = get_token_for_user(account_username)
    if not token: return JSONResponse(status_code=401, content={"status": "error", "message": "No active token"})
    try:
        r = requests.get("https://api.github.com/user/repos?per_page=100&sort=updated", headers={"Authorization": f"Bearer {token}"}, timeout=8)
        if r.status_code == 200:
            return [{"full_name": item.get("full_name"), "name": item.get("name"), "owner": item.get("owner", {}).get("login"), "default_branch": item.get("default_branch", "main"), "is_private": item.get("private", False)} for item in r.json()]
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    return []

@app.get("/api/github/branches")
def api_list_branches(account_username: str, repo_full_name: str):
    token = get_token_for_user(account_username)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.get(f"https://api.github.com/repos/{repo_full_name}/branches", headers=headers, timeout=6)
        if r.status_code == 200: return [b.get("name") for b in r.json()]
    except Exception: pass
    return ["main", "dev", "master"]

# ---------------- Workspace, History & Timeline Endpoints ----------------

@app.get("/api/workspaces/active")
def api_get_workspaces():
    return get_active_workspaces()

@app.get("/api/workspace/status")
def api_get_status(repo_dir_name: str):
    return get_workspace_git_status(repo_dir_name)

@app.get("/api/workspace/history")
def api_get_history(repo_dir_name: str):
    return load_workspace_history(repo_dir_name)

@app.get("/api/workspace/timeline")
def api_get_timeline(repo_dir_name: str):
    hist = load_workspace_history(repo_dir_name)
    events = hist.get("timeline_events", [])
    
    w_path = os.path.join(WORKSPACES_ROOT, repo_dir_name)
    git_commits = []
    if os.path.exists(w_path):
        res = subprocess.run(["git", "-C", w_path, "log", "-n", "15", "--pretty=format:%h|%an|%ad|%s", "--date=iso"], capture_output=True, text=True)
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                parts = line.split("|", 3)
                if len(parts) == 4:
                    git_commits.append({
                        "hash": parts[0],
                        "author": parts[1],
                        "date": parts[2],
                        "subject": parts[3]
                    })
    
    return {"events": events, "commits": git_commits}

@app.post("/api/workspace/create_thread")
def api_create_thread(req: CreateThreadRequest):
    hist = load_workspace_history(req.repo_dir_name)
    new_id = f"thread-{int(os.times().elapsed * 1000)}"
    new_thread = {
        "id": new_id,
        "title": req.title.strip() or "Untitled Thread",
        "created_at": datetime.datetime.utcnow().isoformat(),
        "messages": []
    }
    hist.setdefault("threads", []).insert(0, new_thread)
    hist["active_thread_id"] = new_id
    save_workspace_history(req.repo_dir_name, hist)
    return {"status": "success", "thread": new_thread, "history": hist}

@app.post("/api/workspace/switch_thread")
def api_switch_thread(req: SwitchThreadRequest):
    hist = load_workspace_history(req.repo_dir_name)
    hist["active_thread_id"] = req.thread_id
    save_workspace_history(req.repo_dir_name, hist)
    return {"status": "success", "history": hist}

@app.get("/api/workspace/branches")
def api_get_workspace_branches(repo_dir_name: str):
    return get_workspace_branches(repo_dir_name)

@app.post("/api/workspace/switch_branch")
def api_switch_branch(req: GithubBranchSwitchRequest):
    w_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(w_path):
        return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})
    try:
        subprocess.run(["git", "-C", w_path, "checkout", req.branch], capture_output=True)
        res = subprocess.run(["git", "-C", w_path, "branch", "--show-current"], capture_output=True, text=True)
        current = res.stdout.strip() or req.branch
        return {"status": "success", "active_branch": current}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/workspace/create_branch")
def api_create_branch(req: CreateBranchRequest):
    w_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(w_path):
        return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})
    clean_b = req.branch_name.strip().replace(" ", "-")
    try:
        res = subprocess.run(["git", "-C", w_path, "checkout", "-b", clean_b], capture_output=True, text=True)
        if res.returncode == 0:
            return {"status": "success", "active_branch": clean_b}
        return JSONResponse(status_code=400, content={"status": "error", "message": res.stderr or res.stdout})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/workspace/generate_commit_msg")
def api_gen_commit_msg(req: GenCommitMsgRequest):
    res = generate_commit_msg_from_diff(req.repo_dir_name, req.target_files, req.model_identifier)
    if res.get("status") == "error":
        return JSONResponse(status_code=500, content=res)
    return res

@app.post("/api/workspace/pull")
def api_pull_upstream(req: WorkspaceActionRequest):
    w_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(w_path):
        return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})
    try:
        res = subprocess.run(["git", "-C", w_path, "pull", "origin", req.branch], capture_output=True, text=True, timeout=25)
        if res.returncode == 0:
            return {"status": "success", "message": res.stdout or f"Branch '{req.branch}' is up to date."}
        return JSONResponse(status_code=500, content={"status": "error", "message": res.stderr or res.stdout})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/github/clone")
def api_clone_project(req: GithubCloneRequest):
    token = get_token_for_user(req.account_username)
    if not token: return JSONResponse(status_code=401, content={"status": "error", "message": "Auth required"})
    dir_name = req.repo_full_name.replace("/", "_")
    dest_path = os.path.join(WORKSPACES_ROOT, dir_name)
    auth_url = f"https://oauth2:{token}@github.com/{req.repo_full_name}.git"

    try:
        if os.path.exists(dest_path):
            subprocess.run(["git", "-C", dest_path, "remote", "set-url", "origin", auth_url], capture_output=True)
            subprocess.run(["git", "-C", dest_path, "fetch", "--all"], capture_output=True, timeout=10)
            subprocess.run(["git", "-C", dest_path, "checkout", req.branch], capture_output=True, timeout=10)
            subprocess.run(["git", "-C", dest_path, "pull", "origin", req.branch], capture_output=True, timeout=10)
        else:
            res = subprocess.run(["git", "clone", "-b", req.branch, auth_url, dest_path], capture_output=True, text=True, timeout=30)
            if res.returncode != 0:
                subprocess.run(["git", "clone", auth_url, dest_path], capture_output=True, text=True, timeout=30)
                subprocess.run(["git", "-C", dest_path, "checkout", "-B", req.branch], capture_output=True, timeout=10)

        subprocess.run(["git", "-C", dest_path, "config", "user.name", req.account_username], capture_output=True)
        subprocess.run(["git", "-C", dest_path, "config", "user.email", f"{req.account_username}@users.noreply.github.com"], capture_output=True)
        os.chmod(dest_path, 0o777)
        return {"status": "success", "dir_name": dir_name, "branch": req.branch, "path": dest_path, "display_name": req.repo_full_name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/workspace/files")
def api_workspace_files(repo_dir_name: str):
    w_path = os.path.join(WORKSPACES_ROOT, repo_dir_name)
    if not os.path.exists(w_path): return []
    fl = []
    for root, dirs, files in os.walk(w_path):
        if ".git" in root or "__pycache__" in root: continue
        for f in files:
            if f == ".lmstudio_history.json": continue
            fl.append(os.path.relpath(os.path.join(root, f), w_path))
    fl.sort()
    return fl

@app.post("/api/workspace/create_file")
def api_create_file(req: CreateFileRequest):
    w_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(w_path): return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})
    target = os.path.join(w_path, req.file_path.strip().lstrip("/"))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    try:
        with open(target, "w", encoding="utf-8") as f: f.write(req.initial_content)
        subprocess.run(["git", "-C", w_path, "add", req.file_path], capture_output=True)
        return {"status": "success", "file_path": req.file_path}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/agent/execute")
def api_agent_execute(req: AgentTaskRequest):
    res = process_agent_task(req.repo_dir_name, req.target_files, req.instruction, req.thread_id, req.model_identifier)
    if res.get("status") == "error": return JSONResponse(status_code=500, content=res)
    return res

@app.post("/api/workspace/commit")
def api_commit(req: CommitRequest):
    w_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(w_path):
        return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})
    try:
        b_res = subprocess.run(["git", "-C", w_path, "branch", "--show-current"], capture_output=True, text=True)
        active_branch = b_res.stdout.strip() or "main"

        append_to_changelog(w_path, active_branch, req.commit_message, req.target_files, req.instruction_summary)

        if req.target_files:
            for f in req.target_files:
                subprocess.run(["git", "-C", w_path, "add", f], capture_output=True)
        else:
            subprocess.run(["git", "-C", w_path, "add", "."], capture_output=True)

        res = subprocess.run(["git", "-C", w_path, "commit", "-m", req.commit_message], capture_output=True, text=True)
        if res.returncode == 0 or "nothing to commit" in res.stdout:
            hist = load_workspace_history(req.repo_dir_name)
            hist.setdefault("timeline_events", []).insert(0, {
                "id": f"evt-{int(datetime.datetime.utcnow().timestamp()*1000)}",
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "type": "commit",
                "branch": active_branch,
                "commit_msg": req.commit_message,
                "summary": req.instruction_summary or req.commit_message,
                "modified_files": req.target_files
            })
            save_workspace_history(req.repo_dir_name, hist)

            git_status = get_workspace_git_status(req.repo_dir_name)
            return {"status": "success", "message": f"Committed to branch '{active_branch}'", "branch": active_branch, "git_status": git_status}
        return JSONResponse(status_code=400, content={"status": "error", "message": res.stderr or res.stdout})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/workspace/discard")
def api_discard(req: DiscardRequest):
    w_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(w_path):
        return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})
    try:
        if req.target_files:
            for f in req.target_files:
                subprocess.run(["git", "-C", w_path, "checkout", "--", f], capture_output=True)
                subprocess.run(["git", "-C", w_path, "clean", "-fd", f], capture_output=True)
        else:
            subprocess.run(["git", "-C", w_path, "checkout", "--", "."], capture_output=True)
            subprocess.run(["git", "-C", w_path, "clean", "-fd"], capture_output=True)
        git_status = get_workspace_git_status(req.repo_dir_name)
        return {"status": "success", "message": "Changes discarded.", "git_status": git_status}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/workspace/validate")
def api_validate(req: WorkspaceActionRequest):
    w_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(w_path): return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})
    logs, err = [], False
    for pf in glob.glob(f"{w_path}/**/*.py", recursive=True):
        r = subprocess.run(["python3", "-m", "py_compile", pf], capture_output=True, text=True)
        rel = os.path.relpath(pf, w_path)
        if r.returncode != 0: err = True; logs.append(f"❌ Python Error in {rel}:\n{r.stderr}")
        else: logs.append(f"✓ Python OK: {rel}")
    for sf in glob.glob(f"{w_path}/**/*.sh", recursive=True):
        r = subprocess.run(["bash", "-n", sf], capture_output=True, text=True)
        rel = os.path.relpath(sf, w_path)
        if r.returncode != 0: err = True; logs.append(f"❌ Shell Error in {rel}:\n{r.stderr}")
        else: logs.append(f"✓ Bash OK: {rel}")
    return {"status": "error" if err else "success", "passed": not err, "logs": "\n".join(logs) if logs else "No testable scripts."}

@app.post("/api/workspace/push")
def api_push(req: WorkspaceActionRequest):
    w_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(w_path): return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})
    try:
        res = subprocess.run(["git", "-C", w_path, "push", "origin", req.branch], capture_output=True, text=True, timeout=25)
        if res.returncode == 0:
            git_status = get_workspace_git_status(req.repo_dir_name)
            return {"status": "success", "message": f"Pushed to origin/{req.branch}!", "git_status": git_status}
        return JSONResponse(status_code=500, content={"status": "error", "message": res.stderr or res.stdout})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)