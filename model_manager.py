"""
Madison AI Core Server Entrypoint
Direct Absolute-Path Model Loading, Background Task Queue, and Workspace Agent.
"""

import os
import re
import glob
import json
import shutil
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
    fetch_web_search_snippets,
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
from core.personas import load_all_personas, get_persona_prompt
from core.task_queue import TASK_QUEUE

app = FastAPI(title="Madison AI Workstation")

CHAT_SESSIONS_FILE = os.path.join(STORAGE_PATH, ".chat_sessions.json")
HF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
}

@app.on_event("startup")
async def startup_event():
    TASK_QUEUE.start_worker()

# ---------------- Direct Absolute-Path Model Loading ----------------

def list_local_gguf_models() -> list[dict]:
    """Scans MODELS_PATH directly for existing .gguf files, omitting any stale or deleted paths."""
    models = []
    if not os.path.exists(MODELS_PATH):
        return models

    for root, _, files in os.walk(MODELS_PATH, followlinks=True):
        for f in files:
            if f.endswith(".gguf") and not re.search(r'-0000[2-9]-of-', f):
                path = os.path.join(root, f)
                # Strict verification that the file actually exists on disk
                if not os.path.exists(path):
                    continue
                parent_dir = os.path.basename(root)
                clean_name = re.sub(r'(-0000\d-of-\d{5})?\.gguf$', '', f)
                display_label = f"{parent_dir} / {clean_name}" if parent_dir else clean_name
                
                models.append({
                    "key": path,  # Absolute path as key for reliable lms load
                    "display_name": display_label,
                    "filename": f
                })
    
    models.sort(key=lambda x: x["display_name"].lower())
    return models

def execute_lms_load(target_path: str, context_length: int = 32768, gpu_offload: str = "max") -> tuple[bool, str]:
    """Executes `lms load <absolute_path>` directly with context fallback."""
    lms = get_lms_bin()
    if not target_path or target_path in ("auto", "default"):
        return True, "Auto mode active"

    ctx_ladder = [context_length]
    for fallback in [16384, 8192, 4096]:
        if fallback < context_length:
            ctx_ladder.append(fallback)

    last_error = ""
    for ctx in ctx_ladder:
        cmd = [
            lms, "load", target_path,
            f"--gpu={gpu_offload}",
            f"--context-length={ctx}",
            "--ttl=3600",
            "--yes"
        ]
        try:
            res = subprocess.run(cmd, env=LMS_ENV, capture_output=True, text=True, timeout=90)
            if res.returncode == 0:
                return True, f"Loaded model successfully ({ctx} context)."
            last_error = (res.stderr or res.stdout or "").strip()
        except Exception as e:
            last_error = str(e)

    return False, last_error or "Failed to load model"

def ensure_active_model(model_path: str = "", context_length: int = 32768) -> str:
    loaded = get_loaded_models()
    if model_path and model_path not in ("auto", "default"):
        fname = os.path.basename(model_path).replace('.gguf', '').lower()
        for l in (loaded or []):
            if fname in l.lower() or l.lower() in fname:
                return l
        success, _ = execute_lms_load(model_path, context_length=context_length)
        if success:
            recheck = get_loaded_models()
            return recheck[0] if recheck else model_path

    if loaded:
        return loaded[0]

    ggufs = list_local_gguf_models()
    if ggufs:
        first_path = ggufs[0]["key"]
        execute_lms_load(first_path, context_length=context_length)
        recheck = get_loaded_models()
        return recheck[0] if recheck else first_path

    return "default"

# ---------------- Execution Callables for Queue ----------------

def _execute_chat_task_sync(messages: list, model_id: str, context_length: int, persona_id: str, custom_system_prompt: str, enable_web_search: bool, temperature: float, max_tokens: int, top_p: float):
    active_model = ensure_active_model(model_id, context_length=context_length)
    if not active_model:
        raise RuntimeError("No model could be initialized or loaded in VRAM.")

    msgs = list(messages)
    current_date_str = datetime.datetime.utcnow().strftime("%A, %B %d, %Y")
    base_system = get_persona_prompt(persona_id, custom_system_prompt, domain="chat")
    grounding_system = f"{base_system}\n\n[System Info: Real-world date is {current_date_str}]"

    search_context = ""
    if enable_web_search and msgs:
        user_query = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        if user_query:
            snippets = fetch_web_search_snippets(user_query, max_results=5)
            search_context = f"\n\n--- LIVE SEARCH CONTEXT ({current_date_str}) ---\n{snippets}\n--- END SEARCH CONTEXT ---\n"
            grounding_system += f"{search_context}\nIncorporate the search context above into your response. Reference sources accurately."

    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = f"{grounding_system}\n\n{msgs[0]['content']}"
    else:
        msgs.insert(0, {"role": "system", "content": grounding_system})

    payload = {
        "model": active_model,
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    }
    resp = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=240)
    if resp.status_code != 200:
        raise RuntimeError(f"LM Studio API Error: {resp.text}")
    
    data = resp.json()
    data["search_grounded"] = bool(search_context)
    data["loaded_model"] = active_model
    return data

def _execute_agent_task_sync(repo_dir_name: str, target_files: list, instruction: str, thread_id: str, persona_id: str, custom_system_prompt: str, model_id: str, context_length: int):
    effective_model = ensure_active_model(model_id, context_length=context_length)
    if not effective_model:
        raise RuntimeError("No model available to execute coding task.")
    
    return process_agent_task(
        repo_dir_name=repo_dir_name,
        target_files=target_files,
        instruction=instruction,
        thread_id=thread_id,
        persona_id=persona_id,
        custom_system_prompt=custom_system_prompt,
        model_id=effective_model
    )

# ---------------- Request Models ----------------

class DownloadRequest(BaseModel):
    repo_id: str
    group_name: str
    files: list[str]

class DeleteRequest(BaseModel):
    filename: str

class LoadRequest(BaseModel):
    model_key: str = ""
    model_path: str = ""
    context_length: int = 32768
    gpu_offload: str = "max"
    ttl: int = 3600

class AsyncChatSubmitRequest(BaseModel):
    session_id: str
    messages: list[dict]
    model_identifier: str = ""
    context_length: int = 32768
    persona_id: str = "chat"
    custom_system_prompt: str = ""
    enable_web_search: bool = False
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95

class AsyncAgentSubmitRequest(BaseModel):
    repo_dir_name: str
    target_files: list[str] = []
    instruction: str
    thread_id: str = ""
    persona_id: str = "coding"
    custom_system_prompt: str = ""
    model_identifier: str = ""
    context_length: int = 32768

class MarkTaskReadRequest(BaseModel):
    task_id: str

class SaveChatSessionRequest(BaseModel):
    session_id: str
    title: str = "Chat Conversation"
    persona_id: str = "chat"
    custom_system_prompt: str = ""
    messages: list[dict] = []

class DeleteChatSessionRequest(BaseModel):
    session_id: str

class DeleteWorkspaceRequest(BaseModel):
    repo_dir_name: str

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

# ---------------- Model Registry & Control Endpoints ----------------

@app.get("/api/lms/registered_models")
def api_get_lms_models():
    return {
        "models": list_local_gguf_models(),
        "loaded_models": get_loaded_models()
    }

@app.post("/api/lms/resync")
def api_resync_models():
    return {"status": "success", "models": list_local_gguf_models()}

@app.post("/api/load_model")
def api_load_model(req: LoadRequest):
    target = req.model_path or req.model_key
    success, msg = execute_lms_load(target, context_length=req.context_length, gpu_offload=req.gpu_offload)
    if success:
        return {"status": "success", "loaded_target": target, "output": msg}
    return JSONResponse(status_code=400, content={"status": "error", "message": msg})

@app.post("/api/unload_model")
def api_unload_model():
    lms = get_lms_bin()
    try:
        res = subprocess.run([lms, "unload", "--all"], env=LMS_ENV, capture_output=True, text=True, timeout=5)
        return {"status": "success", "output": res.stdout}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# ---------------- Background Task Queue Endpoints ----------------

@app.post("/api/queue/submit_chat")
async def api_queue_submit_chat(req: AsyncChatSubmitRequest):
    task_id = await TASK_QUEUE.submit_task(
        domain="chat",
        target_id=req.session_id,
        runner_func=_execute_chat_task_sync,
        runner_kwargs={
            "messages": req.messages,
            "model_id": req.model_identifier,
            "context_length": req.context_length,
            "persona_id": req.persona_id,
            "custom_system_prompt": req.custom_system_prompt,
            "enable_web_search": req.enable_web_search,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "top_p": req.top_p
        },
        preferred_model=req.model_identifier
    )
    return {"status": "queued", "task_id": task_id, "session_id": req.session_id}

@app.post("/api/queue/submit_agent")
async def api_queue_submit_agent(req: AsyncAgentSubmitRequest):
    t_id = req.thread_id or "thread-default"
    task_id = await TASK_QUEUE.submit_task(
        domain="coding",
        target_id=f"{req.repo_dir_name}:{t_id}",
        runner_func=_execute_agent_task_sync,
        runner_kwargs={
            "repo_dir_name": req.repo_dir_name,
            "target_files": req.target_files,
            "instruction": req.instruction,
            "thread_id": t_id,
            "persona_id": req.persona_id,
            "custom_system_prompt": req.custom_system_prompt,
            "model_id": req.model_identifier,
            "context_length": req.context_length
        },
        preferred_model=req.model_identifier
    )
    return {"status": "queued", "task_id": task_id, "repo_dir_name": req.repo_dir_name, "thread_id": t_id}

@app.get("/api/queue/task_status")
def api_get_task_status(task_id: str):
    t = TASK_QUEUE.get_task(task_id)
    if not t:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Task not found"})
    return t

@app.get("/api/queue/states")
def api_get_queue_states():
    return TASK_QUEUE.get_active_and_unread_states()

@app.post("/api/queue/mark_read")
def api_mark_task_read(req: MarkTaskReadRequest):
    TASK_QUEUE.mark_task_read(req.task_id)
    return {"status": "success"}

# ---------------- Personas API ----------------

@app.get("/api/personas")
def api_get_personas():
    return load_all_personas()

# ---------------- Chat Sessions API ----------------

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
    except Exception: pass

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
            s["persona_id"] = req.persona_id
            s["custom_system_prompt"] = req.custom_system_prompt
            s["messages"] = req.messages
            s["updated_at"] = datetime.datetime.utcnow().isoformat()
            found = True
            break
    
    if not found:
        sessions.insert(0, {
            "id": req.session_id,
            "title": req.title,
            "persona_id": req.persona_id,
            "custom_system_prompt": req.custom_system_prompt,
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

# ---------------- Hardware & Model Endpoints ----------------

@app.get("/api/system_info")
def api_sys_info():
    return get_system_hardware_info()

@app.get("/api/search")
def api_search_hf(q: str = "", sort_by: str = "downloads", verified_only: bool = False):
    hf_sort = "likes" if sort_by == "likes" else ("lastModified" if sort_by == "lastModified" else "downloads")
    params = {"filter": "gguf", "sort": hf_sort, "direction": "-1", "limit": 60}
    if q.strip(): params["search"] = q.strip()
        
    res = []
    try:
        resp = requests.get("https://huggingface.co/api/models", params=params, headers=HF_HEADERS, timeout=12)
        if resp.status_code == 200: res = resp.json()
    except Exception: pass

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
        resp = requests.get(f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true", headers=HF_HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        path = item.get("path", "")
                        if path.endswith(".gguf"):
                            sz = item.get("size", 0) or (item.get("lfs", {}).get("size", 0) if isinstance(item.get("lfs"), dict) else 0)
                            raw_files[path] = sz
    except Exception: pass

    if not raw_files:
        try:
            resp = requests.get(f"https://huggingface.co/api/models/{repo_id}", headers=HF_HEADERS, timeout=10)
            if resp.status_code == 200:
                res = resp.json()
                siblings = res.get("siblings")
                if isinstance(siblings, list):
                    for s in siblings:
                        if isinstance(s, dict):
                            fname = s.get("rfilename", "")
                            if fname.endswith(".gguf"):
                                raw_files[fname] = s.get("size", 0)
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
    vram_total = hw.get("gpu", {}).get("total_vram_gb", 0)
    ram_total = hw.get("system_ram", {}).get("total_gb", 16)

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

        max_cap_ctx = 131072 if any(k in gname.lower() for k in ["llama-3", "qwen", "nemotron", "gemma"]) else 32768

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
                    max_cap_ctx = 131072 if any(k in f.lower() for k in ["llama-3", "qwen", "nemotron", "gemma"]) else 32768
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

# ---------------- GitHub Vault Endpoints ----------------

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

# ---------------- Workspaces & Git Actions ----------------

@app.get("/api/workspaces/active")
def api_get_workspaces():
    return get_active_workspaces()

@app.post("/api/workspace/delete")
def api_delete_workspace(req: DeleteWorkspaceRequest):
    w_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(w_path):
        return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})
    try:
        shutil.rmtree(w_path)
        return {"status": "success", "message": f"Workspace '{req.repo_dir_name}' deleted."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

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
                    git_commits.append({"hash": parts[0], "author": parts[1], "date": parts[2], "subject": parts[3]})
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
    if not os.path.exists(w_path): return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})
    try:
        subprocess.run(["git", "-C", w_path, "checkout", req.branch], capture_output=True)
        res = subprocess.run(["git", "-C", w_path, "branch", "--show-current"], capture_output=True, text=True)
        return {"status": "success", "active_branch": res.stdout.strip() or req.branch}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/workspace/create_branch")
def api_create_branch(req: CreateBranchRequest):
    w_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(w_path): return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})
    clean_b = req.branch_name.strip().replace(" ", "-")
    try:
        res = subprocess.run(["git", "-C", w_path, "checkout", "-b", clean_b], capture_output=True, text=True)
        if res.returncode == 0: return {"status": "success", "active_branch": clean_b}
        return JSONResponse(status_code=400, content={"status": "error", "message": res.stderr or res.stdout})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/workspace/generate_commit_msg")
def api_gen_commit_msg(req: GenCommitMsgRequest):
    res = generate_commit_msg_from_diff(req.repo_dir_name, req.target_files, req.model_identifier)
    if res.get("status") == "error": return JSONResponse(status_code=500, content=res)
    return res

@app.post("/api/workspace/pull")
def api_pull_upstream(req: WorkspaceActionRequest):
    w_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(w_path): return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})
    try:
        res = subprocess.run(["git", "-C", w_path, "pull", "origin", req.branch], capture_output=True, text=True, timeout=25)
        if res.returncode == 0: return {"status": "success", "message": res.stdout or f"Branch '{req.branch}' is up to date."}
        return JSONResponse(status_code=500, content={"status": "error", "message": res.stderr or res.stdout})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/github/clone")
def api_clone_project(req: GithubCloneRequest):
    token = get_token_for_user(req.account_username)
    if not token:
        return JSONResponse(status_code=401, content={"status": "error", "message": "No active GitHub token found for this account."})
    
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
            res = subprocess.run(["git", "clone", "-b", req.branch, auth_url, dest_path], capture_output=True, text=True, timeout=45)
            if res.returncode != 0:
                subprocess.run(["git", "clone", auth_url, dest_path], capture_output=True, text=True, timeout=45)
                subprocess.run(["git", "-C", dest_path, "checkout", "-B", req.branch], capture_output=True, timeout=10)

        subprocess.run(["git", "-C", dest_path, "config", "user.name", req.account_username], capture_output=True)
        subprocess.run(["git", "-C", dest_path, "config", "user.email", f"{req.account_username}@users.noreply.github.com"], capture_output=True)
        os.chmod(dest_path, 0o777)
        return {"status": "success", "dir_name": dir_name, "branch": req.branch, "path": dest_path, "display_name": req.repo_full_name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Git clone failed: {str(e)}"})

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

@app.post("/api/workspace/commit")
def api_commit(req: CommitRequest):
    w_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(w_path): return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})
    try:
        b_res = subprocess.run(["git", "-C", w_path, "branch", "--show-current"], capture_output=True, text=True)
        active_branch = b_res.stdout.strip() or "main"
        append_to_changelog(w_path, active_branch, req.commit_message, req.target_files, req.instruction_summary)

        if req.target_files:
            for f in req.target_files: subprocess.run(["git", "-C", w_path, "add", f], capture_output=True)
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
            return {"status": "success", "message": f"Committed to branch '{active_branch}'", "branch": active_branch, "git_status": get_workspace_git_status(req.repo_dir_name)}
        return JSONResponse(status_code=400, content={"status": "error", "message": res.stderr or res.stdout})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/workspace/discard")
def api_discard(req: DiscardRequest):
    w_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(w_path): return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})
    try:
        if req.target_files:
            for f in req.target_files:
                subprocess.run(["git", "-C", w_path, "checkout", "--", f], capture_output=True)
                subprocess.run(["git", "-C", w_path, "clean", "-fd", f], capture_output=True)
        else:
            subprocess.run(["git", "-C", w_path, "checkout", "--", "."], capture_output=True)
            subprocess.run(["git", "-C", w_path, "clean", "-fd"], capture_output=True)
        return {"status": "success", "message": "Changes discarded.", "git_status": get_workspace_git_status(req.repo_dir_name)}
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

@app.post("/api/push")
def api_push(req: WorkspaceActionRequest):
    w_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(w_path): return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})
    try:
        res = subprocess.run(["git", "-C", w_path, "push", "origin", req.branch], capture_output=True, text=True, timeout=25)
        if res.returncode == 0:
            return {"status": "success", "message": f"Pushed to origin/{req.branch}!", "git_status": get_workspace_git_status(repo_dir_name=req.repo_dir_name)}
        return JSONResponse(status_code=500, content={"status": "error", "message": res.stderr or res.stdout})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
        
class AgentPlanRequest(BaseModel):
    repo_dir_name: str
    instruction: str
    model_identifier: str = ""

class AgentExecutePlanRequest(BaseModel):
    repo_dir_name: str
    target_files: list[str]
    instruction: str
    thread_id: str = ""
    persona_id: str = "coding"
    custom_system_prompt: str = ""
    model_identifier: str = ""
    context_length: int = 32768

@app.post("/api/agent/plan")
def api_agent_plan(req: AgentPlanRequest):
    effective_model = ensure_active_model(req.model_identifier)
    res = generate_agent_plan(req.repo_dir_name, req.instruction, model_id=effective_model)
    if res.get("status") == "error":
        return JSONResponse(status_code=500, content=res)
    return res

@app.post("/api/agent/execute_plan")
def api_agent_execute_plan(req: AgentExecutePlanRequest):
    effective_model = ensure_active_model(req.model_identifier)
    res = execute_approved_plan(
        repo_dir_name=req.repo_dir_name,
        target_files=req.target_files,
        instruction=req.instruction,
        thread_id=req.thread_id,
        persona_id=req.persona_id,
        custom_system_prompt=req.custom_system_prompt,
        model_id=effective_model,
        context_length=req.context_length
    )
    if res.get("status") == "error":
        return JSONResponse(status_code=500, content=res)
    return res

if __name__ == "__main__":
    uvicorn.run("model_manager:app", host="0.0.0.0", port=8080, log_level="info")