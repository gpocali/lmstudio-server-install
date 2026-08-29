"""
Madison AI Core Server Entrypoint
Direct LM Studio Registry Management, Background Task Queue, and Workspace Agent.
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
    ensure_models_indexed()

# ---------------- Direct LMS Registry & Loading ----------------

def ensure_models_indexed():
    """Indexes all local GGUF models into LM Studio's registry via symbolic links and import."""
    lms = get_lms_bin()
    lms_dirs = [
        os.path.join(STORAGE_PATH, ".cache", "lm-studio", "models"),
        os.path.join(STORAGE_PATH, ".lmstudio", "models")
    ]
    for d in lms_dirs:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(MODELS_PATH):
        return

    for root, _, files in os.walk(MODELS_PATH, followlinks=True):
        for f in files:
            if f.endswith(".gguf") and not re.search(r'-0000[2-9]-of-', f):
                src_file = os.path.join(root, f)
                rel = os.path.relpath(root, MODELS_PATH)
                for base in lms_dirs:
                    target_dir = os.path.join(base, rel)
                    os.makedirs(target_dir, exist_ok=True)
                    link_dest = os.path.join(target_dir, f)
                    if not os.path.exists(link_dest):
                        try: os.symlink(src_file, link_dest)
                        except Exception: pass
                try:
                    subprocess.run([lms, "import", "--yes", "--symbolic-link", src_file], env=LMS_ENV, capture_output=True, timeout=5)
                except Exception: pass

def list_registered_lms_keys() -> list[dict]:
    """Retrieves clean model keys directly from LM Studio CLI (`lms ls`). 
    Deduplicates entries to prefer descriptive names (e.g., 'Author - Model') over short ones."""
    lms = get_lms_bin()
    raw_models = []
    seen_keys = set()

    # 1. Try parsing JSON
    try:
        res = subprocess.run([lms, "ls", "--json"], env=LMS_ENV, capture_output=True, text=True, timeout=6)
        if res.returncode == 0 and res.stdout.strip().startswith("{"):
            data = json.loads(res.stdout)
            for m in data.get("models", data.get("llms", [])):
                k = m.get("key") or m.get("identifier") or m.get("path")
                if k and str(k).strip() not in seen_keys:
                    seen_keys.add(str(k).strip())
                    raw_models.append({
                        "key": str(k).strip(),
                        "display_name": m.get("name") or str(k).strip(),
                        "loaded": m.get("loaded", False)
                    })
    except Exception: pass

    # 2. Text fallback parsing of `lms ls`
    try:
        res = subprocess.run([lms, "ls"], env=LMS_ENV, capture_output=True, text=True, timeout=6)
        if res.returncode == 0:
            in_llm = False
            for line in res.stdout.splitlines():
                raw = line.strip()
                if "LLM" in raw.upper():
                    in_llm = True
                    continue
                if "EMBEDDING" in raw.upper() or raw.startswith("===") or raw.startswith("---"):
                    in_llm = False
                    continue
                if in_llm and raw:
                    clean = re.sub(r'^[├│└─•\*\s\-\>\|]+', '', raw).strip()
                    parts = clean.split()
                    if parts:
                        candidate_key = parts[0].strip()
                        if candidate_key and not candidate_key.startswith("-") and candidate_key.upper() != "IDENTIFIER":
                            if candidate_key not in seen_keys:
                                seen_keys.add(candidate_key)
                                raw_models.append({
                                    "key": candidate_key,
                                    "display_name": candidate_key,
                                    "loaded": "LOADED" in raw.upper()
                                })
    except Exception: pass

    # 3. Disk fallback (only if nothing found)
    if not raw_models and os.path.exists(MODELS_PATH):
        for root, _, files in os.walk(MODELS_PATH, followlinks=True):
            for f in files:
                if f.endswith(".gguf") and not re.search(r'-0000[2-9]-of-', f):
                    clean_name = re.sub(r'(-0000\d-of-\d{5})?\.gguf$', '', f).lower()
                    if clean_name not in seen_keys:
                        seen_keys.add(clean_name)
                        raw_models.append({
                            "key": clean_name,
                            "display_name": clean_name,
                            "loaded": False
                        })

    if not raw_models:
        return []

    # --- Deduplication Logic ---
    # Sort by display name length descending so "Author - ModelName" comes before "ModelName"
    sorted_by_len = sorted(raw_models, key=lambda x: len(x['display_name']), reverse=True)
    unique_models = []

    for m in sorted_by_len:
        # If this model's name is a substring of an already added (longer/more descriptive) name, skip it.
        is_redundant = False
        for existing in unique_models:
            if m['display_name'] in existing['display_name']:
                is_redundant = True
                break
        
        if not is_redundant:
            unique_models.append(m)

    # Return sorted alphabetically for a clean UI experience
    return sorted(unique_models, key=lambda x: x['display_name'].lower())

def execute_lms_load(target_key: str, context_length: int = 32768, gpu_offload: str = "max") -> tuple[bool, str]:
    """Executes `lms load <key>` directly with fallback on context length if VRAM bounds are exceeded."""
    lms = get_lms_bin()
    if not target_key or target_key in ("auto", "default"):
        return True, "Auto mode active"

    ctx_ladder = [context_length]
    for fallback in [16384, 8192, 4096]:
        if fallback < context_length:
            ctx_ladder.append(fallback)

    last_error = ""
    for ctx in ctx_ladder:
        cmd = [
            lms, "load", target_key,
            f"--gpu={gpu_offload}",
            f"--context-length={ctx}",
            "--ttl=3600",
            "--yes"
        ]
        try:
            res = subprocess.run(cmd, env=LMS_ENV, capture_output=True, text=True, timeout=90)
            if res.returncode == 0:
                return True, f"Loaded {target_key} successfully ({ctx} context)."
            last_error = (res.stderr or res.stdout or "").strip()
        except Exception as e:
            last_error = str(e)

    return False, last_error or "Failed to load model"

def ensure_active_model(model_key: str = "", context_length: int = 32768) -> str:
    loaded = get_loaded_models()
    if model_key and model_key not in ("auto", "default"):
        for l in (loaded or []):
            if model_key.lower() in l.lower() or l.lower() in model_key.lower():
                return l
        success, _ = execute_lms_load(model_key, context_length=context_length)
        if success:
            recheck = get_loaded_models()
            return recheck[0] if recheck else model_key

    if loaded:
        return loaded[0]

    reg = list_registered_lms_keys()
    if reg:
        first_key = reg[0]["key"]
        execute_lms_load(first_key, context_length=context_length)
        recheck = get_loaded_models()
        return recheck[0] if recheck else first_key

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

# ---------------- API Endpoints ----------------

@app.get("/hardware/stats")
async def get_hw_stats():
    return {
        "vram": get_system_hardware_info().get("vram_usage", "0GB"),
        "storage": get_storage_usage()
    }

@app.get("/models/list")
async def list_models():
    return {"models": list_registered_lms_keys()}

@app.post("/models/refresh")
async def refresh_models():
    """Triggers a re-index of the local model directory into LM Studio."""
    ensure_models_indexed()
    return {"status": "success", "message": "Models re-indexed successfully"}

@app.post("/models/load")
async def load_model(req: LoadRequest):
    success, msg = execute_lms_load(req.model_key, req.context_length, req.gpu_offload)
    return JSONResponse({"success": success, "message": msg})

@app.get("/models/loaded")
async def get_loaded():
    return {"loaded": get_loaded_models() or []}

# ... [Rest of the file continues with existing endpoints] ...