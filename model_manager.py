"""
LM Studio Remote Model & Multi-Account Code Studio Manager
Manages HuggingFace GGUF models, multiple GitHub accounts, and local AI agent workspaces.
"""

import os
import re
import glob
import json
import ctypes
import shutil
import subprocess
import threading
import concurrent.futures
import requests
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="LM Studio Headless Model & Code Studio")

DOWNLOAD_JOBS = {}
STORAGE_PATH = "/storage/lmstudio"
MODELS_PATH = os.path.join(STORAGE_PATH, "models")
WORKSPACES_ROOT = os.path.join(STORAGE_PATH, "workspaces")
ACCOUNTS_FILE = os.path.join(STORAGE_PATH, ".github_accounts.json")

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(WORKSPACES_ROOT, exist_ok=True)

LMS_ENV = {
    **os.environ,
    "HOME": STORAGE_PATH,
    "LMS_SERVER_HOST": "0.0.0.0",
    "PATH": f"/usr/local/bin:{STORAGE_PATH}/.cache/lm-studio/bin:{STORAGE_PATH}/.lmstudio/bin:/usr/sbin:/usr/bin:/bin"
}

# Configure Git safe directory globally for all subdirectories under workspaces
subprocess.run(["git", "config", "--global", "--add", "safe.directory", "*"], capture_output=True)

VERIFIED_CREATORS = {
    "bartowski", "unsloth", "TheBloke", "MaziyarPanahi", "mradermacher",
    "QuantFactory", "meta-llama", "Qwen", "mistralai", "google",
    "deepseek-ai", "microsoft", "nomic-ai", "cohere", "NousResearch"
}

QUANT_DESCRIPTIONS = {
    "Q4_K_M": "Recommended standard. Medium 4-bit quantization with optimal balance between memory and quality.",
    "Q4_K_S": "Small 4-bit quantization. Uses slightly less memory than Q4_K_M.",
    "Q5_K_M": "High quality 5-bit quantization. Near-original quality with modest memory footprint.",
    "Q5_K_S": "Compact 5-bit quantization. Higher precision than 4-bit.",
    "Q8_0": "Extremely high precision (8-bit). Virtually zero quality loss; requires high VRAM.",
    "Q6_K": "Very high quality 6-bit quantization. Perceptually indistinguishable from 16-bit float.",
    "Q3_K_L": "Large 3-bit quantization. Noticeable quality reduction on small models.",
    "Q3_K_M": "Medium 3-bit quantization. Aggressive compression.",
    "IQ4_XS": "Importance Matrix 4-bit extra small.",
    "IQ4_NL": "Importance Matrix 4-bit non-linear."
}

# ----------------- Data Models -----------------

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

class AgentTaskRequest(BaseModel):
    repo_dir_name: str
    target_file: str
    instruction: str
    model_identifier: str = ""

class WorkspaceActionRequest(BaseModel):
    repo_dir_name: str
    branch: str = "main"

# ----------------- Helper Functions -----------------

def get_lms_bin():
    for p in ["/usr/local/bin/lms", f"{STORAGE_PATH}/.lmstudio/bin/lms", f"{STORAGE_PATH}/.cache/lm-studio/bin/lms"]:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p
    return shutil.which("lms") or "lms"

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

def get_loaded_models():
    loaded = []
    lms = get_lms_bin()
    try:
        res = subprocess.run([lms, "ps"], env=LMS_ENV, capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            for line in res.stdout.strip().split("\n"):
                l_str = line.strip()
                if not l_str or "IDENTIFIER" in l_str or "---" in l_str or "No models" in l_str:
                    continue
                parts = l_str.split()
                if parts:
                    loaded.append(parts[0].lower())
                    if len(parts) > 1:
                        loaded.append(parts[1].lower())
    except Exception:
        pass
    return list(set(loaded))

def get_active_model_for_agent():
    models = get_loaded_models()
    if models:
        return models[0]
    return "default"

def calculate_trust_score(downloads: int, likes: int, is_verified: bool) -> int:
    score = 35 if is_verified else 0
    if downloads >= 100000: score += 40
    elif downloads >= 10000: score += 30
    elif downloads >= 1000: score += 20
    elif downloads >= 100: score += 10
    if likes >= 500: score += 25
    elif likes >= 100: score += 18
    elif likes >= 20: score += 10
    elif likes >= 5: score += 5
    return min(score, 100)

def get_quant_description(variant: str):
    v_upper = variant.upper().replace("-", "_")
    for key, desc in QUANT_DESCRIPTIONS.items():
        if key == v_upper: return desc
    return "Standard GGUF quantization variant."

def get_storage_usage():
    target_path = MODELS_PATH if os.path.exists(MODELS_PATH) else STORAGE_PATH
    try:
        total, used, free = shutil.disk_usage(target_path)
        return {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "percent_used": round((used / total) * 100, 1)
        }
    except Exception:
        return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0, "percent_used": 0.0}

def get_system_hardware_info():
    sys_ram_total_gb, sys_ram_avail_gb = 0.0, 0.0
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        total_m = re.search(r'MemTotal:\s+(\d+)\s+kB', meminfo)
        avail_m = re.search(r'MemAvailable:\s+(\d+)\s+kB', meminfo)
        if total_m: sys_ram_total_gb = round(int(total_m.group(1)) / (1024**2), 2)
        if avail_m: sys_ram_avail_gb = round(int(avail_m.group(1)) / (1024**2), 2)
    except Exception: pass

    gpu_found = False
    gpu_name = "NVIDIA Dedicated GPU"
    gpu_vram_total_gb, gpu_vram_free_gb = 0.0, 0.0

    try:
        for name in ["libnvidia-ml.so.1", "libnvidia-ml.so", "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"]:
            try:
                nvml = ctypes.CDLL(name)
                break
            except Exception:
                nvml = None

        if nvml and (nvml.nvmlInit_v2() == 0 or nvml.nvmlInit() == 0):
            class nvmlMemory_t(ctypes.Structure):
                _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]

            device_count = ctypes.c_uint()
            nvml.nvmlDeviceGetCount_v2(ctypes.byref(device_count))
            tot_bytes, free_bytes, names = 0, 0, []
            for i in range(device_count.value):
                handle = ctypes.c_void_p()
                if nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle)) == 0:
                    name_buf = ctypes.create_string_buffer(64)
                    nvml.nvmlDeviceGetName(handle, name_buf, 64)
                    names.append(name_buf.value.decode('utf-8'))
                    mem = nvmlMemory_t()
                    if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) == 0:
                        tot_bytes += mem.total
                        free_bytes += mem.free
            if tot_bytes > 0:
                gpu_found = True
                gpu_name = ", ".join(names)
                gpu_vram_total_gb = round(tot_bytes / (1024**3), 2)
                gpu_vram_free_gb = round(free_bytes / (1024**3), 2)
            try: nvml.nvmlShutdown()
            except Exception: pass
    except Exception:
        gpu_found = False

    return {
        "system_ram": {"total_gb": sys_ram_total_gb, "available_gb": sys_ram_avail_gb},
        "gpu": {"has_gpu": gpu_found, "gpu_name": gpu_name, "total_vram_gb": gpu_vram_total_gb, "free_vram_gb": gpu_vram_free_gb},
        "storage": get_storage_usage(),
        "loaded_models": get_loaded_models()
    }

def parse_model_metadata(filename: str, repo_id: str):
    weight_match = re.search(r'(\d+(\.\d+)?(?:x\d+)?[bB])', f"{repo_id} {filename}")
    weight = weight_match.group(1).upper() if weight_match else "Unknown"
    quant_match = re.search(r'(IQ\d_[A-Z_]+|Q\d_[A-Z0-9_]+|FP16|BF16|F16|F32)', filename, re.IGNORECASE)
    variant = quant_match.group(1).upper() if quant_match else "Standard"
    return weight, variant

def fetch_single_file_size(repo_id: str, rel_path: str):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{rel_path}"
    try:
        r = requests.head(url, headers={"Accept-Encoding": "identity"}, allow_redirects=True, timeout=5)
        if r.status_code == 200:
            return int(r.headers.get("Content-Length", 0))
    except Exception: pass
    return 0

def run_download_job(repo_id: str, group_name: str, file_paths: list[str]):
    DOWNLOAD_JOBS[group_name] = {
        "status": "downloading",
        "downloaded_bytes": 0,
        "total_bytes": 0,
        "progress_str": "Connecting...",
        "percent": 0.0
    }
    dest_dir = os.path.join(MODELS_PATH, repo_id.replace('/', '_'))
    os.makedirs(dest_dir, exist_ok=True)

    try:
        total_all_shards = sum(fetch_single_file_size(repo_id, p) for p in file_paths)
        DOWNLOAD_JOBS[group_name]["total_bytes"] = total_all_shards
        cum_downloaded = 0
        first_shard_file = None

        for idx, rel_path in enumerate(file_paths, 1):
            dest_file = os.path.join(dest_dir, os.path.basename(rel_path))
            if not first_shard_file:
                first_shard_file = dest_file

            url = f"https://huggingface.co/{repo_id}/resolve/main/{rel_path}"
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(dest_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            cum_downloaded += len(chunk)
                            dl_gb = round(cum_downloaded / (1024**3), 2)
                            tot_gb = round(total_all_shards / (1024**3), 2) if total_all_shards > 0 else 0
                            pct = round((cum_downloaded / total_all_shards) * 100, 1) if total_all_shards > 0 else 0.0
                            shard_note = f" (Part {idx}/{len(file_paths)})" if len(file_paths) > 1 else ""
                            DOWNLOAD_JOBS[group_name].update({
                                "downloaded_bytes": cum_downloaded,
                                "progress_str": f"{dl_gb} GB / {tot_gb} GB ({pct}%){shard_note}",
                                "percent": pct
                            })

        DOWNLOAD_JOBS[group_name]["status"] = "completed"
        DOWNLOAD_JOBS[group_name]["progress_str"] = "100% (Complete)"
        DOWNLOAD_JOBS[group_name]["percent"] = 100.0
        
        lms_cache = os.path.join(STORAGE_PATH, ".cache", "lm-studio", "models")
        os.makedirs(lms_cache, exist_ok=True)
        dest_folder_name = repo_id.replace('/', '_')
        link_target = os.path.join(lms_cache, dest_folder_name)
        if not os.path.exists(link_target):
            try: os.symlink(dest_dir, link_target)
            except Exception: pass

        if first_shard_file:
            subprocess.run([get_lms_bin(), "import", "--yes", "--symbolic-link", first_shard_file], env=LMS_ENV, capture_output=True, timeout=10)

    except Exception as e:
        DOWNLOAD_JOBS[group_name] = {"status": "failed", "progress_str": f"Error: {str(e)}", "percent": 0.0}

# ----------------- Model API Routes -----------------

@app.get("/api/system_info")
def get_sys_info():
    return get_system_hardware_info()

@app.get("/api/search")
def search_hf(q: str = "", sort_by: str = "downloads", verified_only: bool = False):
    hf_sort = "likes" if sort_by == "likes" else ("lastModified" if sort_by == "lastModified" else "downloads")
    params = {"filter": "gguf", "sort": hf_sort, "direction": "-1", "limit": 60}
    if q.strip(): params["search"] = q.strip()

    try:
        resp = requests.get("https://huggingface.co/api/models", params=params, timeout=10)
        res = resp.json() if resp.status_code == 200 else []
    except Exception:
        res = []

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

    if sort_by == "alphabetical":
        results.sort(key=lambda x: x["model_name"].lower())
    elif sort_by == "trust":
        results.sort(key=lambda x: x["trust_score"], reverse=True)
    return results

@app.get("/api/model_files")
def get_model_files(repo_id: str):
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

    grouped_variants = {}
    for rel_path, size_bytes in raw_files.items():
        fname = os.path.basename(rel_path)
        shard_match = re.search(r'(-\d{5}-of-\d{5})', fname)
        clean_name = fname.replace(shard_match.group(1), "") if shard_match else fname
        if clean_name not in grouped_variants:
            grouped_variants[clean_name] = {"group_name": clean_name, "paths": [], "total_bytes": 0}
        grouped_variants[clean_name]["paths"].append(rel_path)
        grouped_variants[clean_name]["total_bytes"] += size_bytes

    for group in grouped_variants.values():
        group["paths"].sort()

    hw = get_system_hardware_info()
    vram_total = hw["gpu"]["total_vram_gb"]
    ram_total = hw["system_ram"]["total_gb"]

    local_files_on_disk = set()
    if os.path.exists(MODELS_PATH):
        for root, _, filenames in os.walk(MODELS_PATH):
            for f in filenames:
                if f.endswith(".gguf"): local_files_on_disk.add(f)

    parsed_files = []
    for gname, gdata in grouped_variants.items():
        weight, variant = parse_model_metadata(gname, repo_id)
        size_gb = round(gdata["total_bytes"] / (1024**3), 2) if gdata["total_bytes"] > 0 else 0.0
        est_mem_req = round(size_gb * 1.2, 2) if size_gb > 0 else 0.0

        fit_status = "unknown"
        if size_gb > 0:
            if vram_total > 0:
                fit_status = "fits_gpu" if est_mem_req <= vram_total else ("split_gpu_ram" if est_mem_req <= (vram_total + ram_total * 0.75) else "exceeds")
            else:
                fit_status = "fits_ram" if est_mem_req <= ram_total * 0.85 else "exceeds"

        shard_basenames = [os.path.basename(p) for p in gdata["paths"]]
        is_downloaded = all(sb in local_files_on_disk for sb in shard_basenames)
        is_downloading = gname in DOWNLOAD_JOBS and DOWNLOAD_JOBS[gname].get("status") == "downloading"
        shard_info = f" ({len(gdata['paths'])} Shards Package)" if len(gdata['paths']) > 1 else ""

        parsed_files.append({
            "group_name": gname, "display_name": gname + shard_info, "paths": gdata["paths"],
            "is_sharded": len(gdata["paths"]) > 1, "shard_count": len(gdata["paths"]),
            "weight": weight, "variant": variant, "description": get_quant_description(variant),
            "size_gb": f"{size_gb} GB" if size_gb > 0 else "Pending...", "raw_size_gb": size_gb,
            "est_vram": f"~{est_mem_req} GB" if est_mem_req > 0 else "N/A",
            "fit_status": fit_status, "is_downloaded": is_downloaded, "is_downloading": is_downloading
        })

    parsed_files.sort(key=lambda x: x["raw_size_gb"] if x["raw_size_gb"] > 0 else 999)
    return {"repo_id": repo_id, "hardware": hw, "files": parsed_files}

@app.post("/api/load_model")
def load_model(req: LoadRequest):
    lms = get_lms_bin()
    fname = os.path.basename(req.model_path)
    parent_dir = os.path.basename(os.path.dirname(req.model_path))

    try: subprocess.run([lms, "import", "--yes", "--symbolic-link", req.model_path], env=LMS_ENV, capture_output=True, timeout=5)
    except Exception: pass

    try: subprocess.run([lms, "unload", "--all"], env=LMS_ENV, capture_output=True, timeout=5)
    except Exception: pass

    candidates = []
    try:
        ls_res = subprocess.run([lms, "ls"], env=LMS_ENV, capture_output=True, text=True, timeout=3)
        if ls_res.returncode == 0:
            for line in ls_res.stdout.split("\n"):
                parts = line.split()
                if parts and not parts[0].startswith("---") and parts[0] != "LLM" and parts[0] != "EMBEDDING":
                    key = parts[0]
                    k_clean = re.sub(r'[^a-zA-Z0-9]', '', key).lower()
                    f_clean = re.sub(r'[^a-zA-Z0-9]', '', fname).lower()
                    if k_clean in f_clean or f_clean in k_clean:
                        candidates.append(key)
    except Exception: pass

    clean_base = re.sub(r'(-0000\d-of-\d{5})?\.gguf$', '', fname)
    candidates.extend([clean_base, fname, parent_dir, req.model_path])
    
    last_error = ""
    for target in candidates:
        try:
            cmd = [lms, "load", target, f"--gpu={req.gpu_offload}", f"--context-length={req.context_length}", f"--ttl={req.ttl}", "--yes"]
            res = subprocess.run(cmd, env=LMS_ENV, capture_output=True, text=True, timeout=15)
            if res.returncode == 0:
                return {"status": "success", "loaded_target": target, "output": res.stdout}
            else:
                last_error = (res.stderr or res.stdout or "").strip()
        except Exception as err:
            last_error = str(err)

    return JSONResponse(status_code=400, content={"status": "error", "message": last_error or "Unable to match model in library"})

@app.post("/api/unload_model")
def unload_model():
    lms = get_lms_bin()
    try:
        res = subprocess.run([lms, "unload", "--all"], env=LMS_ENV, capture_output=True, text=True, timeout=5)
        return {"status": "success", "output": res.stdout}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/download")
def start_download(req: DownloadRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_download_job, req.repo_id, req.group_name, req.files)
    return {"status": "started", "group_name": req.group_name}

@app.post("/api/delete")
def delete_local_model(req: DeleteRequest):
    deleted = False
    if os.path.exists(MODELS_PATH):
        prefix_pattern = req.filename.replace(".gguf", "")
        for root, _, filenames in os.walk(MODELS_PATH):
            for f in filenames:
                if f == req.filename or (prefix_pattern in f and f.endswith(".gguf")):
                    try:
                        os.remove(os.path.join(root, f))
                        deleted = True
                    except Exception: pass
            if not os.listdir(root):
                try: os.rmdir(root)
                except Exception: pass
    if req.filename in DOWNLOAD_JOBS:
        del DOWNLOAD_JOBS[req.filename]
    return {"status": "deleted" if deleted else "not_found", "storage": get_storage_usage()}

@app.get("/api/tasks")
def get_tasks():
    return DOWNLOAD_JOBS

@app.get("/api/local_models")
def get_local_models():
    files = []
    if os.path.exists(MODELS_PATH):
        for root, _, filenames in os.walk(MODELS_PATH):
            for f in filenames:
                if f.endswith(".gguf") and not re.search(r'-0000[2-9]-of-', f):
                    path = os.path.join(root, f)
                    size_gb = round(os.path.getsize(path) / (1024**3), 2)
                    weight, variant = parse_model_metadata(f, root)
                    files.append({
                        "filename": f, "weight": weight, "variant": variant,
                        "size_gb": f"{size_gb} GB", "path": path
                    })
    return {"files": files, "storage": get_storage_usage(), "loaded_models": get_loaded_models()}

# ----------------- Multi-Account GitHub Management -----------------

@app.get("/api/github/accounts")
def list_accounts():
    data = load_accounts_data()
    # Mask tokens for safe client display
    safe_accounts = []
    for a in data.get("accounts", []):
        t = a.get("token", "")
        masked_token = f"ghp_...{t[-4:]}" if len(t) > 4 else "ghp_****"
        safe_accounts.append({
            "username": a.get("username", ""),
            "label": a.get("label", a.get("username", "")),
            "masked_token": masked_token,
            "avatar_url": a.get("avatar_url", "")
        })
    return safe_accounts

@app.post("/api/github/accounts/add")
def add_account(req: AddAccountRequest):
    token = req.token.strip()
    if not token:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Token cannot be empty"})

    try:
        # Validate token with GitHub API
        r = requests.get("https://api.github.com/user", headers={"Authorization": f"Bearer {token}"}, timeout=6)
        if r.status_code != 200:
            return JSONResponse(status_code=401, content={"status": "error", "message": "Invalid GitHub Token. Please verify permissions."})
        
        user_info = r.json()
        username = user_info.get("login", "")
        avatar = user_info.get("avatar_url", "")
        label = req.label.strip() or username

        data = load_accounts_data()
        # Remove if already exists to update
        data["accounts"] = [a for a in data.get("accounts", []) if a.get("username", "").lower() != username.lower()]
        data["accounts"].append({
            "username": username,
            "label": label,
            "token": token,
            "avatar_url": avatar
        })
        save_accounts_data(data)
        return {"status": "success", "username": username, "label": label}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/github/accounts/remove")
def remove_account(req: RemoveAccountRequest):
    data = load_accounts_data()
    orig_len = len(data.get("accounts", []))
    data["accounts"] = [a for a in data.get("accounts", []) if a.get("username", "").lower() != req.username.lower()]
    if len(data["accounts"]) < orig_len:
        save_accounts_data(data)
        return {"status": "success", "message": f"Account @{req.username} removed."}
    return JSONResponse(status_code=404, content={"status": "error", "message": "Account not found"})

@app.get("/api/github/repos")
def list_repos_for_account(account_username: str):
    token = get_token_for_user(account_username)
    if not token:
        return JSONResponse(status_code=401, content={"status": "error", "message": f"No active token for user @{account_username}"})

    try:
        r = requests.get("https://api.github.com/user/repos?per_page=100&sort=updated", headers={"Authorization": f"Bearer {token}"}, timeout=8)
        if r.status_code == 200:
            repos = []
            for item in r.json():
                repos.append({
                    "full_name": item.get("full_name"),
                    "name": item.get("name"),
                    "owner": item.get("owner", {}).get("login"),
                    "default_branch": item.get("default_branch", "main"),
                    "is_private": item.get("private", False)
                })
            return repos
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    return []

@app.get("/api/github/branches")
def list_branches_for_repo(account_username: str, repo_full_name: str):
    token = get_token_for_user(account_username)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.get(f"https://api.github.com/repos/{repo_full_name}/branches", headers=headers, timeout=6)
        if r.status_code == 200:
            return [b.get("name") for b in r.json()]
    except Exception: pass
    return ["main", "dev", "master"]

# ----------------- Workspaces Management -----------------

@app.get("/api/workspaces/active")
def get_active_workspaces():
    """Lists all locally cloned projects on the disk."""
    workspaces = []
    if os.path.exists(WORKSPACES_ROOT):
        for d in os.listdir(WORKSPACES_ROOT):
            w_path = os.path.join(WORKSPACES_ROOT, d)
            if os.path.isdir(w_path) and os.path.exists(os.path.join(w_path, ".git")):
                branch_res = subprocess.run(["git", "-C", w_path, "branch", "--show-current"], capture_output=True, text=True)
                branch = branch_res.stdout.strip() or "main"
                
                # Extract owner/repo from folder slug (format: user_repo)
                display_name = d.replace("_", "/", 1) if "_" in d else d
                workspaces.append({
                    "dir_name": d,
                    "display_name": display_name,
                    "branch": branch,
                    "path": w_path
                })
    return workspaces

@app.post("/api/github/clone")
def clone_or_open_project(req: GithubCloneRequest):
    token = get_token_for_user(req.account_username)
    if not token:
        return JSONResponse(status_code=401, content={"status": "error", "message": f"Authentication required for @{req.account_username}"})

    # Folder pattern: user_repo
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

        # Set user configuration inside the specific repository
        subprocess.run(["git", "-C", dest_path, "config", "user.name", req.account_username], capture_output=True)
        subprocess.run(["git", "-C", dest_path, "config", "user.email", f"{req.account_username}@users.noreply.github.com"], capture_output=True)
        os.chmod(dest_path, 0o777)

        return {"status": "success", "dir_name": dir_name, "branch": req.branch, "path": dest_path}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/workspaces/switch_branch")
def switch_workspace_branch(req: GithubBranchSwitchRequest):
    workspace_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(workspace_path):
        return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace directory not found"})

    try:
        subprocess.run(["git", "-C", workspace_path, "checkout", req.branch], capture_output=True)
        res = subprocess.run(["git", "-C", workspace_path, "branch", "--show-current"], capture_output=True, text=True)
        current = res.stdout.strip()
        if current != req.branch:
            subprocess.run(["git", "-C", workspace_path, "checkout", "-B", req.branch], capture_output=True)
        return {"status": "success", "active_branch": req.branch}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/workspace/files")
def get_workspace_files(repo_dir_name: str):
    workspace_path = os.path.join(WORKSPACES_ROOT, repo_dir_name)
    if not os.path.exists(workspace_path):
        return []

    file_list = []
    for root, dirs, files in os.walk(workspace_path):
        if ".git" in root or "__pycache__" in root: continue
        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), workspace_path)
            file_list.append(rel_path)
    file_list.sort()
    return file_list

@app.post("/api/agent/execute")
def execute_agent_task(req: AgentTaskRequest):
    workspace_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    file_path = os.path.join(workspace_path, req.target_file)

    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"status": "error", "message": f"File {req.target_file} not found in workspace"})

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            original_code = f.read()
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Could not read target file: {str(e)}"})

    system_prompt = (
        "You are an expert AI software engineer. "
        "Your task is to edit source code files precisely according to instruction.\n"
        "RULES:\n"
        "1. Return ONLY the complete, updated source code for the requested file enclosed in a single markdown code block.\n"
        "2. Do NOT truncate or abbreviate code with placeholders like '// ... existing code ...'. Always output the full file.\n"
        "3. Maintain all existing imports, formatting, and functionality unless explicitly instructed to modify them."
    )

    user_prompt = (
        f"File: {req.target_file}\n\n"
        f"Current File Content:\n```\n{original_code}\n```\n\n"
        f"Task Instruction:\n{req.instruction}\n\n"
        f"Provide the complete updated {req.target_file} file:"
    )

    model_id = req.model_identifier or get_active_model_for_agent()

    try:
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 16384
        }
        
        resp = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=120)
        if resp.status_code != 200:
            return JSONResponse(status_code=500, content={"status": "error", "message": f"LM Studio API Error: {resp.text}"})

        ai_response = resp.json()["choices"][0]["message"]["content"]
        
        # Extract code cleanly from markdown fences
        code_match = re.search(r'```(?:[a-zA-Z0-9_\-]+)?\n([\s\S]*?)\n```', ai_response)
        updated_code = code_match.group(1) if code_match else ai_response.strip()

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated_code)

        subprocess.run(["git", "-C", workspace_path, "add", req.target_file], capture_output=True)
        commit_msg = f"AI Update: {req.instruction[:70]}"
        subprocess.run(["git", "-C", workspace_path, "commit", "-m", commit_msg], capture_output=True)

        diff_res = subprocess.run(["git", "-C", workspace_path, "diff", "HEAD~1", "HEAD"], capture_output=True, text=True)
        diff_text = diff_res.stdout or "File modified and committed."

        return {
            "status": "success",
            "commit_message": commit_msg,
            "diff": diff_text[:4000]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Execution error: {str(e)}"})

@app.post("/api/workspace/validate")
def validate_workspace(req: WorkspaceActionRequest):
    workspace_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(workspace_path):
        return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})

    logs = []
    has_error = False

    py_files = glob.glob(f"{workspace_path}/**/*.py", recursive=True)
    for pf in py_files:
        res = subprocess.run(["python3", "-m", "py_compile", pf], capture_output=True, text=True)
        rel = os.path.relpath(pf, workspace_path)
        if res.returncode != 0:
            has_error = True
            logs.append(f"❌ Python Syntax Error in {rel}:\n{res.stderr}")
        else:
            logs.append(f"✓ Python Compilation OK: {rel}")

    sh_files = glob.glob(f"{workspace_path}/**/*.sh", recursive=True)
    for sf in sh_files:
        res = subprocess.run(["bash", "-n", sf], capture_output=True, text=True)
        rel = os.path.relpath(sf, workspace_path)
        if res.returncode != 0:
            has_error = True
            logs.append(f"❌ Shell Script Syntax Error in {rel}:\n{res.stderr}")
        else:
            logs.append(f"✓ Bash Syntax OK: {rel}")

    return {
        "status": "error" if has_error else "success",
        "passed": not has_error,
        "logs": "\n".join(logs) if logs else "No Python or Shell scripts found to validate."
    }

@app.post("/api/workspace/push")
def push_to_github(req: WorkspaceActionRequest):
    workspace_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(workspace_path):
        return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})

    try:
        res = subprocess.run(["git", "-C", workspace_path, "push", "origin", req.branch], capture_output=True, text=True, timeout=25)
        if res.returncode == 0:
            return {"status": "success", "message": f"Successfully pushed commits to branch '{req.branch}' on GitHub!"}
        else:
            return JSONResponse(status_code=500, content={"status": "error", "message": res.stderr or res.stdout})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# ----------------- Frontend HTML -----------------

@app.get("/", response_class=HTMLResponse)
def get_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>LM Studio Remote Model & Multi-Account Code Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen p-4 md:p-8 font-sans">
  <div class="max-w-7xl mx-auto space-y-6">
    
    <!-- Top Bar & Hardware Specs -->
    <header class="flex flex-col lg:flex-row justify-between items-start lg:items-center border-b border-slate-700 pb-5 gap-4">
      <div>
        <h1 class="text-2xl font-bold text-sky-400">LM Studio Code & Model Studio</h1>
        <p class="text-xs text-slate-400">Storage Target: /storage/lmstudio</p>
      </div>
      
      <div class="flex flex-wrap items-center gap-3 text-xs">
        <div class="bg-slate-800 px-3.5 py-2 rounded-lg border border-slate-700 flex items-center gap-2">
          <span class="text-slate-400">Dedicated VRAM:</span>
          <span id="vramStat" class="font-semibold text-emerald-400">Probing NVML...</span>
        </div>
        
        <div class="bg-slate-800 px-3.5 py-2 rounded-lg border border-slate-700 flex items-center gap-2">
          <span class="text-slate-400">System RAM:</span>
          <span id="ramStat" class="font-semibold text-sky-300">Probing memory...</span>
        </div>

        <div class="bg-slate-800 px-3.5 py-2 rounded-lg border border-slate-700 flex items-center gap-2">
          <span class="text-slate-400">Disk Storage:</span>
          <span id="storageStat" class="font-semibold text-amber-400">Checking disk...</span>
        </div>
      </div>
    </header>

    <!-- Navigation Tabs -->
    <div class="flex border-b border-slate-700 gap-2">
      <button onclick="switchTab('models')" id="tabBtnModels" class="px-5 py-2.5 text-sm font-semibold border-b-2 border-sky-400 text-sky-400 transition flex items-center gap-2">
        📦 Model Manager
      </button>
      <button onclick="switchTab('workspaces')" id="tabBtnWorkspaces" class="px-5 py-2.5 text-sm font-semibold border-b-2 border-transparent text-slate-400 hover:text-slate-200 transition flex items-center gap-2">
        🤖 GitHub Multi-Account & AI Studio
      </button>
    </div>

    <!-- ==================== TAB 1: MODEL MANAGER ==================== -->
    <div id="tabModels" class="space-y-6">
      <section class="bg-slate-800 p-6 rounded-lg shadow space-y-4">
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-2">
          <h2 id="catalogHeader" class="text-lg font-semibold">Available Models (Top GGUFs on Hugging Face)</h2>
          <div class="flex flex-wrap gap-1.5 text-xs items-center">
            <span class="text-slate-400 mr-1">Filter:</span>
            <button onclick="quickSearch('')" class="bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded text-slate-300">🔥 All</button>
            <button onclick="searchAuthor('bartowski')" class="bg-sky-950 hover:bg-sky-900 border border-sky-800 text-sky-300 px-2 py-1 rounded">🛡️ bartowski</button>
            <button onclick="searchAuthor('unsloth')" class="bg-sky-950 hover:bg-sky-900 border border-sky-800 text-sky-300 px-2 py-1 rounded">🛡️ unsloth</button>
            <button onclick="searchAuthor('TheBloke')" class="bg-sky-950 hover:bg-sky-900 border border-sky-800 text-sky-300 px-2 py-1 rounded">🛡️ TheBloke</button>
            <button onclick="searchAuthor('Qwen')" class="bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded text-slate-300">Qwen</button>
            <button onclick="quickSearch('Llama-3')" class="bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded text-slate-300">Llama 3</button>
          </div>
        </div>
        
        <div class="flex flex-col md:flex-row gap-3 items-stretch md:items-center">
          <input id="searchInput" type="text" placeholder="Search models or creators..." 
                 onkeydown="if(event.key === 'Enter') searchModels()"
                 class="flex-1 bg-slate-950 border border-slate-700 rounded px-4 py-2 focus:outline-none focus:border-sky-500 text-sm">
          
          <div class="flex flex-wrap items-center gap-3">
            <label class="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer select-none bg-slate-950 px-3 py-2 rounded border border-slate-700">
              <input type="checkbox" id="verifiedOnly" onchange="searchModels()" class="rounded bg-slate-900 border-slate-700 text-sky-600 focus:ring-0">
              <span>🛡️ Verified Creators Only</span>
            </label>

            <div class="flex items-center gap-2">
              <label for="sortSelect" class="text-xs text-slate-400 shrink-0">Sort:</label>
              <select id="sortSelect" onchange="searchModels()" class="bg-slate-950 border border-slate-700 rounded px-3 py-2 text-xs text-slate-200">
                <option value="downloads">Most Downloads</option>
                <option value="trust">⭐ Trust Score</option>
                <option value="likes">Most Likes</option>
                <option value="lastModified">Recent Release</option>
                <option value="alphabetical">Alphabetical (A-Z)</option>
              </select>
            </div>

            <button onclick="searchModels()" class="bg-sky-600 hover:bg-sky-500 px-6 py-2 rounded font-medium text-sm transition">Search</button>
            <button onclick="quickSearch('')" class="bg-slate-700 hover:bg-slate-600 px-3 py-2 rounded font-medium text-sm text-slate-300 transition">Reset</button>
          </div>
        </div>
        
        <div id="searchResults" class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <p class="text-slate-400">Loading models...</p>
        </div>
      </section>

      <!-- Active Tasks & Local Models -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section class="bg-slate-800 p-6 rounded-lg">
          <h2 class="text-lg font-semibold mb-4">Download Tasks</h2>
          <div id="tasksList" class="space-y-3 text-sm text-slate-300">No active downloads</div>
        </section>

        <section class="bg-slate-800 p-6 rounded-lg space-y-4">
          <div class="flex justify-between items-center">
            <div>
              <h2 class="text-lg font-semibold">Installed Models on Server</h2>
              <span id="diskSubStat" class="text-xs text-slate-400">-- / -- GB</span>
            </div>
            <button onclick="unloadActiveModel(this)" class="bg-slate-700 hover:bg-slate-600 border border-slate-600 text-xs px-3 py-1.5 rounded transition">
              ⏹ Unload All
            </button>
          </div>
          <div id="localList" class="space-y-2 text-sm text-slate-300">Scanning...</div>
        </section>
      </div>
    </div>

    <!-- ==================== TAB 2: GITHUB & AGENT STUDIO ==================== -->
    <div id="tabWorkspaces" class="hidden space-y-6">
      
      <!-- Section 1: Instructions & PAT Setup Guide -->
      <section class="bg-slate-800 p-5 rounded-lg border border-slate-700 space-y-3">
        <div class="flex justify-between items-center cursor-pointer" onclick="togglePatGuide()">
          <div class="flex items-center gap-2">
            <span class="text-lg">📖</span>
            <h2 class="text-sm font-bold text-sky-300 uppercase tracking-wide">How to Generate a GitHub Personal Access Token (PAT)</h2>
          </div>
          <button id="toggleGuideBtn" class="text-xs text-slate-400 hover:text-slate-200">Show Instructions ▼</button>
        </div>
        
        <div id="patGuideContent" class="hidden text-xs text-slate-300 space-y-2 border-t border-slate-700 pt-3">
          <ol class="list-decimal list-inside space-y-1.5 text-slate-300">
            <li>Log into the GitHub account you wish to connect.</li>
            <li>Click your <strong>Profile Photo (top-right)</strong> ➔ <strong>Settings</strong>.</li>
            <li>In the left sidebar, scroll to the bottom and click <strong>Developer settings</strong>.</li>
            <li>Click <strong>Personal access tokens</strong> ➔ <strong>Tokens (classic)</strong>.</li>
            <li>Click <strong>Generate new token</strong> ➔ <strong>Generate new token (classic)</strong>.</li>
            <li>Set a descriptive Note (e.g. <code class="bg-slate-900 px-1 py-0.5 rounded text-sky-300">LM Studio Code Studio</code>) and select Expiration.</li>
            <li>Check the <strong><code class="text-emerald-400 font-semibold">repo</code></strong> scope (Full control of private repositories).</li>
            <li>Click <strong>Generate token</strong> at the bottom and copy the token (<code class="text-amber-300">ghp_...</code>) below.</li>
          </ol>
        </div>
      </section>

      <!-- Section 2: Connected Accounts Management -->
      <section class="bg-slate-800 p-6 rounded-lg shadow space-y-4">
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-2 border-b border-slate-700 pb-3">
          <div>
            <h2 class="text-base font-semibold text-slate-100">Registered GitHub Accounts</h2>
            <p class="text-xs text-slate-400">Add or manage multiple personal or organizational accounts.</p>
          </div>
          <button onclick="toggleAddAccountForm()" class="bg-sky-600 hover:bg-sky-500 text-xs px-3.5 py-1.5 rounded font-medium transition">
            + Register New Account
          </button>
        </div>

        <!-- Add Account Collapsible Form -->
        <div id="addAccountForm" class="hidden bg-slate-950 p-4 rounded border border-slate-700 space-y-3">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label class="block text-xs text-slate-400 mb-1">Account Label / Alias:</label>
              <input id="newAccountLabel" type="text" placeholder="e.g. Personal or Work / Client" 
                     class="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-xs focus:border-sky-500 focus:outline-none">
            </div>
            <div>
              <label class="block text-xs text-slate-400 mb-1">GitHub Personal Access Token (PAT):</label>
              <input id="newAccountToken" type="password" placeholder="ghp_xxxxxxxxxxxxxxxxxxxx" 
                     class="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-xs focus:border-sky-500 focus:outline-none font-mono">
            </div>
          </div>
          <div class="flex justify-end gap-2">
            <button onclick="toggleAddAccountForm()" class="bg-slate-800 hover:bg-slate-700 text-xs px-3 py-1.5 rounded">Cancel</button>
            <button onclick="submitAddAccount()" class="bg-emerald-600 hover:bg-emerald-500 text-xs px-4 py-1.5 rounded font-medium">Verify & Save Account</button>
          </div>
        </div>

        <!-- Registered Accounts Grid -->
        <div id="accountsList" class="grid grid-cols-1 md:grid-cols-3 gap-3">
          <p class="text-xs text-slate-500">Loading accounts...</p>
        </div>
      </section>

      <!-- Section 3: Project & Workspace Selector -->
      <section class="bg-slate-800 p-6 rounded-lg shadow space-y-6">
        <div class="border-b border-slate-700 pb-3">
          <h2 class="text-base font-semibold text-slate-100">Project & Workspace Selector</h2>
          <p class="text-xs text-slate-400">Select an account, then open an active local project or clone a new repository.</p>
        </div>

        <!-- Step 1: Select Active Account -->
        <div>
          <label class="block text-xs font-semibold text-sky-400 uppercase tracking-wide mb-1">1. Select GitHub Account:</label>
          <select id="accountDropdown" onchange="onAccountSelected()" class="w-full md:w-1/2 bg-slate-950 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200">
            <option value="">-- Select an account --</option>
          </select>
        </div>

        <!-- Step 2: Choose Project (Active or New) -->
        <div id="projectSelectionBlock" class="hidden grid grid-cols-1 lg:grid-cols-2 gap-6 pt-2">
          
          <!-- Option A: Active Local Edits -->
          <div class="bg-slate-950 p-4 rounded-lg border border-slate-700 space-y-3">
            <div class="flex justify-between items-center">
              <h3 class="text-sm font-semibold text-slate-200">⚡ Active Local Projects</h3>
              <span class="text-[11px] text-slate-400">Already on disk</span>
            </div>
            <div id="activeProjectsList" class="space-y-2 max-h-56 overflow-y-auto pr-1 text-xs">
              <p class="text-slate-500">Scanning local workspaces...</p>
            </div>
          </div>

          <!-- Option B: Clone New / Switch Repository -->
          <div class="bg-slate-950 p-4 rounded-lg border border-slate-700 space-y-3">
            <h3 class="text-sm font-semibold text-slate-200">📥 Clone from Account</h3>
            
            <div class="space-y-2 text-xs">
              <div>
                <label class="block text-slate-400 mb-1">Select Remote Repository:</label>
                <select id="repoSelect" onchange="fetchBranchesForSelectedRepo()" class="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-xs text-slate-200">
                  <option value="">Loading repositories...</option>
                </select>
              </div>

              <div>
                <label class="block text-slate-400 mb-1">Branch:</label>
                <select id="branchSelect" class="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-xs text-slate-200">
                  <option value="main">main</option>
                  <option value="dev">dev</option>
                </select>
              </div>

              <button onclick="cloneOrOpenWorkspace()" class="w-full bg-sky-600 hover:bg-sky-500 py-2 rounded text-xs font-semibold transition mt-2">
                📂 Open / Clone Project Workspace
              </button>
            </div>
          </div>

        </div>
      </section>

      <!-- Section 4: Live AI Code Studio Panel -->
      <section id="agentStudioPanel" class="bg-slate-800 p-6 rounded-lg shadow space-y-6 hidden">
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-2 border-b border-slate-700 pb-4">
          <div>
            <h2 class="text-lg font-semibold text-emerald-400 flex items-center gap-2">
              <span>🤖 Active Workspace:</span>
              <span id="activeProjectLabel" class="text-slate-100 font-mono">None</span>
            </h2>
            <p class="text-xs text-slate-400">Account: <strong id="activeAccountLabel" class="text-amber-300 font-mono">--</strong> • Branch: <strong id="activeBranchLabel" class="text-sky-300 font-mono">main</strong></p>
          </div>
          
          <div class="flex items-center gap-2">
            <button onclick="runSyntaxValidation()" class="bg-slate-700 hover:bg-slate-600 border border-slate-600 text-xs px-3.5 py-2 rounded font-medium transition">
              🧪 Validate Syntax
            </button>
            <button onclick="pushChangesToGitHub()" class="bg-emerald-600 hover:bg-emerald-500 text-xs px-4 py-2 rounded font-medium transition flex items-center gap-1">
              🚀 Push to GitHub
            </button>
          </div>
        </div>

        <!-- Task Configuration -->
        <div class="space-y-4">
          <div>
            <label class="block text-xs text-slate-400 mb-1">Target File to Edit:</label>
            <select id="targetFileSelect" class="w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 font-mono">
              <option value="">-- Select File --</option>
            </select>
          </div>

          <div>
            <label class="block text-xs text-slate-400 mb-1">Instruction for Local AI Agent:</label>
            <textarea id="agentInstructionInput" rows="4" 
                      placeholder="e.g. Add an endpoint to fetch active branches and handle errors gracefully with try/except..."
                      class="w-full bg-slate-950 border border-slate-700 rounded p-3 text-sm focus:border-sky-500 focus:outline-none font-mono"></textarea>
          </div>

          <div class="flex justify-between items-center">
            <span class="text-xs text-slate-400">Connected Model: <strong id="agentModelBadge" class="text-sky-300 font-mono">Scanning...</strong></span>
            <button id="executeAgentBtn" onclick="executeAgentPlan()" class="bg-sky-600 hover:bg-sky-500 px-6 py-2.5 rounded font-semibold text-sm transition flex items-center gap-2">
              ⚡ Execute AI Code Plan
            </button>
          </div>
        </div>

        <!-- Output & Diff Terminal Window -->
        <div class="space-y-2">
          <label class="block text-xs text-slate-400">Live Agent Execution & Git Commit Output:</label>
          <pre id="agentConsoleOutput" class="bg-slate-950 p-4 rounded-lg border border-slate-700 text-xs text-emerald-400 font-mono overflow-x-auto max-h-96 whitespace-pre-wrap">Ready for instructions.</pre>
        </div>
      </section>

    </div>
  </div>

  <script>
    let localModelSet = new Set();
    let activeTasksMap = {};
    let loadedModelsList = [];
    let currentAccountUser = "";
    let currentWorkspaceDir = "";
    let currentBranch = "main";

    function switchTab(tab) {
      if (tab === 'models') {
        document.getElementById('tabModels').classList.remove('hidden');
        document.getElementById('tabWorkspaces').classList.add('hidden');
        document.getElementById('tabBtnModels').className = 'px-5 py-2.5 text-sm font-semibold border-b-2 border-sky-400 text-sky-400 transition flex items-center gap-2';
        document.getElementById('tabBtnWorkspaces').className = 'px-5 py-2.5 text-sm font-semibold border-b-2 border-transparent text-slate-400 hover:text-slate-200 transition flex items-center gap-2';
      } else {
        document.getElementById('tabModels').classList.add('hidden');
        document.getElementById('tabWorkspaces').classList.remove('hidden');
        document.getElementById('tabBtnModels').className = 'px-5 py-2.5 text-sm font-semibold border-b-2 border-transparent text-slate-400 hover:text-slate-200 transition flex items-center gap-2';
        document.getElementById('tabBtnWorkspaces').className = 'px-5 py-2.5 text-sm font-semibold border-b-2 border-sky-400 text-sky-400 transition flex items-center gap-2';
        loadAccounts();
        loadActiveWorkspaces();
      }
    }

    function togglePatGuide() {
      const el = document.getElementById('patGuideContent');
      const btn = document.getElementById('toggleGuideBtn');
      if (el.classList.contains('hidden')) {
        el.classList.remove('hidden');
        btn.textContent = 'Hide Instructions ▲';
      } else {
        el.classList.add('hidden');
        btn.textContent = 'Show Instructions ▼';
      }
    }

    function toggleAddAccountForm() {
      const el = document.getElementById('addAccountForm');
      el.classList.toggle('hidden');
    }

    async function initHardwareInfo() {
      try {
        const res = await fetch('/api/system_info');
        const data = await res.json();
        loadedModelsList = (data.loaded_models || []).map(x => x.toLowerCase());
        
        if (data.gpu && data.gpu.has_gpu) {
          const freeStr = data.gpu.free_vram_gb > 0 ? `${data.gpu.free_vram_gb} GB Free / ` : '';
          document.getElementById('vramStat').innerHTML = 
            `${freeStr}${data.gpu.total_vram_gb} GB <span class="text-slate-400 font-normal">(${data.gpu.gpu_name})</span>`;
        } else {
          document.getElementById('vramStat').innerHTML = `<span class="text-slate-400 font-normal">No Dedicated GPU Detected</span>`;
        }

        if (data.system_ram) {
          document.getElementById('ramStat').innerHTML = `${data.system_ram.available_gb} GB Avail / ${data.system_ram.total_gb} GB Total`;
        }

        if (data.storage) renderStorageMetrics(data.storage);

        const agentBadge = document.getElementById('agentModelBadge');
        if (agentBadge) {
          agentBadge.textContent = loadedModelsList.length > 0 ? loadedModelsList[0] : "Default (Auto-Load)";
        }
      } catch(e) {
        document.getElementById('vramStat').textContent = 'Error probing VRAM';
      }
    }

    function renderStorageMetrics(storage) {
      document.getElementById('storageStat').innerHTML = `${storage.used_gb} GB Used / ${storage.total_gb} GB (${storage.free_gb} GB Free)`;
      document.getElementById('diskSubStat').innerHTML = `Storage: <span class="text-amber-300">${storage.used_gb} GB</span> / ${storage.total_gb} GB (${storage.percent_used}%)`;
    }

    function quickSearch(tag) {
      document.getElementById('searchInput').value = tag;
      searchModels();
    }

    function searchAuthor(author) {
      document.getElementById('searchInput').value = author;
      document.getElementById('verifiedOnly').checked = false;
      searchModels();
    }

    async function searchModels() {
      const q = document.getElementById('searchInput').value.trim();
      const sortBy = document.getElementById('sortSelect').value;
      const verifiedOnly = document.getElementById('verifiedOnly').checked;
      const container = document.getElementById('searchResults');
      const header = document.getElementById('catalogHeader');
      const sortLabel = document.getElementById('sortSelect').selectedOptions[0].text;
      
      header.textContent = q === "" ? `All Available Models (${sortLabel})` : `Search Results for "${q}" (${sortLabel})`;
      container.innerHTML = '<p class="text-slate-400">Loading catalog from Hugging Face...</p>';
      
      try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(q)}&sort_by=${encodeURIComponent(sortBy)}&verified_only=${verifiedOnly}`);
        let models = await res.json();
        container.innerHTML = '';
        
        if (!models || models.length === 0) {
          container.innerHTML = '<p class="text-slate-400">No GGUF models found.</p>';
          return;
        }

        if (sortBy === 'alphabetical') {
          models.sort((a, b) => (a.model_name || '').localeCompare(b.model_name || '', undefined, { sensitivity: 'base' }));
        }

        models.forEach(m => {
          const card = document.createElement('div');
          card.className = 'bg-slate-950 p-4 rounded border border-slate-700 space-y-3';
          
          const verifiedBadge = m.is_verified 
            ? `<button onclick="searchAuthor('${m.maker}')" class="text-[10px] font-semibold px-2 py-0.5 rounded bg-emerald-950 hover:bg-emerald-900 text-emerald-300 border border-emerald-800 transition cursor-pointer">🛡️ ${m.maker}</button>`
            : `<button onclick="searchAuthor('${m.maker}')" class="text-[10px] font-semibold px-2 py-0.5 rounded bg-sky-950 hover:bg-sky-900 text-sky-300 border border-sky-800 transition cursor-pointer">${m.maker}</button>`;

          const trustBadge = `<span class="text-[10px] bg-slate-800 text-amber-300 border border-slate-700 px-1.5 py-0.5 rounded font-mono">⭐ ${m.trust_score}/100</span>`;

          card.innerHTML = `
            <div class="flex justify-between items-start">
              <div class="overflow-hidden pr-2">
                <div class="flex items-center gap-1.5">
                  ${verifiedBadge}
                  ${trustBadge}
                </div>
                <h3 class="font-bold text-slate-100 text-base mt-1.5 truncate" title="${m.model_name}">${m.model_name}</h3>
                <div class="text-[11px] text-slate-400 mt-0.5">
                  Author: <button onclick="searchAuthor('${m.maker}')" class="font-semibold text-sky-400 hover:text-sky-300 hover:underline cursor-pointer">${m.maker}</button> • Updated: ${m.lastModified || 'Recent'}
                </div>
              </div>
              <div class="text-right text-xs text-slate-400 shrink-0">
                <div>⬇ ${(m.downloads || 0).toLocaleString()}</div>
                <div>❤ ${(m.likes || 0).toLocaleString()}</div>
              </div>
            </div>
            <button onclick="toggleFiles('${m.id}', this)" class="toggle-btn w-full text-xs bg-slate-800 hover:bg-slate-700 border border-slate-600 px-3 py-1.5 rounded transition">
                  Inspect Quantizations & Memory Fit
            </button>
            <div class="file-container mt-3 hidden space-y-2"></div>
          `;
          container.appendChild(card);
        });
      } catch(err) {
        container.innerHTML = '<p class="text-rose-400">Error retrieving models from Hugging Face.</p>';
      }
    }

    async function toggleFiles(repoId, btn) {
      const parent = btn.parentElement;
      const fileContainer = parent.querySelector('.file-container');
      
      if (!fileContainer.classList.contains('hidden')) {
        fileContainer.classList.add('hidden');
        btn.textContent = 'Inspect Quantizations & Memory Fit';
        btn.className = 'toggle-btn w-full text-xs bg-slate-800 hover:bg-slate-700 border border-slate-600 px-3 py-1.5 rounded transition';
        return;
      }

      fileContainer.classList.remove('hidden');
      btn.textContent = '▲ Minimize Quantizations';
      btn.className = 'toggle-btn w-full text-xs bg-slate-700 hover:bg-slate-600 border border-slate-500 text-sky-300 px-3 py-1.5 rounded transition';
      fileContainer.innerHTML = '<span class="text-xs text-slate-500">Fetching file metadata & calculating VRAM footprint...</span>';
      
      const res = await fetch(`/api/model_files?repo_id=${encodeURIComponent(repoId)}`);
      const data = await res.json();
      fileContainer.innerHTML = '';

      if (!data.files || data.files.length === 0) {
        fileContainer.innerHTML = '<span class="text-xs text-slate-500">No .gguf files found in repository.</span>';
        return;
      }

      const table = document.createElement('div');
      table.className = 'space-y-2';

      data.files.forEach(f => {
        let fitBadge = '';
        if (f.fit_status === 'fits_gpu') {
          fitBadge = '<span class="text-[10px] bg-emerald-950 text-emerald-300 border border-emerald-800 px-1.5 py-0.5 rounded font-medium">Fits GPU VRAM</span>';
        } else if (f.fit_status === 'split_gpu_ram') {
          fitBadge = '<span class="text-[10px] bg-amber-950 text-amber-300 border border-amber-800 px-1.5 py-0.5 rounded font-medium">Split GPU + RAM</span>';
        } else if (f.fit_status === 'fits_ram') {
          fitBadge = '<span class="text-[10px] bg-sky-950 text-sky-300 border border-sky-800 px-1.5 py-0.5 rounded font-medium">Fits System RAM</span>';
        } else if (f.fit_status === 'exceeds') {
          fitBadge = '<span class="text-[10px] bg-rose-950 text-rose-300 border border-rose-800 px-1.5 py-0.5 rounded font-medium">Exceeds Memory</span>';
        }

        const isDownloaded = f.is_downloaded || localModelSet.has(f.group_name);
        const isDownloading = f.is_downloading || (activeTasksMap[f.group_name] && activeTasksMap[f.group_name].status === 'downloading');

        let btnHtml = '';
        const btnId = `btn-${btoa(f.group_name).replace(/=/g, '')}`;
        const filesPayload = encodeURIComponent(JSON.stringify(f.paths));

        if (isDownloaded) {
          btnHtml = `
            <button id="${btnId}" disabled class="bg-slate-800 text-slate-400 border border-slate-700 font-medium px-3 py-1.5 rounded text-xs shrink-0 cursor-not-allowed flex items-center gap-1">
              ✓ Installed
            </button>`;
        } else if (isDownloading) {
          btnHtml = `
            <button id="${btnId}" disabled class="bg-sky-950 text-sky-300 border border-sky-800 font-medium px-3 py-1.5 rounded text-xs shrink-0 cursor-not-allowed animate-pulse flex items-center gap-1">
              ⏳ Downloading...
            </button>`;
        } else {
          btnHtml = `
            <button id="${btnId}" onclick="triggerDownload('${repoId}', '${f.group_name}', '${filesPayload}', this)" class="bg-emerald-600 hover:bg-emerald-500 text-white font-medium px-3 py-1.5 rounded text-xs shrink-0 transition">
              Download
            </button>`;
        }

        const shardBadge = f.is_sharded ? `<span class="text-[10px] bg-sky-950 text-sky-300 border border-sky-800 px-1.5 py-0.5 rounded ml-1 font-mono">${f.shard_count} Shards</span>` : '';

        const row = document.createElement('div');
        row.className = 'flex flex-col md:flex-row justify-between items-start md:items-center bg-slate-900 p-2.5 rounded border border-slate-800 text-xs gap-2';
        row.innerHTML = `
          <div class="space-y-1 overflow-hidden pr-2">
            <div class="font-medium text-sky-200 truncate flex items-center" title="${f.display_name}">
              <span class="truncate">${f.group_name}</span>
              ${shardBadge}
            </div>
            <div class="flex flex-wrap items-center gap-2 text-slate-400 text-[11px]">
              <span>Weight: <strong class="text-slate-200">${f.weight}</strong></span>
              <span>•</span>
              <span>Variant: <strong class="text-slate-200">${f.variant}</strong></span>
              <span>•</span>
              <span>Size: <strong class="text-emerald-400">${f.size_gb}</strong></span>
              <span>(${f.est_vram} Req)</span>
              ${fitBadge}
            </div>
            <div class="text-[11px] text-slate-400 italic bg-slate-950/60 px-2 py-0.5 rounded border border-slate-800/80">
              ℹ ${f.description}
            </div>
          </div>
          ${btnHtml}
        `;
        table.appendChild(row);
      });
      fileContainer.appendChild(table);
    }

    async function triggerDownload(repoId, groupName, filesPayloadEncoded, btn) {
      btn.disabled = true;
      btn.className = "bg-sky-950 text-sky-300 border border-sky-800 font-medium px-3 py-1 rounded text-xs shrink-0 cursor-not-allowed animate-pulse flex items-center gap-1";
      btn.innerHTML = "⏳ Downloading...";

      const filePaths = JSON.parse(decodeURIComponent(filesPayloadEncoded));
      await fetch('/api/download', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({repo_id: repoId, group_name: groupName, files: filePaths})
      });
      updateTasks();
    }

    async function loadModelIntoGPU(modelPath, btn) {
      btn.disabled = true;
      btn.textContent = '⏳ Loading (32k ctx)...';
      btn.className = 'bg-sky-950 text-sky-300 border border-sky-800 px-2.5 py-1 rounded text-xs animate-pulse';

      try {
        const res = await fetch('/api/load_model', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({model_path: modelPath, gpu_offload: 'max', context_length: 32768, ttl: 3600})
        });
        const data = await res.json();
        if (!res.ok || data.status !== 'success') {
          alert('Load Failure Output from LMS:\\n\\n' + (data.message || JSON.stringify(data)));
        }
      } catch(e) {
        alert('Communication Error: ' + e.message);
      }
      await initHardwareInfo();
      await fetchLocalModels();
    }

    async function unloadActiveModel(btn) {
      if (btn) btn.textContent = '⏳ Unloading...';
      try {
        await fetch('/api/unload_model', {method: 'POST'});
      } catch(e) {}
      if (btn) btn.textContent = '⏹ Unload All';
      await initHardwareInfo();
      await fetchLocalModels();
    }

    async function deleteModel(filename) {
      if (!confirm(`Are you sure you want to delete ${filename} to free up space?`)) return;
      const res = await fetch('/api/delete', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filename: filename})
      });
      const data = await res.json();
      localModelSet.delete(filename);
      if (data.storage) renderStorageMetrics(data.storage);
      fetchLocalModels();
    }

    async function updateTasks() {
      const res = await fetch('/api/tasks');
      const data = await res.json();
      activeTasksMap = data;
      const list = document.getElementById('tasksList');
      
      if (Object.keys(data).length === 0) {
        list.innerHTML = '<span class="text-slate-500">No active downloads</span>';
        return;
      }

      list.innerHTML = Object.entries(data).map(([file, info]) => {
        const pct = info.percent || 0;
        const isDone = info.status === 'completed';
        const barColor = isDone ? 'bg-emerald-500' : 'bg-sky-500';

        return `
          <div class="bg-slate-950 p-3 rounded border border-slate-700 space-y-2">
            <div class="flex justify-between items-center text-xs">
              <div class="font-medium truncate pr-2">${file}</div>
              <div class="text-sky-300 font-mono shrink-0">${info.progress_str || info.status}</div>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
              <div class="${barColor} h-1.5 rounded-full transition-all duration-300" style="width: ${pct}%"></div>
            </div>
          </div>
        `;
      }).join('');
    }

    async function fetchLocalModels() {
      const res = await fetch('/api/local_models');
      const data = await res.json();
      const list = document.getElementById('localList');
      localModelSet.clear();
      loadedModelsList = (data.loaded_models || []).map(x => x.toLowerCase());

      if (data.storage) renderStorageMetrics(data.storage);

      if (!data.files || data.files.length === 0) {
        list.innerHTML = '<span class="text-slate-500">No GGUF models on disk</span>';
        return;
      }

      data.files.forEach(m => localModelSet.add(m.filename));

      list.innerHTML = data.files.map(m => {
        const fLower = m.filename.toLowerCase().replace('.gguf', '');
        const pLower = m.path.toLowerCase();
        
        const isLoaded = loadedModelsList.some(loaded => {
          if (!loaded || loaded.length < 3) return false;
          const cleanSlug = loaded.replace(/[^a-z0-9]/g, '');
          const cleanF = fLower.replace(/[^a-z0-9]/g, '');
          return cleanF.includes(cleanSlug) || cleanSlug.includes(cleanF) || pLower.includes(loaded);
        });
        
        let actionBtn = '';
        if (isLoaded) {
          actionBtn = `
            <div class="flex items-center gap-1.5">
              <span class="bg-emerald-950 text-emerald-300 border border-emerald-800 px-2 py-1 rounded text-xs flex items-center gap-1 font-semibold">
                ⚡ Loaded
              </span>
              <button onclick="unloadActiveModel(this)" class="bg-slate-800 hover:bg-slate-700 border border-slate-600 text-slate-300 px-2 py-1 rounded text-xs transition" title="Unload from VRAM">
                ⏹
              </button>
            </div>`;
        } else {
          actionBtn = `
            <button onclick="loadModelIntoGPU('${m.path}', this)" class="bg-sky-700 hover:bg-sky-600 text-white px-2.5 py-1 rounded text-xs transition flex items-center gap-1">
              🚀 Load to GPU
            </button>`;
        }

        return `
          <div class="flex justify-between items-center bg-slate-950 p-3 rounded border ${isLoaded ? 'border-emerald-700/80 bg-emerald-950/20' : 'border-slate-700'} text-xs gap-3">
            <div class="truncate pr-2">
              <div class="font-medium text-slate-200 truncate" title="${m.filename}">${m.filename}</div>
              <div class="text-slate-400 text-[10px] mt-0.5">Weight: ${m.weight} | Variant: ${m.variant}</div>
            </div>
            <div class="flex items-center gap-2 shrink-0">
              <span class="bg-slate-800 px-2 py-1 rounded text-slate-300 font-mono">${m.size_gb}</span>
              ${actionBtn}
              <button onclick="deleteModel('${m.filename}')" class="bg-rose-950 hover:bg-rose-900 border border-rose-800 text-rose-300 px-2 py-1 rounded text-xs transition" title="Delete from disk">
                🗑️
              </button>
            </div>
          </div>
        `;
      }).join('');
    }

    // ---------------- Multi-Account & Workspace Frontend Logic ----------------

    async function loadAccounts() {
      const container = document.getElementById('accountsList');
      const dropdown = document.getElementById('accountDropdown');
      
      try {
        const res = await fetch('/api/github/accounts');
        const accounts = await res.json();
        
        if (!accounts || accounts.length === 0) {
          container.innerHTML = '<p class="text-xs text-slate-500 col-span-3">No GitHub accounts registered yet. Click "+ Register New Account" above.</p>';
          dropdown.innerHTML = '<option value="">-- No accounts available --</option>';
          document.getElementById('projectSelectionBlock').classList.add('hidden');
          return;
        }

        container.innerHTML = accounts.map(a => `
          <div class="bg-slate-950 p-3 rounded border border-slate-700 flex justify-between items-center text-xs">
            <div class="flex items-center gap-2.5 overflow-hidden">
              <img src="${a.avatar_url || 'https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png'}" class="w-7 h-7 rounded-full border border-slate-700">
              <div class="truncate">
                <div class="font-bold text-slate-200 truncate">${a.label}</div>
                <div class="text-[11px] text-slate-400">@${a.username} • <span class="font-mono text-[10px] text-slate-500">${a.masked_token}</span></div>
              </div>
            </div>
            <button onclick="removeAccount('${a.username}')" class="text-rose-400 hover:text-rose-300 px-2 py-1 bg-slate-900 rounded border border-slate-800 text-[11px] shrink-0" title="Remove Account">
              ✕
            </button>
          </div>
        `).join('');

        dropdown.innerHTML = '<option value="">-- Choose Account --</option>' + accounts.map(a => `
          <option value="${a.username}">${a.label} (@${a.username})</option>
        `).join('');

        if (currentAccountUser) {
          dropdown.value = currentAccountUser;
        }
      } catch(e) {
        container.innerHTML = '<p class="text-xs text-rose-400">Error loading accounts.</p>';
      }
    }

    async function submitAddAccount() {
      const label = document.getElementById('newAccountLabel').value.trim();
      const token = document.getElementById('newAccountToken').value.trim();

      if (!token) return alert('Please enter a valid GitHub token');

      try {
        const res = await fetch('/api/github/accounts/add', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({token: token, label: label})
        });
        const data = await res.json();
        if (res.ok) {
          alert(`Account @${data.username} registered successfully!`);
          document.getElementById('newAccountLabel').value = '';
          document.getElementById('newAccountToken').value = '';
          toggleAddAccountForm();
          loadAccounts();
        } else {
          alert('Failed to add account: ' + data.message);
        }
      } catch(e) {
        alert('Communication Error: ' + e.message);
      }
    }

    async function removeAccount(username) {
      if (!confirm(`Are you sure you want to remove account @${username}?`)) return;
      try {
        const res = await fetch('/api/github/accounts/remove', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({username: username})
        });
        const data = await res.json();
        if (res.ok) {
          loadAccounts();
        } else {
          alert('Error: ' + data.message);
        }
      } catch(e) {
        alert('Communication Error: ' + e.message);
      }
    }

    async function onAccountSelected() {
      const username = document.getElementById('accountDropdown').value;
      currentAccountUser = username;
      const block = document.getElementById('projectSelectionBlock');

      if (!username) {
        block.classList.add('hidden');
        return;
      }

      block.classList.remove('hidden');
      await loadActiveWorkspaces();
      await loadRemoteReposForAccount(username);
    }

    async function loadActiveWorkspaces() {
      const container = document.getElementById('activeProjectsList');
      try {
        const res = await fetch('/api/workspaces/active');
        const workspaces = await res.json();
        
        if (!workspaces || workspaces.length === 0) {
          container.innerHTML = '<p class="text-slate-500">No active workspaces on disk yet.</p>';
          return;
        }

        container.innerHTML = workspaces.map(w => `
          <div class="flex justify-between items-center bg-slate-900 p-2.5 rounded border border-slate-800">
            <div class="truncate pr-2">
              <span class="font-bold text-sky-300">${w.display_name}</span>
              <span class="text-[10px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded font-mono ml-1.5">${w.branch}</span>
            </div>
            <button onclick="openExistingWorkspace('${w.dir_name}', '${w.display_name}', '${w.branch}')" class="bg-emerald-700 hover:bg-emerald-600 px-2.5 py-1 rounded text-xs text-white font-medium shrink-0">
              Open ➔
            </button>
          </div>
        `).join('');
      } catch(e) {
        container.innerHTML = '<p class="text-slate-500">Error loading local workspaces.</p>';
      }
    }

    async function loadRemoteReposForAccount(username) {
      const select = document.getElementById('repoSelect');
      select.innerHTML = '<option value="">Loading account repositories...</option>';
      try {
        const res = await fetch(`/api/github/repos?account_username=${encodeURIComponent(username)}`);
        const repos = await res.json();
        if (Array.isArray(repos) && repos.length > 0) {
          select.innerHTML = repos.map(r => `
            <option value="${r.full_name}">${r.is_private ? '🔒 ' : ''}${r.full_name}</option>
          `).join('');
          fetchBranchesForSelectedRepo();
        } else {
          select.innerHTML = '<option value="">No repositories found for account</option>';
        }
      } catch(e) {
        select.innerHTML = '<option value="">Error fetching repos</option>';
      }
    }

    async function fetchBranchesForSelectedRepo() {
      const repoFullName = document.getElementById('repoSelect').value;
      const branchSelect = document.getElementById('branchSelect');
      if (!repoFullName || !currentAccountUser) return;
      try {
        const res = await fetch(`/api/github/branches?account_username=${encodeURIComponent(currentAccountUser)}&repo_full_name=${encodeURIComponent(repoFullName)}`);
        const branches = await res.json();
        branchSelect.innerHTML = branches.map(b => `<option value="${b}">${b}</option>`).join('');
      } catch(e) {}
    }

    async function cloneOrOpenWorkspace() {
      const repoFullName = document.getElementById('repoSelect').value;
      const branch = document.getElementById('branchSelect').value;
      const consoleOut = document.getElementById('agentConsoleOutput');
      if (!repoFullName) return alert('Select a repository first');

      consoleOut.textContent = `[Git] Preparing workspace for ${repoFullName} on branch '${branch}'...\\n`;

      try {
        const res = await fetch('/api/github/clone', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({account_username: currentAccountUser, repo_full_name: repoFullName, branch: branch})
        });
        const data = await res.json();
        if (res.ok) {
          openExistingWorkspace(data.dir_name, repoFullName, data.branch);
          consoleOut.textContent += `[Git] Cloned / active at ${data.path}\\n`;
        } else {
          alert('Workspace error: ' + data.message);
        }
      } catch(e) {
        alert('Communication Error: ' + e.message);
      }
    }

    async function openExistingWorkspace(dirName, displayName, branch) {
      currentWorkspaceDir = dirName;
      currentBranch = branch;

      document.getElementById('activeProjectLabel').textContent = displayName;
      document.getElementById('activeAccountLabel').textContent = `@${currentAccountUser || 'local'}`;
      document.getElementById('activeBranchLabel').textContent = branch;
      document.getElementById('agentStudioPanel').classList.remove('hidden');

      await loadWorkspaceFiles();
    }

    async function loadWorkspaceFiles() {
      const select = document.getElementById('targetFileSelect');
      select.innerHTML = '<option value="">Loading files...</option>';
      try {
        const res = await fetch(`/api/workspace/files?repo_dir_name=${encodeURIComponent(currentWorkspaceDir)}`);
        const files = await res.json();
        if (Array.isArray(files) && files.length > 0) {
          select.innerHTML = files.map(f => `<option value="${f}">${f}</option>`).join('');
        } else {
          select.innerHTML = '<option value="">No editable files found</option>';
        }
      } catch(e) {}
    }

    async function executeAgentPlan() {
      const targetFile = document.getElementById('targetFileSelect').value;
      const instruction = document.getElementById('agentInstructionInput').value.trim();
      const consoleOut = document.getElementById('agentConsoleOutput');
      const btn = document.getElementById('executeAgentBtn');

      if (!targetFile) return alert('Please select a target file to edit');
      if (!instruction) return alert('Please enter an instruction for the AI agent');

      btn.disabled = true;
      btn.textContent = '⏳ AI Agent Generating & Committing...';
      consoleOut.textContent = `[Agent] Loading ${targetFile} into local LLM context (32k tokens)...\\n`;
      consoleOut.textContent += `[Agent] Task: "${instruction}"\\n`;

      try {
        const res = await fetch('/api/agent/execute', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            repo_dir_name: currentWorkspaceDir,
            target_file: targetFile,
            instruction: instruction
          })
        });
        const data = await res.json();
        if (res.ok) {
          consoleOut.textContent += `\\n[✓ Success] ${data.commit_message}\\n\\n`;
          consoleOut.textContent += `--- GIT DIFF ---\\n${data.diff}\\n`;
        } else {
          consoleOut.textContent += `\\n[❌ Error] ${data.message}`;
        }
      } catch(e) {
        consoleOut.textContent += `\\n[❌ Communication Error] ${e.message}`;
      }
      btn.disabled = false;
      btn.textContent = '⚡ Execute AI Code Plan';
    }

    async function runSyntaxValidation() {
      const consoleOut = document.getElementById('agentConsoleOutput');
      consoleOut.textContent += `\\n[Validator] Running Python compilation & Bash syntax checks...\\n`;
      try {
        const res = await fetch('/api/workspace/validate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({repo_dir_name: currentWorkspaceDir, branch: currentBranch})
        });
        const data = await res.json();
        consoleOut.textContent += `\\n${data.logs}\\n`;
      } catch(e) {
        consoleOut.textContent += `\\n[Validator Error] ${e.message}\\n`;
      }
    }

    async function pushChangesToGitHub() {
      if (!confirm(`Are you ready to push committed changes to remote branch '${currentBranch}' on GitHub?`)) return;
      const consoleOut = document.getElementById('agentConsoleOutput');
      consoleOut.textContent += `\\n[Git] Pushing commits to GitHub origin/${currentBranch}...\\n`;
      try {
        const res = await fetch('/api/workspace/push', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({repo_dir_name: currentWorkspaceDir, branch: currentBranch})
        });
        const data = await res.json();
        if (res.ok) {
          alert('Push Successful: ' + data.message);
          consoleOut.textContent += `[✓ Git Push Complete] ${data.message}\\n`;
        } else {
          alert('Push Failed: ' + data.message);
          consoleOut.textContent += `[❌ Git Push Failed] ${data.message}\\n`;
        }
      } catch(e) {
        consoleOut.textContent += `[❌ Error] ${e.message}\\n`;
      }
    }

    initHardwareInfo();
    fetchLocalModels().then(() => { searchModels(); });
    setInterval(updateTasks, 1000);
    setInterval(fetchLocalModels, 4000);
  </script>
</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)