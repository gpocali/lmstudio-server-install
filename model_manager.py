"""
LM Studio Remote Model & Multi-Account Code Studio Manager
Provides Model Management, GitHub PAT Account Vault, and a Full Left-Pane IDE Chat Studio.
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

app = FastAPI(title="LM Studio Code & Model Studio")

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

class CreateFileRequest(BaseModel):
    repo_dir_name: str
    file_path: str
    initial_content: str = ""

class AgentTaskRequest(BaseModel):
    repo_dir_name: str
    target_files: list[str] = []
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
        r = requests.get("https://api.github.com/user", headers={"Authorization": f"Bearer {token}"}, timeout=6)
        if r.status_code != 200:
            return JSONResponse(status_code=401, content={"status": "error", "message": "Invalid GitHub Token. Please verify permissions."})
        
        user_info = r.json()
        username = user_info.get("login", "")
        avatar = user_info.get("avatar_url", "")
        label = req.label.strip() or username

        data = load_accounts_data()
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

# ----------------- Workspaces & File Tree -----------------

@app.get("/api/workspaces/active")
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

@app.post("/api/github/clone")
def clone_or_open_project(req: GithubCloneRequest):
    token = get_token_for_user(req.account_username)
    if not token:
        return JSONResponse(status_code=401, content={"status": "error", "message": f"Authentication required for @{req.account_username}"})

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

@app.post("/api/workspace/create_file")
def create_workspace_file(req: CreateFileRequest):
    workspace_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(workspace_path):
        return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})

    target_file = os.path.join(workspace_path, req.file_path.strip().lstrip("/"))
    os.makedirs(os.path.dirname(target_file), exist_ok=True)

    try:
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(req.initial_content)
        
        subprocess.run(["git", "-C", workspace_path, "add", req.file_path], capture_output=True)
        subprocess.run(["git", "-C", workspace_path, "commit", "-m", f"Create new file: {req.file_path}"], capture_output=True)
        return {"status": "success", "file_path": req.file_path}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# ----------------- Multi-File AI Agent Execution -----------------

@app.post("/api/agent/execute")
def execute_agent_task(req: AgentTaskRequest):
    workspace_path = os.path.join(WORKSPACES_ROOT, req.repo_dir_name)
    if not os.path.exists(workspace_path):
        return JSONResponse(status_code=404, content={"status": "error", "message": "Workspace not found"})

    # Read selected context files
    context_blocks = []
    for rel_file in req.target_files:
        full_p = os.path.join(workspace_path, rel_file)
        if os.path.exists(full_p):
            try:
                with open(full_p, "r", encoding="utf-8") as f:
                    content = f.read()
                context_blocks.append(f"### File: {rel_file}\n```\n{content}\n```")
            except Exception: pass

    context_str = "\n\n".join(context_blocks) if context_blocks else "No existing files selected as context."

    system_prompt = (
        "You are an expert AI software architect and full-stack engineer.\n"
        "Your task is to analyze instructions and output complete updated or newly created source code files.\n\n"
        "OUTPUT FORMAT REQUIREMENTS:\n"
        "For EACH file you create or modify, format your response strictly as:\n"
        "### File: <relative_path_to_file>\n"
        "```\n"
        "<complete file contents without any truncation, placeholders, or omission comments>\n"
        "```\n\n"
        "You may output explanations and reasoning before or after file blocks."
    )

    user_prompt = (
        f"Active Workspace Context Files:\n\n{context_str}\n\n"
        f"Task Instruction:\n{req.instruction}\n\n"
        "Please provide the complete implementation for all required files:"
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
        
        resp = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=150)
        if resp.status_code != 200:
            return JSONResponse(status_code=500, content={"status": "error", "message": f"LM Studio API Error: {resp.text}"})

        ai_response = resp.json()["choices"][0]["message"]["content"]
        
        # Parse all file blocks: ### File: <path>\n```...\n```
        file_pattern = re.compile(r'###\s*File:\s*([^\n\r]+)[\r\n]+```(?:[a-zA-Z0-9_\-]+)?[\r\n]+([\s\S]*?)[\r\n]+```', re.MULTILINE)
        matches = file_pattern.findall(ai_response)
        
        modified_files = []
        if matches:
            for file_rel_path, file_content in matches:
                clean_rel = file_rel_path.strip().lstrip("/")
                dest_file_path = os.path.join(workspace_path, clean_rel)
                os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
                with open(dest_file_path, "w", encoding="utf-8") as f:
                    f.write(file_content)
                subprocess.run(["git", "-C", workspace_path, "add", clean_rel], capture_output=True)
                modified_files.append(clean_rel)
        elif len(req.target_files) == 1:
            # Fallback if model omitted ### File header on single file task
            code_match = re.search(r'```(?:[a-zA-Z0-9_\-]+)?\n([\s\S]*?)\n