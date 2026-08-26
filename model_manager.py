import os
import re
import glob
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

app = FastAPI(title="LM Studio Headless Model Manager")

DOWNLOAD_JOBS = {}
STORAGE_PATH = "/storage/lmstudio"
MODELS_PATH = os.path.join(STORAGE_PATH, "models")
LMS_ENV = {
    **os.environ,
    "HOME": STORAGE_PATH,
    "LMS_SERVER_HOST": "0.0.0.0",
    "PATH": f"/usr/local/bin:{STORAGE_PATH}/.cache/lm-studio/bin:{STORAGE_PATH}/.lmstudio/bin:/usr/sbin:/usr/bin:/bin"
}

VERIFIED_CREATORS = {
    "bartowski", "unsloth", "TheBloke", "MaziyarPanahi", "mradermacher",
    "QuantFactory", "meta-llama", "Qwen", "mistralai", "google",
    "deepseek-ai", "microsoft", "nomic-ai", "cohere", "NousResearch"
}

QUANT_DESCRIPTIONS = {
    "Q4_K_M": "Recommended standard. Medium 4-bit quantization with optimal balance between low memory and high quality.",
    "Q4_K_S": "Small 4-bit quantization. Uses slightly less memory than Q4_K_M with minor quality trade-off.",
    "Q5_K_M": "High quality 5-bit quantization. Near-original quality with modest memory footprint.",
    "Q5_K_S": "Compact 5-bit quantization. Higher precision than 4-bit with slightly lower memory than Q5_K_M.",
    "Q8_0": "Extremely high precision (8-bit). Virtually zero quality loss; requires high VRAM.",
    "Q6_K": "Very high quality 6-bit quantization. Perceptually indistinguishable from 16-bit float for most tasks.",
    "Q3_K_L": "Large 3-bit quantization. Lower memory footprint; noticeable quality reduction on small models.",
    "Q3_K_M": "Medium 3-bit quantization. Aggressive compression for fitting very large models into limited VRAM.",
    "Q3_K_S": "Small 3-bit quantization. Minimal memory usage; high perplexity loss.",
    "Q2_K": "2-bit quantization. Maximum compression; significant loss in reasoning and grammar.",
    "IQ4_XS": "Importance Matrix 4-bit extra small. Better quality than legacy Q4_0 with smaller file size.",
    "IQ4_NL": "Importance Matrix 4-bit non-linear. High quality preservation for modern architectures.",
    "IQ3_M": "Importance Matrix 3-bit medium. Outperforms traditional Q3_K quantizations in quality.",
    "IQ3_S": "Importance Matrix 3-bit small. Optimized for fitting into tight memory boundaries.",
    "IQ2_M": "Importance Matrix 2-bit medium. State-of-the-art 2-bit compression using importance matrix.",
    "IQ1_S": "Extreme 1-bit quantization. Fits massive architectures into minimum memory; severe degradation.",
    "FP16": "Full 16-bit unquantized float. Maximum quality and fidelity; highest memory and compute requirements.",
    "BF16": "Bfloat16 unquantized format. Native training precision with full dynamic range."
}

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

def get_lms_bin():
    for p in ["/usr/local/bin/lms", f"{STORAGE_PATH}/.lmstudio/bin/lms", f"{STORAGE_PATH}/.cache/lm-studio/bin/lms"]:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p
    return shutil.which("lms") or "lms"

def get_loaded_models():
    """Extracts identifiers of loaded models non-blockingly from `lms ps`."""
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
    if "Q4" in v_upper: return "4-bit quantization (Balanced performance and efficiency)."
    elif "Q5" in v_upper: return "5-bit quantization (High fidelity)."
    elif "Q8" in v_upper: return "8-bit quantization (Near-lossless precision)."
    return "Standard GGUF model quantization variant."

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
    sys_ram_total_gb = 0.0
    sys_ram_avail_gb = 0.0
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        total_m = re.search(r'MemTotal:\s+(\d+)\s+kB', meminfo)
        avail_m = re.search(r'MemAvailable:\s+(\d+)\s+kB', meminfo)
        if total_m: sys_ram_total_gb = round(int(total_m.group(1)) / (1024**2), 2)
        if avail_m: sys_ram_avail_gb = round(int(avail_m.group(1)) / (1024**2), 2)
    except Exception:
        pass

    gpu_found = False
    gpu_name = "NVIDIA Dedicated GPU"
    gpu_vram_total_gb = 0.0
    gpu_vram_free_gb = 0.0

    try:
        nvml_lib_names = ["libnvidia-ml.so.1", "libnvidia-ml.so", "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"]
        nvml = None
        for name in nvml_lib_names:
            try:
                nvml = ctypes.CDLL(name)
                break
            except Exception:
                continue

        if nvml:
            class nvmlMemory_t(ctypes.Structure):
                _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]

            if nvml.nvmlInit_v2() == 0 or nvml.nvmlInit() == 0:
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
    except Exception:
        pass
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
            try:
                os.symlink(dest_dir, link_target)
            except Exception:
                pass

        if first_shard_file:
            subprocess.run([get_lms_bin(), "import", "--yes", "--symbolic-link", first_shard_file], env=LMS_ENV, capture_output=True)

    except Exception as e:
        DOWNLOAD_JOBS[group_name] = {"status": "failed", "progress_str": f"Error: {str(e)}", "percent": 0.0}

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
    """Unloads active models and loads the target model into GPU VRAM."""
    lms = get_lms_bin()
    fname = os.path.basename(req.model_path)
    ident = req.identifier or fname.replace(".gguf", "")

    # 1. Unload running models first
    subprocess.run([lms, "unload", "--all"], env=LMS_ENV, capture_output=True)

    # 2. Try loading by absolute file path
    cmd = [
        lms, "load", req.model_path,
        f"--gpu={req.gpu_offload}",
        f"--context-length={req.context_length}",
        f"--ttl={req.ttl}",
        "--yes"
    ]
    res = subprocess.run(cmd, env=LMS_ENV, capture_output=True, text=True)

    # 3. Fallback to model filename if needed
    if res.returncode != 0:
        cmd_fallback = [
            lms, "load", fname,
            f"--gpu={req.gpu_offload}",
            f"--context-length={req.context_length}",
            f"--ttl={req.ttl}",
            "--yes"
        ]
        res_fallback = subprocess.run(cmd_fallback, env=LMS_ENV, capture_output=True, text=True)
        if res_fallback.returncode == 0:
            return {"status": "success", "output": res_fallback.stdout, "identifier": ident}

    if res.returncode == 0:
        return {"status": "success", "output": res.stdout, "identifier": ident}
    else:
        err_msg = (res.stderr or res.stdout or "Command returned non-zero exit status").strip()
        return JSONResponse(status_code=500, content={"status": "error", "message": err_msg})

@app.post("/api/unload_model")
def unload_model():
    """Unloads all running models from memory."""
    lms = get_lms_bin()
    try:
        res = subprocess.run([lms, "unload", "--all"], env=LMS_ENV, capture_output=True, text=True)
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

@app.get("/", response_class=HTMLResponse)
def get_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>LM Studio Remote Model Manager</title>
      <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-900 text-slate-100 min-h-screen p-6 md:p-8">
      <div class="max-w-7xl mx-auto space-y-8">
        
        <header class="flex flex-col lg:flex-row justify-between items-start lg:items-center border-b border-slate-700 pb-5 gap-4">
          <div>
            <h1 class="text-2xl font-bold text-sky-400">LM Studio Model Manager</h1>
            <p class="text-xs text-slate-400">Storage Target: /storage/lmstudio/models</p>
          </div>
          
          <div class="flex flex-wrap items-center gap-3 text-xs">
            <div id="gpuCard" class="bg-slate-800 px-3.5 py-2 rounded-lg border border-slate-700 flex items-center gap-2">
              <span class="text-slate-400">Dedicated VRAM:</span>
              <span id="vramStat" class="font-semibold text-emerald-400">Probing NVML...</span>
            </div>
            
            <div id="ramCard" class="bg-slate-800 px-3.5 py-2 rounded-lg border border-slate-700 flex items-center gap-2">
              <span class="text-slate-400">System RAM:</span>
              <span id="ramStat" class="font-semibold text-sky-300">Probing memory...</span>
            </div>

            <div id="storageCard" class="bg-slate-800 px-3.5 py-2 rounded-lg border border-slate-700 flex items-center gap-2">
              <span class="text-slate-400">Disk Storage:</span>
              <span id="storageStat" class="font-semibold text-amber-400">Checking disk...</span>
            </div>
            
            <button onclick="initHardwareInfo(); fetchLocalModels();" class="bg-slate-700 hover:bg-slate-600 px-3 py-2 rounded text-xs text-slate-200 transition">
              Refresh Stats
            </button>
          </div>
        </header>

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

        <!-- Active Jobs & Local Models Grid -->
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

      <script>
        let localModelSet = new Set();
        let activeTasksMap = {};
        let loadedModelsList = [];

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
              document.getElementById('vramStat').innerHTML = 
                `<span class="text-slate-400 font-normal">No Dedicated GPU Detected</span>`;
            }

            if (data.system_ram) {
              document.getElementById('ramStat').innerHTML = 
                `${data.system_ram.available_gb} GB Avail / ${data.system_ram.total_gb} GB Total`;
            }

            if (data.storage) {
              renderStorageMetrics(data.storage);
            }
          } catch(e) {
            document.getElementById('vramStat').textContent = 'Error probing VRAM';
          }
        }

        function renderStorageMetrics(storage) {
          document.getElementById('storageStat').innerHTML = 
            `${storage.used_gb} GB Used / ${storage.total_gb} GB (${storage.free_gb} GB Free)`;
          document.getElementById('diskSubStat').innerHTML = 
            `Storage: <span class="text-amber-300">${storage.used_gb} GB</span> / ${storage.total_gb} GB (${storage.percent_used}%)`;
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
            if (data.status !== 'success') {
              alert('Load Failure Output from LMS:\n\n' + (data.message || JSON.stringify(data)));
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
            
            // Compare normalized slugs
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
                    ⚡ Loaded (32k)
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

        initHardwareInfo();
        fetchLocalModels().then(() => { searchModels(); });
        setInterval(updateTasks, 1000);
        setInterval(fetchLocalModels, 4000);
      </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)