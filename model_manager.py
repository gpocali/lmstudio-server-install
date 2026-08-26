import os
import re
import glob
import ctypes
import shutil
import subprocess
import threading
import requests
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="LM Studio Headless Model Manager")

DOWNLOAD_JOBS = {}
STORAGE_PATH = "/storage/lmstudio"
MODELS_PATH = os.path.join(STORAGE_PATH, "models")

class DownloadRequest(BaseModel):
    repo_id: str
    filename: str

def get_system_hardware_info():
    """
    Detects System RAM and Dedicated GPU VRAM using:
    1. Direct NVML C-Library bindings (same method as nvtop).
    2. nvidia-smi CLI.
    3. sysfs / PCI bus inspection specifically filtering for dedicated GPUs.
    """
    # 1. System RAM Detection
    sys_ram_total_gb = 0.0
    sys_ram_avail_gb = 0.0
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        total_m = re.search(r'MemTotal:\s+(\d+)\s+kB', meminfo)
        avail_m = re.search(r'MemAvailable:\s+(\d+)\s+kB', meminfo)
        if total_m:
            sys_ram_total_gb = round(int(total_m.group(1)) / (1024**2), 2)
        if avail_m:
            sys_ram_avail_gb = round(int(avail_m.group(1)) / (1024**2), 2)
    except Exception:
        pass

    # 2. GPU Detection
    gpu_found = False
    gpu_name = "NVIDIA Dedicated GPU"
    gpu_vram_total_gb = 0.0
    gpu_vram_free_gb = 0.0

    # Method A: Direct NVML ctypes library query
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
                _fields_ = [
                    ('total', ctypes.c_ulonglong),
                    ('free', ctypes.c_ulonglong),
                    ('used', ctypes.c_ulonglong)
                ]

            if nvml.nvmlInit_v2() == 0 or nvml.nvmlInit() == 0:
                device_count = ctypes.c_uint()
                nvml.nvmlDeviceGetCount_v2(ctypes.byref(device_count))
                
                tot_bytes = 0
                free_bytes = 0
                names = []

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

                try:
                    nvml.nvmlShutdown()
                except Exception:
                    pass
    except Exception:
        gpu_found = False

    # Method B: Fallback to nvidia-smi CLI
    if not gpu_found:
        nvidia_smi_path = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"
        if os.path.exists(nvidia_smi_path):
            try:
                cmd = [nvidia_smi_path, "--query-gpu=name,memory.total,memory.free", "--format=csv,nounits,noheader"]
                output = subprocess.check_output(cmd, encoding='utf-8').strip()
                if output:
                    lines = output.split('\n')
                    total_mb = 0.0
                    free_mb = 0.0
                    names = []
                    for line in lines:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3:
                            names.append(parts[0])
                            total_mb += float(parts[1])
                            free_mb += float(parts[2])
                    if total_mb > 0:
                        gpu_found = True
                        gpu_name = ", ".join(list(dict.fromkeys(names)))
                        gpu_vram_total_gb = round(total_mb / 1024, 2)
                        gpu_vram_free_gb = round(free_mb / 1024, 2)
            except Exception:
                pass

    # Method C: Dedicated PCI Inspection (Prioritizing Vendor 0x10de)
    if not gpu_found:
        pci_devices = sorted(glob.glob('/sys/bus/pci/devices/*'))
        for dev in pci_devices:
            vendor_file = os.path.join(dev, "vendor")
            resource_file = os.path.join(dev, "resource")
            if os.path.exists(vendor_file) and os.path.exists(resource_file):
                try:
                    with open(vendor_file, 'r') as f:
                        vendor = f.read().strip().lower()
                    if "0x10de" in vendor:
                        with open(resource_file, 'r') as f:
                            res_lines = f.readlines()
                        max_bar_bytes = 0
                        for line in res_lines:
                            parts = line.strip().split()
                            if len(parts) >= 3:
                                start = int(parts[0], 16)
                                end = int(parts[1], 16)
                                if end > start:
                                    bar_size = end - start + 1
                                    if bar_size > max_bar_bytes and bar_size >= (1024**3):
                                        max_bar_bytes = bar_size
                        if max_bar_bytes > 0:
                            gpu_found = True
                            gpu_vram_total_gb = round(max_bar_bytes / (1024**3), 2)
                            gpu_vram_free_gb = gpu_vram_total_gb
                            gpu_name = "NVIDIA Dedicated GPU"
                            break
                except Exception:
                    pass

    return {
        "system_ram": {
            "total_gb": sys_ram_total_gb,
            "available_gb": sys_ram_avail_gb
        },
        "gpu": {
            "has_gpu": gpu_found,
            "gpu_name": gpu_name,
            "total_vram_gb": gpu_vram_total_gb,
            "free_vram_gb": gpu_vram_free_gb
        }
    }

def parse_model_metadata(filename: str, repo_id: str):
    """Parse Weight, Quantization, and Format details from repo and filename."""
    weight_match = re.search(r'(\d+(\.\d+)?(?:x\d+)?[bB])', f"{repo_id} {filename}")
    weight = weight_match.group(1).upper() if weight_match else "Unknown"

    quant_match = re.search(r'(IQ\d_[A-Z_]+|Q\d_[A-Z0-9_]+|FP16|BF16|F16|F32)', filename, re.IGNORECASE)
    variant = quant_match.group(1).upper() if quant_match else "Standard"

    return weight, variant

def run_download_job(repo_id: str, filename: str):
    DOWNLOAD_JOBS[filename] = {"status": "downloading", "progress": "In Progress"}
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    dest_dir = os.path.join(MODELS_PATH, repo_id.replace('/', '_'))
    os.makedirs(dest_dir, exist_ok=True)
    dest_file = os.path.join(dest_dir, filename)

    try:
        cmd = ["curl", "-L", "-C", "-", "-o", dest_file, url]
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode == 0:
            DOWNLOAD_JOBS[filename] = {"status": "completed", "progress": "100%"}
            subprocess.run(["/usr/local/bin/lms", "import", dest_file], capture_output=True)
        else:
            DOWNLOAD_JOBS[filename] = {"status": "failed", "progress": process.stderr}
    except Exception as e:
        DOWNLOAD_JOBS[filename] = {"status": "failed", "progress": str(e)}

@app.get("/api/system_info")
def get_sys_info():
    return get_system_hardware_info()

@app.get("/api/search")
def search_hf(q: str = ""):
    # If query is empty, pull the top 30 most downloaded trending GGUF models
    if not q or q.strip() == "":
        url = "https://huggingface.co/api/models?filter=gguf&sort=downloads&direction=-1&limit=30"
    else:
        url = f"https://huggingface.co/api/models?search={q.strip()}&filter=gguf&sort=downloads&direction=-1&limit=30"
        
    res = requests.get(url).json()
    results = []
    if isinstance(res, list):
        for m in res:
            repo_id = m.get("id", "")
            maker = repo_id.split('/')[0] if '/' in repo_id else "Community"
            model_name = repo_id.split('/')[1] if '/' in repo_id else repo_id
            results.append({
                "id": repo_id,
                "maker": maker,
                "model_name": model_name,
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "lastModified": m.get("lastModified", "")[:10]
            })
    return results

@app.get("/api/model_files")
def get_model_files(repo_id: str):
    url = f"https://huggingface.co/api/models/{repo_id}"
    res = requests.get(url).json()
    siblings = res.get("siblings", [])
    hw = get_system_hardware_info()
    vram_total = hw["gpu"]["total_vram_gb"]
    ram_total = hw["system_ram"]["total_gb"]

    parsed_files = []
    for s in siblings:
        fname = s.get("rfilename", "")
        if fname.endswith(".gguf"):
            weight, variant = parse_model_metadata(fname, repo_id)
            size_bytes = s.get("size", 0)
            size_gb = round(size_bytes / (1024**3), 2) if size_bytes else None

            est_mem_req = round(size_gb * 1.2, 2) if size_gb else None
            
            fit_status = "unknown"
            if est_mem_req:
                if vram_total > 0:
                    if est_mem_req <= vram_total:
                        fit_status = "fits_gpu"
                    elif est_mem_req <= (vram_total + ram_total * 0.75):
                        fit_status = "split_gpu_ram"
                    else:
                        fit_status = "exceeds"
                else:
                    if est_mem_req <= ram_total * 0.85:
                        fit_status = "fits_ram"
                    else:
                        fit_status = "exceeds"

            parsed_files.append({
                "filename": fname,
                "weight": weight,
                "variant": variant,
                "size_gb": f"{size_gb} GB" if size_gb else "Dynamic",
                "est_vram": f"~{est_mem_req} GB" if est_mem_req else "N/A",
                "fit_status": fit_status
            })

    return {
        "repo_id": repo_id,
        "hardware": hw,
        "files": parsed_files
    }

@app.post("/api/download")
def start_download(req: DownloadRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_download_job, req.repo_id, req.filename)
    return {"status": "started", "file": req.filename}

@app.get("/api/tasks")
def get_tasks():
    return DOWNLOAD_JOBS

@app.get("/api/local_models")
def get_local_models():
    files = []
    if os.path.exists(MODELS_PATH):
        for root, _, filenames in os.walk(MODELS_PATH):
            for f in filenames:
                if f.endswith(".gguf"):
                    path = os.path.join(root, f)
                    size_gb = round(os.path.getsize(path) / (1024**3), 2)
                    weight, variant = parse_model_metadata(f, root)
                    files.append({
                        "filename": f,
                        "weight": weight,
                        "variant": variant,
                        "size_gb": f"{size_gb} GB",
                        "path": path
                    })
    return files

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
        
        <!-- Header & Hardware Metrics -->
        <header class="flex flex-col lg:flex-row justify-between items-start lg:items-center border-b border-slate-700 pb-5 gap-4">
          <div>
            <h1 class="text-2xl font-bold text-sky-400">LM Studio Model Manager</h1>
            <p class="text-xs text-slate-400">Storage Target: /storage/lmstudio/models</p>
          </div>
          
          <div class="flex flex-wrap items-center gap-3 text-xs">
            <!-- Dedicated GPU VRAM Badge -->
            <div id="gpuCard" class="bg-slate-800 px-3.5 py-2 rounded-lg border border-slate-700 flex items-center gap-2">
              <span class="text-slate-400">Dedicated VRAM:</span>
              <span id="vramStat" class="font-semibold text-emerald-400">Probing NVML...</span>
            </div>
            
            <!-- System RAM Badge -->
            <div id="ramCard" class="bg-slate-800 px-3.5 py-2 rounded-lg border border-slate-700 flex items-center gap-2">
              <span class="text-slate-400">System RAM:</span>
              <span id="ramStat" class="font-semibold text-sky-300">Probing memory...</span>
            </div>
            
            <button onclick="initHardwareInfo(); fetchLocalModels();" class="bg-slate-700 hover:bg-slate-600 px-3 py-2 rounded text-xs text-slate-200 transition">
              Refresh Stats
            </button>
          </div>
        </header>

        <!-- Search & Catalog Section -->
        <section class="bg-slate-800 p-6 rounded-lg shadow space-y-4">
          <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-2">
            <h2 id="catalogHeader" class="text-lg font-semibold">Available Models (Top GGUFs on Hugging Face)</h2>
            <div class="flex flex-wrap gap-1.5 text-xs">
              <button onclick="quickSearch('')" class="bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded text-slate-300">🔥 All Top</button>
              <button onclick="quickSearch('Llama-3')" class="bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded text-slate-300">Llama 3</button>
              <button onclick="quickSearch('Qwen2.5')" class="bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded text-slate-300">Qwen 2.5</button>
              <button onclick="quickSearch('Mistral')" class="bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded text-slate-300">Mistral</button>
              <button onclick="quickSearch('DeepSeek')" class="bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded text-slate-300">DeepSeek</button>
              <button onclick="quickSearch('Coder')" class="bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded text-slate-300">Coding</button>
              <button onclick="quickSearch('bartowski')" class="bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded text-slate-300">bartowski</button>
            </div>
          </div>
          
          <div class="flex gap-3">
            <input id="searchInput" type="text" placeholder="Search by model or creator (e.g. Llama-3.1, bartowski, Qwen2.5, DeepSeek)..." 
                   onkeydown="if(event.key === 'Enter') searchModels()"
                   class="flex-1 bg-slate-950 border border-slate-700 rounded px-4 py-2 focus:outline-none focus:border-sky-500 text-sm">
            <button onclick="searchModels()" class="bg-sky-600 hover:bg-sky-500 px-6 py-2 rounded font-medium text-sm transition">Search</button>
            <button onclick="quickSearch('')" class="bg-slate-700 hover:bg-slate-600 px-3 py-2 rounded font-medium text-sm text-slate-300 transition" title="Reset to Trending">Reset</button>
          </div>
          
          <div id="searchResults" class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <p class="text-slate-400">Loading popular Hugging Face models...</p>
          </div>
        </section>

        <!-- Active Jobs & Local Models Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <section class="bg-slate-800 p-6 rounded-lg">
            <h2 class="text-lg font-semibold mb-4">Download Tasks</h2>
            <div id="tasksList" class="space-y-2 text-sm text-slate-300">No active downloads</div>
          </section>

          <section class="bg-slate-800 p-6 rounded-lg">
            <h2 class="text-lg font-semibold mb-4">Downloaded Models on Disk (/storage)</h2>
            <div id="localList" class="space-y-2 text-sm text-slate-300">Scanning...</div>
          </section>
        </div>
      </div>

      <script>
        async function initHardwareInfo() {
          try {
            const res = await fetch('/api/system_info');
            const data = await res.json();
            
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
          } catch(e) {
            document.getElementById('vramStat').textContent = 'Error probing VRAM';
            document.getElementById('ramStat').textContent = 'Error probing RAM';
          }
        }

        function quickSearch(tag) {
          document.getElementById('searchInput').value = tag;
          searchModels();
        }

        async function searchModels() {
          const q = document.getElementById('searchInput').value.trim();
          const container = document.getElementById('searchResults');
          const header = document.getElementById('catalogHeader');
          
          if (q === "") {
            header.textContent = "Available Models (Top GGUFs on Hugging Face)";
          } else {
            header.textContent = `Search Results for "${q}"`;
          }

          container.innerHTML = '<p class="text-slate-400">Querying Hugging Face API...</p>';
          const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
          const models = await res.json();
          container.innerHTML = '';
          
          if (!models || models.length === 0) {
            container.innerHTML = '<p class="text-slate-400">No GGUF models found matching your query.</p>';
            return;
          }

          models.forEach(m => {
            const card = document.createElement('div');
            card.className = 'bg-slate-950 p-4 rounded border border-slate-700 space-y-3';
            card.innerHTML = `
              <div class="flex justify-between items-start">
                <div class="overflow-hidden pr-2">
                  <span class="text-[10px] font-semibold px-2 py-0.5 rounded bg-sky-950 text-sky-300 border border-sky-800">${m.maker}</span>
                  <h3 class="font-bold text-slate-100 text-base mt-1 truncate" title="${m.model_name}">${m.model_name}</h3>
                </div>
                <div class="text-right text-xs text-slate-400 shrink-0">
                  <div>⬇ ${m.downloads.toLocaleString()}</div>
                  <div>❤ ${m.likes}</div>
                </div>
              </div>
              <button onclick="fetchFiles('${m.id}', this)" class="w-full text-xs bg-slate-800 hover:bg-slate-700 border border-slate-600 px-3 py-1.5 rounded transition">
                Inspect Quantizations & Memory Fit
              </button>
              <div class="file-container mt-3 hidden space-y-2"></div>
            `;
            container.appendChild(card);
          });
        }

        async function fetchFiles(repoId, btn) {
          const parent = btn.parentElement;
          const fileContainer = parent.querySelector('.file-container');
          fileContainer.classList.remove('hidden');
          fileContainer.innerHTML = '<span class="text-xs text-slate-500">Calculating memory footprint...</span>';
          
          const res = await fetch(`/api/model_files?repo_id=${encodeURIComponent(repoId)}`);
          const data = await res.json();
          fileContainer.innerHTML = '';

          if (data.files.length === 0) {
            fileContainer.innerHTML = '<span class="text-xs text-slate-500">No .gguf files found in repository root.</span>';
            return;
          }

          const table = document.createElement('div');
          table.className = 'space-y-1.5';

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

            const row = document.createElement('div');
            row.className = 'flex flex-col md:flex-row justify-between items-start md:items-center bg-slate-900 p-2 rounded border border-slate-800 text-xs gap-2';
            row.innerHTML = `
              <div class="space-y-0.5 overflow-hidden">
                <div class="font-medium text-sky-200 truncate" title="${f.filename}">${f.filename}</div>
                <div class="flex flex-wrap items-center gap-2 text-slate-400 text-[11px]">
                  <span>Weight: <strong class="text-slate-200">${f.weight}</strong></span>
                  <span>•</span>
                  <span>Variant: <strong class="text-slate-200">${f.variant}</strong></span>
                  <span>•</span>
                  <span>Size: <strong class="text-slate-200">${f.size_gb}</strong></span>
                  <span>(${f.est_vram} Req)</span>
                  ${fitBadge}
                </div>
              </div>
              <button onclick="triggerDownload('${repoId}', '${f.filename}')" class="bg-emerald-600 hover:bg-emerald-500 text-white font-medium px-3 py-1 rounded text-xs shrink-0">
                Download
              </button>
            `;
            table.appendChild(row);
          });
          fileContainer.appendChild(table);
        }

        async function triggerDownload(repoId, filename) {
          await fetch('/api/download', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({repo_id: repoId, filename: filename})
          });
          updateTasks();
        }

        async function updateTasks() {
          const res = await fetch('/api/tasks');
          const data = await res.json();
          const list = document.getElementById('tasksList');
          if (Object.keys(data).length === 0) {
            list.innerHTML = '<span class="text-slate-500">No active downloads</span>';
            return;
          }
          list.innerHTML = Object.entries(data).map(([file, info]) => `
            <div class="bg-slate-950 p-2 rounded border border-slate-700">
              <div class="font-medium truncate">${file}</div>
              <div class="text-xs text-sky-400 mt-1">${info.status.toUpperCase()}: ${info.progress}</div>
            </div>
          `).join('');
        }

        async function fetchLocalModels() {
          const res = await fetch('/api/local_models');
          const data = await res.json();
          const list = document.getElementById('localList');
          if (data.length === 0) {
            list.innerHTML = '<span class="text-slate-500">No GGUF models on disk</span>';
            return;
          }
          list.innerHTML = data.map(m => `
            <div class="flex justify-between items-center bg-slate-950 p-2 rounded border border-slate-700 text-xs">
              <div class="truncate pr-2">
                <div class="font-medium text-slate-200 truncate">${m.filename}</div>
                <div class="text-slate-400 text-[10px]">Weight: ${m.weight} | Variant: ${m.variant}</div>
              </div>
              <span class="bg-slate-800 px-2 py-1 rounded text-slate-300 font-mono shrink-0">${m.size_gb}</span>
            </div>
          `).join('');
        }

        // Initialize on page load
        initHardwareInfo();
        searchModels(); // Loads default top trending catalog
        setInterval(updateTasks, 3000);
        fetchLocalModels();
      </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)