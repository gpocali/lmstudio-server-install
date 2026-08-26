import os
import re
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

def get_system_vram_gb():
    """Detect NVIDIA GPU VRAM using nvidia-smi, fallback to 0 if not available."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,nounits,noheader"]
        output = subprocess.check_output(cmd, encoding='utf-8').strip()
        lines = output.split('\n')
        total_mb = 0
        free_mb = 0
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 2:
                total_mb += float(parts[0].strip())
                free_mb += float(parts[1].strip())
        return {
            "has_gpu": True,
            "total_vram_gb": round(total_mb / 1024, 2),
            "free_vram_gb": round(free_mb / 1024, 2)
        }
    except Exception:
        # Fallback to system RAM if no NVIDIA GPU is detected
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            total_match = re.search(r'MemTotal:\s+(\d+)', meminfo)
            free_match = re.search(r'MemAvailable:\s+(\d+)', meminfo)
            total_gb = round(int(total_match.group(1)) / (1024**2), 2) if total_match else 0
            free_gb = round(int(free_match.group(1)) / (1024**2), 2) if free_match else 0
            return {
                "has_gpu": False,
                "total_vram_gb": total_gb,
                "free_vram_gb": free_gb
            }
        except Exception:
            return {"has_gpu": False, "total_vram_gb": 0, "free_vram_gb": 0}

def parse_model_metadata(filename: str, repo_id: str):
    """Parse Weight, Quantization, and Format details from repo and filename."""
    # Extract Weight (e.g. 7B, 8B, 70B, 1.5B, 0.5B, 128x8B)
    weight_match = re.search(r'(\d+(\.\d+)?(?:x\d+)?[bB])', f"{repo_id} {filename}")
    weight = weight_match.group(1).upper() if weight_match else "Unknown"

    # Extract Quantization Variant (e.g., Q4_K_M, Q8_0, IQ3_XXS, FP16)
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
    return get_system_vram_gb()

@app.get("/api/search")
def search_hf(q: str = "llama"):
    url = f"https://huggingface.co/api/models?search={q}&filter=gguf&sort=downloads&direction=-1&limit=20"
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
    vram_info = get_system_vram_gb()
    available_vram = vram_info["total_vram_gb"]

    parsed_files = []
    for s in siblings:
        fname = s.get("rfilename", "")
        if fname.endswith(".gguf"):
            # Estimate size if available in payload
            weight, variant = parse_model_metadata(fname, repo_id)
            
            # Fetch file metadata size via head request if necessary or sibling size
            size_bytes = s.get("size", 0)
            size_gb = round(size_bytes / (1024**3), 2) if size_bytes else None

            # Determine VRAM Compatibility
            # Approximate VRAM required for KV cache + runtime overhead = Size * 1.2
            est_vram_req = round(size_gb * 1.2, 2) if size_gb else None
            
            fit_status = "unknown"
            if est_vram_req and available_vram > 0:
                if est_vram_req <= available_vram:
                    fit_status = "fits"
                elif est_vram_req <= available_vram * 1.25:
                    fit_status = "tight"
                else:
                    fit_status = "exceeds"

            parsed_files.append({
                "filename": fname,
                "weight": weight,
                "variant": variant,
                "size_gb": f"{size_gb} GB" if size_gb else "Dynamic",
                "est_vram": f"~{est_vram_req} GB" if est_vram_req else "N/A",
                "fit_status": fit_status
            })

    # Sort files by variant/size
    return {
        "repo_id": repo_id,
        "vram_info": vram_info,
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
    <body class="bg-slate-900 text-slate-100 min-h-screen p-8">
      <div class="max-w-7xl mx-auto space-y-8">
        
        <!-- Header & Hardware Stats -->
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-slate-700 pb-4 gap-4">
          <div>
            <h1 class="text-2xl font-bold text-sky-400">LM Studio Model Manager</h1>
            <p class="text-xs text-slate-400">Target Path: /storage/lmstudio/models</p>
          </div>
          <div id="vramBanner" class="flex items-center gap-3 bg-slate-800 px-4 py-2 rounded-lg border border-slate-700 text-sm">
            <span class="text-slate-400">Hardware VRAM:</span>
            <span id="vramStat" class="font-semibold text-emerald-400">Detecting...</span>
          </div>
        </header>

        <!-- Search Bar -->
        <section class="bg-slate-800 p-6 rounded-lg shadow space-y-4">
          <div class="flex justify-between items-center">
            <h2 class="text-lg font-semibold">Search Hugging Face Repositories (GGUF)</h2>
            <button onclick="fetchLocalModels()" class="text-xs bg-slate-700 hover:bg-slate-600 px-3 py-1.5 rounded">Refresh Disk</button>
          </div>
          <div class="flex gap-3">
            <input id="searchInput" type="text" placeholder="Search by model or maker (e.g. Llama-3.1, bartowski, Qwen2.5, DeepSeek)..." 
                   class="flex-1 bg-slate-950 border border-slate-700 rounded px-4 py-2 focus:outline-none focus:border-sky-500">
            <button onclick="searchModels()" class="bg-sky-600 hover:bg-sky-500 px-6 py-2 rounded font-medium">Search</button>
          </div>
          <div id="searchResults" class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4"></div>
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
        let serverVramGB = 0;

        async function initSystemInfo() {
          try {
            const res = await fetch('/api/system_info');
            const data = await res.json();
            serverVramGB = data.total_vram_gb;
            const label = data.has_gpu ? 'Total GPU VRAM' : 'System RAM (CPU mode)';
            document.getElementById('vramStat').innerHTML = `${data.total_vram_gb} GB <span class="text-xs text-slate-400">(${label})</span>`;
          } catch(e) {
            document.getElementById('vramStat').textContent = 'Detection Unavailable';
          }
        }

        async function searchModels() {
          const q = document.getElementById('searchInput').value;
          const container = document.getElementById('searchResults');
          container.innerHTML = '<p class="text-slate-400">Querying Hugging Face API...</p>';
          const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
          const models = await res.json();
          container.innerHTML = '';
          if (models.length === 0) {
            container.innerHTML = '<p class="text-slate-400">No GGUF models found.</p>';
            return;
          }
          models.forEach(m => {
            const card = document.createElement('div');
            card.className = 'bg-slate-950 p-4 rounded border border-slate-700 space-y-3';
            card.innerHTML = `
              <div class="flex justify-between items-start">
                <div>
                  <span class="text-xs font-semibold px-2 py-0.5 rounded bg-sky-950 text-sky-300 border border-sky-800">${m.maker}</span>
                  <h3 class="font-bold text-slate-100 text-base mt-1">${m.model_name}</h3>
                </div>
                <div class="text-right text-xs text-slate-400">
                  <div>⬇ ${m.downloads.toLocaleString()}</div>
                  <div>❤ ${m.likes}</div>
                </div>
              </div>
              <button onclick="fetchFiles('${m.id}', this)" class="w-full text-xs bg-slate-800 hover:bg-slate-700 border border-slate-600 px-3 py-1.5 rounded transition">
                Inspect Quantizations & VRAM Fit
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
          fileContainer.innerHTML = '<span class="text-xs text-slate-500">Retrieving repository files & calculating VRAM footprint...</span>';
          
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
            if (f.fit_status === 'fits') {
              fitBadge = '<span class="text-[10px] bg-emerald-950 text-emerald-300 border border-emerald-800 px-1.5 py-0.5 rounded">Fits VRAM</span>';
            } else if (f.fit_status === 'tight') {
              fitBadge = '<span class="text-[10px] bg-amber-950 text-amber-300 border border-amber-800 px-1.5 py-0.5 rounded">Tight / Offload</span>';
            } else if (f.fit_status === 'exceeds') {
              fitBadge = '<span class="text-[10px] bg-rose-950 text-rose-300 border border-rose-800 px-1.5 py-0.5 rounded">Exceeds VRAM</span>';
            }

            const row = document.createElement('div');
            row.className = 'flex flex-col md:flex-row justify-between items-start md:items-center bg-slate-900 p-2 rounded border border-slate-800 text-xs gap-2';
            row.innerHTML = `
              <div class="space-y-0.5 overflow-hidden">
                <div class="font-medium text-sky-200 truncate" title="${f.filename}">${f.filename}</div>
                <div class="flex items-center gap-2 text-slate-400 text-[11px]">
                  <span>Weight: <strong class="text-slate-200">${f.weight}</strong></span>
                  <span>•</span>
                  <span>Variant: <strong class="text-slate-200">${f.variant}</strong></span>
                  <span>•</span>
                  <span>Size: <strong class="text-slate-200">${f.size_gb}</strong></span>
                  <span>(${f.est_vram} VRAM)</span>
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

        initSystemInfo();
        setInterval(updateTasks, 3000);
        fetchLocalModels();
      </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)