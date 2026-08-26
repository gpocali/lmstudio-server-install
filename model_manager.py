sudo tee /storage/lmstudio/model_manager.py > /dev/null <<'EOF'
import os
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

def run_download_job(repo_id: str, filename: str):
    DOWNLOAD_JOBS[filename] = {"status": "downloading", "progress": "In Progress"}
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    dest_dir = os.path.join(MODELS_PATH, repo_id.replace('/', '_'))
    os.makedirs(dest_dir, exist_ok=True)
    dest_file = os.path.join(dest_dir, filename)

    try:
        # Download using curl with resume support
        cmd = ["curl", "-L", "-C", "-", "-o", dest_file, url]
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode == 0:
            DOWNLOAD_JOBS[filename] = {"status": "completed", "progress": "100%"}
            # Notify lms runtime to refresh indexed models
            subprocess.run(["/usr/local/bin/lms", "import", dest_file], capture_output=True)
        else:
            DOWNLOAD_JOBS[filename] = {"status": "failed", "progress": process.stderr}
    except Exception as e:
        DOWNLOAD_JOBS[filename] = {"status": "failed", "progress": str(e)}

@app.get("/api/search")
def search_hf(q: str = "llama"):
    # Query Hugging Face API for GGUF tagged models
    url = f"https://huggingface.co/api/models?search={q}&filter=gguf&sort=downloads&direction=-1&limit=15"
    res = requests.get(url).json()
    results = []
    for m in res:
        results.append({
            "id": m.get("id"),
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "lastModified": m.get("lastModified", "")
        })
    return results

@app.get("/api/model_files")
def get_model_files(repo_id: str):
    url = f"https://huggingface.co/api/models/{repo_id}"
    res = requests.get(url).json()
    siblings = res.get("siblings", [])
    gguf_files = [s["rfilename"] for s in siblings if s["rfilename"].endswith(".gguf")]
    return {"repo_id": repo_id, "files": gguf_files}

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
                    files.append({"filename": f, "size_gb": f"{size_gb} GB", "path": path})
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
      <div class="max-w-6xl mx-auto space-y-8">
        <header class="flex justify-between items-center border-b border-slate-700 pb-4">
          <div>
            <h1 class="text-2xl font-bold text-sky-400">LM Studio Model Manager</h1>
            <p class="text-xs text-slate-400">Target Path: /storage/lmstudio/models</p>
          </div>
          <button onclick="fetchLocalModels()" class="bg-slate-700 hover:bg-slate-600 px-3 py-1.5 rounded text-sm">Refresh Disk</button>
        </header>

        <!-- Search Bar -->
        <section class="bg-slate-800 p-6 rounded-lg shadow space-y-4">
          <h2 class="text-lg font-semibold">Search Hugging Face (GGUF)</h2>
          <div class="flex gap-3">
            <input id="searchInput" type="text" placeholder="e.g. Meta-Llama-3.1-8B, Qwen2.5, bartowski..." 
                   class="flex-1 bg-slate-950 border border-slate-700 rounded px-4 py-2 focus:outline-none focus:border-sky-500">
            <button onclick="searchModels()" class="bg-sky-600 hover:bg-sky-500 px-6 py-2 rounded font-medium">Search</button>
          </div>
          <div id="searchResults" class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4"></div>
        </section>

        <!-- Active Jobs & Local Models Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <section class="bg-slate-800 p-6 rounded-lg">
            <h2 class="text-lg font-semibold mb-4">Download Tasks</h2>
            <div id="tasksList" class="space-y-2 text-sm text-slate-300">No active downloads</div>
          </section>

          <section class="bg-slate-800 p-6 rounded-lg">
            <h2 class="text-lg font-semibold mb-4">Downloaded Models (/storage)</h2>
            <div id="localList" class="space-y-2 text-sm text-slate-300">Scanning...</div>
          </section>
        </div>
      </div>

      <script>
        async function searchModels() {
          const q = document.getElementById('searchInput').value;
          const container = document.getElementById('searchResults');
          container.innerHTML = '<p class="text-slate-400">Searching Hugging Face...</p>';
          const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
          const models = await res.json();
          container.innerHTML = '';
          models.forEach(m => {
            const card = document.createElement('div');
            card.className = 'bg-slate-950 p-4 rounded border border-slate-700 space-y-2';
            card.innerHTML = `
              <div class="font-bold text-sky-300">${m.id}</div>
              <div class="text-xs text-slate-400">Downloads: ${m.downloads.toLocaleString()} | Likes: ${m.likes}</div>
              <button onclick="fetchFiles('${m.id}', this)" class="text-xs bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded">View Quantizations</button>
              <div class="file-container mt-2 hidden space-y-1"></div>
            `;
            container.appendChild(card);
          });
        }

        async function fetchFiles(repoId, btn) {
          const parent = btn.parentElement;
          const fileContainer = parent.querySelector('.file-container');
          fileContainer.classList.remove('hidden');
          fileContainer.innerHTML = '<span class="text-xs text-slate-500">Loading files...</span>';
          const res = await fetch(`/api/model_files?repo_id=${encodeURIComponent(repoId)}`);
          const data = await res.json();
          fileContainer.innerHTML = '';
          data.files.forEach(f => {
            const row = document.createElement('div');
            row.className = 'flex justify-between items-center bg-slate-900 px-2 py-1 rounded text-xs';
            row.innerHTML = `
              <span class="truncate pr-2">${f}</span>
              <button onclick="triggerDownload('${repoId}', '${f}')" class="bg-emerald-600 hover:bg-emerald-500 px-2 py-0.5 rounded text-white font-medium">Download</button>
            `;
            fileContainer.appendChild(row);
          });
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
            list.innerHTML = 'No active tasks';
            return;
          }
          list.innerHTML = Object.entries(data).map(([file, info]) => `
            <div class="bg-slate-950 p-2 rounded border border-slate-700">
              <div class="font-medium truncate">${file}</div>
              <div class="text-xs text-sky-400">${info.status}: ${info.progress}</div>
            </div>
          `).join('');
        }

        async function fetchLocalModels() {
          const res = await fetch('/api/local_models');
          const data = await res.json();
          const list = document.getElementById('localList');
          if (data.length === 0) {
            list.innerHTML = 'No GGUF models on disk';
            return;
          }
          list.innerHTML = data.map(m => `
            <div class="flex justify-between items-center bg-slate-950 p-2 rounded border border-slate-700">
              <span class="truncate font-medium">${m.filename}</span>
              <span class="text-xs bg-slate-800 px-2 py-1 rounded text-slate-300">${m.size_gb}</span>
            </div>
          `).join('');
        }

        setInterval(updateTasks, 3000);
        fetchLocalModels();
      </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF