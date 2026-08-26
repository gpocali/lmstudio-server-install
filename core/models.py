import os
import re
import shutil
import requests
import subprocess
from core.hardware import get_lms_bin, get_system_hardware_info, get_storage_usage, STORAGE_PATH, MODELS_PATH, LMS_ENV

DOWNLOAD_JOBS = {}

VERIFIED_CREATORS = {
    "bartowski", "unsloth", "TheBloke", "MaziyarPanahi", "mradermacher",
    "QuantFactory", "meta-llama", "Qwen", "mistralai", "google",
    "deepseek-ai", "microsoft", "nomic-ai", "cohere", "NousResearch"
}

QUANT_DESCRIPTIONS = {
    "Q4_K_M": "Recommended standard. Medium 4-bit quantization with optimal balance.",
    "Q4_K_S": "Small 4-bit quantization.",
    "Q5_K_M": "High quality 5-bit quantization.",
    "Q5_K_S": "Compact 5-bit quantization.",
    "Q8_0": "Extremely high precision (8-bit).",
    "Q6_K": "Very high quality 6-bit quantization.",
    "Q3_K_L": "Large 3-bit quantization.",
    "Q3_K_M": "Medium 3-bit quantization."
}

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
            if not first_shard_file: first_shard_file = dest_file

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