HF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
}

@app.get("/api/search")
def api_search_hf(q: str = "", sort_by: str = "downloads", verified_only: bool = False):
    hf_sort = "likes" if sort_by == "likes" else ("lastModified" if sort_by == "lastModified" else "downloads")
    params = {"filter": "gguf", "sort": hf_sort, "direction": "-1", "limit": 60}
    if q.strip(): 
        params["search"] = q.strip()
        
    res = []
    try:
        resp = requests.get("https://huggingface.co/api/models", params=params, headers=HF_HEADERS, timeout=12)
        if resp.status_code == 200:
            res = resp.json()
    except Exception as e:
        print(f"HF Search Error: {e}")

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
def api_model_files(repo_id: str):
    raw_files = {}
    try:
        resp = requests.get(f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true", headers=HF_HEADERS, timeout=10)
        if resp.status_code == 200:
            for item in resp.json():
                path = item.get("path", "")
                if path.endswith(".gguf"):
                    sz = item.get("size", 0) or (item.get("lfs", {}).get("size", 0) if isinstance(item.get("lfs"), dict) else 0)
                    raw_files[path] = sz
    except Exception: pass

    if not raw_files:
        try:
            res = requests.get(f"https://huggingface.co/api/models/{repo_id}", headers=HF_HEADERS, timeout=10).json()
            for s in res.get("siblings", []):
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