import os
import re
import ctypes
import shutil
import subprocess

STORAGE_PATH = "/storage/lmstudio"
MODELS_PATH = os.path.join(STORAGE_PATH, "models")
LMS_ENV = {
    **os.environ,
    "HOME": STORAGE_PATH,
    "LMS_SERVER_HOST": "0.0.0.0",
    "PATH": f"/usr/local/bin:{STORAGE_PATH}/.cache/lm-studio/bin:{STORAGE_PATH}/.lmstudio/bin:/usr/sbin:/usr/bin:/bin"
}

def get_lms_bin():
    for p in ["/usr/local/bin/lms", f"{STORAGE_PATH}/.lmstudio/bin/lms", f"{STORAGE_PATH}/.cache/lm-studio/bin/lms"]:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p
    return shutil.which("lms") or "lms"

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

def calculate_optimal_context(model_size_gb: float, vram_total_gb: float = 0.0, max_cap: int = 131072) -> int:
    """
    Computes the maximum safe context length that fits in GPU VRAM without OOM or spilling to RAM.
    """
    if vram_total_gb <= 0:
        hw = get_system_hardware_info()
        vram_total_gb = hw.get("gpu", {}).get("total_vram_gb", 0.0)

    if vram_total_gb > 0:
        headroom = vram_total_gb - model_size_gb - 1.2
        if headroom >= 7.5 and max_cap >= 131072:
            return 131072
        elif headroom >= 3.8 and max_cap >= 65536:
            return 65536
        elif headroom >= 1.8 and max_cap >= 32768:
            return 32768
        elif headroom >= 0.9 and max_cap >= 16384:
            return 16384
        elif headroom >= 0.4 and max_cap >= 8192:
            return 8192
        else:
            return min(4096, max_cap)
    else:
        # Fallback for CPU-only systems based on available System RAM
        hw = get_system_hardware_info()
        avail_ram = hw.get("system_ram", {}).get("available_gb", 8.0)
        ram_headroom = avail_ram - model_size_gb
        if ram_headroom >= 12.0 and max_cap >= 65536:
            return 65536
        elif ram_headroom >= 6.0 and max_cap >= 32768:
            return 32768
        elif ram_headroom >= 3.0 and max_cap >= 16384:
            return 16384
        else:
            return min(8192, max_cap)