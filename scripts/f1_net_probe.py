#!/usr/bin/env python3
"""gradmotion network probe: download speed from the indexes uv sync needs (PyTorch cu128 index, PyPI,
GitHub). Prints one line per url: http_code, seconds, bytes/s, bytes. Exits 0."""
import subprocess
import time

URLS = [
    "https://download.pytorch.org/whl/cu128/torch/",
    "https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl",
    "https://pypi.org/simple/numpy/",
    "https://files.pythonhosted.org/packages/source/n/numpy/numpy-2.2.6.tar.gz",
    "https://github.com/Roboparty/PartyMimic.git/info/refs?service=git-upload-pack",
]
for url in URLS:
    t0 = time.time()
    r = subprocess.run(["curl", "-s", "-L", "-o", "/dev/null", "-m", "60", "-r", "0-8000000",
                        "-w", "%{http_code} %{time_total} %{speed_download} %{size_download}", url],
                       capture_output=True, text=True)
    print(f"[probe] {url.split('/')[2]:40s} code/speed/size: {r.stdout.strip()} (wall {time.time() - t0:.1f}s)", flush=True)
print("[probe] done", flush=True)
