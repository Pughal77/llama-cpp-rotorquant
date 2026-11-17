#!/usr/bin/env python3
import json
import os
import subprocess
import sys

MODEL_PATH = "/home/raijinshounen/myRoot/mySEJourney/model-inference/models/Qwen3.5-4B-Q4_0.gguf"
DEV = "ROCm0"
RESULTS_JSON = "scratch/benchmark_results.json"

ENV = os.environ.copy()
ENV["LD_LIBRARY_PATH"] = f"/opt/rocm/lib:/opt/rocm/lib64:{ENV.get('LD_LIBRARY_PATH', '')}"
ENV["HIP_VISIBLE_DEVICES"] = "0"

TYPES = [
    "f16",
    "f8_e4m3",
    "q8_0",
    "q4_0",
    "turbo4",
    "planar4",
    "iso4",
    "turbo3",
    "planar3",
    "iso3",
    "turbo2"
]

def run_bench(kv_type):
    print(f"[*] Running llama-bench for {kv_type} on {DEV}...")
    cmd = (
        f"./build/bin/llama-bench -dev {DEV} -m {MODEL_PATH} -fa on "
        f"-p 512,2048,4096 -n 128 -r 1 -ctk {kv_type} -ctv {kv_type} -o json"
    )
    p = subprocess.run(cmd, shell=True, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout
    res = {}
    if p.returncode == 0:
        idx = out.find("[")
        if idx != -1:
            try:
                data = json.loads(out[idx:])
                for item in data:
                    np = item.get("n_prompt", 0)
                    ng = item.get("n_gen", 0)
                    ts = round(item.get("avg_ts", 0.0), 2)
                    if ng == 0:
                        res[f"pp{np}"] = ts
                    else:
                        res[f"tg{ng}"] = ts
            except Exception as e:
                print(f"[!] JSON parse error: {e}")
    else:
        print(f"[!] Command failed: {out[:200]}")
    return res

def main():
    if not os.path.exists(RESULTS_JSON):
        print(f"[!] {RESULTS_JSON} not found!")
        sys.exit(1)

    with open(RESULTS_JSON, "r") as f:
        results = json.load(f)

    for kv_type in TYPES:
        speed_data = run_bench(kv_type)
        print(f"  -> {kv_type}: {speed_data}")
        if kv_type in results:
            results[kv_type]["speed"] = speed_data

        with open(RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=2)

    print("\n[+] Updated speed data in", RESULTS_JSON)

if __name__ == "__main__":
    main()
