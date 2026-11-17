#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import time

MODEL_PATH = "/home/raijinshounen/myRoot/mySEJourney/model-inference/models/Qwen3.5-4B-Q4_0.gguf"
DEV = "ROCm0"
CORPUS_PATH = "scratch/eval_corpus.txt"
BASE_LOGITS_PATH = "scratch/base_f16_logits.bin"
RESULTS_JSON = "scratch/benchmark_results.json"

ENV = os.environ.copy()
ENV["LD_LIBRARY_PATH"] = f"/opt/rocm/lib:/opt/rocm/lib64:{ENV.get('LD_LIBRARY_PATH', '')}"
ENV["HIP_VISIBLE_DEVICES"] = "0"

TYPES = [
    ("f16", 16.0, "Baseline (Uncompressed)"),
    ("f8_e4m3", 8.0, "FP8 Floating Point"),
    ("q8_0", 8.5, "8-bit Standard Quant"),
    ("q4_0", 4.5, "4-bit Standard Uniform"),
    ("turbo4", 4.5, "4-bit TurboQuant (WHT)"),
    ("planar4", 4.5, "4-bit PlanarQuant (2D Givens)"),
    ("iso4", 4.5, "4-bit IsoQuant (4D Quaternion)"),
    ("turbo3", 3.5, "3-bit TurboQuant (WHT)"),
    ("planar3", 3.5, "3-bit PlanarQuant (2D Givens)"),
    ("iso3", 3.5, "3-bit IsoQuant (4D Quaternion)"),
    ("turbo2", 2.5, "2-bit TurboQuant (WHT)"),
]

def run_cmd(cmd):
    p = subprocess.run(cmd, shell=True, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.stdout, p.returncode

def ensure_corpus():
    if not os.path.exists(CORPUS_PATH) or os.path.getsize(CORPUS_PATH) < 5000:
        os.makedirs("scratch", exist_ok=True)
        run_cmd(f"cat README.md docs/*.md docs/development/*.md 2>/dev/null | head -n 2500 > {CORPUS_PATH}")

def generate_base_logits():
    print(f"[*] Generating base FP16 reference logits on {DEV}...")
    cmd = (
        f"./build/bin/llama-perplexity -dev {DEV} -m {MODEL_PATH} -f {CORPUS_PATH} "
        f"-c 2048 --chunks 3 -ctk f16 -ctv f16 -ngl 99 --save-all-logits {BASE_LOGITS_PATH}"
    )
    out, code = run_cmd(cmd)
    if code != 0:
        print(f"[!] Error generating base logits:\n{out}")
        sys.exit(1)
    print("[+] Base logits successfully generated.")

def benchmark_speed(kv_type):
    print(f"[*] Running llama-bench for {kv_type} on {DEV}...")
    cmd = (
        f"./build/bin/llama-bench -dev {DEV} -m {MODEL_PATH} -fa on "
        f"-p 512,2048,4096 -n 128 -r 2 -ctk {kv_type} -ctv {kv_type} -o json"
    )
    out, code = run_cmd(cmd)
    res = {}
    if code == 0:
        try:
            # llama-bench outputs json array
            data = json.loads(out)
            for item in data:
                test_name = item.get("test", "")
                ts = item.get("t/s", 0.0)
                res[test_name] = ts
        except Exception as e:
            # Parse text fallback if json fails
            for line in out.splitlines():
                if "pp512" in line or "pp2048" in line or "pp4096" in line or "tg128" in line:
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if len(parts) >= 2:
                        res[parts[0]] = parts[1]
    else:
        print(f"[!] Bench error for {kv_type}: {out[:200]}")
    return res

def benchmark_kl_divergence(kv_type):
    print(f"[*] Measuring KL divergence & Top-1 agreement for {kv_type}...")
    cmd = (
        f"./build/bin/llama-perplexity -dev {DEV} -m {MODEL_PATH} -c 2048 "
        f"-ctk {kv_type} -ctv {kv_type} -ngl 99 --kl-divergence --kl-divergence-base {BASE_LOGITS_PATH}"
    )
    out, code = run_cmd(cmd)
    kld_val = None
    rms_val = None
    top1_val = None
    ppl_delta = None

    for line in out.splitlines():
        if "Mean    KLD:" in line:
            m = re.search(r"Mean\s+KLD:\s+([0-9\.]+)", line)
            if m: kld_val = float(m.group(1))
        elif "RMS Δp" in line or "RMS \u0394p" in line or "RMS" in line and "p" in line:
            m = re.search(r"RMS.*?:\s+([0-9\.]+)", line)
            if m: rms_val = float(m.group(1))
        elif "Same top p:" in line:
            m = re.search(r"Same top p:\s+([0-9\.]+)", line)
            if m: top1_val = float(m.group(1))
        elif "Mean PPL(Q)-PPL(base)" in line:
            m = re.search(r"Mean PPL\(Q\)-PPL\(base\)\s+:\s+([-\+0-9\.]+)", line)
            if m: ppl_delta = float(m.group(1))

    return {
        "mean_kld": kld_val,
        "rms_dp_pct": rms_val,
        "top1_match_pct": top1_val,
        "ppl_delta": ppl_delta,
    }

def benchmark_passkey(kv_type):
    print(f"[*] Running multi-depth passkey retrieval for {kv_type}...")
    trials = [
        {"ctx": 4096, "junk": 140, "pos": 20},
        {"ctx": 4096, "junk": 140, "pos": 50},
        {"ctx": 4096, "junk": 140, "pos": 80},
        {"ctx": 8192, "junk": 300, "pos": 25},
        {"ctx": 8192, "junk": 300, "pos": 75},
    ]
    passed = 0
    total = len(trials)
    details = []

    for t in trials:
        cmd = (
            f"./build/bin/llama-passkey -dev {DEV} -m {MODEL_PATH} -fa 1 "
            f"-c {t['ctx']} --junk {t['junk']} --pos {t['pos']} -ngl 99 -ctk {kv_type} -ctv {kv_type}"
        )
        out, code = run_cmd(cmd)
        
        # Extract passkey number
        pk_m = re.search(r"passkey\s*=\s*(\d+)", out)
        if not pk_m:
            details.append({"trial": t, "result": "ERROR", "key": None, "generated": out[-100:]})
            continue
        expected_key = pk_m.group(1)
        
        # Check generated answer
        gen_m = re.search(r"The pass key is\s*(\d+)", out)
        got_key = gen_m.group(1) if gen_m else None
        
        # Also check speed
        speed_m = re.search(r"speed:\s*([0-9\.]+)\s*t/s", out)
        speed = float(speed_m.group(1)) if speed_m else None

        ok = (got_key == expected_key)
        if ok: passed += 1
        details.append({
            "ctx": t["ctx"],
            "depth_pct": t["pos"],
            "expected": expected_key,
            "got": got_key,
            "success": ok,
            "speed_ts": speed
        })

    return {
        "pass_rate_pct": (passed / total) * 100.0,
        "passed": passed,
        "total": total,
        "details": details
    }

def main():
    ensure_corpus()
    generate_base_logits()

    all_results = {}
    for kv_type, bpw, desc in TYPES:
        print(f"\n========================================================")
        print(f"Evaluating: {kv_type} ({bpw} bpw) - {desc}")
        print(f"========================================================")

        t0 = time.time()
        speed_data = benchmark_speed(kv_type)
        fidelity_data = benchmark_kl_divergence(kv_type)
        retrieval_data = benchmark_passkey(kv_type)
        elapsed = time.time() - t0

        all_results[kv_type] = {
            "type": kv_type,
            "bpw": bpw,
            "description": desc,
            "elapsed_sec": round(elapsed, 2),
            "speed": speed_data,
            "fidelity": fidelity_data,
            "retrieval": retrieval_data
        }

        # Save incremental
        with open(RESULTS_JSON, "w") as f:
            json.dump(all_results, f, indent=2)

    print("\n[+] Benchmark completed successfully. Saved to:", RESULTS_JSON)

if __name__ == "__main__":
    main()
