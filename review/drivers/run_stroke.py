#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6 — Table 5.2（Vanilla RNN @ MNIST-Stroke）重現 driver。
資料：review_workspace/data/mnist_seq/sequences/（de Jong stroke 序列，已取得）。
結構同 run_all.py：唯讀 repo、冪等續跑、不整批崩潰、per-config timeout、D8 thread-cap。
論文 grid：m∈{30,35,40,45} × {ReLU,Tanh} × h∈{16,32,64,128} = 32 配置。
"""
import os, sys, json, time, glob, subprocess

REPO = "/home/fbi0826/Lin/ZeroSplitVerifier"
WS   = "/home/fbi0826/Lin/review_workspace"
PY   = f"{WS}/.venv/bin/python"
SEQ  = f"{WS}/data/mnist_seq/sequences/"
LOGDIR = f"{WS}/logs"; os.makedirs(LOGDIR, exist_ok=True)
PROGRESS = f"{LOGDIR}/progress_stroke.jsonl"

EVR = ["--N", "50", "--p", "2", "--eps-min", "0.005", "--eps-max", "0.1", "--max-splits", "5"]
N_WORKERS = "30"
TIMEOUT = 604800  # 7天 backstop（長序列可能很久）

ENV = dict(os.environ)
ENV.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1", "NUMBA_THREADING_LAYER": "workqueue", "VECLIB_MAXIMUM_THREADS": "1"})

def now(): return time.strftime("%Y-%m-%d %H:%M:%S")
def log(m): print(f"[{now()}] {m}", flush=True)
def record(e):
    with open(PROGRESS, "a") as f: f.write(json.dumps(e, ensure_ascii=False) + "\n")

def configs():
    return [dict(h=h, m=m, act=a)
            for h in [16, 32, 64, 128]      # h 小先跑（便宜優先）
            for m in [30, 35, 40, 45]
            for a in ["relu", "tanh"]]

def key(c): return f"5.2/rnn/mnist-seq/h{c['h']}/m{c['m']}/{c['act']}"
def model_file(c): return f"{WS}/models/mnist_seq_classifier/rnn_seq_{c['m']}_{c['h']}_{c['act']}/rnn"
def result_glob(c):
    pat = f"{WS}/results/rnn_seq/**/*{c['act']}_hidden{c['h']}_ts{c['m']}_*N50_splits5.json"
    g = glob.glob(pat, recursive=True)
    return g[0] if g else None

def train_cmd(c):
    return [PY, "vanilla_rnn/train_rnn_mnist_seq.py", "--hidden-size", str(c['h']),
            "--time-step", str(c['m']), "--activation", c['act'],
            "--data-dir", SEQ, "--save-dir", f"{WS}/models/mnist_seq_classifier/"]

def verify_cmd(c):
    return [PY, "vanilla_rnn/rnn_zerosplit_verifier.py", "--dataset", "mnist-seq",
            "--hidden-size", str(c['h']), "--time-step", str(c['m']), "--activation", c['act'],
            "--work-dir", f"{WS}/models/mnist_seq_classifier/rnn_seq_{c['m']}_{c['h']}_{c['act']}/",
            "--model-name", "rnn", "--data-dir", SEQ,
            *EVR, "--n-workers", N_WORKERS, "--save-dir", f"{WS}/results/rnn_seq/"]

def run_one(c):
    k = key(c)
    if result_glob(c):
        log(f"SKIP {k} (已有結果)"); return
    t0 = time.time()
    if not os.path.exists(model_file(c)):
        log(f"TRAIN {k}")
        r = subprocess.run(train_cmd(c), cwd=REPO, env=ENV, timeout=TIMEOUT,
                           stdout=open(f"{LOGDIR}/train_{k.replace('/','_')}.log", "w"), stderr=subprocess.STDOUT)
        if r.returncode != 0 or not os.path.exists(model_file(c)):
            log(f"FAIL {k} 訓練 rc={r.returncode}"); record(dict(key=k, status="train_fail", t=now())); return
    log(f"VERIFY {k}")
    try:
        r = subprocess.run(verify_cmd(c), cwd=REPO, env=ENV, timeout=TIMEOUT,
                           stdout=open(f"{LOGDIR}/verify_{k.replace('/','_')}.log", "w"), stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT {k}"); record(dict(key=k, status="timeout", t=now())); return
    js = result_glob(c)
    if r.returncode != 0 or not js:
        log(f"FAIL {k} verify rc={r.returncode}"); record(dict(key=k, status="verify_fail", t=now())); return
    summ = json.load(open(js)).get("evr_summary", {})
    dt = round(time.time() - t0, 1)
    log(f"DONE {k} {summ} ({dt}s)")
    record(dict(key=k, status="done", summary=summ, json=js, sec=dt, t=now()))

def main():
    cfgs = configs()
    log(f"=== run_stroke 啟動：{len(cfgs)} 配置 (Table 5.2 MNIST-Stroke) ===")
    for i, c in enumerate(cfgs, 1):
        log(f"[{i}/{len(cfgs)}] {key(c)}")
        try: run_one(c)
        except Exception as e:
            log(f"ERROR {key(c)}: {type(e).__name__}: {e}")
            record(dict(key=key(c), status="error", err=str(e), t=now()))
    log("=== run_stroke 完畢 ===")

if __name__ == "__main__":
    main()
