#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZeroSplitVerifier 可再現性審查 — 主控 driver（對照論文第5章 Tables 5.3/5.4/5.1）。

設計（見 /home/fbi0826/Lin/CLAUDE.md §14）：
  - 被審 repo 唯讀；所有輸出落在 review_workspace（WS）。
  - 冪等/可續跑：結果 JSON 已存在就跳過 → 殺掉重啟自動接續。
  - 不因單一配置失敗而整批崩潰：每配置 try/except + per-config timeout，失敗記錄後續跑。
  - D8 死結解法：所有 train/verify 都帶 thread-cap 環境變數，安全使用多 worker。
  - 進度寫 logs/progress.jsonl（機器讀）+ stdout（人讀，tmux 導向 driver.log）。

執行（tmux detach）：
  tmux new-session -d -s zsv \
    "cd /home/fbi0826/Lin/ZeroSplitVerifier && \
     /home/fbi0826/Lin/review_workspace/.venv/bin/python \
     /home/fbi0826/Lin/review_workspace/drivers/run_all.py \
     >> /home/fbi0826/Lin/review_workspace/logs/driver.log 2>&1"
"""
import os, sys, json, time, glob, hashlib, subprocess

REPO = "/home/fbi0826/Lin/ZeroSplitVerifier"
WS   = "/home/fbi0826/Lin/review_workspace"
PY   = f"{WS}/.venv/bin/python"
LOGDIR = f"{WS}/logs"
os.makedirs(LOGDIR, exist_ok=True)
PROGRESS = f"{LOGDIR}/progress.jsonl"

# 論文對齊參數（§A/§F）
EVR = ["--N", "50", "--p", "2", "--eps-min", "0.005", "--eps-max", "0.1", "--max-splits", "5"]
N_WORKERS = "30"            # 每配置 sample 級多進程（每 worker 單執行緒；32 核留 2 給 OS）
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"

# per-config timeout（秒）：超時 → 記錄後續跑，不卡死整批
TIMEOUT = {"rnn_mnist": 1800, "rnn_cifar10": 10800, "lstm": 604800}  # LSTM 7天 backstop＝實質不設限（使用者選），僅防真正卡死

# D8 解法：完整 thread-cap，讓 fork-pool 不死結（RNN 需 OMP=1；LSTM 另需 NUMBA=1）
ENV = dict(os.environ)
ENV.update({
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1", "NUMBA_THREADING_LAYER": "workqueue",
    "VECLIB_MAXIMUM_THREADS": "1",
})

def now(): return time.strftime("%Y-%m-%d %H:%M:%S")
def log(m): print(f"[{now()}] {m}", flush=True)
def record(e):
    with open(PROGRESS, "a") as f: f.write(json.dumps(e, ensure_ascii=False) + "\n")

# ---------- 配置矩陣（順序：5.3 快 → 5.4 慢/與CIFAR下載重疊 → 5.1 需資料）----------
def configs():
    cfgs = []
    for m in [1, 2, 4, 7]:               # Table 5.3 RNN MNIST（32）
        for h in [4, 8, 16, 32]:
            for act in ["relu", "tanh"]:
                cfgs.append(dict(table="5.3", net="rnn", dataset="mnist", h=h, m=m, act=act))
    for h in [4, 8, 16, 32]:             # Table 5.4 LSTM MNIST（16）— h 小先跑（便宜 LSTM 填下載空檔）
        for m in [1, 2, 4, 7]:
            cfgs.append(dict(table="5.4", net="lstm", dataset="mnist", h=h, m=m, act="sigtanh"))
    for h in [16, 32, 64, 128]:          # Table 5.1 RNN CIFAR-10（32）— h 小先跑
        for m in [8, 12, 24, 32]:
            for act in ["relu", "tanh"]:
                cfgs.append(dict(table="5.1", net="rnn", dataset="cifar10", h=h, m=m, act=act))
    return cfgs

def key(c): return f"{c['table']}/{c['net']}/{c['dataset']}/h{c['h']}/m{c['m']}/{c['act']}"

# ---------- 路徑 / 指令 ----------
def model_file(c):
    if c["net"] == "lstm":
        return f"{WS}/models/mnist_lstm/lstm_{c['m']}_{c['h']}/lstm"
    sub = "mnist_classifier" if c["dataset"] == "mnist" else "cifar10_classifier"
    return f"{WS}/models/{sub}/rnn_{c['m']}_{c['h']}_{c['act']}/rnn"

def save_dir(c):
    return {"mnist": f"{WS}/results/rnn_mnist/", "cifar10": f"{WS}/results/rnn_cifar10/"}[c["dataset"]] \
           if c["net"] == "rnn" else f"{WS}/results/lstm_mnist/"

def result_exists(c):
    pat = f"{save_dir(c)}**/*hidden{c['h']}_ts{c['m']}_*N50_splits5.json"
    for p in glob.glob(pat, recursive=True):
        if c["net"] == "rnn" and f"_{c['act']}_" not in os.path.basename(p):
            continue
        return p
    return None

def train_cmd(c):
    if c["net"] == "lstm":
        return [PY, "lstm/train_mnist_lstm.py", "--hidden-size", str(c['h']),
                "--time-step", str(c['m']), "--data-dir", "../data/mnist",
                "--save-dir", f"{WS}/models/mnist_lstm/"]
    if c["dataset"] == "mnist":
        return [PY, "vanilla_rnn/train_rnn_mnist_classifier.py", "--hidden-size", str(c['h']),
                "--time-step", str(c['m']), "--activation", c['act'],
                "--save_dir", f"{WS}/models/mnist_classifier/"]          # 注意底線
    return [PY, "vanilla_rnn/train_rnn_cifar10.py", "--hidden-size", str(c['h']),
            "--time-step", str(c['m']), "--activation", c['act'],
            "--data-dir", "../data", "--save-dir", f"{WS}/models/cifar10_classifier/"]

def verify_cmd(c):
    if c["net"] == "lstm":
        return [PY, "lstm/lstm_zerosplit_verifier.py", "--dataset", "mnist",
                "--hidden-size", str(c['h']), "--time-step", str(c['m']),
                "--work-dir", f"{WS}/models/mnist_lstm/",                # 父目錄，verifier 自接 lstm_M_H/lstm
                "--data-dir", "../data/mnist", "--lut-dir", "./lookup_tables",
                *EVR, "--n-workers", N_WORKERS, "--save-dir", f"{WS}/results/lstm_mnist/"]
    common = ["--hidden-size", str(c['h']), "--time-step", str(c['m']), "--activation", c['act'],
              "--model-name", "rnn", *EVR, "--n-workers", N_WORKERS]
    if c["dataset"] == "mnist":
        return [PY, "vanilla_rnn/rnn_zerosplit_verifier.py", "--dataset", "mnist", *common,
                "--work-dir", f"{WS}/models/mnist_classifier/rnn_{c['m']}_{c['h']}_{c['act']}/",
                "--data-dir", "../data/mnist", "--save-dir", f"{WS}/results/rnn_mnist/"]
    return [PY, "vanilla_rnn/rnn_zerosplit_verifier.py", "--dataset", "cifar10", "--use-rgb", *common,
            "--work-dir", f"{WS}/models/cifar10_classifier/rnn_{c['m']}_{c['h']}_{c['act']}/",
            "--data-dir", "../data", "--save-dir", f"{WS}/results/rnn_cifar10/"]   # --use-rgb 必須（D-cifar）

def timeout_for(c):
    return TIMEOUT["lstm"] if c["net"] == "lstm" else TIMEOUT[f"rnn_{c['dataset']}"]

# ---------- CIFAR 資料就緒守門（背景下載完成才跑 CIFAR）----------
def cifar_ready():
    tar = "/home/fbi0826/Lin/data/cifar-10-python.tar.gz"
    if os.path.isdir("/home/fbi0826/Lin/data/cifar-10-batches-py"):
        return True
    if not os.path.exists(tar):
        return False
    h = hashlib.md5()
    with open(tar, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest() == CIFAR_MD5

def wait_cifar(max_wait=3 * 3600):
    t0 = time.time()
    while not cifar_ready():
        if time.time() - t0 > max_wait:
            return False
        log("  …等待 CIFAR 資料下載完成（背景 curl）…")
        time.sleep(120)
    return True

# ---------- 跑單一配置 ----------
def run_one(c):
    k = key(c)
    hit = result_exists(c)
    if hit:
        log(f"SKIP {k}  (已有結果 {os.path.basename(hit)})")
        return
    if c["dataset"] == "cifar10" and not wait_cifar():
        log(f"FAIL {k}  CIFAR 資料未就緒，跳過")
        record(dict(key=k, status="skip_no_data", t=now())); return

    t0 = time.time()
    # 1) 訓練（若缺模型）
    if not os.path.exists(model_file(c)):
        log(f"TRAIN {k}")
        r = subprocess.run(train_cmd(c), cwd=REPO, env=ENV,
                           stdout=open(f"{LOGDIR}/train_{k.replace('/','_')}.log", "w"),
                           stderr=subprocess.STDOUT, timeout=timeout_for(c))
        if r.returncode != 0 or not os.path.exists(model_file(c)):
            log(f"FAIL {k}  訓練失敗 rc={r.returncode}")
            record(dict(key=k, status="train_fail", rc=r.returncode, t=now())); return

    # 2) 驗證
    log(f"VERIFY {k}  (n-workers={N_WORKERS}, timeout={timeout_for(c)}s)")
    vlog = f"{LOGDIR}/verify_{k.replace('/','_')}.log"
    try:
        r = subprocess.run(verify_cmd(c), cwd=REPO, env=ENV,
                           stdout=open(vlog, "w"), stderr=subprocess.STDOUT,
                           timeout=timeout_for(c))
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT {k}  (>{timeout_for(c)}s)")
        record(dict(key=k, status="timeout", t=now(), sec=round(time.time()-t0,1))); return

    js = result_exists(c)
    if r.returncode != 0 or not js:
        log(f"FAIL {k}  驗證 rc={r.returncode} json={bool(js)}")
        record(dict(key=k, status="verify_fail", rc=r.returncode, t=now())); return

    summ = json.load(open(js)).get("evr_summary", {})
    dt = round(time.time() - t0, 1)
    log(f"DONE {k}  {summ}  ({dt}s)")
    record(dict(key=k, status="done", summary=summ, json=js, sec=dt, t=now()))

def main():
    cfgs = configs()
    t53 = [c for c in cfgs if c["table"] == "5.3"]   # RNN MNIST（快）
    t54 = [c for c in cfgs if c["table"] == "5.4"]   # LSTM（慢）
    t51 = [c for c in cfgs if c["table"] == "5.1"]   # RNN CIFAR（需下載資料）
    total = len(cfgs)
    state = {"i": 0}
    log(f"=== run_all 啟動：{total} 配置（5.3×{len(t53)} / 5.4×{len(t54)} / 5.1×{len(t51)}）===")

    def do(c):
        state["i"] += 1
        log(f"[{state['i']}/{total}] {key(c)}")
        try:
            run_one(c)
        except Exception as e:
            log(f"ERROR {key(c)}: {type(e).__name__}: {e}")
            record(dict(key=key(c), status="error", err=f"{type(e).__name__}: {e}", t=now()))

    # 1) RNN MNIST 全跑（最快、先有結果）
    for c in t53:
        do(c)
    # 2) 交錯：CIFAR 資料一就緒就插隊跑；否則先推進 LSTM（用 LSTM 填滿 ~3.5h 下載等待）
    while t54 or t51:
        if t51 and cifar_ready():
            do(t51.pop(0))
        elif t54:
            do(t54.pop(0))
        elif t51:
            log("  只剩 CIFAR 而資料未就緒 → 等待下載完成")
            wait_cifar()
            do(t51.pop(0))
    log("=== run_all 全部配置處理完畢 ===")

if __name__ == "__main__":
    main()
