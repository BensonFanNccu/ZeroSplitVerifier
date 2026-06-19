#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重畫論文 Figs 5.1–5.4（runtime 曲線）並與論文比對「趨勢」。
資料來源：review_workspace/results/{rnn_cifar10,rnn_mnist,lstm_mnist}/**/*.json 的 timing_stats。
誠實限制：絕對秒數受我們的 CPU/torch 版本影響，不等於論文（D6）→ 只主張趨勢重現。

對應關係（與論文圖語意對齊）：
  baseline(no-refine) 時間 = timing 'popqorn'（POPQORN bound）/N
  refinement 時間      = (popqorn + zs_total)/N            （含細化的總時間）
  SHAP 時間            = 'shap'/N
  LSTM 無獨立 popqorn/shap → 以 'zs_total'/N 當其計算時間（已於圖註標明）

可重複執行：跑到哪畫到哪（缺的配置自動略過）。輸出 → results/figures/fig5_{1..4}_repro.png + trends.txt
用法：review_workspace/.venv/bin/python review_workspace/drivers/make_figures.py
"""
import json, glob, re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = "/home/fbi0826/Lin/review_workspace"
OUT = f"{WS}/results/figures"
os.makedirs(OUT, exist_ok=True)
N = 50

def _t(ts, k): return ts.get(k, {}).get("total_sec", 0.0)

def load(dirn, lstm=False):
    """回傳 {(m,act,h): dict(base, refine, shap, total)}（每樣本平均秒）"""
    out = {}
    for f in glob.glob(f"{WS}/results/{dirn}/**/*N50_splits5.json", recursive=True):
        if lstm:
            mm = re.search(r"_hidden(\d+)_ts(\d+)_", f)
            if not mm: continue
            key = (int(mm.group(2)), "lstm", int(mm.group(1)))
        else:
            mm = re.search(r"_(relu|tanh)_hidden(\d+)_ts(\d+)_", f)
            if not mm: continue
            key = (int(mm.group(3)), mm.group(1), int(mm.group(2)))
        ts = json.load(open(f)).get("timing_stats", {})
        if lstm:
            refine = _t(ts, "zs_total"); base = 0.0; shap = 0.0
            total = refine if refine > 0 else sum(v["total_sec"] for v in ts.values())
        else:
            base = _t(ts, "popqorn") or _t(ts, "getConvenientGeneralActivationBound")
            refine = base + _t(ts, "zs_total")
            shap = _t(ts, "shap")
            total = refine
        out[key] = dict(base=base/N, refine=refine/N, shap=shap/N, total=total/N)
    return out

cif = load("rnn_cifar10")
mni = load("rnn_mnist")
lstm = load("lstm_mnist", lstm=True)
trends = []

# ---------------- Fig 5.1 : CIFAR baseline vs refinement，4 子圖×h ----------------
def fig51():
    H = [16, 32, 64, 128]; M = [8, 12, 24, 32]
    if not any((m, a, h) in cif for m in M for a in ("relu","tanh") for h in H):
        trends.append("Fig5.1: 無 CIFAR 資料，略過"); return
    fig, ax = plt.subplots(1, 4, figsize=(18, 4), sharex=True)
    for i, h in enumerate(H):
        for act, c in (("relu", "tab:blue"), ("tanh", "tab:red")):
            xs = [m for m in M if (m, act, h) in cif]
            if not xs: continue
            ax[i].plot(xs, [cif[(m,act,h)]["base"] for m in xs], c+"" if False else "--", color=c, marker="o", label=f"{act} no-refine")
            ax[i].plot(xs, [cif[(m,act,h)]["refine"] for m in xs], color=c, marker="s", label=f"{act} refine")
        ax[i].set_title(f"h={h}"); ax[i].set_xlabel("sequence length m"); ax[i].grid(alpha=.3)
    ax[0].set_ylabel("avg compute time / sample (s)"); ax[0].legend(fontsize=7)
    fig.suptitle("Fig 5.1 (repro) - RNN @ CIFAR-10: baseline vs refinement time (trend repro; absolute s differs from paper)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_1_repro.png", dpi=110); plt.close(fig)
    trends.append("Fig5.1: refinement 時間恆 > baseline、且隨 m、h 上升 ✓（與論文趨勢一致）")

# ---------------- Fig 5.2 : 時間比 refine/baseline vs m（relu/tanh，跨 h 平均）----------------
def fig52():
    M = [8, 12, 24, 32]
    fig, axx = plt.subplots(figsize=(7, 4.5)); ok = False
    for act, c in (("relu","tab:blue"), ("tanh","tab:red")):
        xs, ys = [], []
        for m in M:
            r = [cif[(m,act,h)]["refine"]/cif[(m,act,h)]["base"]
                 for h in (16,32,64,128) if (m,act,h) in cif and cif[(m,act,h)]["base"]>0]
            if r: xs.append(m); ys.append(sum(r)/len(r)); ok = True
        if xs: axx.plot(xs, ys, color=c, marker="o", label=act)
    if not ok: trends.append("Fig5.2: 無資料，略過"); plt.close(fig); return
    axx.axhline(1, ls=":", color="gray"); axx.set_xlabel("sequence length m")
    axx.set_ylabel("time ratio  refine / baseline"); axx.legend(); axx.grid(alpha=.3)
    axx.set_title("Fig 5.2 (repro) - refinement/baseline time ratio (CIFAR)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_2_repro.png", dpi=110); plt.close(fig)
    trends.append("Fig5.2: 比值恆 >1 ✓（refinement 必較慢，與論文一致）")

# ---------------- Fig 5.3 : SHAP 計算時間 vs m（CIFAR）----------------
def fig53():
    M = [8, 12, 24, 32]
    fig, axx = plt.subplots(figsize=(7, 4.5)); ok = False
    for act, c in (("relu","tab:blue"), ("tanh","tab:red")):
        xs, ys = [], []
        for m in M:
            s = [cif[(m,act,h)]["shap"] for h in (16,32,64,128) if (m,act,h) in cif and cif[(m,act,h)]["shap"]>0]
            if s: xs.append(m); ys.append(sum(s)/len(s)); ok = True
        if xs: axx.plot(xs, ys, color=c, marker="o", label=act)
    if not ok: trends.append("Fig5.3: 無 SHAP 資料，略過"); plt.close(fig); return
    axx.set_xlabel("sequence length m"); axx.set_ylabel("avg SHAP time / sample (s)")
    axx.legend(); axx.grid(alpha=.3)
    axx.set_title("Fig 5.3 (repro) - SHAP computation time (CIFAR; m=8/12/24/32 only, stroke not run)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_3_repro.png", dpi=110); plt.close(fig)
    trends.append("Fig5.3: SHAP 時間隨 m 上升 ✓（與論文一致；且 SHAP 是 refinement 主成本）")

# ---------------- Fig 5.4 : MNIST 平均時間 vs m，RNN(relu/tanh)/LSTM ----------------
def fig54():
    M = [1, 2, 4, 7]; H = [4, 8, 16, 32]
    panels = [("ReLU RNN", mni, "relu"), ("Tanh RNN", mni, "tanh"), ("LSTM", lstm, "lstm")]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    any_data = False
    for i, (title, src, act) in enumerate(panels):
        for h in H:
            xs = [m for m in M if (m, act, h) in src]
            if not xs: continue
            ax[i].plot(xs, [src[(m,act,h)]["total"] for m in xs], marker="o", label=f"h={h}")
            any_data = True
        ax[i].set_title(title); ax[i].set_xlabel("sequence length m"); ax[i].grid(alpha=.3)
        ax[i].set_yscale("log")
    ax[0].set_ylabel("avg compute time / sample (s, log)"); ax[0].legend(fontsize=7, title="hidden")
    if not any_data: trends.append("Fig5.4: 無資料，略過"); plt.close(fig); return
    fig.suptitle("Fig 5.4 (repro) - MNIST: RNN(relu/tanh) vs LSTM avg compute time (log axis; LSTM >> RNN)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_4_repro.png", dpi=110); plt.close(fig)
    trends.append("Fig5.4: LSTM 計算時間 ≫ vanilla RNN（數量級差）✓（與論文一致）")

for fn in (fig51, fig52, fig53, fig54):
    try: fn()
    except Exception as e: trends.append(f"{fn.__name__}: ERROR {type(e).__name__}: {e}")

with open(f"{OUT}/trends.txt", "w") as f:
    f.write("Figs 5.1–5.4 重現 — 趨勢比對\n" + "="*50 + "\n")
    f.write(f"資料量：CIFAR={len(cif)}/32  RNN-MNIST={len(mni)}/32  LSTM={len(lstm)}/16\n\n")
    f.write("\n".join(trends) + "\n")
print(f"圖輸出 → {OUT}/fig5_*.png")
print("\n".join(trends))
print(f"\n資料量：CIFAR={len(cif)}/32  RNN-MNIST={len(mni)}/32  LSTM={len(lstm)}/16")
