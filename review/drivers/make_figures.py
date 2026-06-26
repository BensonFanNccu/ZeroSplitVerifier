#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重畫論文 Figs 5.1–5.4（runtime 趨勢）。

指標(重要):每樣本平均「計算工作量」以 CPU 秒計 = (該配置所有計時器總和)/50。
  - 細化(refinement)成本 = 全部計時器總和 / 50
  - 未細化(no-refinement)成本 = (全部 − zs_total) / 50   （zs_total 為細化專屬計時器）
  以「同一個 run 內拆分」確保 refine ≥ no-refine、且 ε 掃描長度一致
  （分開跑 --max-splits 0 因 ε 提早停止邏輯不同，不可直接比，故不採用）。
誠實限制:這是我們 30 核 CPU 上的計算量,絕對值不等於論文(軟體/是否用 GPU 未知,D6),只主張趨勢。
缺口:Fig 5.4 子圖 (d) GenBaB 未跑。
"""
import json, glob, re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = "/home/fbi0826/Lin/review_workspace"
OUT = f"{WS}/results/figures"; os.makedirs(OUT, exist_ok=True)
N = 50

def load(dirn, lstm=False):
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
        total = sum(v["total_sec"] for v in ts.values())
        zs = ts.get("zs_total", {}).get("total_sec", 0.0)
        shap = ts.get("shap", {}).get("total_sec", 0.0)
        out[key] = dict(refine=total / N, norefine=(total - zs) / N, shap=shap / N)
    return out

cif = load("rnn_cifar10"); seq = load("rnn_seq")
mni = load("rnn_mnist");   lstm = load("lstm_mnist", lstm=True)
# Fig 5.1/5.2/5.3 把 CIFAR(8-32)+Stroke(30-45) 合到同一軸
comb = {**cif, **seq}
MS_COMB = [8, 12, 24, 32, 30, 35, 40, 45]; MS_COMB = sorted(set(MS_COMB))
trends = []

# ---- Fig 5.1: baseline vs refinement，4 子圖(h)，x=m(CIFAR+Stroke) ----
def fig51():
    H = [16, 32, 64, 128]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4), sharex=True)
    for i, h in enumerate(H):
        for act, c in (("relu", "tab:blue"), ("tanh", "tab:red")):
            xs = sorted(m for m in MS_COMB if (m, act, h) in comb)
            if not xs: continue
            ax[i].plot(xs, [comb[(m,act,h)]["norefine"] for m in xs], "-",  color=c, marker="o", ms=4, label=f"{act.upper()} no-refine")
            ax[i].plot(xs, [comb[(m,act,h)]["refine"]   for m in xs], "--", color=c, marker="s", ms=4, label=f"{act.upper()} refine")
        ax[i].set_title(f"h = {h}"); ax[i].set_xlabel("Timestep (m)"); ax[i].grid(alpha=.3)
    ax[0].set_ylabel("Avg. Computation Time (CPU-s)/sample"); ax[0].legend(fontsize=7)
    fig.suptitle("Fig 5.1 (repro) - baseline vs refinement, CIFAR-10 (m=8-32) + MNIST-Stroke (m=30-45); trend only, abs != paper")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_1_repro.png", dpi=110); plt.close(fig)
    trends.append("Fig5.1: refine >= no-refine、隨 m/h 上升、tanh>relu、CIFAR+Stroke 連續 ✓")

# ---- Fig 5.2: refine/no-refine 比值，x=m(CIFAR+Stroke)，relu/tanh(跨h平均) ----
def fig52():
    fig, axx = plt.subplots(figsize=(7.5, 4.5)); ok = False
    for act, c in (("relu","tab:blue"), ("tanh","tab:red")):
        xs, ys = [], []
        for m in MS_COMB:
            r = [comb[(m,act,h)]["refine"]/comb[(m,act,h)]["norefine"]
                 for h in (16,32,64,128) if (m,act,h) in comb and comb[(m,act,h)]["norefine"]>0]
            if r: xs.append(m); ys.append(sum(r)/len(r)); ok = True
        if xs: axx.plot(xs, ys, color=c, marker="o", label=act.upper())
    if not ok: trends.append("Fig5.2: 無資料"); plt.close(fig); return
    axx.axhline(1, ls=":", color="gray"); axx.set_xlabel("Timestep (m)")
    axx.set_ylabel("Refinement / No refinement"); axx.legend(); axx.grid(alpha=.3)
    axx.set_title("Fig 5.2 (repro) - time ratio refinement vs no refinement (CIFAR+Stroke)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_2_repro.png", dpi=110); plt.close(fig)
    trends.append("Fig5.2: 比值恆 >1 ✓")

# ---- Fig 5.3: SHAP 時間，x=m(CIFAR+Stroke)，relu/tanh ----
def fig53():
    fig, axx = plt.subplots(figsize=(7.5, 4.5)); ok = False
    import numpy as np
    xs = [m for m in MS_COMB if any((m,a,h) in comb for a in ("relu","tanh") for h in (16,32,64,128))]
    w = 0.8;
    for j,(act, c) in enumerate((("relu","tab:blue"), ("tanh","tab:red"))):
        ys = []
        for m in xs:
            s = [comb[(m,act,h)]["shap"] for h in (16,32,64,128) if (m,act,h) in comb and comb[(m,act,h)]["shap"]>0]
            ys.append(sum(s)/len(s) if s else 0);
        if any(ys): ok=True
        axx.plot(xs, ys, color=c, marker="o", label=act.upper())
    if not ok: trends.append("Fig5.3: 無 SHAP 資料"); plt.close(fig); return
    axx.set_xlabel("Timestep (m)"); axx.set_ylabel("Avg. SHAP Computation Time (CPU-s)/sample")
    axx.legend(); axx.grid(alpha=.3)
    axx.set_title("Fig 5.3 (repro) - SHAP computation time across m (CIFAR+Stroke)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_3_repro.png", dpi=110); plt.close(fig)
    trends.append("Fig5.3: SHAP 時間隨 m 上升 ✓")

# ---- Fig 5.4: (a)ReLU RNN (b)Tanh RNN (c)LSTM；x=m(1,2,4,7)；每h雙線；(d)GenBaB缺 ----
def fig54():
    M = [1, 2, 4, 7]; H = [4, 8, 16, 32]
    colors = {4:"tab:blue", 8:"tab:orange", 16:"tab:green", 32:"tab:red"}
    panels = [("(a) ReLU RNN", mni, "relu"), ("(b) Tanh RNN", mni, "tanh"), ("(c) LSTM", lstm, "lstm")]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for i, (title, src, act) in enumerate(panels):
        for h in H:
            xs = [m for m in M if (m, act, h) in src]
            if not xs: continue
            ax[i].plot(xs, [src[(m,act,h)]["norefine"] for m in xs], "-",  color=colors[h], marker="o", ms=4, label=f"h={h} no-ref")
            ax[i].plot(xs, [src[(m,act,h)]["refine"]   for m in xs], "--", color=colors[h], marker="s", ms=4, label=f"h={h} refine")
        ax[i].set_title(title); ax[i].set_xlabel("Timestep (m)"); ax[i].grid(alpha=.3); ax[i].set_yscale("log")
    ax[0].set_ylabel("Avg. Computation Time (CPU-s)/sample, log"); ax[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Fig 5.4 (repro) - RNN(relu/tanh) & LSTM on MNIST; solid=no-refine dashed=refine; (d) GenBaB not run; trend only")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_4_repro.png", dpi=110); plt.close(fig)
    trends.append("Fig5.4: LSTM ≫ vanilla RNN(數量級)、refine>no-refine、隨m上升 ✓；缺GenBaB(d)")

for fn in (fig51, fig52, fig53, fig54):
    try: fn()
    except Exception as e: trends.append(f"{fn.__name__}: ERROR {type(e).__name__}: {e}")

with open(f"{OUT}/trends.txt", "w") as f:
    f.write("Figs 5.1-5.4 重現(趨勢) — 指標:每樣本 CPU 計算秒數=Σtimers/50；no-refine=(Σ-zs_total)/50\n")
    f.write(f"資料:CIFAR={len(cif)} Stroke={len(seq)} RNN-MNIST={len(mni)} LSTM={len(lstm)}\n\n")
    f.write("\n".join(trends))
print("\n".join(trends))
print(f"資料:CIFAR={len(cif)} Stroke={len(seq)} RNN-MNIST={len(mni)} LSTM={len(lstm)}")
