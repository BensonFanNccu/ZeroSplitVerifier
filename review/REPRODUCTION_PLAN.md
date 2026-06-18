# 重現計畫書 — 對照論文第 5 章（Experiments）

> **目的**：在開跑任何長實驗前，把「我們要重現什麼、怎麼跑、跟論文哪裡一致、哪裡必然有落差」
> 白紙黑字寫清楚，請使用者**逐項確認**後才動工。
> **被審 repo（`ZeroSplitVerifier/`）全程唯讀**；一切輸出落在 `review_workspace/`（下稱 `$WS`）。
> 論文來源：`ZeroSplitVerifier/Lin2026Robustness.pdf`（52 頁）。本計畫對照其 **§5.1 實驗設定、Tables 5.1–5.5、Figs 5.1–5.4**。

---

## A. 論文的實驗協定（§5.1，我們據此設定參數）

| 項目 | 論文設定（出處） | 我們的對齊方式 | 一致? |
|---|---|---|---|
| 擾動範數 | per-timestep **L2**（§5.1.2） | `--p 2` | ✅ |
| ε 掃描 | **0.005 → 0.1，step 0.001**（§5.1.3，EVR sweep, Li et al. 2023） | `--eps-min 0.005 --eps-max 0.1`；step=0.001 為 `verify_evr(precision=0.001)` 內建（rnn:642 / lstm:365） | ✅ 完全一致 |
| 樣本數 | 每配置 **50** 個（§5.1.2） | `--N 50` | ✅ |
| 樣本選取 | 「**前 50 個**、**不依正確性過濾**」（§5.1.2） | 程式實為 **seed=2025 隨機 50、不過濾**（`utils/sample_data.py`）→ 見 **§D 落差 D1** | ⚠️ 順序不同 |
| 細化預算 | top-5 候選、**「at most 31 refinements」**（§5.1.4） | `--max-splits 5`（= 遞迴深度 5，2⁵−1=31，`_evr_recursive` rnn:410-498） | ✅ 等價 |
| 細化目標 | 跨零之 pre-activation 神經元；SHAP(IG 式) 依時間序選（§4.3, §5.1.4） | verifier 內建（`locate_timestep_shap`），不改 | ✅ |
| 認證準則 | top-1 標籤保持：真類 logit 下界 > 所有他類上界（Eq 4.1） | verifier 內建（untargeted，以 top-1 **預測**類對全部他類；rnn `_is_verified`） | ✅ |
| 算力 | **未記載**（§5.1.5 僅「PyTorch on POPQORN」） | **CPU**（`--cuda` 不帶；理由見 §D 落差 D6） | ⚠️ 論文無從對齊 |
| 隨機種子 / 訓練超參數 / clean accuracy | **全部未記載** | 見 **§D 落差 D5**（致命：模型不可重現） | ⚠️ |

---

## B. 重現範圍（哪些表要跑、哪些暫緩）

| 論文表/圖 | 內容 | 本次處置 |
|---|---|---|
| **Table 5.3** | Vanilla RNN @ **MNIST(image)**，m∈{1,2,4,7} × {ReLU,Tanh} × h∈{4,8,16,32} | ✅ **跑**（最小、最先，當驗證樣板） |
| **Table 5.1** | Vanilla RNN @ **CIFAR-10**，m∈{8,12,24,32} × {ReLU,Tanh} × h∈{16,32,64,128} | ✅ **跑**（成本中-高） |
| **Table 5.4** | **LSTM** @ MNIST，m∈{1,2,4,7} × Sigmoid/Tanh × h∈{4,8,16,32} | ✅ **跑**（成本最高，可能 >1 天） |
| **Table 5.2** | Vanilla RNN @ **MNIST Stroke**（pen 軌跡），m∈{30,35,40,45} × {ReLU,Tanh} × h∈{16,32,64,128} | ⏸️ **暫緩**：需外部 stroke 資料集（Liwicki 2012），不自動下載 |
| **Table 5.5** | GenBaB(α,β-CROWN) @ MNIST LSTM（外部 baseline） | 📝 **只記錄**：外部 repo、路徑寫死 `/home/sausage/...`，不實跑 |
| **Fig 5.1–5.4** | runtime / time-ratio / SHAP-time 曲線 | 📈 **僅趨勢**：絕對秒數因硬體未記載不可重現（D6）；比對「趨勢方向」（refinement 較慢、LSTM≫RNN、tanh>relu、SHAP 時間隨 m 增） |

---

## B2. 論文「圖」的重現（Figs 5.1–5.4）— 使用者目標明列

論文的圖都是 **runtime / timing 曲線**。因論文未記載硬體（D6），**絕對秒數不可比**，
但**圖的形狀/趨勢可重現**：verifier 每次 run 都會把 `timing_stats` 寫進 JSON
（`popqorn`=baseline bound 時間、`zs_total`=refinement 時間、`shap`=SHAP 計算時間），
我們聚合各配置後用 matplotlib 重畫同款圖，與論文並排比對趨勢。

| 圖 | 論文內容 | 我們的重現方式 | 比對層級 |
|---|---|---|---|
| **Fig 5.1** | baseline vs refinement 平均時間 × m（分 h、act） | 由各配置 `popqorn` / `zs_total` 平均時間繪製 | 趨勢（refinement 較慢、隨 m/h 上升） |
| **Fig 5.2** | 時間比 refinement/baseline × m | `zs_total` / `popqorn` 比值 | 趨勢（比值恆 >1） |
| **Fig 5.3** | SHAP 計算時間 × m | `shap` 平均時間 | 趨勢（隨 m 上升） |
| **Fig 5.4** | MNIST 上 RNN(relu/tanh)/LSTM 平均時間 × m,h | 同上、跨網路族 | 趨勢（LSTM ≫ vanilla RNN） |

- **產出**：`$WS/results/figures/fig5_{1..4}_repro.png` ＋ 一份「趨勢一致/不一致」對照說明。
- **作法**：driver 全程保留每配置 timing；grid 跑完後由 `$WS/drivers/make_figures.py` 一次生成。
- **誠實限制**：絕對秒數受我們的 CPU（torch 2.12+cpu）影響，不等於論文；**只主張趨勢重現**。

---

## C. 逐表執行矩陣（精確指令）

通用 verifier 參數（對齊 §A）：`--N 50 --p 2 --eps-min 0.005 --eps-max 0.1 --max-splits 5`（CPU，不帶 `--cuda`）。
所有 `--work-dir / --save-dir` 指向 `$WS`，repo 唯讀。

### C-1. Table 5.3 — Vanilla RNN @ MNIST　（32 配置 = 4 h × 4 m × 2 act）
```bash
# 訓練（每配置一個模型；MNIST 自動下載到 ../data/mnist）
python vanilla_rnn/train_rnn_mnist_classifier.py --hidden-size {4,8,16,32} --time-step {1,2,4,7} --activation {relu,tanh}
#   訓練超參數（程式內定，論文未報）：SGD lr=0.01 momentum, epochs=20, batch=64
# 驗證
python vanilla_rnn/rnn_zerosplit_verifier.py --dataset mnist \
    --hidden-size H --time-step M --activation A \
    --work-dir $WS/models/mnist_classifier/rnn_M_H_A/ \
    --N 50 --p 2 --eps-min 0.005 --eps-max 0.1 --max-splits 5 \
    --save-dir $WS/results/rnn_mnist/
```

### C-2. Table 5.1 — Vanilla RNN @ CIFAR-10　（32 配置 = 4 h × 4 m × 2 act）
```bash
python vanilla_rnn/train_rnn_cifar10.py --hidden-size {16,32,64,128} --time-step {8,12,24,32} --activation {relu,tanh}
#   訓練超參數（程式內定）：Adam lr=1e-3, epochs=30, batch=64
python vanilla_rnn/rnn_zerosplit_verifier.py --dataset cifar10 \
    --hidden-size H --time-step M --activation A \
    --work-dir $WS/models/cifar10_classifier/rnn_M_H_A/ \
    --N 50 --p 2 --eps-min 0.005 --eps-max 0.1 --max-splits 5 \
    --save-dir $WS/results/rnn_cifar10/
```

### C-3. Table 5.4 — LSTM @ MNIST　（16 配置 = 4 h × 4 m；Sigmoid/Tanh 為 LSTM 內含閘）
```bash
python lstm/train_mnist_lstm.py --hidden-size {4,8,16,32} --time-step {1,2,4,7}
#   訓練超參數（程式內定）：Adam lr=1e-3, epochs=50, batch=128, dropout=0.5, StepLR(20,0.1)
python lstm/lstm_zerosplit_verifier.py --dataset mnist \
    --hidden-size H --time-step M \
    --work-dir $WS/models/mnist_lstm/lstm_M_H/ \
    --N 50 --p 2 --eps-min 0.005 --eps-max 0.1 --max-splits 5 \
    --lut-dir ./lookup_tables \
    --save-dir $WS/results/lstm_mnist/
```
> ⚠️ 上述 `--work-dir` 的確切資料夾命名、以及 train 腳本實際存模型的路徑，**會在「Phase 3 最小端到端驗證」當場鎖定**（先用 MNIST RNN h=4/ts=1 跑通，確認 train↔verifier 接法，再一般化成 driver）。

---

## D. 對論文的「已聲明落差」（誠實揭露；這就是審查的重點產物）

| # | 落差 | 對數值的影響 | 我們如何處理 |
|---|---|---|---|
| **D1** | 樣本選取：論文「前 50」 vs 程式「seed=2025 隨機 50」 | 改變「哪 50 個」→ ×/↑/% 偏移 | **照程式實況跑**（不改 code），逐格標注；偏移可重現、可記錄 |
| **D5** | 訓練無 seed + 論文無超參數/accuracy → 自行重訓的權重 ≠ 論文權重 | 近邊界樣本集不同 → ×/↑/% 必然偏移 | **降級為「趨勢重現」**：不追求逐格數值相等，比對趨勢一致性 |
| **D6** | 論文未記載硬體/軟體版本 | runtime 絕對值不可重現 | Fig 只比趨勢；完整記錄我們的環境（torch 2.12.1+cpu 等，見 CLAUDE.md §5c） |

> 其餘 D2（ε 範圍）、D4（深度預算）已透過 §A 的參數對齊**解決**；D3（auto_test 預設 grid≠論文）以「自寫外部 driver 直呼 verifier」**繞過**，皆不改 repo。

---

## E. 指標與比對方法

程式輸出每樣本一個 flag（README §Result Flags）：`pq_all_pass` / `zs_better` / `both_fail`。
對映到論文的 ×／↑／%（§4 指標定義）：

| 論文符號 | 意義 | 由程式 flag 計算 |
|---|---|---|
| `×` | refinement 被觸發的樣本數（= baseline 在某 ε 失敗） | `count(zs_better) + count(both_fail)` |
| `↑` | 被 refinement 救回的樣本數 | `count(zs_better)` |
| `%` | refinement rate（論文主打指標） | `↑ / ×` |
| （不計入×） | baseline 全程通過 | `pq_all_pass` |

**比對方式**：每格做「論文值 vs 我們值」並排表；因 D1/D5，**判定標準為趨勢一致**，重點檢驗論文的核心宣稱：
1. refinement rate `%` **隨 sequence length m 增大而下降**（累積鬆弛誤差）；
2. ZeroSplit 能救回**非平凡比例**的 baseline 失敗樣本（% 顯著 > 0）；
3. 趨勢在不同 hidden size 下大致成立。
（會用 repo 的 `parse_evr.py` / `parse_evr_lstm.py` 彙整成 xlsx，與論文表並排。）

---

## F. 與論文「完全一致」的保證項（給你快速核對）
- ✅ ε 掃描 = 0.005→0.1 step 0.001（逐點，與論文同）
- ✅ N=50、L2 (p=2)
- ✅ max-splits=5（= 論文「≤31 refinements」）
- ✅ top-5 SHAP 候選、跨零篩選、時間序細化（verifier 內建，未改）
- ✅ 認證準則 = top-1 標籤保持、untargeted、對全部他類取 margin
- ✅ 全 CPU、單機

## G. 執行順序與成本預估（>1 天風險所在）
1. **Table 5.3（MNIST RNN）** — 最小，先跑（也當 driver 樣板）。預估數十分鐘級。
2. **Table 5.1（CIFAR-10 RNN）** — h128/m32 + tanh+refinement 最貴（論文 Fig 5.1 約 ~30s/樣本）→ 數小時級。
3. **Table 5.4（LSTM MNIST）** — 論文 Fig 5.4 指 LSTM refinement 比 vanilla RNN **貴 2–3 個數量級** → **這部分最可能 >1 天**。
- driver 採「便宜→昂貴」排序、冪等續跑、per-config timeout（細節見 CLAUDE.md §14）。

## H. repo 完整性保證
- 被審 `ZeroSplitVerifier/` **零寫入**（每步後 `git status` 應為空）。
- 所有模型/資料/結果/log/driver 落在 `$WS`；以 `--work-dir/--save-dir/--data-dir/MODEL_ROOT` 外置。
- 不安裝 `torchtext`（僅與論文無關的 news-classification 模組相關）。

## I. 「算重現成功」的驗收標準（趨勢層級）
- [ ] 三張在範圍內的表（5.3/5.1/5.4）每格都有「論文 vs 我們」數值。
- [ ] 論文三項核心趨勢（§E）在我們的結果中成立或被明確反證。
- [ ] 所有偏離以 D1/D5/D6 解釋，無「不明原因」落差。
- [ ] **Fig 5.1–5.4 以我們的 timing 重畫，趨勢與論文一致或被明確反證（§B2）。**
- [ ] 全程 repo 唯讀無污染。

## J. 請你確認的項目（sign-off）
1. □ 範圍（§B）：跑 5.3/5.1/5.4、暫緩 5.2、只記錄 5.5 ── 同意？
2. □ 參數對齊（§A/§F）：eps 0.005–0.1@0.001、N=50、L2、max-splits=5、CPU ── 符合你讀到的論文第 5 章？
3. □ 落差揭露（§D）：D1/D5/D6 接受以「趨勢重現」處理？
4. □ 指標換算（§E）：×/↑/% 由 flags 計算的方式正確？
5. □ 任何你希望加跑/改動的配置（例如某張表想加做某個 hidden）？

> 確認後我才會：鎖定 train↔verifier 接法（Phase 3）→ 寫 `run_all.py` → tmux detach 啟動（Phase 4）。
