# 可再現性審查報告 — ZeroSplitVerifier (Lin 2026)

> **被審對象**：`ZeroSplitVerifier/`（fork of `BensonFanNccu/ZeroSplitVerifier`），論文 `Lin2026Robustness.pdf`
> （“Robustness Verification of RNN with Abstraction Refinement”，NCCU Li-Jen Lin，2026，52 頁）。
> **審查者立場**：第三方可再現性審查。**全程未更動被審 repo 的任何原始程式碼**（master 與上游一致，可 `git diff` 驗證；
> 所有審查產物在 repo 外的 `review_workspace/`，計畫書在 fork 的 `review` 分支）。
> **狀態**：🟢 進行中（2025-06-19）。RNN 兩張表(5.3/5.1)已完整；LSTM(5.4) 7/16；圖待 grid 跑完生成。本檔會持續更新。

---

## 1. 結論摘要（Executive Summary）

| 項目 | 結論 |
|---|---|
| **程式能否重現論文趨勢** | **能**（RNN 已證實）。在「自建環境 + 自行重訓 + 自寫外部 driver」下，Table 5.3/5.1 的 ×/↑/% 與論文**逐格貼合**、核心趨勢清楚重現。 |
| **能否 exact 重現數字** | **不能**（結構性障礙，非程式錯）。論文**未報訓練 seed/超參數/環境/clean accuracy**（D5/D6），權重不可重建 → 數值必帶誤差。重現等級＝**趨勢重現**。 |
| **照文件直接跑能否成功** | **不能**：(a) auto_test 預設 grid≠論文(D3)；(b) `conda torch-env` 不存在、無 requirements 鎖版(環境)；(c) **多進程預設會 deadlock(D8)**。皆需審查者繞過。 |
| **整體可再現性評級** | **中等偏上**：方法與核心 verifier 健全、CLI 完整參數化使「不改碼重現論文 grid」可行；但**環境/訓練/資料/文件**多處缺口需手動補齊。 |

**一句話**：*這份程式能重現論文的「結論與趨勢」，但無法重現「確切數字」，主因是論文端的環境與訓練資訊缺失，而非程式缺陷；同時文件與預設配置有數處會讓人「照做卻跑不起來」的坑。*

---

## 2. 審查方法與紀律

- **唯讀紀律**：被審 `ZeroSplitVerifier/` 一個位元未改（README 要求「從 repo root 跑」，故 `cd` 進去執行，但所有寫出路徑都用絕對路徑導向 `review_workspace/`）。每步後 `git status` 為空。
- **重現策略（因 D3）**：不使用 repo 的 `auto_test_*.py`（其預設 grid 與論文對不上），改**自寫外部 driver**（`review_workspace/drivers/run_all.py`）迴圈論文 grid，逐一以完整 CLI 呼叫核心 verifier（`rnn_/lstm_zerosplit_verifier.py`）。
- **參數對齊論文§5.1**：`--N 50 --p 2`（L2）、`--eps-min 0.005 --eps-max 0.1`（step 0.001 為 `verify_evr` 內建，與論文一致）、`--max-splits 5`（＝論文「at most 31 refinements」＝深度 5 的二元樹 2⁵−1）。CPU（論文未提 GPU；`--cuda` 為 opt-in）。
- **指標換算**：論文 `×`＝觸發 refinement 的樣本數＝程式 flag `zs_better`+`both_fail`；`↑`＝救回數＝`zs_better`；`%`＝↑/×。

---

## 3. 實驗環境（論文未提供，這是我們補記的）

論文對環境**零記載**（CPU/GPU/RAM/OS、Python/PyTorch 版本、seed 全 NOT STATED；僅一句「implemented in PyTorch on top of POPQORN」）。我們的環境：

- 硬體：32 cores / 91 GB RAM；驗證一律 **CPU**（torch CPU build）。
- `review_workspace/.venv`（系統 Python 3.12.3）：**torch 2.12.1+cpu、torchvision 0.27.1+cpu、numpy 2.4.4、shap 0.52.0、pandas 3.0.3、openpyxl 3.1.5、matplotlib 3.11.0、scikit-learn 1.9.0、numba 0.65.1**。
- 不安裝 `torchtext`（僅與論文無關的 news-classification 模組相關）。
- > 註：我們的套件版本比 repo 寫作年代新很多；RNN/LSTM 驗證、訓練、解析路徑實測皆相容（numpy2/pandas3 無踩雷）。

---

## 4. 重現結果（對照論文五表）

### 4.1 Table 5.3 — Vanilla RNN @ MNIST　✅ 完整（32/32）
**一致性**：平均 |Δ×|=**2.4**（max 6）、平均 |Δ%|=**11.2 pp**、↑ 落在論文±2 內者 **24/32**。

```
每格 ×/↑/%      h4          h8          h16         h32
 m=1 RELU  論 9/1/11  2/0/0   0/0/-   0/0/-     我 11/1/9  6/0/0  0/0/-  0/0/-
 m=2 TANH  論 22/7/32 12/4/33 5/5/100 2/2/100   我 28/10/36 17/6/35 6/6/100 5/5/100
 m=4 TANH  論 22/7/32 24/20/83 17/17/100 8/8/100 我 28/11/39 27/21/78 18/18/100 9/9/100
 m=7 TANH  論 45/24/53 35/30/86 27/23/85 41/41/100 我 47/21/45 36/28/78 30/26/87 41/41/100
 m=7 RELU  論 43/8/19 25/16/64 10/9/90 4/2/50    我 45/7/16 24/9/38 14/8/57 6/3/50
```
**分析**：`×`（baseline 失敗數）幾乎完全重現 → 驗證 POPQORN baseline 與取樣機制成功重現；`↑`/`%` 同量級，**tanh 重現特別好**、relu 噪訊較大（如 m7-relu h8：64% vs 38%）。差異完全來自 **D1**（seed=2025 隨機 50 vs 論文前 50）+ **D5**（重訓權重不同）。

### 4.2 Table 5.1 — Vanilla RNN @ CIFAR-10　✅ 完整（32/32）
**一致性**：平均 |Δ×|=**0.1**（max 2，幾乎完全相同）、平均 |Δ%|=**5.1 pp**、↑ 落在±2 內者 **21/32**。**比 MNIST 更穩**（×飽和於 50，受 D1 取樣影響小）。

```
每格 ×/↑/%       h16         h32         h64         h128
 m=8 RELU  論 43/24/56 47/33/70 50/32/64 50/25/50  我 44/26/59 47/28/60 50/36/72 48/28/58
 m=12 TANH 論 50/34/68 50/27/54 50/15/30 50/11/22  我 50/35/70 50/28/56 50/15/30 50/14/28
 m=32 TANH 論 50/16/32 50/10/20 50/4/8  50/1/2     我 50/16/32 50/10/20 50/9/18 50/2/4
```
**分析**：**論文核心宣稱「refinement rate % 隨序列長度 m 增大而下降」清楚重現**（tanh h128：m8→m32 由 50%→2%）。這是本審查對論文主結論最強的正面證據。

### 4.3 Table 5.4 — LSTM @ MNIST　🔄 進行中（7/16）
**初步發現（待補完）**：救回率 `%` 仍同量級（m1h4: 25 vs 25；m4h4: 39 vs 48），**但 `×`（觸發數）系統性偏低**（論文 44–50，我們 6–28）。
```
每格 ×/↑/%       h4          h8          h16   h32
 m=1  論 44/11/25 40/22/55 27/23/85 16/14/88   我 20/5/25  6/5/83  (跑) (跑)
 m=4  論 50/24/48 50/23/46 50/24/48 50/18/36   我 28/11/39 25/9/36 (跑) (跑)
```
→ **這是 D5 的放大、且本身是一條審查發現**：RNN 兩表的 × 幾乎吻合，LSTM 的 × 卻明顯偏離，顯示 **LSTM 對「訓練不可重現」更敏感**（LSTM 訓練 50 epochs + dropout 0.5 + 無 seed，權重與論文差異更大）。LSTM 也**極耗時**（單配置 1.5–5.5 小時，h4/m7 達 19,600 秒），印證論文「LSTM 比 vanilla RNN 慢 2–3 個數量級」。完整 16 格預計再數日。

### 4.4 Table 5.2（MNIST Stroke）／Table 5.5（GenBaB）— 未納入
- **5.2**：需外部 stroke-sequence 資料集（Liwicki 2012），不自動下載 → **標記「需外部資料、暫緩」**。
- **5.5**：GenBaB(α,β-CROWN) 為外部 repo、橋接腳本路徑寫死 `/home/sausage/...` → **只記錄、不實跑**。

### 4.5 圖 Figs 5.1–5.4（runtime）— 待生成
verifier 每次 run 已將 timing（`popqorn`=baseline、`zs_total`=refinement、`shap`）寫入 JSON。grid 跑完後將重畫並**比對趨勢**（絕對秒數因論文未記載硬體不可比，D6）。

---

## 5. 核心發現：paper ↔ code 矛盾與缺口（D1–D8）

> 這是本審查最有價值的產出。每條：陳述／證據／對重現的影響／狀態。

| # | 發現 | 證據 (file:line) | 影響 | 狀態 |
|---|---|---|---|---|
| **D1** | 取樣：論文§5.1.2「前 50 個」，程式為 **seed=2025 + DataLoader shuffle**＝隨機 50。〔主路徑不過濾正確性，與論文一致；「filter correct」僅在 GenBaB 路徑〕 | `utils/sample_data.py:10,16,28,69` | 中-高：改變哪 50 個 → ×/↑/% 偏移（可重現可記錄） | 已釐清 |
| **D2** | ε 範圍：auto_test 各 driver 預設不一（0.03/0.05/0.1…）。**但 step=0.001 內建於 `verify_evr`、與論文一致**；range 可由 CLI 對齊。 | `verify_evr`(rnn:642/lstm:365) | 低（已解）：driver 帶 `--eps-min/max` 即精確對齊 | 已解 |
| **D3** | `auto_test_*.py` 預設 grid ≠ 論文（hidden/timestep 都不同，為縮減版開發配置）。 | `auto_test_*.py:10-18` | 高：不能直接跑 auto_test 重現論文 → 須自寫 driver | 已繞過 |
| **D4** | 深度預算：auto_test 不一致。**`--max-splits`＝遞迴深度，論文「31 refinements」＝深度 5**。 | `rnn_..._verifier.py:410-498` | 低（已解）：統一 `--max-splits 5` | 已解 |
| **D5** | **訓練不可重現**：所有 `train_*.py` 無 seed，論文未報超參數/clean accuracy。 | `grep manual_seed train_*` → 空 | **致命**：權重不同 → 數值必偏離（LSTM 尤甚，見 4.3）→ 只能趨勢重現 | 結構性 |
| **D6** | **環境零記載**：硬體/版本/seed 全無。 | 論文全文 | 中：runtime 絕對值不可重現；我們補記環境（§3） | 結構性 |
| **D7** | timestep 三方打架：`train_rnn_mnist_seq.py` 預設 ts=55、README ts50、論文 30/35/40/45。 | `train_rnn_mnist_seq.py:98` | 低-中 | 記錄 |
| **D8** | **多進程死結**：`mp.Pool` 未設 start method → Linux 預設 `fork`，在 torch/OpenMP(+LSTM 的 numba) 多執行緒後 fork → **deadlock**。**照 README/auto_test 預設（n-workers>1）跑會直接卡死。** | `rnn:669`、`lstm:417` | 對使用性=高；對我們=低（已解） | **免改碼解法**：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`（LSTM 再加 `NUMBA_NUM_THREADS=1`）→ fork-pool 正常且取得真多進程平行 |

### 5b. 正面事實（加分）
- ✅ 核心 verifier **CLI 完整參數化** → 不改碼即可重現論文 grid（本審查可行的關鍵）。
- ✅ LSTM 分裂點 **LUT 已 commit**（`lookup_tables/*.pkl`），免離線重建。
- ✅ ε step、max-splits 語意、untargeted top-1 認證、三資料集 dispatch、CPU 預設 — 皆與論文一致。
- ✅ `--toy-rnn` 零依賴 smoke test 存在。

---

## 6. 可再現性分級（per-component）

| 元件 | 等級 | 說明 |
|---|---|---|
| 核心 verifier（RNN/LSTM）| 🟢 **可重現** | CLI 健全；趨勢重現良好（5.3/5.1 已證） |
| Table 5.3 / 5.1（RNN）| 🟢 **趨勢可重現** | 逐格貼合、核心趨勢成立 |
| Table 5.4（LSTM）| 🟡 **可重現但高成本 + 偏離較大** | 極耗時（數日）；× 因 D5 系統性偏低 |
| 環境重建 | 🟠 **需手動** | 無 requirements/env 鎖版；`torch-env` 不存在 → 自建 |
| 訓練/權重 | 🔴 **不可重現** | 無 seed、無超參數（D5） |
| 照 README/auto_test 直接跑 | 🔴 **會失敗** | D3 grid 不符 + D8 死結 |
| Table 5.2 / 5.5 | ⚪ **未納入** | 需外部資料/外部 repo |
| Figs 5.1–5.4 | 🟡 **趨勢可重現** | 絕對值不可（D6）；趨勢待生成 |

---

## 7. 給作者的可再現性建議

1. **附 `requirements.txt`/`environment.yml` 並鎖版**；於論文補記硬體/OS/Python/PyTorch 版本（解 D6）。
2. **所有 `train_*.py` 設 random seed，並釋出訓練好的 checkpoint** ＋ 報 clean accuracy（解 D5，這是 exact 重現的唯一途徑）。
3. **修多進程死結（D8）**：`mp.set_start_method('spawn')` 或在 fork 前限制執行緒；否則 README 預設指令會 deadlock。
4. **讓 `auto_test_*.py` 預設 grid＝論文表格**，或提供「論文重現」開關（解 D3）。
5. 統一 timestep 文件（D7）；移除/標註寫死路徑（`/home/sausage/...`、`C:/Users/...`）。

---

## 8. 附錄

- **可重現指令**：見 `review_workspace/drivers/run_all.py` 與 CLAUDE.md §6b/§11。
- **完整逐格數據**：`review_workspace/results/{rnn_mnist,rnn_cifar10,lstm_mnist}/**/*.json`。
- **重現計畫書**（對照論文§5）：fork `review` 分支 `review/REPRODUCTION_PLAN.md`。
- **執行進度**：`review_workspace/logs/progress.jsonl`、`driver.log`。

*（本報告為進行中草稿；LSTM 5.4 完整結果與 Figs 5.1–5.4 將於 detached run 完成後補入。）*
