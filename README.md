# SymCascade

**云-边-端三层下的符号-神经级联缓存推理调度器**
A cache-state-aware dual-path discriminative routing cascade for cloud-edge-end LLM agent inference.

> 论文研究代码 / Conference submission codebase
> 设计文档：[`docs/superpowers/specs/2026-06-27-symcascade-design.md`](docs/superpowers/specs/2026-06-27-symcascade-design.md)
> 实现计划：[`docs/superpowers/plans/2026-06-27-symcascade-implementation.md`](docs/superpowers/plans/2026-06-27-symcascade-implementation.md)

---

## 1. 这是什么

SymCascade 在云-边-端三层异构部署下，用**缓存状态感知的双路判别分流**替换传统单链逐层升级级联：缓存未命中后，在线 ROI 判别器依据 query 与**当前缓存状态**将请求分流到符号链（PDDL 骨架重规划 / LLM 生成 PDDL + Fast Downward 求解）或神经链（Gemma 置信度门控 / 云端 Gemini 兜底），让多数请求在抵达云端前被拦截。

### 核心创新

| 编号 | 名称 | 定位 |
|------|------|------|
| **I1** | 双路判别分流 + 缓存状态感知在线 ROI 判别器（自成长调度器） | 主创新 1 |
| **C2** | PDDL 骨架缓存 + 受约束重规划（域内两阶段验证） | 主创新 2 |
| **C1** | 符号规划器作求解后端（PDDL + Fast Downward） | 支撑性 |
| **C3** | Gemma-Gemini 同源置信度门控 | 消融变量 |
| **I4** | 在线级联 regret 界 `O(ε·T·(1-h(t))·α)` | 理论项 |

### 级联管线

```
请求到达
   │
   ▼
[L0: 语义缓存] ──命中──► 直接返回 (零 LLM)
   │未命中
   ▼
[D: 在线 ROI 判别器]  (输入: query emb ⊕ 缓存状态特征)
   │
   ├──档1: 缓存有匹配骨架──► [L1: PDDL 骨架受约束重规划]  (CPU, 零 LLM)
   │                              ├──成功──► 返回 + 更新判别器
   │                              └──失败──► 回退档2
   ├──档2: 可形式化无骨架──► [L2: LLM 生成 PDDL + FD 求解]  (1 边缘 LLM + CPU)
   │                              ├──成功──► 返回 + 抽取骨架存缓存 + 更新判别器
   │                              └──失败──► 回退档3
   └──档3: 不可形式化──► [L3: Gemma 置信度门控]  (边缘 GPU, thinking=False)
                                ├──高置信(conformal)──► Gemma 答
                                └──低置信──► [L4: 云端 Gemini 兜底]
```

---

## 2. 仓库结构

```
symcascade/
├── core/                    # 调度核心
│   ├── types.py             # Tier, Query, StageResult, ROI, CacheState
│   ├── stage.py             # Stage protocol + FallbackError
│   └── orchestrator.py      # Cascade: L0→D→L1/L2/L3→L4 + fallback + observer hook
├── cache/
│   ├── semantic_cache.py    # L0 语义缓存 (Query|str key, cosine)
│   └── skeleton_cache.py    # C2 骨架存储 + state_for() 缓存状态特征
├── discriminator/           # I1
│   ├── features.py          # build_features (query emb ⊕ cache state)
│   ├── roi_estimator.py     # GBDT ROI 回归 + refit()
│   ├── online_learner.py    # 滑动窗口 + EWMA + drift 检测
│   ├── conformal.py         # ConformalTierSelector (split conformal, 1-α)
│   ├── router.py            # DiscriminatorRouter (符号 bonus)
│   └── online_router.py     # OnlineDiscriminatorRouter (闭环: route+observe+refit+drift)
├── symbolic/                # C1+C2
│   ├── skeleton.py          # Skeleton/SkeletonAction/extract_skeleton
│   ├── matcher.py           # 编辑距离 + 谓词重叠
│   ├── replanner.py         # SkeletonReplanner (L1)
│   ├── pddl_gen.py          # PDDLGenStage (L2)
│   ├── pddl_llm.py          # LLMPDDLGenerator (LLM→PDDL + 可选 VAL)
│   └── fd_solver.py         # FDSolver (subprocess 调 fast-downward, GPL 隔离)
└── neural/                  # L3/L4 + C3 + 后端适配器
    ├── gemma_backend.py     # GemmaBackend (logprobs→confidence)
    ├── confidence_gate.py   # ConfidenceGate + ConfidenceCalibrator (C3)
    ├── embedder.py          # SentenceTransformerEmbedder (BGE-M3 → L0 embed_fn)
    ├── vllm_client.py       # VLLMGemmaClient (vllm.LLM → VLLMClient)
    └── cloud_stage.py       # GeminiClient + CloudStage (L4 floor)
```

---

## 3. 环境要求

### 硬件

| 组件 | 最低 | 推荐 |
|------|------|------|
| 符号链 (L1/L2/FD) | CPU | 任何 CPU |
| 神经链 L3 (Gemma 4 12B) | 1× GPU ≥24GB (FP16) | 1× A100 40GB / 量化后 24GB |
| 神经链 L4 (Gemini) | 网络 + API key | 同左 |
| Embedder (BGE-M3) | CPU (慢) | GPU 或 CPU |

> 无 GPU 时：L3 可降级为 llama.cpp GGUF 量化版（设计文档 5.1 备选），但需自行替换 `vllm_client.py`。

### 软件

- Python ≥ 3.11（实测 3.14）
- Fast Downward（提供 `fast-downward` CLI）
- vLLM（仅 L3 真实运行需要）
- sentence-transformers（仅 BGE-M3 embedder 真实运行需要）
- google-genai（仅 L4 真实运行需要）

---

## 4. 安装

### 4.1 克隆

```bash
git clone https://github.com/liaoweichu/symcascade.git
cd symcascade
```

### 4.2 Python 依赖

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`pyproject.toml` 默认只装 `numpy` + `scikit-learn` + `pytest`，足以跑**全部单元测试**（所有重依赖都是 lazy-import + 可注入）。

真实实验需额外装：

```bash
pip install vllm sentence-transformers google-genai mapie
```

### 4.3 Fast Downward（符号链必需）

```bash
pip install fast-downward      # 提供 fast-downward CLI
# 或从源码构建：https://github.com/aibasel/downward
fast-downward --help           # 验证
```

> **GPL 隔离**：Fast Downward 是 GPL-3.0，本项目通过 subprocess 调用（进程隔离），不 import 其 Python，保持 SymCascade 与 GPL 解耦。见 [`symcascade/symbolic/fd_solver.py`](symcascade/symbolic/fd_solver.py)。

### 4.4 环境变量

```bash
export GEMINI_API_KEY="your-key"          # L4 云端兜底
export HF_HOME="$HOME/.cache/huggingface" # 模型缓存（可选）
export CUDA_VISIBLE_DEVICES=0             # L3 GPU（可选）
```

### 4.5 验证安装

```bash
python -m pytest -q
# 期望: 95 passed
```

若 95 测试全绿，说明核心调度 + 全部适配器就绪。

---

## 5. 快速开始（端到端最小示例）

```python
import os
from symcascade.core.types import Query
from symcascade.core.orchestrator import Cascade
from symcascade.cache.semantic_cache import SemanticCache
from symcascade.cache.skeleton_cache import SkeletonCache
from symcascade.neural.embedder import SentenceTransformerEmbedder
from symcascade.neural.vllm_client import VLLMGemmaClient
from symcascade.neural.gemma_backend import GemmaBackend
from symcascade.neural.confidence_gate import ConfidenceGate
from symcascade.neural.cloud_stage import GeminiClient, CloudStage
from symcascade.symbolic.fd_solver import FDSolver
from symcascade.symbolic.replanner import SkeletonReplanner
from symcascade.symbolic.pddl_gen import PDDLGenStage
from symcascade.symbolic.pddl_llm import LLMPDDLGenerator
from symcascade.discriminator.online_router import OnlineDiscriminatorRouter
from symcascade.discriminator.roi_estimator import ROIEstimator
from symcascade.discriminator.conformal import ConformalTierSelector

# --- 后端 ---
embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-m3")
cache = SemanticCache(sim_threshold=0.9, embed_fn=embedder.as_embed_fn())
skel_cache = SkeletonCache(threshold=0.75)

gemma = GemmaBackend(VLLMGemmaClient(model_name="google/gemma-3-12b-it"))
l3 = ConfidenceGate(threshold=0.7).as_stage(gemma)
l4 = CloudStage(GeminiClient(api_key=os.environ["GEMINI_API_KEY"]))

fd = FDSolver(timeout=30)
DOMAIN_PDDL = open("path/to/domain.pddl").read()
l1 = SkeletonReplanner(cache=skel_cache, fd=fd, domain_pddl=DOMAIN_PDDL,
                       problem_pddl_fn=lambda q: "")  # Phase 1: ALFWorld adapter 填充
l2 = PDDLGenStage(llm=LLMPDDLGenerator(llm=gemma, domain_pddl=DOMAIN_PDDL),
                  fd=fd, cache=skel_cache, domain_pddl=DOMAIN_PDDL,
                  problem_pddl_fn=lambda q: "")

# --- 判别器 (I1) ---
estimator = ROIEstimator()
router = OnlineDiscriminatorRouter(
    estimator=estimator,
    conformal=ConformalTierSelector(alpha=0.1),
    online=...,  # 见 online_learner.OnlineLearner
    fallback_tier=Tier.L4,
)

# --- Cascade ---
cascade = Cascade(
    cache=cache, discriminator=router,
    l1=l1, l2=l2, l3=l3, l4=l4,
    cache_state_fn=lambda: skel_cache.state_for(...),
)
result = cascade.run(Query(text="put the apple in the fridge", embedding=embedder.embed("...")))
print(result.answer, result.success, result.cost)
```

> `...` 处依赖 ALFWorld 任务结构，完整可运行示例在 Phase 1 的 `experiments/alfworld/` 完成后给出。

---

## 6. 实验复现路线图

下表标注每个实验的**实现状态**。✅ 表示命令可直接运行；🚧 表示 harness 开发中（命令为预期形态）。

| Phase | 实验 | 状态 | 产物 |
|-------|------|------|------|
| 0 | 后端适配器 + 核心调度 | ✅ | 95 单元测试通过 |
| 1 | E1 级联效果（ALFWorld 主载体） | 🚧 | 云端调用↓% + 质量保持% |
| 1 | E6 自成长曲线（I1 卖点） | 🚧 | 缓存规模 vs 云端调用率单调下降曲线 |
| 2 | E4 消融（2³ 析因，I1/C2/C3） | 🚧 | 逐模块质量/成本/延迟变化表 |
| 3 | E1 基线对比（B1-B7） | 🚧 | 级联 vs 基线对比表 |
| 4 | E2/E3/E7 + 统计检验 + 出图 | 🚧 | Pareto 前沿 + 延迟 CDF + Sankey |
| 5 | E5 扩展 + C3 同源验证 | 🚧 | HotpotQA/GSM8K/GAIA 结果 |

---

## 7. 实验详细复现步骤

> 通用约定：所有实验脚本将位于 `experiments/<benchmark>/`，结果输出到 `results/<experiment>/`，配置在 `experiments/configs/`。统一入口 `python -m experiments.run --config <yaml>`。

### 7.1 数据准备

#### ALFWorld（符号规划主载体，验证 C2 受约束重规划）

```bash
pip install alfworld
alfworld-download          # 下载 ALFWorld 资源到 $ALFWORLD_DATA
export ALFWORLD_DATA="$PWD/data/alfworld"
```

ALFWorld 6 类任务（pick, clean, heat, cool, examine, pick-two）原生 PDDL，是 C2 域内迁移（同骨架不同初始状态）的主载体。

#### HotpotQA / GSM8K（神经链验证 C3）

```bash
# datasets 库自动下载
pip install datasets
python -c "from datasets import load_dataset; load_dataset('hotpotqa','distractor'); load_dataset('gsm8k','main')"
```

#### GAIA（综合诚实评测）

```bash
# 需从 HuggingFace 申请：https://huggingface.co/datasets/gaia-benchmark/GAIA
# 下载到 data/gaia/
```

### 7.2 Phase 1 — E1 级联效果 + E6 自成长曲线（ALFWorld）

**目的**：验证双路判别分流降本保质；展示"缓存填充 → 云端调用率单调下降"。

```bash
# 🚧 待 experiments/alfworld/ 实现
python -m experiments.alfworld.run \
    --config experiments/configs/alfworld_full.yaml \
    --seed 42 \
    --output results/e1_alfworld_full/
```

**预期产出**：
- `results/e1_alfworld_full/metrics.json`：云端调用率、质量（任务成功率）、平均级联深度、各档命中率
- `results/e1_alfworld_full/self_growth_curve.png`：缓存规模 vs 云端调用率（E6，I1 卖点曲线）
- `results/e1_alfworld_full/cascade_depth_hist.png`：级联深度直方图（目标平均深度 < 1.5）

**配置项**（`alfworld_full.yaml`）：
```yaml
benchmark: alfworld
n_tasks: 134
discriminator: {alpha: 0.1, retrain_every: 200, sliding_window: 2000}
skeleton_cache: {threshold: 0.75, topk: 3}
gemma: {model: "google/gemma-3-12b-it", thinking: false}
cloud: {model: "gemini-3-pro"}
seeds: [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # ≥10 次 for 统计检验
```

### 7.3 Phase 2 — E4 消融（2³ 析因）

**目的**：验证 I1/C2/C3 各组件贡献。按 spec 4.4 表格跑 7 配置。

```bash
# 🚧 待 experiments/ablation/ 实现
for cfg in full minus_i1_disc minus_i1_online minus_i1_cachestate \
           minus_c2_skel minus_c2_replan minus_c3_samesource; do
  python -m experiments.alfworld.run \
      --config experiments/configs/ablation_${cfg}.yaml \
      --output results/e4_ablation/${cfg}/
done
python -m experiments.ablation.summarize --input results/e4_ablation/ \
    --output results/e4_ablation/table.tex
```

**消融配置矩阵**（spec 4.4）：

| 配置 | I1 判别器 | C2 骨架 | C3 同源 |
|------|-----------|---------|---------|
| `full` | 在线+缓存状态 | ✓ | ✓ |
| `minus_i1_disc` | 随机分流 | ✓ | ✓ |
| `minus_i1_online` | 静态判别器 | ✓ | ✓ |
| `minus_i1_cachestate` | 在线无缓存特征 | ✓ | ✓ |
| `minus_c2_skel` | 在线+缓存状态 | ✗ | ✓ |
| `minus_c2_replan` | 在线+缓存状态 | 只缓存不复用 | ✓ |
| `minus_c3_samesource` | 在线+缓存状态 | ✓ | 异构门控 |

### 7.4 Phase 3 — E1 基线对比（B1-B7）

**目的**：与 7 个基线对比，证明 SymCascade 在质量保持下显著降云端调用。

```bash
# 🚧 待 experiments/baselines/ 实现
for baseline in b1_cloud_only b2_edge_only b3_frugalgpt b4_routellm \
                b6_apc b7_llm_dp; do
  python -m experiments.baselines.run \
      --baseline ${baseline} \
      --config experiments/configs/alfworld_full.yaml \
      --output results/e1_baselines/${baseline}/
done
python -m experiments.baselines.compare \
    --symcascade results/e1_alfworld_full/ \
    --baselines results/e1_baselines/ \
    --output results/e1_baselines/comparison.tex
```

**基线说明**（spec 4.3）：

| 基线 | 机制 | 主对照 |
|------|------|--------|
| B1 纯云端 | 全部送 Gemini 3 | 上界质量 |
| B2 纯边缘 | 全部送 Gemma 4 12B | 下界成本 |
| B3 FrugalGPT | 小→大 LLM 级联+蒸馏缓存 | 混合负载 |
| B4 RouteLLM | 学习型单次路由 | 混合负载 |
| B6 APC | Agentic Plan Caching（小 LLM 适配） | C2 直接对照 |
| B7 LLM-DP | LLM 生成 PDDL + FD 求解 | C1 符号链基线 |

### 7.5 Phase 4 — 统计检验 + 出图（E2/E3/E7）

**目的**：补齐显著性检验短板（spec 4.6 第三层），生成论文图表。

```bash
# 🚧 待 experiments/analysis/ 实现
# E2 成本-质量 Pareto
python -m experiments.analysis.pareto \
    --input results/e1_alfworld_full/ results/e1_baselines/ \
    --output results/e2_pareto/frontier.png

# E3 延迟分布
python -m experiments.analysis.latency \
    --input results/e1_alfworld_full/ \
    --output results/e3_latency/cdf.png

# E7 判别器分析（Sankey）
python -m experiments.analysis.discriminator \
    --input results/e1_alfworld_full/ \
    --output results/e7_sankey/flow.png

# 统计显著性（≥10 seeds 已在 Phase 1 跑完）
python -m experiments.analysis.significance \
    --input results/e1_alfworld_full/ results/e1_baselines/ \
    --tests wilcoxon bootstrap \
    --correction holm \
    --output results/significance.tex
```

**统计方法**（spec 4.6）：
- ≥10 次独立运行（不同种子），报告 mean±std
- Wilcoxon signed-rank test（配对非参数，不假设正态）
- Bootstrap CI（B=1000-5000）
- Holm 校正（多重比较）
- Cliff's delta（效应量）

### 7.6 Phase 5 — 扩展验证（C3 + 综合）

**目的**：在 HotpotQA/GSM8K 验证 C3 神经链门控；GAIA 做最终诚实评测；混合负载验证 I1+C2 跨任务类型复用。

```bash
# 🚧 待 experiments/extension/ 实现
python -m experiments.extension.run --benchmark hotpotqa --config experiments/configs/c3_ablation.yaml
python -m experiments.extension.run --benchmark gsm8k   --config experiments/configs/c3_ablation.yaml
python -m experiments.extension.run --benchmark gaia    --config experiments/configs/gaia_full.yaml

# 混合负载（I1+C2 主验证，spec 4.2）
python -m experiments.mixed.run \
    --config experiments/configs/mixed_3_7.yaml \
    --output results/e1_mixed/
python -m experiments.mixed.run \
    --config experiments/configs/mixed_5_5.yaml \
    --output results/e1_mixed_5_5/
python -m experiments.mixed.run \
    --config experiments/configs/mixed_7_3.yaml \
    --output results/e1_mixed_7_3/
```

**混合负载配比**（可形式化:不可形式化 = 3:7 / 5:5 / 7:3，spec 4.2 敏感性分析）。

---

## 8. 评测指标与评判方法

### 第一层：客观指标（避免 LLM-as-judge 噪声，spec 4.6）

| Benchmark | 指标 | judge |
|-----------|------|-------|
| ALFWorld | 任务成功率 | 程序化检查，无 judge |
| HotpotQA | EM / F1 | 与 ground truth 比对 |
| GSM8K | 数值 exact-match | 与 ground truth 比对 |
| GAIA | exact-match | 刻意避免 judge 保证可复现 |

### 第二层：Pareto 多目标

- 超体积 HV（bootstrap 95% CI，参考点取 nadir）
- Pareto 支配判断 + ε-指示子
- 呈现：Pareto 前沿图 + 采样数值表

### 级联系统特有指标（E6/E7）

- 各档命中率 / 升级率（升级率须 < 33-66%，spec 3.3）
- 级联深度直方图（目标平均深度 < 1.5）
- 成本/质量/延迟各层贡献分解（Sankey 图）
- **自成长曲线**：缓存规模 vs 云端调用率（I1 核心卖点，期望单调下降）

---

## 9. 已验证的核心能力（单元测试层面）

以下能力已由 95 个单元测试覆盖（`python -m pytest -q` 全绿）：

| 能力 | 测试文件 |
|------|----------|
| L0 语义缓存（exact + cosine 语义命中） | `tests/cache/test_semantic_cache.py` |
| C2 骨架抽取/匹配/缓存 + 缓存状态特征 | `tests/cache/test_skeleton_cache.py`, `tests/symbolic/test_skeleton.py`, `tests/symbolic/test_matcher.py` |
| L1 受约束重规划（骨架命中→FD） | `tests/symbolic/test_replanner.py` |
| L2 PDDL 生成 + FD 求解 + 骨架回填 | `tests/symbolic/test_pddl_gen.py`, `tests/symbolic/test_pddl_llm.py` |
| L3 Gemma 置信度门控（conformal 校准） | `tests/neural/test_gemma_backend.py`, `tests/neural/test_confidence_gate.py` |
| L4 云端兜底 | `tests/neural/test_cloud_stage.py` |
| I1 ROI 估计 + 在线学习 + drift 检测 | `tests/discriminator/test_roi_estimator.py`, `tests/discriminator/test_online_learner.py` |
| I1 conformal tier 选择 | `tests/discriminator/test_conformal.py` |
| I1 在线判别器闭环（route+observe+refit+drift 恢复） | `tests/discriminator/test_online_router.py` |
| Cascade 编排 + fallback 链 + observer hook | `tests/core/test_orchestrator.py` |
| 后端适配器（embedder/vllm/cloud, lazy-import 可注入） | `tests/neural/test_embedder.py`, `tests/neural/test_vllm_client.py` |
| e2e 集成（缓存命中/符号链/神经链/级联回退） | `tests/test_e2e_cascade.py` |

---

## 10. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 管线结构 | 双路判别分流（I1） | 原单链层序次优；判别器解决可形式化分流 |
| 判别目标 | 缓存状态感知在线 ROI | 回避二元判别 trivial、ROI 冷启动难、标签漂移三硬伤 |
| C2 载体 | 域内两阶段 | ALFWorld 证重规划，混合负载证模糊匹配；不碰跨域迁移 |
| C3 定位 | 降级为消融变量 | 同源红利可能被证伪，不押宝核心 |
| I4 | 在线 regret 界 | 原 ICML2025 定理因 I1 结构变化失效；regret 契合在线学习 |
| GPL | subprocess 隔离 Fast Downward | 进程隔离不触发 GPL 传染 |

完整决策记录见 spec 第 9 节。

---

## 11. 复现问题排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: sklearn` | 未装核心依赖 | `pip install -e ".[dev]"` |
| `fast-downward: command not found` | 未装 FD | `pip install fast-downward` |
| L3 OOM | GPU 显存不足 | 量化（AWQ/GPTQ）或换 llama.cpp GGUF |
| L4 401 | GEMINI_API_KEY 未设/无效 | `export GEMINI_API_KEY=...` |
| 测试因 `sentence_transformers`/`vllm` 失败 | 误装了真实后端 | 测试用 mock，不应触发真实 import；检查是否误改 lazy-import |
| skeleton 命中率低 | threshold 过高 | 调 `SkeletonCache(threshold=...)`，ALFWorld 建议从 0.75 起调 |

---

## 12. 许可证

本项目代码 Apache-2.0 兼容。Fast Downward 为 GPL-3.0，通过 subprocess 进程隔离调用，不传染本项目。

## 13. 引用

论文完成后补充 BibTeX。

---

## 14. Roadmap

- [x] Phase 0：后端适配器 + 核心调度（95 测试）
- [ ] Phase 1：ALFWorld harness（E1 + E6 自成长曲线）
- [ ] Phase 2：消融 E4（7 配置）
- [ ] Phase 3：基线 B1-B7
- [ ] Phase 4：统计检验 + 出图（E2/E3/E7）
- [ ] Phase 5：HotpotQA/GSM8K/GAIA + 混合负载
- [ ] I4：regret 界理论证明（纯推导，独立于代码）

> 设计/实现状态：核心调度与全部后端适配器已完成并通过测试；benchmark harness 与实验脚本按上述 Roadmap 推进。各实验命令在对应 Phase 完成后即可直接运行。
