# Uncertainty-gated selection for block-sparse attention

Reference implementation, paper, and experiment results for the *value-of-information router* — a backbone-agnostic, single-hyperparameter add-on for block-sparse attention selectors that lifts selector quality on multi-hop and query-latent long-context retrieval.

> **Status:** research preview. The method, the kernel, and the experimental results in `paper/summary_for_review.pdf` are stable; the paper text is in late draft form (abstract and conclusion still tinted red). Numbers reported below match the JSONs in `results/`.

## TL;DR

Block-sparse attention selects which key blocks each query attends to via a per-query top-$k$ rule. When the $k$-th and $(k{+}1)$-th block are nearly tied, that cutoff is a coin flip — and dropping the wrong one loses an answer hop with no recovery. We measure the normalised cutoff margin $\sigma$ per Q-tile and, for the bottom $q$-fraction of tiles per layer, double the attended set. The rule is independent of how block scores are computed and composes with existing scoring backbones.

**Headline at 32K, Qwen2.5-14B-Instruct** (paired recall — fraction of dense-correct examples the sparse policy also solves):

| backbone | policy              | RULER NIAH-multikey ($n_{dc}=100$) | LongBench-v2 medium ($n_{dc}=40$) |
|----------|---------------------|---:|---:|
| —        | dense (ceiling)     | 1.00 | 1.00 |
| K-mean   | top-$k$             | 0.51 | 0.47 |
| K-mean   | router              | 0.63 | 0.60 |
| K-max    | Quest               | 0.93 | 0.68 |
| K-max    | **router-on-Quest** | **0.98** | **0.75** |

Router-on-Quest vs top-$k$ on LB-v2 medium ($n=215$, the full dataset subset): **+28 pp paired, McNemar $p<0.01$** — the first statistically clean composition lift on a multi-hop benchmark.

**Cross-model + long context on Qwen2.5-7B-Instruct-1M** (paired recall, RULER NIAH-multikey, $n_{dc}=100$):

| context | top-$k$ | Quest | **router-on-Quest** | dense |
|---|---:|---:|---:|---:|
| 32K  | 0.28 | 0.94 | **1.00** | 1.00 |
| 128K | 0.09 | 0.77 | **0.81** | 1.00 |

The Quest backbone is essentially model-invariant ($0.93 \to 0.94$ across the two models on RULER NIAH at 32K). Mean-pool selection collapses at 128K; the Quest backbone's lift over top-$k$ widens with context ($+66 \to +68$ pp).

**Speed.** The fused selection-plus-kernel pipeline runs at $0.97\times$ dense at 32K, $0.76$–$0.85\times$ at 64K, and $0.57$–$0.62\times$ at 128K (Qwen-7B-1M, A100-80GB). The crossover widens structurally with context length: attention is $35\%$ of dense prefill at 32K, $51\%$ at 64K, $63\%$ at 128K.

See `paper/summary_for_review.pdf` for the full method, math, and discussion.

## What's here

```
release/
├── paper/                     # the paper (.tex + .pdf)
├── src/                       # all method and harness code
└── results/                   # JSON dumps from the experiments cited in the paper
```

### `src/`

| file | role |
|---|---|
| `sparse_attention.py` | selector logic — per-tile scoring (K-mean / Quest K-max), the σ router, dispatch into the kernel |
| `triton_block_attn.py` | the Triton block-sparse attention kernel (FlashAttention-style online softmax over per-tile `kv_idx`) |
| `poc_core.py` | evaluation / sweep harness; computes paired recall, hit rate, etc. |
| `pointer_haystack.py` | the **Pointer-Chase Haystack** (PCH) diagnostic benchmark generator |
| `ruler_tasks.py` | RULER NIAH-multikey + VT bucket generation |
| `longbench_tasks.py` | LongBench v1 + v2 loaders, scoring, and MC parsing |
| `result_cache.py` | content-addressed cache for cheap reruns |
| `modal_app.py` | Modal entrypoint for the PCH evaluation |
| `modal_app_ruler.py` | Modal entrypoint for RULER |
| `modal_app_longbench.py` | Modal entrypoint for LongBench v1 |
| `modal_app_longbench_v2.py` | Modal entrypoint for LongBench v2 |
| `modal_app_profile.py` | CUDA-event prefill profiler (32K / 64K / 128K efficiency tables) |

### `results/`

JSON dumps for the experiments cited in the paper. Each record is one (task, policy, example) trial; see `poc_core.py` for the schema.

**RULER NIAH-multikey + VT, Qwen2.5-14B-Instruct, $n=100$, 32K** (paper Table 2):
- `ruler_smoke_20260607_161126.json` — dense / top-$k$ / router (K-mean)
- `ruler_smoke_20260607_193623.json` — Quest baseline added
- `ruler_smoke_20260607_222700.json` — router-on-Quest added (full panel)

**LongBench-v2 medium, Qwen2.5-14B-Instruct, $n=215$, 32K** (paper Table 3 — full dataset subset):
- `longbench_v2_20260611_004311.json` — dense / top-$k$ / Quest / router (K-mean)
- `longbench_v2_20260611_083950.json` — router-on-Quest added

(Earlier $n=100$ runs at the same context — `longbench_v2_20260607_*.json` — are kept for historical reference; the paper's headline LB-v2 numbers come from the $n=215$ pair above.)

**Budget-match ablation, RULER NIAH-multikey, $n=100$, 32K** (paper Table 4):
- `ruler_smoke_20260609_204524.json` — Quest swept at $k_{\text{budget}} \in \{33, 40, 47, 52, 66\}$ + router-on-Quest baseline. Decomposes the router's $+5$ pp lift over Quest into $\approx+3$ pp from average budget vs $\approx+2$ pp from selectivity.

**Cross-model panel, Qwen2.5-7B-Instruct-1M, $n=100$, 32K** (paper §5.7):
- `longbench_v2_20260611_104807.json` — LB-v2 medium, mean-router
- `longbench_v2_20260611_111212.json` — LB-v2 medium, router-on-Quest added
- `ruler_smoke_20260611_115759.json` — RULER NIAH+VT, mean-router
- `ruler_smoke_20260611_121350.json` — RULER NIAH+VT, router-on-Quest added

**Long context, Qwen2.5-7B-Instruct-1M, 128K** (paper §5.8):
- `ruler_smoke_20260611_164405.json` — RULER NIAH+VT, $n=100$, mean-router
- `ruler_smoke_20260611_232830.json` — router-on-Quest added (re-run in detached mode after the first attempt's client lost connection)
- `profile_20260611_234501.json` — CUDA-event prefill profile, $n=8$ per task

**Earlier profiles** (paper §5.6):
- `profile_20260601_204936.json` — 32K
- `profile_20260601_211844.json` — 64K

## Requirements

- Python 3.10+
- CUDA-capable GPU with sufficient memory for the chosen model:
  - Qwen2.5-14B-Instruct: A100-80GB used for all reported numbers; A100-40GB works at 16K, marginal at 32K
  - Qwen2.5-7B-Instruct-1M: A100-80GB sufficient up to 128K (requires the `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` env var; already set in the Modal image)
- See `requirements.txt`. Key pins:
  - `datasets==2.21.0` (LongBench v1 ships a script-based loader that breaks under `datasets>=3`)
  - `torch>=2.3`, `triton>=2.3`
  - `transformers>=4.45` (for Qwen2.5 support)
  - `modal` (optional; only needed for the Modal-hosted reproducibility path)

## Reproducing the headline numbers

The experiments were run on Modal. To reproduce locally, replace the `@app.function` decorators in `modal_app_*.py` with plain function definitions and point the entry script at a local GPU.

All four sparse policies are produced by two runs per harness, differing only in `--router-score`: the first run computes `dense`, `top-k`, `quest`, and `router` (on the K-mean backbone); the second run computes `router` on the Quest backbone (other policies are cache-hit from run 1).

```bash
# RULER NIAH-multikey + VT, n=100, 32K, Qwen2.5-14B (Table 2)
modal run modal_app_ruler.py --ctxlen 32768 --n-per 100 \
    --policies "dense,topk,quest,router" \
    --router-values "0.40" --router-score "mean"
modal run modal_app_ruler.py --ctxlen 32768 --n-per 100 \
    --policies "dense,topk,quest,router" \
    --router-values "0.40" --router-score "quest"

# LongBench-v2 medium, n=215 (dataset cap), 32K, Qwen2.5-14B (Table 3)
modal run modal_app_longbench_v2.py --length medium --n-per 300 \
    --policies "dense,topk,quest,router" \
    --router-values "0.40" --router-score "mean"
modal run modal_app_longbench_v2.py --length medium --n-per 300 \
    --policies "dense,topk,quest,router" \
    --router-values "0.40" --router-score "quest"

# Cross-model on Qwen2.5-7B-Instruct-1M, 32K (§5.7)
modal run modal_app_ruler.py --ctxlen 32768 --n-per 100 \
    --policies "dense,topk,quest,router" \
    --router-values "0.40" --router-score "quest" \
    --model "Qwen/Qwen2.5-7B-Instruct-1M"

# Long context, 128K, Qwen2.5-7B-Instruct-1M (§5.8)
modal run modal_app_ruler.py --ctxlen 131072 --n-per 100 \
    --policies "dense,topk,quest,router" \
    --router-values "0.40" --router-score "quest" \
    --model "Qwen/Qwen2.5-7B-Instruct-1M"

# Prefill profile at 128K, Qwen2.5-7B-Instruct-1M
modal run modal_app_profile.py --ctxlen 131072 --n-per 8 \
    --model "Qwen/Qwen2.5-7B-Instruct-1M"

# Budget-match ablation (Table 4): re-runs Quest at several budgets to
# decompose the router's lift into "more average budget" vs "selectivity".
# The dense / topk / quest / router rows in the base panel are reused
# from cache; the extra Quest sweeps are the only cold runs.
modal run modal_app_ruler.py --ctxlen 32768 --n-per 100 \
    --policies "dense,topk,quest,router" \
    --router-values "0.40" --router-score "quest" \
    --budget-sweep-policy "quest" \
    --budget-values "40,47,52,66"
```

The ablation harness adds an audit table at the end of each run, derived from `policy` labels and `matched_budget` via the `_policy_eff_budget` helper — no mutation of the cache-keyed result records, so prior cached panels validate without re-running. The `--model` flag accepts any HF-hosted Qwen-family or compatible LLM.

Outputs land in `results/` as `<task>_<timestamp>.json`. The same `poc_core.summary_table` / `paired_table` printers reproduce the tables in the paper.

## Citing

The paper is archived on Zenodo with a persistent DOI:

- **Concept DOI** (always resolves to the latest version): [`10.5281/zenodo.20630587`](https://doi.org/10.5281/zenodo.20630587)
- v2 (current): [`10.5281/zenodo.20663967`](https://doi.org/10.5281/zenodo.20663967)
- v1: [`10.5281/zenodo.20630588`](https://doi.org/10.5281/zenodo.20630588)

BibTeX (cite the concept DOI for "always latest"; swap in a version DOI to pin):

```bibtex
@misc{rossi2026voirouter,
  author    = {Rossi, Thomas},
  title     = {Uncertainty-gated selection for block-sparse attention},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20630587},
  url       = {https://doi.org/10.5281/zenodo.20630587}
}
```

See `CITATION.cff` for the machine-readable form.

## License

Code: MIT (see `LICENSE`). Paper PDF and `.tex`: CC-BY-4.0.

## Acknowledgements

Built on top of the SSA recipe from [Subquadratic](https://subq.ai) and the Quest block-scoring rule from [Quest (MLSys '24)](https://arxiv.org/abs/2406.10774).
