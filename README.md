# Uncertainty-gated selection for block-sparse attention

Reference implementation, paper, and experiment results for the *value-of-information router* — a backbone-agnostic, single-hyperparameter add-on for block-sparse attention selectors that lifts selector quality on multi-hop and query-latent long-context retrieval.

> **Status:** research preview. The method, the kernel, and the experimental results in `paper/summary_for_review.pdf` are stable; the paper text is in late draft form (abstract and conclusion still tinted red). Numbers reported below match the JSONs in `results/`.

## TL;DR

Block-sparse attention selects which key blocks each query attends to via a per-query top-$k$ rule. When the $k$-th and $(k{+}1)$-th block are nearly tied, that cutoff is a coin flip — and dropping the wrong one loses an answer hop with no recovery. We measure the normalised cutoff margin $\sigma$ per Q-tile and, for the bottom $q$-fraction of tiles per layer, double the attended set. The rule is independent of how block scores are computed and composes with existing scoring backbones.

**Headline at $n=100$, Qwen2.5-14B-Instruct, 32K context** (paired recall — fraction of dense-correct examples the sparse policy also solves):

| backbone | policy            | RULER NIAH-multikey | LongBench-v2 medium |
|----------|-------------------|--------------------:|--------------------:|
| —        | dense (ceiling)   | 1.00                | 1.00                |
| K-mean   | top-$k$           | 0.51                | 0.40                |
| K-mean   | router            | 0.63                | 0.60                |
| K-max    | Quest             | 0.93                | 0.60                |
| K-max    | **router-on-Quest** | **0.98**          | **0.75**            |

Within 2 pp of dense on RULER NIAH at $k_{\text{budget}}=33$. The fused selection-plus-kernel pipeline runs at $0.76$–$0.85\times$ dense wall time at 64K.

See `paper/summary_for_review.pdf` for the full method, math, and discussion.

## What's here

```
release/
├── paper/                     # the paper (.tex + .pdf)
├── src/                       # all method and harness code
└── results/                   # JSON dumps from the headline experiments
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
| `modal_app_profile.py` | CUDA-event prefill profiler (used for the 32K/64K efficiency table) |

### `results/`

JSON dumps for the experiments quoted in the paper:

- `ruler_smoke_20260607_161126.json` — RULER NIAH-multikey, dense / top-$k$ / router (K-mean), $n=100$, 32K
- `ruler_smoke_20260607_193623.json` — RULER NIAH-multikey, Quest baseline added, $n=100$, 32K
- `ruler_smoke_20260607_222700.json` — RULER NIAH-multikey + VT, router-on-Quest, $n=100$, 32K
- `longbench_v2_20260607_132056.json` — LongBench-v2 medium, dense / top-$k$ / router (K-mean), $n=100$, 32K
- `longbench_v2_20260607_184826.json` — LongBench-v2 medium, Quest baseline added, $n=100$, 32K
- `longbench_v2_20260607_222750.json` — LongBench-v2 medium, router-on-Quest, $n=100$, 32K
- `profile_20260601_204936.json`, `profile_20260601_211844.json` — 32K and 64K prefill CUDA-event profiles ($n=8$)

The JSON layout is one record per (task, policy, example): see `poc_core.py` for the schema.

## Requirements

- Python 3.10+
- CUDA-capable GPU with sufficient memory for Qwen2.5-14B-Instruct (A100-80GB used for all reported numbers; A100-40GB works at 16K, marginal at 32K)
- See `requirements.txt`. Key pins:
  - `datasets==2.21.0` (LongBench v1 ships a script-based loader that breaks under `datasets>=3`)
  - `torch>=2.3`, `triton>=2.3`
  - `transformers>=4.45` (for Qwen2.5 support)
  - `modal` (optional; only needed for the Modal-hosted reproducibility path)

## Reproducing the headline numbers

The experiments were run on Modal. To reproduce locally, replace the `@app.function` decorators in `modal_app_*.py` with plain function definitions and point the entry script at a local GPU.

```bash
# RULER NIAH-multikey, n=100, 32K, all four sparse policies
python -m modal_app_ruler --ctxlen 32768 --n-per 50 --policies dense topk quest router

# LongBench-v2 medium, n=100, 32K, all four sparse policies
python -m modal_app_longbench_v2 --ctxlen 32768 --n 100 --policies dense topk quest router

# 64K prefill profile
python -m modal_app_profile --ctxlen 65536 --n 8
```

Outputs land in `results/` as `<task>_<timestamp>.json`. The same `poc_core.summary_table` / `paired_table` printers reproduce the tables in the paper.

## Citing

\TODO{Add a BibTeX block once the arXiv ID is assigned.} See `CITATION.cff` for the machine-readable form.

## License

Code: MIT (see `LICENSE`). Paper PDF and `.tex`: CC-BY-4.0.

## Acknowledgements

Built on top of the SSA recipe from [Subquadratic](https://subq.ai) and the Quest block-scoring rule from [Quest (MLSys '24)](https://arxiv.org/abs/2406.10774).
