"""
modal_app_profile.py
====================

Prefill-only CUDA-event profiler. Runs a forward pass (no generation) for
n_per RULER examples at fixed ctx, switching policies between forwards, and
reports per-attention-layer time broken down into:
  - dense_sdpa  (the dense SDPA call, for dense policy)
  - v2_select   (the _block_select_kv_idx_v2 call, sparse policies)
  - v2_kernel   (the block_sparse_attn_v2 call, sparse policies)

Decode (Q==1) is excluded from the profile by design. The point is to
attribute the 32K prefill cost across attention components so we can see
where the dense/sparse gap actually lives.

Usage:
  modal run modal_app_profile.py --n-per 4
"""

import os as _os
import time

import modal

MODEL = "Qwen/Qwen2.5-14B-Instruct"
_GPU = _os.environ.get("SUBQ_GPU", "A100-80GB")

image = (
    # nvidia/cuda devel image: ships nvcc so causal-conv1d (Gated DeltaNet
    # fused conv branch) builds from source. debian_slim lacks nvcc and the
    # build fails at the bdist_wheel step.
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04",
        add_python="3.11",
    )
    # clang: causal-conv1d's source build (no prebuilt wheel for cu13/torch
    # 2.12) requires clang++ >=7. ninja: pytorch's cpp_extension build system.
    .apt_install("clang", "ninja-build", "git")
    .pip_install("torch", "transformers==5.12.1", "accelerate",
                 "safetensors", "hf_transfer", "ninja", "packaging",
                 "wheel", "setuptools",
                 "flash-linear-attention", "causal-conv1d")
    # Tried + abandoned during 2026-06-28/29 image iteration:
    # - megablocks 0.10.0: fails to compile against CUDA 13.0's cub
    #   (csrc/cumsum.h uses pre-CUDA-12 cub::DeviceScan::ExclusiveSum
    #   signature with separate stream + debug_synchronous args).
    # - flash-attn 2.8.3: no prebuilt wheel for cu13/torch 2.12 and the
    #   source build OOMs on Modal's image builder (parallel nvcc on 30+
    #   kernels needs >20GB RAM/job).
    # Future work: vLLM fused_moe, grouped_gemm, or HF kernels lib with the
    # correct LayerRepository(repo_id, layer_name, revision) spec.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("sparse_attention", "ruler_tasks", "poc_core",
                             "pointer_haystack", "result_cache",
                             "triton_block_attn")
)

cache = modal.Volume.from_name("voi-router-hf-cache", create_if_missing=True)
app = modal.App("voi-router-profile")


@app.function(image=image, gpu=_GPU, volumes={"/cache": cache},
              timeout=3600)
def profile(n_per: int, ctx: int, buckets: list, fixed_budget: int,
            router_quantile: float, seed: int, model_name: str = MODEL):
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import sparse_attention
    from sparse_attention import SEL, PROF
    import poc_core

    rng = random.Random(seed)
    examples = [(task, kw, poc_core.make_example(rng, task, kw, ctx))
                for task, kw in buckets for _ in range(n_per)]
    print(f"{len(examples)} examples, ctx~{ctx}, fixed_budget={fixed_budget}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="block_sparse"
    ).to("cuda").eval()
    print(f"model loaded, gpu={torch.cuda.get_device_name(0)}", flush=True)

    SEL.kernel_v2 = True
    SEL.budget_blocks = fixed_budget
    SEL.router_mode = "quantile"
    SEL.router_grain = "row"
    SEL.router_quantile = router_quantile

    # Pre-tokenize so tokenization isn't in the timing.
    encoded = []
    for task, kw, ex in examples:
        enc, _ = poc_core.locate(tok, ex)
        ids = enc["input_ids"].to("cuda")
        am = enc["attention_mask"].to("cuda")
        encoded.append((task, ids, am))

    # Configurations to profile. Order: warm up, then measure.
    configs = [
        ("dense", "dense"),
        ("topk-v2", "topk"),
        (f"router-v2 q={router_quantile}", "router"),
    ]

    # Warmup pass: one forward per config to compile triton kernels and warm
    # autotune. Use the first example. NOT recorded.
    print("warmup...", flush=True)
    SEL.log = False
    warm_task, warm_ids, warm_am = encoded[0]
    for _, mode in configs:
        SEL.mode = mode
        with torch.no_grad():
            model(input_ids=warm_ids, attention_mask=warm_am, use_cache=False)
        torch.cuda.synchronize()

    results = {}
    for label, mode in configs:
        SEL.mode = mode
        PROF.reset()
        PROF.enabled = True
        torch.cuda.synchronize()

        wall = []
        for task, ids, am in encoded:
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                model(input_ids=ids, attention_mask=am, use_cache=False)
            torch.cuda.synchronize()
            wall.append(time.time() - t0)
            print(f"  [{label}] {task} forward={wall[-1]*1000:.0f}ms",
                  flush=True)

        PROF.enabled = False
        summary = PROF.summary()
        results[label] = {
            "wall_ms": [w * 1000 for w in wall],
            "wall_mean_ms": sum(wall) * 1000 / len(wall),
            "components": summary,
        }
        print(f"=== {label} === wall mean = "
              f"{results[label]['wall_mean_ms']:.0f}ms", flush=True)
        for name, st in summary.items():
            print(f"    {name:<12} calls={st['calls']:>3}  "
                  f"mean={st['mean_ms']:.2f}ms  "
                  f"total={st['total_ms']:.0f}ms  "
                  f"p50={st['p50_ms']:.2f}ms", flush=True)

    return results


@app.local_entrypoint()
def main(n_per: int = 4, ctxlen: int = 32768, fixed_budget: int = 33,
         router_quantile: float = 0.40, seed: int = 0,
         num_keys: int = 4, num_hops: int = 3,
         num_distractor_chains: int = 3, model: str = MODEL):
    buckets = [
        ("niah_multikey", {"num_keys": num_keys}),
        ("vt", {"num_hops": num_hops,
                "num_distractor_chains": num_distractor_chains}),
    ]
    results = profile.remote(n_per, ctxlen, buckets, fixed_budget,
                             router_quantile, seed, model)

    # Print a clean per-policy summary.
    print("\n" + "=" * 64)
    print("PROFILE SUMMARY (per prefill forward, 48 attention layers)")
    print("=" * 64)
    for label, r in results.items():
        print(f"\n{label}:")
        print(f"  wall (mean over {len(r['wall_ms'])} examples): "
              f"{r['wall_mean_ms']:.0f}ms")
        attn_total_ms = sum(c["total_ms"] for c in r["components"].values())
        if r["wall_ms"]:
            attn_per_example = attn_total_ms / len(r["wall_ms"])
            other = r["wall_mean_ms"] - attn_per_example
            print(f"  attention layers (sum per example): "
                  f"{attn_per_example:.0f}ms")
            print(f"  rest of model (per example): {other:.0f}ms")
        for name, st in r["components"].items():
            print(f"    {name:<12} mean/layer={st['mean_ms']:.2f}ms  "
                  f"per-example total={st['total_ms']/len(r['wall_ms']):.0f}ms")

    import json
    import os
    os.makedirs("results", exist_ok=True)
    path = f"results/profile_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved -> {path}")
