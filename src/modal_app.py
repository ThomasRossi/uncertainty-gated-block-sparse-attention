"""
modal_app.py
============

Run the SSA selection proof-of-concept on a Modal GPU.

  modal run modal_app.py                      # small smoke test (defaults)
  modal run modal_app.py --n-per 30 --ctx 32768   # the real run

The model is cached in a Modal Volume, so it downloads only once. Results are
returned to the local machine and saved under results/.
"""

import json
import time

import modal

MODEL = "Qwen/Qwen2.5-14B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.9.0", "accelerate",
                 "safetensors", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
    .add_local_python_source("sparse_attention", "ruler_tasks", "poc_core",
                             "pointer_haystack", "result_cache",
                             "triton_block_attn")
)

cache = modal.Volume.from_name("voi-router-hf-cache", create_if_missing=True)
app = modal.App("voi-router-pch")


@app.function(image=image, gpu="A100-80GB", volumes={"/cache": cache},
              timeout=14400)
def experiment(n_per: int, ctx: int, buckets: list,
               policies: list, fixed_budget: int,
               peek_layer_min: int, peek_layer_max: int,
               peek_metric: str, peek_xh_threshold: float,
               router_mode: str, router_grain: str, router_values: list,
               seed: int, kernel_v2: bool,
               code_version: str, dense_code_version: str):
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import sparse_attention
    import poc_core
    from result_cache import ResultCache

    rng = random.Random(seed)
    examples = [(task, kw, poc_core.make_example(rng, task, kw, ctx))
                for task, kw in buckets for _ in range(n_per)]
    print(f"{len(examples)} examples, ctx~{ctx}, policies={policies}, "
          f"fixed_budget={fixed_budget}, "
          f"peek_layer_range=[{peek_layer_min},{peek_layer_max}], "
          f"code_version={code_version} "
          f"dense_code_version={dense_code_version}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="block_sparse"
    ).to("cuda").eval()
    cache.commit()
    sparse_attention.install_rollout_hooks(model)
    sparse_attention.SEL.peek_layer_min = peek_layer_min
    sparse_attention.SEL.peek_layer_max = peek_layer_max
    sparse_attention.SEL.peek_metric = peek_metric
    sparse_attention.SEL.peek_xh_threshold = peek_xh_threshold
    print(f"model loaded, gpu={torch.cuda.get_device_name()}", flush=True)

    result_cache = ResultCache("/cache/result_cache.json")
    print(f"loaded result cache: {len(result_cache)} entries", flush=True)

    fb = fixed_budget if fixed_budget > 0 else None
    # Base sweep: everything except router (which we sweep across values below).
    base_pols = [p for p in policies if p != "router"]
    sparse_attention.SEL.router_mode = router_mode
    sparse_attention.SEL.router_grain = router_grain
    sparse_attention.SEL.kernel_v2 = kernel_v2
    if router_values:
        if router_mode == "quantile":
            sparse_attention.SEL.router_quantile = router_values[0]
        else:
            sparse_attention.SEL.router_tau = router_values[0]
    result = poc_core.run_sweep(model, tok, examples, "cuda",
                                policies=tuple(base_pols), fixed_budget=fb,
                                cache=result_cache, model_name=MODEL,
                                code_version=code_version,
                                dense_code_version=dense_code_version)
    # Router sweep over router_values. Each value gets its own cache key
    # (router_tau or router_quantile is part of cache_key_for_run). Rows are
    # tagged router_{mode}{value} so the local summary can split them.
    if "router" in policies and router_values:
        matched = result["matched_budget"]
        sparse_attention.SEL.budget_blocks = matched
        for val in router_values:
            if router_mode == "quantile":
                sparse_attention.SEL.router_quantile = val
                label_knob = f"q{val}"
            else:
                sparse_attention.SEL.router_tau = val
                label_knob = f"tau{val}"
            print(f"\n=== router[{router_mode}] {label_knob} ===", flush=True)
            for i, (task, kw, ex) in enumerate(examples):
                r = poc_core.run_one(model, tok, ex, "router", "cuda",
                                     cache=result_cache, model_name=MODEL,
                                     code_version=code_version,
                                     dense_code_version=dense_code_version)
                row = dict(task=task, diff=str(kw),
                           policy=f"router_{label_knob}", **r)
                result["rows"].append(row)
                print(f"[router {label_knob} {i+1}/{len(examples)}] "
                      f"{task} {kw} recall={r['recall']:.2f} "
                      f"hit={r['hit']:.2f} {r['dt']:.1f}s", flush=True)
        result_cache.save()
        cache.commit()
    print(result_cache.summary(), flush=True)

    result["buckets"] = buckets
    result["ctx"] = ctx
    result["n_per"] = n_per
    result["policies"] = list(policies)
    result["router_mode"] = router_mode
    result["router_values"] = list(router_values)
    result["code_version"] = code_version
    return result


@app.local_entrypoint()
def main(n_per: int = 100, ctxlen: int = 8192,
         policies: str = "topk,dense,rollout_peek", fixed_budget: int = 0,
         hops: str = "1,2,3,4",
         peek_layer_min: int = 0, peek_layer_max: int = 999,
         peek_metric: str = "peakedness",
         peek_xh_threshold: float = 0.5,
         router_mode: str = "abs",
         router_grain: str = "cell",
         router_values: str = "0.10",
         seed: int = 0,
         kernel_v2: bool = False):
    # Pointer-Chase Haystack: hop depth dials query-latency + compounding.
    hop_list = [int(h.strip()) for h in hops.split(",")]
    buckets = [("pointer_chase", {"num_hops": h}) for h in hop_list]
    pols = [p.strip() for p in policies.split(",")]
    # Code-version hash: any change to the result-determining sources
    # invalidates the cache. Conservative but safe.
    from result_cache import hash_files
    # `code_version` invalidates ALL sparse policy entries on any selector /
    # kernel edit. `dense_code_version` is narrower -- dense's forward depends
    # only on the prompt-generating + dispatch code, NOT on the selector or
    # the Triton kernel -- so dense entries survive kernel work.
    prompt_files = ["poc_core.py", "ruler_tasks.py", "pointer_haystack.py"]
    sparse_files = prompt_files + ["sparse_attention.py", "triton_block_attn.py"]
    code_version = hash_files(sparse_files)
    dense_code_version = hash_files(prompt_files)
    val_list = [float(v.strip()) for v in router_values.split(",") if v.strip()]
    result = experiment.remote(n_per, ctxlen, buckets, pols, fixed_budget,
                               peek_layer_min, peek_layer_max,
                               peek_metric, peek_xh_threshold,
                               router_mode, router_grain, val_list,
                               seed, kernel_v2,
                               code_version, dense_code_version)

    import poc_core
    print("\n" + poc_core.summary_table(buckets, result["rows"]))
    print(f"matched budget: {result['matched_budget']} blocks\n")
    print("PAIRED — selector-isolated (among dense-correct examples):")
    print(poc_core.paired_table(buckets, result["rows"]))

    # Local router-sweep summary (poc_core's tables skip unknown policies).
    drows = [r for r in result["rows"] if r["policy"] == "dense"]
    trows = [r for r in result["rows"] if r["policy"] == "topk"]
    if val_list:
        n = len(drows) if drows else (len(trows) if trows else 0)
        dc_idx = [i for i, r in enumerate(drows) if r["recall"] >= 1.0]
        print(f"\nrouter[{router_mode}] sweep:")
        if trows and drows:
            print(f"  baseline: topk recall="
                  f"{sum(r['recall'] for r in trows)/len(trows):.3f}  "
                  f"dense recall="
                  f"{sum(r['recall'] for r in drows)/len(drows):.3f}  "
                  f"dense-correct n={len(dc_idx)}")
        knob = "q" if router_mode == "quantile" else "tau"
        for val in val_list:
            label = f"router_{knob}{val}"
            rrows = [r for r in result["rows"] if r["policy"] == label]
            if not rrows:
                continue
            rec = sum(r["recall"] for r in rrows) / len(rrows)
            hit = sum(r["hit"] for r in rrows) / len(rrows)
            dt = sum(r["dt"] for r in rrows) / len(rrows)
            line = (f"  {knob}={val:>5}: recall={rec:.3f}  hit={hit:.3f}  "
                    f"mean_dt={dt:.2f}s  n={len(rrows)}")
            if dc_idx and len(rrows) == n:
                r_kept = sum(rrows[i]["recall"] >= 1.0
                             for i in dc_idx) / len(dc_idx)
                line += f"  paired={r_kept:.2f}"
            print(line)

    import os
    os.makedirs("results", exist_ok=True)
    path = f"results/modal_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"saved -> {path}")
