"""
modal_app_ruler.py
==================

RULER smoke harness — NIAH (multikey) + variable-tracking (multi-hop), at
ctx=32K, comparing dense / topk-v2 / router-v2 on Qwen2.5-14B-Instruct.

This is the smoke version: tiny n_per just to verify the path runs end to end
on Modal. Promote n_per and tighten knobs once the smoke passes.

  modal run modal_app_ruler.py                 # n_per=2, smoke
  modal run modal_app_ruler.py --n-per 50      # real run later

Sweep order matches modal_app.py: base policies (dense, topk) run first,
router is swept across router_values. Phase E.1 v2 config defaults below.
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
app = modal.App("voi-router-ruler")


@app.function(image=image, gpu="A100-80GB", volumes={"/cache": cache},
              timeout=14400)
def experiment(n_per: int, ctx: int, buckets: list,
               policies: list, fixed_budget: int,
               router_mode: str, router_grain: str, router_values: list,
               router_score: str,
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
          f"fixed_budget={fixed_budget}, kernel_v2={kernel_v2}, "
          f"router={router_mode}/{router_grain} vals={router_values} "
          f"code_version={code_version}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="block_sparse"
    ).to("cuda").eval()
    cache.commit()
    sparse_attention.install_rollout_hooks(model)
    print(f"model loaded, gpu={torch.cuda.get_device_name()}", flush=True)

    result_cache = ResultCache("/cache/result_cache.json")
    print(f"loaded result cache: {len(result_cache)} entries", flush=True)

    fb = fixed_budget if fixed_budget > 0 else None
    base_pols = [p for p in policies if p != "router"]
    sparse_attention.SEL.router_mode = router_mode
    sparse_attention.SEL.router_grain = router_grain
    sparse_attention.SEL.router_score = router_score
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
            if router_score != "mean":
                label_knob = f"{label_knob}_{router_score}"
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
def main(n_per: int = 2, ctxlen: int = 32768,
         policies: str = "topk,dense,router",
         fixed_budget: int = 33,
         num_keys: int = 4,
         num_hops: int = 3,
         num_distractor_chains: int = 3,
         router_mode: str = "quantile",
         router_grain: str = "row",
         router_values: str = "0.20",
         router_score: str = "mean",
         seed: int = 0,
         kernel_v2: bool = True):
    # RULER smoke: two tasks at the same ctx -- NIAH (single-hop budget
    # contention) and VT (multi-hop chain tracing).
    buckets = [
        ("niah_multikey", {"num_keys": num_keys}),
        ("vt", {"num_hops": num_hops,
                "num_distractor_chains": num_distractor_chains}),
    ]
    pols = [p.strip() for p in policies.split(",")]
    val_list = [float(v.strip()) for v in router_values.split(",") if v.strip()]

    from result_cache import hash_files
    prompt_files = ["poc_core.py", "ruler_tasks.py", "pointer_haystack.py"]
    sparse_files = prompt_files + ["sparse_attention.py", "triton_block_attn.py"]
    code_version = hash_files(sparse_files)
    dense_code_version = hash_files(prompt_files)

    result = experiment.remote(n_per, ctxlen, buckets, pols, fixed_budget,
                               router_mode, router_grain, val_list,
                               router_score,
                               seed, kernel_v2,
                               code_version, dense_code_version)

    import poc_core
    print("\n" + poc_core.summary_table(buckets, result["rows"]))
    print(f"matched budget: {result['matched_budget']} blocks\n")
    print("PAIRED — selector-isolated (among dense-correct examples):")
    print(poc_core.paired_table(buckets, result["rows"]))

    # Router-sweep summary (same shape as modal_app.py).
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
    path = f"results/ruler_smoke_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"saved -> {path}")
