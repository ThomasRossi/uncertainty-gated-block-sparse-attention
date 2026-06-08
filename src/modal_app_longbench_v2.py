"""
modal_app_longbench_v2.py
=========================

LongBench-v2 (THUDM/LongBench-v2, arXiv:2412.15204) transfer test.

503 multiple-choice items, contexts up to 2M words, organised into three
length buckets (short / medium / long) by the dataset authors. We default to
`length=medium` (~32K-target prefill) to match our profiled config. The same
selector panel as the RULER / LongBench-v1 sweeps applies:

  dense / topk-v2 / router-v2 (quantile/row, q from --router-values)

Scoring is multiple-choice (extract first A/B/C/D in the output). Threshold
1.0 for the paired metric -- MC is already binary.

  modal run modal_app_longbench_v2.py                  # n_per=30 medium
  modal run modal_app_longbench_v2.py --length long    # ~64K bucket
"""

import json
import time

import modal

MODEL = "Qwen/Qwen2.5-14B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.9.0", "accelerate",
                 "safetensors", "hf_transfer",
                 "datasets==2.21.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache",
          "HF_DATASETS_CACHE": "/cache/datasets_v221"})
    .add_local_python_source("sparse_attention", "ruler_tasks", "poc_core",
                             "pointer_haystack", "result_cache",
                             "triton_block_attn", "longbench_tasks")
)

cache = modal.Volume.from_name("voi-router-hf-cache", create_if_missing=True)
app = modal.App("voi-router-longbench-v2")


@app.function(image=image, gpu="A100-80GB", volumes={"/cache": cache},
              timeout=14400)
def experiment(length: str, difficulty: str, n_per: int, ctx: int,
               policies: list, fixed_budget: int,
               router_mode: str, router_grain: str, router_values: list,
               router_score: str,
               kernel_v2: bool, seed: int,
               code_version: str, dense_code_version: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import sparse_attention
    import poc_core
    import longbench_tasks as lb
    from result_cache import ResultCache

    examples = lb.load_longbench_v2(
        length=length if length else None,
        difficulty=difficulty if difficulty else None,
        max_examples=n_per, seed=seed)
    print(f"loaded {len(examples)} examples, length={length}, "
          f"difficulty={difficulty}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="block_sparse"
    ).to("cuda").eval()
    cache.commit()
    sparse_attention.install_rollout_hooks(model)

    lb.truncate_examples(examples, tok, max_prompt_tokens=ctx - 256)
    print(f"truncated to fit ctx={ctx}", flush=True)
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
    # max_new=8 is plenty for MC (just need the letter); keeps decode cost down.
    result = poc_core.run_sweep(model, tok, examples, "cuda",
                                policies=tuple(base_pols), fixed_budget=fb,
                                max_new=8,
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
                                     max_new=8,
                                     cache=result_cache, model_name=MODEL,
                                     code_version=code_version,
                                     dense_code_version=dense_code_version)
                row = dict(task=task, diff=str(kw),
                           policy=f"router_{label_knob}", **r)
                result["rows"].append(row)
                print(f"[router {label_knob} {i+1}/{len(examples)}] "
                      f"{task} {kw} recall={r['recall']:.2f} "
                      f"{r['dt']:.1f}s", flush=True)
    result_cache.save()
    cache.commit()
    print(result_cache.summary(), flush=True)

    buckets = []
    seen = set()
    for task, kw, _ in examples:
        key = (task, str(kw))
        if key not in seen:
            seen.add(key)
            buckets.append((task, kw))
    result["buckets"] = buckets
    result["ctx"] = ctx
    result["n_per"] = n_per
    result["length"] = length
    result["difficulty"] = difficulty
    result["policies"] = list(policies)
    result["router_mode"] = router_mode
    result["router_values"] = list(router_values)
    result["router_score"] = router_score
    result["kernel_v2"] = kernel_v2
    result["code_version"] = code_version
    # MC is binary; paired threshold = exact match.
    result["paired_threshold"] = 1.0
    return result


@app.local_entrypoint()
def main(length: str = "medium", difficulty: str = "",
         n_per: int = 30,
         ctxlen: int = 32768,
         policies: str = "topk,dense,router",
         fixed_budget: int = 33,
         router_mode: str = "quantile",
         router_grain: str = "row",
         router_values: str = "0.40",
         router_score: str = "mean",
         kernel_v2: bool = True,
         seed: int = 42):
    pol_list = [p.strip() for p in policies.split(",")]
    val_list = [float(v.strip()) for v in router_values.split(",") if v.strip()]

    from result_cache import hash_files
    prompt_files = ["poc_core.py", "ruler_tasks.py", "pointer_haystack.py",
                    "longbench_tasks.py"]
    sparse_files = prompt_files + ["sparse_attention.py", "triton_block_attn.py"]
    code_version = hash_files(sparse_files)
    dense_code_version = hash_files(prompt_files)

    result = experiment.remote(length, difficulty, n_per, ctxlen, pol_list,
                               fixed_budget, router_mode, router_grain,
                               val_list, router_score, kernel_v2, seed,
                               code_version, dense_code_version)

    import poc_core
    print("\n" + poc_core.summary_table(result["buckets"], result["rows"]))
    print(f"\nmatched budget: {result['matched_budget']} blocks")
    print(f"paired threshold (MC): {result['paired_threshold']}\n")
    print("PAIRED — selector-isolated (among dense-correct, exact MC):")
    print(poc_core.paired_table(result["buckets"], result["rows"],
                                threshold=result["paired_threshold"]))

    import os
    os.makedirs("results", exist_ok=True)
    path = f"results/longbench_v2_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"saved -> {path}")
