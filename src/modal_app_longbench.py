"""
modal_app_longbench.py
======================

LongBench transfer test (§7.7). Runs the same dense / topk / rollout_peek
panel as modal_app.py, but with examples loaded from LongBench-family tasks:

  * musique-2hop   (multi-hop, structurally close to PCH at hop=2)
  * musique-4hop   (multi-hop, primary test)
  * narrativeqa    (single-doc, negative control)
  * qasper         (single-doc, negative control)

Headline metric: paired selector-preservation (F1 >= 0.5 binarisation).

  modal run modal_app_longbench.py                          # full panel
  modal run modal_app_longbench.py --tasks musique-4hop     # single task

This script is NEW; it does not modify the PCH pipeline.
"""

import json
import time

import modal

MODEL = "Qwen/Qwen2.5-14B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.9.0", "accelerate",
                 "safetensors", "hf_transfer",
                 # datasets >=3 dropped dataset-script support; THUDM/LongBench
                 # v1 is script-based, so pin to a version that still loads it.
                 "datasets==2.21.0")
    # HF_DATASETS_CACHE: isolated subdir so we don't clash with arrow caches
    # written by an incompatible (datasets >=3) version on the same volume.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache",
          "HF_DATASETS_CACHE": "/cache/datasets_v221"})
    .add_local_python_source("sparse_attention", "ruler_tasks", "poc_core",
                             "pointer_haystack", "result_cache",
                             "triton_block_attn", "longbench_tasks")
)

cache = modal.Volume.from_name("voi-router-hf-cache", create_if_missing=True)
app = modal.App("voi-router-longbench")


@app.function(image=image, gpu="A100-80GB", volumes={"/cache": cache},
              timeout=14400)
def experiment(tasks: list, n_per: int, ctx: int,
               policies: list, fixed_budget: int,
               peek_layer_min: int, peek_layer_max: int,
               router_mode: str, router_grain: str, router_values: list,
               kernel_v2: bool,
               code_version: str, dense_code_version: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import sparse_attention
    import poc_core
    import longbench_tasks as lb
    from result_cache import ResultCache

    # -- load examples --------------------------------------------------------
    examples = []
    for t in tasks:
        if t == "musique-2hop":
            examples += lb.load_musique(num_hops=2, max_examples=n_per)
        elif t == "musique-4hop":
            examples += lb.load_musique(num_hops=4, max_examples=n_per)
        elif t == "narrativeqa":
            examples += lb.load_narrativeqa(max_examples=n_per)
        elif t == "qasper":
            examples += lb.load_qasper(max_examples=n_per)
        else:
            raise ValueError(f"unknown task: {t}")
    print(f"loaded {len(examples)} examples across tasks={tasks}", flush=True)

    # -- model + tokenizer ----------------------------------------------------
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="block_sparse"
    ).to("cuda").eval()
    cache.commit()
    sparse_attention.install_rollout_hooks(model)
    sparse_attention.SEL.peek_layer_min = peek_layer_min
    sparse_attention.SEL.peek_layer_max = peek_layer_max

    # -- truncate prompts to fit ctx -----------------------------------------
    # Leave headroom for the chat template + generation; ~256 tokens is plenty.
    lb.truncate_examples(examples, tok, max_prompt_tokens=ctx - 256)
    print(f"truncated to fit ctx={ctx}", flush=True)
    print(f"model loaded, gpu={torch.cuda.get_device_name()}", flush=True)

    result_cache = ResultCache("/cache/result_cache.json")
    print(f"loaded result cache: {len(result_cache)} entries", flush=True)

    # -- run sweep ------------------------------------------------------------
    fb = fixed_budget if fixed_budget > 0 else None
    base_pols = [p for p in policies if p != "router"]
    sparse_attention.SEL.router_mode = router_mode
    sparse_attention.SEL.router_grain = router_grain
    sparse_attention.SEL.kernel_v2 = kernel_v2
    if router_values:
        if router_mode == "quantile":
            sparse_attention.SEL.router_quantile = router_values[0]
        else:
            sparse_attention.SEL.router_tau = router_values[0]
    # LongBench QA answers tend to be short -- 64 generation tokens is enough
    # and keeps decode cost down.
    result = poc_core.run_sweep(model, tok, examples, "cuda",
                                policies=tuple(base_pols), fixed_budget=fb,
                                max_new=64,
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
            print(f"\n=== router[{router_mode}] {label_knob} ===", flush=True)
            for i, (task, kw, ex) in enumerate(examples):
                r = poc_core.run_one(model, tok, ex, "router", "cuda",
                                     max_new=64,
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

    # Reconstruct buckets in the order examples were appended
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
    result["tasks"] = list(tasks)
    result["policies"] = list(policies)
    result["router_mode"] = router_mode
    result["router_values"] = list(router_values)
    result["kernel_v2"] = kernel_v2
    result["code_version"] = code_version
    result["paired_threshold"] = 0.5
    return result


@app.local_entrypoint()
def main(tasks: str = "musique-2hop,musique-4hop,narrativeqa,qasper",
         n_per: int = 30,
         ctxlen: int = 32768,
         policies: str = "topk,dense,router",
         fixed_budget: int = 33,
         peek_layer_min: int = 0,
         peek_layer_max: int = 999,
         router_mode: str = "quantile",
         router_grain: str = "row",
         router_values: str = "0.40",
         kernel_v2: bool = True):
    task_list = [t.strip() for t in tasks.split(",")]
    pol_list = [p.strip() for p in policies.split(",")]
    val_list = [float(v.strip()) for v in router_values.split(",") if v.strip()]

    from result_cache import hash_files
    prompt_files = ["poc_core.py", "ruler_tasks.py", "pointer_haystack.py",
                    "longbench_tasks.py"]
    sparse_files = prompt_files + ["sparse_attention.py", "triton_block_attn.py"]
    code_version = hash_files(sparse_files)
    dense_code_version = hash_files(prompt_files)

    result = experiment.remote(task_list, n_per, ctxlen, pol_list, fixed_budget,
                               peek_layer_min, peek_layer_max,
                               router_mode, router_grain, val_list, kernel_v2,
                               code_version, dense_code_version)

    import poc_core
    print("\n" + poc_core.summary_table(result["buckets"], result["rows"]))
    print(f"\nmatched budget: {result['matched_budget']} blocks")
    print(f"paired threshold (F1): {result['paired_threshold']}\n")
    print("PAIRED — selector-isolated (among dense-solved, F1>=0.5):")
    print(poc_core.paired_table(result["buckets"], result["rows"],
                                threshold=result["paired_threshold"]))

    import os
    os.makedirs("results", exist_ok=True)
    path = f"results/longbench_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"saved -> {path}")
