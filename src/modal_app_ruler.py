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
    .pip_install("torch", "transformers==5.12.1", "accelerate",
                 "safetensors", "hf_transfer", "triton",
                 "sentencepiece", "tiktoken")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("sparse_attention", "ruler_tasks", "poc_core",
                             "pointer_haystack", "result_cache",
                             "triton_block_attn", "q_chunk_patch",
                             "dump_block_scores")
)

cache = modal.Volume.from_name("voi-router-hf-cache", create_if_missing=True)
app = modal.App("voi-router-ruler-smoke")


def _policy_eff_budget(policy: str, matched_budget: int):
    """Derive mean real blocks/tile from policy label + matched_budget.

    v2 selection always doubles budget_blocks -> k_bud_base = 2*b. For router
    modes, k_bud_max = 2*k_bud_base; expected blocks/tile = 2*b * (1+q) where
    q is the quantile (trigger fraction == quantile by construction, up to
    discretisation). Returns None for dense (full context)."""
    if policy == "dense":
        return None
    if policy.startswith("router_q"):
        # label is e.g. "router_q0.4" or "router_q0.4_quest"
        rest = policy[len("router_q"):].split("_", 1)[0]
        try:
            q = float(rest)
        except ValueError:
            q = 0.0
        return 2 * matched_budget * (1 + q)
    if policy.startswith("router_tau"):
        # tau-mode router: we don't know empirical trigger here -- report
        # the kernel slot cap (upper bound).
        return 4 * matched_budget
    # quest_b46, quest_b40, etc. -- explicit budget override.
    if "_b" in policy:
        try:
            b = int(policy.rsplit("_b", 1)[1])
            return 2 * b
        except ValueError:
            pass
    # topk, quest, belief, etc. at the matched budget.
    return 2 * matched_budget


import os as _os
_GPU = _os.environ.get("SUBQ_GPU", "A100-80GB")
_TIMEOUT = int(_os.environ.get("SUBQ_TIMEOUT", "14400"))


@app.function(image=image, gpu=_GPU, volumes={"/cache": cache},
              timeout=_TIMEOUT)
def experiment(n_per: int, ctx: int, buckets: list,
               policies: list, fixed_budget: int,
               router_mode: str, router_grain: str, router_values: list,
               router_score: str,
               seed: int, kernel_v2: bool,
               code_version: str, dense_code_version: str,
               budget_values: list, budget_sweep_policy: str,
               model_name: str = MODEL,
               dump_block_scores_path: str = "",
               no_cache: bool = False,
               router_score_sweep: list = None):
    """router_score_sweep: optional list of scoring backbones to loop the
    router over (e.g. ["mean", "quest"]). Falls back to [router_score] for
    backward compat with single-score callers."""
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import os
    import sparse_attention
    import poc_core
    from result_cache import ResultCache

    # Install the Q-tile-chunked v2 selection at very long context to bound the
    # intermediate `[B, H, Q, NB]` matmul that OOMs at 1M. Numerically
    # equivalent to the in-tree version; lives outside `code_version` so the
    # result cache stays valid.
    if ctx >= int(os.environ.get("SUBQ_QCHUNK_CTX_MIN", "262144")):
        import q_chunk_patch
        q_chunk_patch.install()

    rng = random.Random(seed)
    examples = [(task, kw, poc_core.make_example(rng, task, kw, ctx))
                for task, kw in buckets for _ in range(n_per)]
    print(f"{len(examples)} examples, ctx~{ctx}, policies={policies}, "
          f"fixed_budget={fixed_budget}, kernel_v2={kernel_v2}, "
          f"router={router_mode}/{router_grain} vals={router_values} "
          f"code_version={code_version}", flush=True)

    print(f"loading model: {model_name}", flush=True)
    tok_kwargs = {"fix_mistral_regex": True} if "mistral" in model_name.lower() else {}
    tok = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    # Multi-GPU: enforce balanced layer sharding so KV cache + activations
    # split evenly. "auto" doesn't shard a 14 GB model — it fits all weights
    # on cuda:0 (or in our case cuda:1) and lets activations pile up on a
    # single card. "balanced_low_0" forces an even layer-by-layer split
    # while keeping cuda:0 lighter so embed_in / lm_head have room.
    model_kwargs = dict(dtype=torch.bfloat16, attn_implementation="block_sparse")
    ndev = torch.cuda.device_count()
    if ndev > 1:
        # Reserve ~10 GB per device for activations the accelerate planner
        # can't see (KV cache grows during runtime, not at load time).
        per_dev = "120GiB"
        model_kwargs["device_map"] = "balanced_low_0"
        model_kwargs["max_memory"] = {i: per_dev for i in range(ndev)}
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     **model_kwargs).eval()
        dm = getattr(model, "hf_device_map", None) or getattr(
            getattr(model, "model", None), "hf_device_map", None)
        print(f"multi-gpu device_map: {dm}", flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs).to("cuda").eval()
    cache.commit()
    # install_rollout_hooks attaches a per-layer pre-hook that stashes the
    # input hidden state via `SEL.hidden_at[idx] = hs.detach()`. At 1M with
    # 28 layers and 3584-wide hidden, that's ~196 GB of dead state per
    # prefill for any policy that isn't rollout_peek. None of the policies
    # in this smoke (dense/topk/quest/router) need rollout_peek, so skip the
    # install entirely. Re-enable only when a rollout_peek pass is needed.
    if any(p in ("rollout_peek",) for p in policies):
        sparse_attention.install_rollout_hooks(model)
    print(f"model loaded, ngpu={torch.cuda.device_count()}, "
          f"gpu0={torch.cuda.get_device_name(0)}", flush=True)

    # Diagnostic dump for the BAI/VoI appendix: install AFTER model load, BEFORE
    # any prefill, so every sparse-selection call is captured. Lives outside the
    # cache-key file set (sparse_files in `hash_files` above) so existing
    # result_cache entries are not invalidated by its presence. A diagnostic run
    # additionally bypasses the result_cache (no_cache=True is implied) because
    # a cache hit short-circuits the forward and there would be no scores to
    # dump.
    if dump_block_scores_path:
        import dump_block_scores as _dbs
        _dbs.install_dump_hook(dump_block_scores_path)
        no_cache = True

    result_cache = None if no_cache else ResultCache("/cache/result_cache.json")
    if result_cache is None:
        print("result cache: BYPASSED (no_cache=True)", flush=True)
    else:
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
    # With device_map sharding, the embed_tokens may not be on cuda:0.
    # Inputs must go to the embedding's device; everything else follows.
    input_dev = str(next(model.get_input_embeddings().parameters()).device)
    print(f"input device: {input_dev}", flush=True)
    result = poc_core.run_sweep(model, tok, examples, input_dev,
                                policies=tuple(base_pols), fixed_budget=fb,
                                cache=result_cache, model_name=model_name,
                                code_version=code_version,
                                dense_code_version=dense_code_version)
    if "router" in policies and router_values:
        matched = result["matched_budget"]
        sparse_attention.SEL.budget_blocks = matched
        scores_to_run = router_score_sweep or [router_score]
        for score in scores_to_run:
            sparse_attention.SEL.router_score = score
            for val in router_values:
                if router_mode == "quantile":
                    sparse_attention.SEL.router_quantile = val
                    label_knob = f"q{val}"
                else:
                    sparse_attention.SEL.router_tau = val
                    label_knob = f"tau{val}"
                if score != "mean":
                    label_knob = f"{label_knob}_{score}"
                print(f"\n=== router[{router_mode}] {label_knob} ===", flush=True)
                for i, (task, kw, ex) in enumerate(examples):
                    r = poc_core.run_one(model, tok, ex, "router", input_dev,
                                         cache=result_cache, model_name=model_name,
                                         code_version=code_version,
                                         dense_code_version=dense_code_version)
                    row = dict(task=task, diff=str(kw),
                               policy=f"router_{label_knob}", **r)
                    result["rows"].append(row)
                    print(f"[router {label_knob} {i+1}/{len(examples)}] "
                          f"{task} {kw} recall={r['recall']:.2f} "
                          f"hit={r['hit']:.2f} {r['dt']:.1f}s", flush=True)
        if result_cache is not None:
            result_cache.save()
            cache.commit()
    # Budget ablation sweep: re-run `budget_sweep_policy` at each budget value
    # so router-at-base vs uniform-at-matched-budget is one JSON away.
    if budget_values and budget_sweep_policy:
        for b in budget_values:
            sparse_attention.SEL.budget_blocks = int(b)
            label = f"{budget_sweep_policy}_b{b}"
            print(f"\n=== budget sweep: {label} ===", flush=True)
            for i, (task, kw, ex) in enumerate(examples):
                r = poc_core.run_one(model, tok, ex, budget_sweep_policy,
                                     input_dev,
                                     cache=result_cache, model_name=model_name,
                                     code_version=code_version,
                                     dense_code_version=dense_code_version)
                row = dict(task=task, diff=str(kw), policy=label, **r)
                result["rows"].append(row)
                print(f"[{label} {i+1}/{len(examples)}] "
                      f"{task} {kw} recall={r['recall']:.2f} "
                      f"hit={r['hit']:.2f} {r['dt']:.1f}s", flush=True)
        if result_cache is not None:
            result_cache.save()
            cache.commit()
    if result_cache is not None:
        print(result_cache.summary(), flush=True)

    # Flush the diagnostic shards to disk if dumping was on. Path lives under
    # /cache so it persists on the Modal Volume; download with
    # `modal volume get voi-router-hf-cache <path>`.
    if dump_block_scores_path:
        import dump_block_scores as _dbs
        _dbs.flush_and_teardown()
        cache.commit()

    result["buckets"] = buckets
    result["ctx"] = ctx
    result["n_per"] = n_per
    result["policies"] = list(policies)
    result["router_mode"] = router_mode
    result["router_values"] = list(router_values)
    result["budget_values"] = list(budget_values)
    result["budget_sweep_policy"] = budget_sweep_policy
    result["code_version"] = code_version
    return result


@app.local_entrypoint()
def main(n_per: int = 2, ctxlen: int = 32768,
         policies: str = "topk,dense,router",
         fixed_budget: int = 33,
         num_keys: int = 4,
         num_hops: int = 3,
         num_distractor_chains: int = 3,
         tasks: str = "niah_multikey,vt",
         router_mode: str = "quantile",
         router_grain: str = "row",
         router_values: str = "0.20",
         router_score: str = "mean",
         router_score_sweep: str = "",
         budget_values: str = "",
         budget_sweep_policy: str = "quest",
         seed: int = 0,
         kernel_v2: bool = True,
         model: str = MODEL,
         dump_block_scores: str = "",
         no_cache: bool = False):
    # RULER smoke: two tasks at the same ctx -- NIAH (single-hop budget
    # contention) and VT (multi-hop chain tracing).
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    bucket_specs = {
        "niah_multikey": ("niah_multikey", {"num_keys": num_keys}),
        "vt": ("vt", {"num_hops": num_hops,
                      "num_distractor_chains": num_distractor_chains}),
    }
    buckets = [bucket_specs[t] for t in task_list]
    pols = [p.strip() for p in policies.split(",")]
    val_list = [float(v.strip()) for v in router_values.split(",") if v.strip()]
    bud_list = [int(v.strip()) for v in budget_values.split(",") if v.strip()]
    score_sweep = [s.strip() for s in router_score_sweep.split(",") if s.strip()] or None

    from result_cache import hash_files
    prompt_files = ["poc_core.py", "ruler_tasks.py", "pointer_haystack.py"]
    sparse_files = prompt_files + ["sparse_attention.py", "triton_block_attn.py"]
    code_version = hash_files(sparse_files)
    dense_code_version = hash_files(prompt_files)

    result = experiment.remote(n_per, ctxlen, buckets, pols, fixed_budget,
                               router_mode, router_grain, val_list,
                               router_score,
                               seed, kernel_v2,
                               code_version, dense_code_version,
                               bud_list, budget_sweep_policy, model,
                               dump_block_scores, no_cache,
                               router_score_sweep=score_sweep)

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

    # Budget-sweep panel: uniform `budget_sweep_policy` at varied budgets.
    # poc_core's tables drop unknown policy labels; this prints them.
    if bud_list:
        n = len(drows)
        dc_idx_b = [i for i, r in enumerate(drows) if r["recall"] >= 1.0]
        print(f"\nbudget sweep ({budget_sweep_policy} at varied budgets):")
        for b in bud_list:
            label = f"{budget_sweep_policy}_b{b}"
            brows = [r for r in result["rows"] if r["policy"] == label]
            if not brows:
                continue
            rec = sum(r["recall"] for r in brows) / len(brows)
            hit = sum(r["hit"] for r in brows) / len(brows)
            dt = sum(r["dt"] for r in brows) / len(brows)
            line = (f"  b={b:>3}: recall={rec:.3f}  hit={hit:.3f}  "
                    f"mean_dt={dt:.2f}s  n={len(brows)}")
            if dc_idx_b and len(brows) == n:
                kept = sum(brows[i]["recall"] >= 1.0
                           for i in dc_idx_b) / len(dc_idx_b)
                line += f"  paired={kept:.2f}"
            print(line)

    # Effective-budget audit: derive blocks/tile per policy from policy label
    # + matched_budget. v2 selection (sparse_attention.py:688) always doubles
    # budget_blocks; router adds quantile-fraction expansion on top.
    print(f"\nEFFECTIVE BUDGET per policy "
          f"(matched_budget={result['matched_budget']}, kernel_v2 always 2x):")
    pol_seen = []
    for r in result["rows"]:
        if r["policy"] not in pol_seen:
            pol_seen.append(r["policy"])
    for p in pol_seen:
        n = sum(1 for r in result["rows"] if r["policy"] == p)
        eff = _policy_eff_budget(p, result["matched_budget"])
        eff_s = "full ctx" if eff is None else f"{eff:.1f}"
        print(f"  {p:<24} eff_budget={eff_s:>9}  n={n}")

    import os
    os.makedirs("results", exist_ok=True)
    path = f"results/ruler_smoke_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"saved -> {path}")
