"""
modal_app_qwen36_smoke.py
=========================

End-to-end wiring smoke for Qwen3.6-35B-A3B (model_type qwen3_5_moe). Loads
the bf16 weights, sets attn_implementation="block_sparse" (our existing
registered dispatch), runs RULER NIAH multikey n=4 @ 8K across four
policies:

    dense (SDPA) | topk-v2 | quest | router-on-Quest

Three claims this should validate:

  (1) AttentionInterface dispatch fires inside Qwen3_5MoeAttention.forward.
      Verified by a per-layer call counter: only the 10 full-attention layers
      (3, 7, 11, ..., 39) should appear.
  (2) Gate (sigmoid * attn_output) + QK-Norm + o_proj wrap correctly around
      our dispatch. Verified by sane logits + non-zero dense recall.
  (3) Selector ordering reproduces in a hybrid-architecture model:
      topk << Quest << router-on-Quest, matching MLA / Qwen / Nemo families.

Budget: ~$1.50, ~10-15 min (H200:1 hourly ~$5; one-time weight download
~3-4 min, NIAH n=4 prefill+decode ~5 min).
"""

import json
import time

import modal

MODEL = "Qwen/Qwen3.6-35B-A3B"
TRANSFORMERS_VERSION = "5.12.1"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", f"transformers=={TRANSFORMERS_VERSION}", "accelerate",
        "safetensors", "hf_transfer", "triton", "sentencepiece", "tiktoken",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source(
        "sparse_attention", "triton_block_attn", "poc_core",
        "ruler_tasks", "pointer_haystack", "result_cache",
    )
)

cache = modal.Volume.from_name("voi-router-hf-cache", create_if_missing=True)
app = modal.App("voi-router-qwen36-smoke")


@app.function(image=image, gpu="H200:1", volumes={"/cache": cache},
              timeout=3600, memory=64 * 1024)
def smoke(n_per: int, ctx: int, num_keys: int, seed: int, budget: int,
          code_version: str, dense_code_version: str):
    import random
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # Import sparse_attention -- triggers AttentionInterface.register("block_sparse", ...)
    import sparse_attention
    from sparse_attention import SEL
    import poc_core
    from result_cache import ResultCache

    SEL.kernel_v2 = True

    print(f"=== loading {MODEL} (bf16) ===", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = AutoConfig.from_pretrained(MODEL)
    # Belt-and-suspenders: set on top-level config AND text sub-config.
    if hasattr(cfg, "text_config"):
        cfg.text_config._attn_implementation = "block_sparse"
    cfg._attn_implementation = "block_sparse"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        config=cfg,
        dtype=torch.bfloat16,
        attn_implementation="block_sparse",
        low_cpu_mem_usage=True,
        device_map="cuda",
    ).eval()
    cache.commit()
    used_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  loaded in {time.time()-t0:.1f}s, gpu alloc {used_gb:.2f} GB",
          flush=True)
    print(f"  model class: {type(model).__name__}", flush=True)
    print(f"  config._attn_implementation: "
          f"{getattr(model.config, '_attn_implementation', '<absent>')}", flush=True)
    if hasattr(model.config, "text_config"):
        print(f"  text_config._attn_implementation: "
              f"{getattr(model.config.text_config, '_attn_implementation', '<absent>')}",
              flush=True)

    # Identify full-attn layer indices (linear-attn layers have no self_attn).
    full_attn_layers = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            full_attn_layers.append(i)
            attn_cls = type(layer.self_attn).__name__
    print(f"  full-attn layers ({len(full_attn_layers)}): "
          f"{full_attn_layers}", flush=True)
    print(f"  attn class: {attn_cls}", flush=True)

    # ---- 1) call-counter probe: confirm dispatch fires only on full-attn layers.
    print(f"\n=== 1) call-counter probe (dispatch verification) ===", flush=True)
    call_log = []
    orig = sparse_attention.block_sparse_attention
    def _counted(module, q, k, v, attention_mask, scaling, dropout=0.0, **kw):
        call_log.append((getattr(module, "layer_idx", -1),
                         q.shape[-2], q.shape[1], q.shape[-1]))
        return orig(module, q, k, v, attention_mask, scaling, dropout=dropout, **kw)
    sparse_attention.block_sparse_attention = _counted
    # Re-register so the dispatch picks up the wrapper.
    from transformers import AttentionInterface
    AttentionInterface.register("block_sparse", _counted)

    SEL.mode = "dense"
    enc = tok("The capital of France is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**enc, use_cache=False)
    print(f"  one prefill: {len(call_log)} dispatch calls", flush=True)
    if call_log:
        seen_idxs = sorted(set(c[0] for c in call_log))
        print(f"  unique layer_idx seen: {seen_idxs}", flush=True)
        print(f"  expected full-attn layers: {full_attn_layers}", flush=True)
        assert seen_idxs == full_attn_layers, (
            f"dispatch fired on unexpected layers: "
            f"seen={seen_idxs} expected={full_attn_layers}")
        sample = call_log[0]
        print(f"  sample call (layer {sample[0]}): "
              f"M={sample[1]} num_q_heads={sample[2]} head_dim={sample[3]}",
              flush=True)
        # Sanity-check the next-token prediction looks reasonable.
        logits = out.logits[0, -1].float()
        top5 = logits.topk(5)
        print(f"  dense top-5: ids={top5.indices.tolist()} "
              f"-> {[tok.decode([i]) for i in top5.indices.tolist()]}", flush=True)
    else:
        raise RuntimeError("dispatch never fired -- attn_implementation wiring broken")

    # Restore original (uncounted) for the sweep so we don't bloat call_log.
    sparse_attention.block_sparse_attention = orig
    AttentionInterface.register("block_sparse", orig)

    # ---- 2) RULER NIAH multikey n=n_per @ ctx, policy sweep.
    print(f"\n=== 2) RULER NIAH multikey n={n_per} ctx={ctx} budget={budget} ===",
          flush=True)
    rng = random.Random(seed)
    examples = [("niah_multikey", {"num_keys": num_keys},
                 poc_core.make_example(rng, "niah_multikey",
                                       {"num_keys": num_keys}, ctx))
                for _ in range(n_per)]
    print(f"  {len(examples)} examples generated", flush=True)

    SEL.budget_blocks = budget
    SEL.kernel_v2 = True
    # Router-on-Quest config -- matches the paper's headline policy.
    SEL.router_score = "quest"
    SEL.router_mode = "quantile"
    SEL.router_quantile = 0.40

    result_cache = ResultCache("/cache/result_cache_qwen36.json")
    print(f"  result cache: {len(result_cache)} entries", flush=True)

    # Policies in increasing strength order. "router" picks up the Quest
    # backbone via SEL.router_score = "quest".
    policies = ("dense", "topk", "quest", "router")
    result = poc_core.run_sweep(
        model, tok, examples, "cuda",
        policies=policies, fixed_budget=budget,
        cache=result_cache, model_name=MODEL,
        code_version=code_version, dense_code_version=dense_code_version,
    )
    result_cache.save()
    cache.commit()
    print(result_cache.summary(), flush=True)

    buckets = [("niah_multikey", {"num_keys": num_keys})]
    print("\n" + poc_core.summary_table(buckets, result["rows"]))
    return dict(
        full_attn_layers=full_attn_layers,
        dispatch_calls_per_prefill=len(call_log),
        rows=result["rows"], n_per=n_per, ctx=ctx, budget=budget,
        gpu_alloc_gb=used_gb,
    )


@app.local_entrypoint()
def main(n_per: int = 4, ctxlen: int = 8192, num_keys: int = 4,
         budget: int = 8, seed: int = 0):
    from result_cache import hash_files
    prompt_files = ["poc_core.py", "ruler_tasks.py", "pointer_haystack.py"]
    sparse_files = prompt_files + ["sparse_attention.py", "triton_block_attn.py"]
    code_version = hash_files(sparse_files)
    dense_code_version = hash_files(prompt_files)

    result = smoke.remote(n_per, ctxlen, num_keys, seed, budget,
                          code_version, dense_code_version)

    print("\n=== smoke summary ===", flush=True)
    print(f"  full-attn layers patched: {result['full_attn_layers']}")
    print(f"  dispatch calls per prefill: {result['dispatch_calls_per_prefill']} "
          f"(expected {len(result['full_attn_layers'])})")
    print(f"  gpu alloc: {result['gpu_alloc_gb']:.1f} GB")
    rows = result["rows"]
    if rows:
        by_policy = {}
        for r in rows:
            by_policy.setdefault(r.get("mode") or r.get("policy"), []).append(r)
        for p, rs in by_policy.items():
            rec = sum(r["recall"] for r in rs) / len(rs)
            print(f"  {p:14s} recall (n={len(rs)}) = {rec:.3f}")

    import os
    os.makedirs("results", exist_ok=True)
    path = f"results/qwen36_smoke_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"saved -> {path}")
