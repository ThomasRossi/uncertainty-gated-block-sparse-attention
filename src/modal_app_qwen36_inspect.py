"""
modal_app_qwen36_inspect.py
===========================

Step-1 ground truth for Qwen3.6-35B-A3B (HF id: Qwen/Qwen3.6-35B-A3B,
model_type "qwen3_5_moe", arch "Qwen3_5MoeForConditionalGeneration").
Two-stage inspect:

  Stage A (CPU, ~$0.02, ~2 min) -- config + architecture skeleton only.
    * AutoConfig.from_pretrained: confirm trust_remote_code requirement.
    * Pin a recent transformers; if Qwen3_5Moe* class is missing, bail with
      a clear "upgrade transformers to >=X" message instead of burning GPU.
    * AutoModelForCausalLM.from_config on the text_config sub-config:
      instantiates the architecture skeleton on CPU with random weights.
      Gets us class names + per-layer attention module shapes for free.
    * Per-layer enumeration: confirm cfg.text_config.layer_types matches
      what model.model.layers[i].self_attn is actually doing (string-match
      "linear_attention" vs "full_attention" -> two different attn classes).
    * Full attn module repr for one DeltaNet layer + one full-attn layer,
      including named_parameters shapes, so we know:
        - the GQA proj shapes (q_proj/k_proj/v_proj/o_proj for full)
        - whether attn_output_gate adds a g_proj sibling
        - partial-rope dim layout (64 of 256)

  Stage B (H100-80GB, ~$0.50, ~5 min) -- weight load + chat template only.
    Guarded by --weights flag; default skips. Loads bf16 weights to confirm:
        - real model loads cleanly (no trust_remote_code surprises)
        - tokenizer.chat_template exists
        - GPU memory footprint matches our prediction (~70 GB bf16)
        - one teacher-forced forward of "hello" produces sane logits

Total stage-A only run: ~$0.05. Stage A+B: ~$0.60. Cheaper than the K2/MLA
inspect rounds because we're not chasing FP8 dequant.
"""

import modal

MODEL = "Qwen/Qwen3.6-35B-A3B"

# Transformers version: Qwen3_5Moe lands in 5.x stable. Pin newest we've
# verified loads cleanly in our other apps; if the class is missing we
# fail loud in stage A before paying for GPU.
TRANSFORMERS_VERSION = "5.12.1"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        f"transformers=={TRANSFORMERS_VERSION}",
        "accelerate",
        "safetensors",
        "hf_transfer",
        "sentencepiece",
        "tiktoken",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/cache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

cache = modal.Volume.from_name("voi-router-hf-cache", create_if_missing=True)
app = modal.App("voi-router-qwen36-inspect")


def _print_attn_module(label, mod):
    print(f"\n== {label}: type={type(mod).__module__}.{type(mod).__name__}", flush=True)
    cfg_keys = [k for k in mod.__dict__.keys() if not k.startswith("_")]
    print(f"   instance attrs: {cfg_keys}", flush=True)
    print(f"   children:", flush=True)
    for name, sub in mod.named_children():
        w = getattr(sub, "weight", None)
        shape = tuple(w.shape) if w is not None else None
        print(f"     {name:24s} {type(sub).__name__:24s} weight={shape}", flush=True)
    print(f"   parameters (recurse):", flush=True)
    for name, p in mod.named_parameters(recurse=True):
        print(f"     {name:40s} {tuple(p.shape)}  {p.dtype}", flush=True)


@app.function(image=image, volumes={"/cache": cache}, timeout=900, cpu=4,
              memory=32 * 1024)
def inspect_cpu():
    """Stage A: config + skeleton, CPU only."""
    print(f"=== Stage A: Qwen3.6 config + skeleton (CPU) ===", flush=True)
    print(f"  transformers pin: {TRANSFORMERS_VERSION}", flush=True)
    import transformers
    print(f"  transformers actually loaded: {transformers.__version__}", flush=True)

    from transformers import AutoConfig

    # ---- 1) Config (cheap, just JSON).
    print(f"\n--- 1) AutoConfig.from_pretrained({MODEL!r}) ---", flush=True)
    try:
        cfg = AutoConfig.from_pretrained(MODEL)
        trust_remote = False
    except Exception as e:
        print(f"  without trust_remote_code: {e!r}", flush=True)
        cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        trust_remote = True
    print(f"  trust_remote_code needed: {trust_remote}", flush=True)
    print(f"  model_type = {cfg.model_type}", flush=True)
    print(f"  architectures = {getattr(cfg, 'architectures', None)}", flush=True)

    # The text-only sub-config is what an LM-only patch path operates on.
    tcfg = getattr(cfg, "text_config", cfg)
    print(f"\n  text_config.model_type = {getattr(tcfg, 'model_type', None)}", flush=True)
    interesting = [
        "num_hidden_layers", "hidden_size", "num_attention_heads",
        "num_key_value_heads", "head_dim", "attn_output_gate",
        "partial_rotary_factor", "rope_parameters", "full_attention_interval",
        "linear_num_key_heads", "linear_num_value_heads",
        "linear_key_head_dim", "linear_value_head_dim",
        "linear_conv_kernel_dim", "moe_intermediate_size",
        "num_experts", "num_experts_per_tok",
        "shared_expert_intermediate_size", "max_position_embeddings",
    ]
    for k in interesting:
        if hasattr(tcfg, k):
            print(f"    {k:30s} = {getattr(tcfg, k)!r}", flush=True)

    layer_types = getattr(tcfg, "layer_types", None)
    if layer_types:
        print(f"\n  layer_types ({len(layer_types)}):", flush=True)
        for i, lt in enumerate(layer_types):
            print(f"    [{i:2d}] {lt}", flush=True)
        counts = {lt: layer_types.count(lt) for lt in set(layer_types)}
        print(f"  layer_type counts: {counts}", flush=True)

    # ---- 2) Skeleton instantiation. We do NOT download weights.
    print(f"\n--- 2) skeleton (no weights) ---", flush=True)
    # Try the text-only AutoModelForCausalLM path first. If the arch only
    # registers under VisionToText, fall back.
    skeleton = None
    skel_kwargs = dict(trust_remote_code=True) if trust_remote else {}
    try:
        from transformers import AutoModelForCausalLM
        skeleton = AutoModelForCausalLM.from_config(cfg, **skel_kwargs)
        print(f"  AutoModelForCausalLM.from_config: ok ({type(skeleton).__name__})",
              flush=True)
    except Exception as e:
        print(f"  AutoModelForCausalLM.from_config failed: {e!r}", flush=True)
        # Try text-only sub-config.
        try:
            from transformers import AutoModelForCausalLM
            skeleton = AutoModelForCausalLM.from_config(tcfg, **skel_kwargs)
            print(f"  AutoModelForCausalLM.from_config(text_config): ok "
                  f"({type(skeleton).__name__})", flush=True)
        except Exception as e2:
            print(f"  text_config path also failed: {e2!r}", flush=True)
            try:
                from transformers import AutoModel
                skeleton = AutoModel.from_config(cfg, **skel_kwargs)
                print(f"  AutoModel.from_config: ok ({type(skeleton).__name__})",
                      flush=True)
            except Exception as e3:
                print(f"  AutoModel.from_config also failed: {e3!r}", flush=True)
                raise RuntimeError(
                    "could not instantiate Qwen3.6 skeleton -- transformers "
                    f"{transformers.__version__} likely missing Qwen3_5Moe* class. "
                    "Bump pin or try trust_remote_code if/when available."
                )

    # ---- 3) Find the text decoder layers regardless of multimodal wrapping.
    print(f"\n--- 3) locating decoder layers ---", flush=True)
    decoder = None
    for path in ["model.layers", "model.model.layers",
                 "model.language_model.layers",
                 "language_model.model.layers"]:
        cur = skeleton
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            decoder = cur
            print(f"  found decoder layers at: {path}  ({len(decoder)} layers)",
                  flush=True)
            break
    if decoder is None:
        print(f"  could NOT find decoder layers; full skeleton structure:",
              flush=True)
        for name, mod in skeleton.named_modules():
            if "layer" in name.lower() and len(name.split(".")) <= 4:
                print(f"    {name}: {type(mod).__name__}", flush=True)
        raise RuntimeError("decoder layers not found in skeleton")

    # ---- 4) Per-layer attention class identity check.
    print(f"\n--- 4) per-layer self_attn class ---", flush=True)
    seen = {}
    for i, layer in enumerate(decoder):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        if attn is None:
            print(f"  layer {i}: NO self_attn / attention attr; children = "
                  f"{[n for n,_ in layer.named_children()]}", flush=True)
            continue
        kind = type(attn).__name__
        seen.setdefault(kind, []).append(i)
    for kind, idxs in seen.items():
        rng = f"{idxs[0]}..{idxs[-1]}" if len(idxs) > 1 else str(idxs[0])
        print(f"  {kind:48s} x{len(idxs):3d}  (layers {rng})", flush=True)
        if layer_types:
            sample_i = idxs[0]
            print(f"    layer_types[{sample_i}] = {layer_types[sample_i]!r}",
                  flush=True)

    # ---- 5) Deep dive into one of each attention class.
    print(f"\n--- 5) attn module deep dive (one of each kind) ---", flush=True)
    for kind, idxs in seen.items():
        i = idxs[0]
        layer = decoder[i]
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        _print_attn_module(f"layer[{i}] ({kind})", attn)
        # The gate proj for attn_output_gate (full-attn only) typically shows up
        # as an extra Linear we don't see in vanilla Qwen.
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj",
                     "g_proj", "gate_proj", "output_gate",
                     "num_heads", "num_attention_heads", "num_key_value_heads",
                     "head_dim", "hidden_size", "rotary_emb",
                     "attn_output_gate", "layer_idx", "scaling"]:
            if hasattr(attn, attr):
                v = getattr(attn, attr)
                v_repr = (f"{type(v).__name__}" if hasattr(v, "named_children")
                          else repr(v))
                print(f"     attn.{attr:24s} = {v_repr}", flush=True)

    # ---- 6) AttentionInterface registry (for monkey-patch slot).
    print(f"\n--- 6) AttentionInterface registry ---", flush=True)
    try:
        from transformers import AttentionInterface
        keys = []
        for attr in ["_registry", "registry", "implementations"]:
            r = getattr(AttentionInterface, attr, None)
            if isinstance(r, dict):
                keys = sorted(r.keys())
                break
        print(f"  registered keys: {keys}", flush=True)
    except Exception as e:
        print(f"  introspection failed: {e!r}", flush=True)

    print(f"\n=== stage A complete ===", flush=True)
    return {
        "ok": True,
        "trust_remote": trust_remote,
        "transformers_version": transformers.__version__,
        "model_type": cfg.model_type,
        "attn_class_counts": {k: len(v) for k, v in seen.items()},
        "layer_types": layer_types,
        "skeleton_class": type(skeleton).__name__,
    }


@app.function(image=image, gpu="H100:1", volumes={"/cache": cache}, timeout=1800,
              memory=64 * 1024)
def inspect_weights():
    """Stage B: weight load + tokenizer + tiny forward. ~70 GB bf16 -> H100-80GB."""
    print(f"=== Stage B: Qwen3.6 weight load + smoke forward ===", flush=True)
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    print(f"  transformers: {transformers.__version__}", flush=True)

    cfg = AutoConfig.from_pretrained(MODEL)
    print(f"  config OK; model_type={cfg.model_type}", flush=True)

    print(f"\n--- loading tokenizer ---", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"  tokenizer class: {type(tok).__name__}", flush=True)
    has_chat = bool(getattr(tok, "chat_template", None))
    print(f"  chat_template present: {has_chat}", flush=True)
    if has_chat:
        sample = tok.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            add_generation_prompt=True, tokenize=False)
        print(f"  rendered sample (first 300 chars): {sample[:300]!r}", flush=True)

    print(f"\n--- loading weights (bf16) ---", flush=True)
    import time
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    cache.commit()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    used_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  gpu memory after load: {used_gb:.2f} GB", flush=True)
    print(f"  model class: {type(model).__name__}", flush=True)

    # Pick out the decoder layers regardless of multimodal wrapping.
    decoder = None
    for path in ["model.layers", "model.model.layers",
                 "model.language_model.layers",
                 "language_model.model.layers"]:
        cur = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False; break
            cur = getattr(cur, part)
        if ok:
            decoder = cur
            print(f"  decoder layers at: {path}  ({len(decoder)} layers)", flush=True)
            break
    if decoder is not None:
        # Print one of each attention class WITH real weights, to confirm
        # everything is materialised (not meta) and shapes match config.
        seen = {}
        for i, layer in enumerate(decoder):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
            if attn is None: continue
            seen.setdefault(type(attn).__name__, []).append(i)
        for kind, idxs in seen.items():
            i = idxs[0]
            attn = (getattr(decoder[i], "self_attn", None)
                    or getattr(decoder[i], "attention", None))
            _print_attn_module(f"weighted layer[{i}] ({kind})", attn)

    # ---- tiny forward, prefill only, check logits look sane.
    print(f"\n--- tiny forward (prefill 'hello') ---", flush=True)
    inp = tok("hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits if hasattr(out, "logits") else out[0]
    print(f"  logits shape: {tuple(logits.shape)}  dtype={logits.dtype}", flush=True)
    last_tok = int(logits[0, -1].argmax().item())
    print(f"  argmax next-token id: {last_tok}  decoded: {tok.decode([last_tok])!r}",
          flush=True)
    print(f"  logits[0,-1,:8]: {logits[0,-1,:8].float().tolist()}", flush=True)

    print(f"\n=== stage B complete ===", flush=True)
    return {
        "ok": True,
        "model_class": type(model).__name__,
        "gpu_alloc_gb": used_gb,
        "has_chat_template": has_chat,
        "next_token_id": last_tok,
    }


@app.local_entrypoint()
def main(weights: bool = False):
    print(f"=== Qwen3.6 inspect: stage A ===", flush=True)
    a = inspect_cpu.remote()
    print(f"\n--- stage A result ---")
    print(a)
    if not weights:
        print(f"\n(skipping stage B; rerun with --weights to load 70 GB on H100)")
        return
    print(f"\n=== Qwen3.6 inspect: stage B ===", flush=True)
    b = inspect_weights.remote()
    print(f"\n--- stage B result ---")
    print(b)
