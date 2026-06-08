"""
result_cache.py
===============

Input-keyed cache for `run_one` results. A `run_one` call is a pure function
of (model, prompt, policy, budget, peek_config, code_version) -> result, so
once we've paid the GPU cost for a configuration there is no reason to pay
again on a later run.

The cache is a single JSON file (`/cache/result_cache.json` in the Modal
Volume), keyed by a SHA-1 hash of the input-determining kwargs. Loaded at
the start of `experiment()`, persisted at the end. Safe to delete the file
to start from scratch.

Invalidation: every entry's key includes a `code_version` -- a hash of the
contents of the source files that determine results (sparse_attention.py,
poc_core.py, ruler_tasks.py, pointer_haystack.py). Any change to those files
shifts `code_version` and effectively invalidates everything. Conservative
but safe; no manual flag needed.
"""

import hashlib
import json
import os


class ResultCache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.data = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(**kwargs):
        """Stable SHA-1 of the input-determining kwargs."""
        s = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha1(s.encode()).hexdigest()

    def get(self, key):
        v = self.data.get(key)
        if v is not None:
            self.hits += 1
        else:
            self.misses += 1
        return v

    def put(self, key, value):
        self.data[key] = value

    def save(self):
        if not self.path:
            return
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)

    def __len__(self):
        return len(self.data)

    def summary(self):
        total = self.hits + self.misses
        rate = self.hits / total if total else 0.0
        return (f"cache: {len(self.data)} entries, "
                f"{self.hits} hits / {self.misses} misses ({rate:.0%})")


def hash_files(paths):
    """SHA-1 hash of the concatenated content of `paths`. Order matters."""
    h = hashlib.sha1()
    for p in paths:
        with open(p, "rb") as f:
            h.update(p.encode())
            h.update(b"\0")
            h.update(f.read())
            h.update(b"\0")
    return h.hexdigest()[:12]


def cache_key_for_run(model_name, ex, mode, max_new, code_version, SEL,
                      dense_code_version=None):
    """
    Construct the cache key for one `run_one` invocation.

    Inputs that determine the result:
      - The model identity.
      - The prompt content (via prompt hash; content-addressed, no order-
        dependence on example generation).
      - The policy (mode) and its budget.
      - For belief / rollout / rollout_peek: their respective config knobs.
      - max_new (number of generated tokens; deterministic given do_sample=False).
      - code_version: invalidates on any change to the result-determining
        source files. For dense, `dense_code_version` (if provided) is used
        instead -- dense bypasses the selector and our custom kernel, so it
        does not need to invalidate when sparse_attention.py or
        triton_block_attn.py change.
    """
    prompt_hash = hashlib.sha1(ex.prompt.encode()).hexdigest()[:16]
    cv = (dense_code_version if (mode == "dense" and dense_code_version)
          else code_version)
    d = {
        "model": model_name,
        "policy": mode,
        "prompt_hash": prompt_hash,
        "max_new": max_new,
        "code_version": cv,
    }
    if mode != "dense":
        d["budget"] = SEL.budget_blocks
        # kernel_v2 only included when on, so existing v1 cache entries keep
        # matching (default kernel_v2=False -> key absent).
        if getattr(SEL, "kernel_v2", False):
            d["kernel_v2"] = True
    if mode == "belief":
        d["top_p"] = SEL.top_p
        d["k_min"] = SEL.k_min
        d["k_max"] = SEL.k_max
    elif mode == "rollout":
        d["explore_bonus"] = SEL.explore_bonus
        d["ema_alpha"] = SEL.ema_alpha
    elif mode == "rollout_peek":
        d["peek_swap_w"] = SEL.peek_swap_w
        d["peek_rows_per_layer"] = SEL.peek_rows_per_layer
        d["peek_layer_min"] = SEL.peek_layer_min
        d["peek_layer_max"] = SEL.peek_layer_max
        d["peek_metric"] = SEL.peek_metric
        if SEL.peek_metric == "cross_head_agreement":
            d["peek_xh_threshold"] = SEL.peek_xh_threshold
    elif mode == "router":
        # Default mode/grain (abs/cell): existing entries stay valid. Other
        # combinations add extra keys so they don't collide with defaults.
        if getattr(SEL, "router_mode", "abs") == "abs":
            d["router_tau"] = SEL.router_tau
        else:
            d["router_mode"] = SEL.router_mode
            d["router_quantile"] = SEL.router_quantile
        if getattr(SEL, "router_grain", "cell") != "cell":
            d["router_grain"] = SEL.router_grain
        if getattr(SEL, "router_score", "mean") != "mean":
            d["router_score"] = SEL.router_score
    return ResultCache.make_key(**d)
