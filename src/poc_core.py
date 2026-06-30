"""
poc_core.py
===========

Device-agnostic experiment core for the SSA selection proof-of-concept.
Shared by the Mac path and the Modal GPU path (modal_app.py).

Runs Qwen2.5 on RULER tasks under three attention policies (dense / topk /
belief) and reports recall plus needle-block hit rate.
"""

import time

import torch

import pointer_haystack as ph
import ruler_tasks as rt
from sparse_attention import BLOCK, SEL


def make_example(rng, task, kw, ctx):
    if task == "vt":
        return rt.make_vt(rng, ctx, **kw)
    if task == "pointer_chase":
        return ph.make_pointer_chase(rng, ctx, **kw)
    return rt.make_niah_multikey(rng, ctx, **kw)


def locate(tok, ex):
    """Tokenize the templated prompt; return (encoding, gold needle block set)."""
    # Qwen3+ reasoning models default to enable_thinking=True and emit a
    # <think>...</think> trace that consumes max_new_tokens before the answer.
    # Suppress only when the template references the variable; rendered text is
    # otherwise identical, so non-reasoning model cache keys are preserved.
    extra = {}
    uses_thinking = "enable_thinking" in (getattr(tok, "chat_template", None) or "")
    if uses_thinking:
        extra["enable_thinking"] = False
    # MC tasks on chat reasoning models: even with thinking disabled, the model
    # discursively walks through "Option (A)...", so the first A/B/C/D the
    # regex hits is not the answer. Bypass via assistant prefill: strip the
    # trailing "The correct answer is" from the user message, prefill the
    # assistant turn with "The correct answer is (", and let the model emit
    # one letter. Gated to reasoning models so non-reasoning cache keys stay
    # valid.
    if uses_thinking and getattr(ex, "task", None) == "longbench_v2":
        marker = "The correct answer is"
        user_text = ex.prompt.rstrip()
        if user_text.endswith(marker):
            user_text = user_text[:-len(marker)].rstrip()
            text = tok.apply_chat_template(
                [{"role": "user", "content": user_text},
                 {"role": "assistant", "content": "The correct answer is ("}],
                continue_final_message=True, tokenize=False, **extra)
        else:
            text = tok.apply_chat_template(
                [{"role": "user", "content": ex.prompt}],
                add_generation_prompt=True, tokenize=False, **extra)
    else:
        text = tok.apply_chat_template(
            [{"role": "user", "content": ex.prompt}],
            add_generation_prompt=True, tokenize=False, **extra)
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt",
              add_special_tokens=False)
    offsets = enc["offset_mapping"][0].tolist()
    blocks = set()
    for needle in ex.needles:
        c = text.find(needle)
        if c < 0:
            continue
        for ti, (s, e) in enumerate(offsets):
            if e > c and s < c + len(needle):
                blocks.add(ti // BLOCK)
    return enc, blocks


def needle_hit_rate(needle_blocks):
    """Mean over layers of (gold needle blocks kept) / (gold needle blocks)."""
    if not SEL.kept_blocks or not needle_blocks:
        return 1.0
    rates = []
    for kept in SEL.kept_blocks.values():
        hit = sum(1 for b in needle_blocks if b < len(kept) and kept[b])
        rates.append(hit / len(needle_blocks))
    return sum(rates) / len(rates)


def run_one(model, tok, ex, mode, device, max_new=40,
            cache=None, model_name=None, code_version=None,
            dense_code_version=None):
    # If we've already run this exact configuration, return the saved result.
    if cache is not None:
        from result_cache import cache_key_for_run
        ck = cache_key_for_run(model_name, ex, mode, max_new, code_version, SEL,
                               dense_code_version=dense_code_version)
        cached = cache.get(ck)
        if cached is not None:
            cached = dict(cached)
            cached["dt"] = 0.0          # report cache hit as zero wall cost
            cached["cached"] = True
            return cached
    enc, nblocks = locate(tok, ex)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    SEL.log = True
    SEL.reset_log()
    t0 = time.time()
    if mode == "rollout_peek":
        # Pre-pass under topk to cache K̄ at every layer, so the peek pass can
        # score candidates against the *real* K̄_{L+1} instead of L's proxy.
        # Counted in `dt` because it is part of the policy's true wall cost.
        SEL.mode = "topk"
        SEL.log = False
        with torch.no_grad():
            model(input_ids=ids, attention_mask=am, use_cache=False)
        SEL.log = True
        SEL.kept_blocks.clear()
        SEL.budget_samples.clear()
    SEL.mode = mode
    with torch.no_grad():
        out = model.generate(ids, attention_mask=am, max_new_tokens=max_new,
                             do_sample=False)
    dt = time.time() - t0
    txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    budget = (sum(SEL.budget_samples) / len(SEL.budget_samples)
              if SEL.budget_samples else float("nan"))
    result = dict(recall=rt.score(ex, txt), hit=needle_hit_rate(nblocks),
                  dt=dt, n_tok=int(ids.shape[1]), budget=budget, output=txt)
    if cache is not None:
        cache.put(ck, result)
    return result


def run_sweep(model, tok, examples, device, max_new=40, verbose=True,
              policies=("topk", "dense", "rollout"), fixed_budget=None,
              cache=None, model_name=None, code_version=None,
              dense_code_version=None):
    """
    examples: list of (task, kw_dict, Example).

    If `fixed_budget` is None: run a belief pass first to derive the matched
    budget, then iterate `policies` at that budget. Belief rows come first.
    If `fixed_budget` is given: skip the belief pass and use it directly;
    rows contain only the requested policies.

    Returns dict(matched_budget, rows). Row order within `rows` is the order
    policies were run (belief first if applicable, then `policies` in order).
    """
    rows = []
    if fixed_budget is None:
        belief_budgets = []
        for i, (task, kw, ex) in enumerate(examples):
            r = run_one(model, tok, ex, "belief", device, max_new,
                        cache=cache, model_name=model_name,
                        code_version=code_version,
                        dense_code_version=dense_code_version)
            belief_budgets.append(r["budget"])
            rows.append(dict(task=task, diff=str(kw), policy="belief", **r))
            if verbose:
                print(f"[belief {i+1}/{len(examples)}] {task} {kw} "
                      f"recall={r['recall']:.2f} hit={r['hit']:.2f} "
                      f"{r['dt']:.1f}s", flush=True)
        matched = max(1, round(sum(belief_budgets) / len(belief_budgets)))
        # Avoid double-running belief if it also appears in `policies`.
        policies = tuple(p for p in policies if p != "belief")
    else:
        matched = int(fixed_budget)
    SEL.budget_blocks = matched
    if verbose:
        print(f"matched fixed budget = {matched} blocks", flush=True)

    for policy in policies:
        for i, (task, kw, ex) in enumerate(examples):
            r = run_one(model, tok, ex, policy, device, max_new,
                        cache=cache, model_name=model_name,
                        code_version=code_version,
                        dense_code_version=dense_code_version)
            rows.append(dict(task=task, diff=str(kw), policy=policy, **r))
            if verbose:
                print(f"[{policy} {i+1}/{len(examples)}] {task} {kw} "
                      f"recall={r['recall']:.2f} hit={r['hit']:.2f} "
                      f"{r['dt']:.1f}s", flush=True)

    return dict(matched_budget=matched, rows=rows)


def summary_table(buckets, rows):
    present = {r["policy"] for r in rows}
    base_order = ("dense", "topk", "quest", "belief", "rollout", "rollout_peek")
    order = [p for p in base_order if p in present]
    # Append router sweep policies in first-seen order, after the base policies.
    seen_router = []
    for r in rows:
        if r["policy"].startswith("router") and r["policy"] not in seen_router:
            seen_router.append(r["policy"])
    order += seen_router
    lines = [f"{'task':<16}{'difficulty':<18}{'policy':<14}{'recall':>9}{'hit':>8}",
             "-" * 65]
    for task, kw in buckets:
        kws = str(kw)
        for policy in order:
            sel = [r for r in rows if r["task"] == task
                   and r["diff"] == kws and r["policy"] == policy]
            if not sel:
                continue
            rec = sum(r["recall"] for r in sel) / len(sel)
            hit = sum(r["hit"] for r in sel) / len(sel)
            lines.append(f"{task:<16}{kws:<18}{policy:<14}{rec:>9.3f}{hit:>8.3f}")
        lines.append("-" * 65)
    return "\n".join(lines)


def paired_table(buckets, rows, threshold=1.0):
    """
    Selector-isolated metric -- the headline number.

    run_sweep emits rows in policy blocks over the SAME example list, in order
    [belief, topk, dense, (rollout)]. Restrict to examples dense solves (the
    model can do them) and report how often each sparse policy preserves the
    answer -- this controls for the model's own reasoning error and leaves
    only the selector's contribution.

    `threshold`: an example counts as "solved" iff recall >= threshold.
    Default 1.0 (exact-match PCH); use 0.5 for F1-scored LongBench tasks.
    """
    # Discover policies in first-occurrence order (matches the run_sweep order).
    seen = []
    for r in rows:
        if r["policy"] not in seen:
            seen.append(r["policy"])
    n_policies = len(seen)
    n = len(rows) // n_policies
    by = {p: rows[i * n:(i + 1) * n] for i, p in enumerate(seen)}
    if "dense" not in by:
        return "(no dense policy in rows -- cannot pair)"
    dense = by["dense"]
    base_nondense = [p for p in ("topk", "quest", "belief", "rollout", "rollout_peek")
                     if p in by]
    router_nondense = [p for p in seen
                       if p.startswith("router") and p in by]
    nondense = base_nondense + router_nondense
    n_per = n // len(buckets)

    col_w = max(13, max((len(p) + 6 for p in nondense), default=13))
    header = f"{'task':<15}{'difficulty':<17}{'n dense-ok':>11}"
    for p in nondense:
        header += f"{(p + ' kept'):>{col_w}}"
    lines = [header, "-" * len(header)]

    for b, (task, kw) in enumerate(buckets):
        dc = [i for i in range(b * n_per, (b + 1) * n_per)
              if dense[i]["recall"] >= threshold]
        if not dc:
            lines.append(f"{task:<15}{str(kw):<17}{0:>11}")
            continue
        row = f"{task:<15}{str(kw):<17}{len(dc):>11}"
        for p in nondense:
            pres = sum(by[p][i]["recall"] >= threshold for i in dc) / len(dc)
            row += f"{pres:>{col_w}.2f}"
        lines.append(row)
    return "\n".join(lines)
