"""dump_block_scores.py
==========================

Cache-safe diagnostic for the BAI/VoI appendix. Monkey-patches
``sparse_attention._block_select_kv_idx_v2`` at runtime to record per-tile
order statistics ``(g, r, sigma, g_rho)`` *without* altering its return value.

Lives **outside** the cache-key fingerprint (``sparse_files`` in the modal
harnesses) so the source files that feed ``code_version`` stay byte-identical
and existing ``result_cache.json`` entries remain valid. The diagnostic run
itself must bypass the cache (``cache=None`` in ``poc_core.run_*``) because a
cache hit short-circuits the prefill — no forward, no scores to dump.

Usage from a harness::

    import dump_block_scores
    dump_block_scores.install_dump_hook("/cache/block_scores_diag.pkl")
    ...  # run prefills with SEL.kernel_v2=True and SEL.mode in
         #  {topk, router, quest}; dense/decode (Q==1) calls are skipped.
    dump_block_scores.flush_and_teardown()

Each ``_block_select_kv_idx_v2`` call appends one shard::

    {"call_idx": int,        # monotonic across the whole run
     "n_blocks": int,
     "mode_id": int,         # see _mode_id()
     "budget_blocks": int,   # SEL.budget_blocks at call time
     "g":     np.ndarray,    # [B, H, Qt] float32, cutoff gap   s_(k-1) - s_(k)
     "r":     np.ndarray,    # [B, H, Qt] float32, spread       s_(0)   - s_(k)
     "sigma": np.ndarray,    # [B, H, Qt] float32, g/r (clamped >=0)
     "g_rho": np.ndarray,    # [B, H, Qt] float32, expanded gap s_(rho k -1) - s_(rho k)
     "valid": np.ndarray}    # [Qt] bool, whether n_visible_blocks > rho_k for that tile

The cutoff ``k`` matches v2 production: ``k_bud_base = 2 * SEL.budget_blocks``
(the v2 line 689 doubling to match v1's per-tile-union size). The expansion
factor ``rho = 2`` matches the router's ``k_bud_max = 2 * k_bud_base``.
"""

import pickle

import numpy as np
import torch
import torch.nn.functional as F

import sparse_attention


_STATE = {
    "enabled": False,
    "out_path": None,
    "orig_fn": None,
    "shards": [],
    "call_idx": 0,
}


def install_dump_hook(out_path):
    """Wrap ``sparse_attention._block_select_kv_idx_v2`` so each call records a
    diagnostic shard. Idempotent; second call is a no-op."""
    if _STATE["enabled"]:
        print(f"[dump_block_scores] already installed (out={_STATE['out_path']}); "
              f"ignoring install with out={out_path}", flush=True)
        return
    _STATE["orig_fn"] = sparse_attention._block_select_kv_idx_v2
    _STATE["out_path"] = out_path
    _STATE["shards"] = []
    _STATE["call_idx"] = 0
    _STATE["enabled"] = True
    sparse_attention._block_select_kv_idx_v2 = _wrapped
    print(f"[dump_block_scores] installed, dumping to {out_path}", flush=True)


def flush_and_teardown():
    """Restore the original selection fn and write the pickle to disk."""
    if not _STATE["enabled"]:
        return
    sparse_attention._block_select_kv_idx_v2 = _STATE["orig_fn"]
    _STATE["enabled"] = False
    _flush()


def _flush():
    if not _STATE["shards"] or _STATE["out_path"] is None:
        return
    with open(_STATE["out_path"], "wb") as f:
        pickle.dump(_STATE["shards"], f, protocol=pickle.HIGHEST_PROTOCOL)
    n_tiles = sum(s["g"].size for s in _STATE["shards"])
    print(f"[dump_block_scores] wrote {len(_STATE['shards'])} calls "
          f"({n_tiles} (B,H,Qt) cells) → {_STATE['out_path']}", flush=True)


def _wrapped(query, key_states, scaling, block_m=64):
    out = _STATE["orig_fn"](query, key_states, scaling, block_m=block_m)
    if _STATE["enabled"]:
        try:
            _record(query, key_states, scaling, block_m)
        except Exception as e:  # never crash the prefill
            print(f"[dump_block_scores] WARN record failed at call "
                  f"{_STATE['call_idx']}: {type(e).__name__}: {e}", flush=True)
        _STATE["call_idx"] += 1
    return out


def _record(query, key_states, scaling, block_m):
    """Mirror of the production block-scoring path; computes (g, r, sigma, g_rho)
    per (B, H, Qt) tile from the same tile_score the production uses."""
    with torch.no_grad():
        SEL = sparse_attention.SEL
        BLOCK = sparse_attention.BLOCK

        B, H, Q, D = query.shape
        K_len = key_states.shape[-2]
        NB = (K_len + BLOCK - 1) // BLOCK
        pad = NB * BLOCK - K_len
        Qt = (Q + block_m - 1) // block_m
        qpad = Qt * block_m - Q

        kp = key_states
        if pad:
            kp = F.pad(kp, (0, 0, 0, pad))
        kp_blocks = kp.view(B, H, NB, BLOCK, D)
        q_padded = query if qpad == 0 else F.pad(query, (0, 0, 0, qpad))

        use_quest = (SEL.mode == "quest"
                     or (SEL.mode == "router" and SEL.router_score == "quest"))
        if use_quest:
            K_min = kp_blocks.amin(dim=3)
            K_max = kp_blocks.amax(dim=3)
            q_pos = q_padded.clamp(min=0)
            q_neg = q_padded.clamp(max=0)
            bs = (torch.matmul(q_pos, K_max.transpose(-1, -2))
                  + torch.matmul(q_neg, K_min.transpose(-1, -2))) * scaling
        else:
            kp_mean = kp_blocks.mean(dim=3)
            bs = torch.matmul(q_padded, kp_mean.transpose(-1, -2)) * scaling

        tile_score = bs.view(B, H, Qt, block_m, NB).amax(dim=-2)

        # Causal mask at tile granularity (matches production line 658-666).
        q_tile_last = (torch.arange(Qt, device=query.device) * block_m
                       + block_m - 1).clamp(max=K_len - 1)
        blk_start = torch.arange(NB, device=query.device) * BLOCK
        visible = (blk_start.view(1, NB)
                   <= q_tile_last.view(Qt, 1))                       # [Qt, NB]
        tile_score = tile_score.masked_fill(
            ~visible.view(1, 1, Qt, NB), float("-inf"))

        # v2 cutoff: k_bud_base = 2 * SEL.budget_blocks (production line 689).
        # Router expansion is rho=2 (line 695), so rho*k = 4 * budget_blocks.
        k_b = min(2 * int(SEL.budget_blocks), NB)
        rho_k = min(2 * k_b, NB)
        if rho_k <= k_b:
            return  # no room for a g_rho gap on this layer

        n_vis = visible.sum(dim=-1)                                  # [Qt]
        valid_tile = (n_vis > rho_k)                                 # [Qt]
        if not valid_tile.any():
            return

        # Full sort over NB (NB <= 1024 at our scales -- cheap).
        sorted_ts, _ = torch.sort(tile_score, dim=-1, descending=True)
        s_0 = sorted_ts[..., 0]
        s_km1 = sorted_ts[..., k_b - 1]
        s_k = sorted_ts[..., k_b]
        s_rkm1 = sorted_ts[..., rho_k - 1]
        s_rk = sorted_ts[..., rho_k]

        g = (s_km1 - s_k).float()
        r = (s_0 - s_k).clamp(min=1e-9).float()
        sigma = (g / r).clamp(0.0, 2.0)
        g_rho = (s_rkm1 - s_rk).float()

        # tau-hat (Eq. (B.5) noise scale): first-difference dispersion on
        # per-row block scores along the query dimension. Adjacent rows are
        # consecutive tokens in the prompt, so under the smoothness assumption
        # mu_{t+1,b} ~ mu_{t,b}, the difference is dominated by noise:
        #     E[(s_{t+1,b} - s_{t,b})^2] = 2*tau^2.
        # We sample n_pairs adjacent-row pairs spread across the query length
        # (Q can be 32K+; full diff would double the bs memory). Aggregated per
        # (B, H) so the analyzer can take per-layer-per-head statistics.
        bs_real = bs[..., :Q, :]                                     # [B,H,Q,NB]
        if Q >= 2:
            n_pairs = min(1024, Q - 1)
            stride = max(1, (Q - 1) // n_pairs)
            i0 = torch.arange(n_pairs, device=query.device) * stride
            i0 = i0.clamp(max=Q - 2)
            pair_lo = bs_real.index_select(-2, i0)
            pair_hi = bs_real.index_select(-2, i0 + 1)
            diff = (pair_hi - pair_lo).float()                       # [B,H,n_pairs,NB]
            tau_sq = 0.5 * diff.pow(2).mean(dim=(-2, -1))            # [B,H]
            tau_hat = tau_sq.clamp(min=0).sqrt().cpu().numpy()
        else:
            import numpy as _np
            tau_hat = _np.zeros((B, H), dtype=_np.float32)

        _STATE["shards"].append({
            "call_idx": _STATE["call_idx"],
            "n_blocks": int(NB),
            "mode_id": _mode_id(SEL),
            "budget_blocks": int(SEL.budget_blocks),
            "g": g.cpu().numpy(),
            "r": r.cpu().numpy(),
            "sigma": sigma.cpu().numpy(),
            "g_rho": g_rho.cpu().numpy(),
            "valid": valid_tile.cpu().numpy(),
            "tau_hat": tau_hat,                                      # [B, H]
        })


def _mode_id(SEL):
    """0=topk, 1=router-mean, 2=router-quest, 3=quest, 4=other."""
    if SEL.mode == "topk":
        return 0
    if SEL.mode == "router":
        return 2 if SEL.router_score == "quest" else 1
    if SEL.mode == "quest":
        return 3
    return 4


# Inverse map for analysis scripts.
MODE_LABELS = {0: "topk", 1: "router_mean", 2: "router_quest",
               3: "quest", 4: "other"}
