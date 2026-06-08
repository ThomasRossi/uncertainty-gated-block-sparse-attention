"""
sparse_attention.py
===================

A block-sparse attention selector for Qwen2.5, registered as a custom
transformers attention implementation. The real-model analogue of the selector
studied in ssa_pomdp_multilayer.py.

How selection works (Quest-style)
---------------------------------
Per layer, per head, per query position:
  1. summarise each key block by its mean key vector;
  2. score blocks by query . block-summary;
  3. keep a budget of top-scoring blocks; the rest are masked before softmax.
Block 0 (attention sink) and the query's own block are always kept.

Policies:
  * "dense"   -- no masking; the quality ceiling.
  * "topk"    -- fixed block budget per query (what SSA runs today).
  * "belief"  -- adaptive budget via top-p on the score distribution
                 (value-of-information on the current-layer signal).
  * "rollout" -- Phase-A rollout-flavoured policy: same budget as topk, but
                 the LAST query row's block scores are penalised by an EMA of
                 prior layers' selections. This biases the selector away from
                 blocks already heavily attended across the stack, freeing
                 budget to surface query-latent ones. A heuristic approximation
                 of the POMDP value-of-information term -- it captures
                 "exploration / information-gathering" cheaply without doing
                 a per-candidate forward peek into layer L+1 (the full
                 depth-1 rollout would need that and is a separate engineering
                 build).

The masked attention itself runs through fused scaled_dot_product_attention,
so there is no huge [B,H,Q,K] materialisation -- this keeps an 8K-context
prefill tractable on an MPS Mac. Block scoring (mean-pooled keys) is a Quest-
style approximation; the GPU run can afford exact max-pooled scoring.

The module-level SEL object configures the policy and collects per-layer
diagnostics for the needle-block analysis in run_poc.py.
"""

import itertools

import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb, repeat_kv,
)

BLOCK = 64   # tokens per selection block


class Selector:
    """Global selector configuration + diagnostics."""

    def __init__(self):
        self.mode = "dense"          # "dense"|"topk"|"belief"|"rollout"|"rollout_peek"|"router"|"quest"
        self.budget_blocks = 32      # topk / rollout / rollout_peek / router: blocks/query
        self.top_p = 0.9             # belief: cumulative block-score mass
        self.k_min, self.k_max = 8, 112   # belief: budget clamp (blocks)
        # rollout (Phase A) -- exploration bonus across layers
        self.explore_bonus = 0.3
        self.ema_alpha = 0.3
        # router (Phase D) -- per-(head,row) uncertainty gate on topk.
        # Margin σ = (s[k-1] - s[k]) / (s[0] - s[k]); when σ is small, the
        # (h, q) cell falls back to per-row dense (all visible blocks). This
        # is the upper bound on what ANY planner can do for that row, so if
        # router doesn't beat topk, no lookahead will either.
        # Two threshold modes:
        #   "abs"      -- trigger iff σ < router_tau (absolute threshold).
        #   "quantile" -- trigger the bottom router_quantile fraction of
        #                 eligible cells per layer (calibrated trigger rate,
        #                 independent of margin distribution shape).
        # Two granularities:
        #   "cell" -- decision per (head, row).
        #   "row"  -- aggregate σ over heads (mean) then one decision per row,
        #            applied to all heads. Tests whether per-row uncertainty
        #            is a sharper signal than the per-cell one.
        self.router_mode = "abs"
        self.router_grain = "cell"
        self.router_tau = 0.10
        self.router_quantile = 0.20
        # Backbone block-scoring used by the router. "mean" matches the original
        # (q · mean(K_block)) signal; "quest" uses Quest's min/max upper bound
        # (max(q · K_min, q · K_max)) so the router's uncertainty gate operates
        # on top of a Quest-style scoring backbone. Tests whether scoring and
        # budget allocation are orthogonal: router-on-Quest > Quest would mean
        # the uncertainty gate adds independent lift.
        self.router_score = "mean"
        # kernel_v2: skip materialising the [B, H, Q, NB] keep tensor and the
        # _compute_kv_idx sort. Produce kv_idx directly via tile-pooled topk
        # (one decision per Q-tile of BLOCK_M rows). Quality changes from
        # per-row precision to per-tile (SSA-style); speed gains are large at
        # long context where the keep-materialisation and sort dominate.
        # Topk-only for now; router_v2 is a separate build.
        self.kernel_v2 = False
        # rollout_peek (Phase B) -- depth-1 forward-peek planner
        self.peek_swap_w = 2         # contested-band half-width: C(2W, W) candidates
        self.peek_rows_per_layer = 32  # how many rows to plan per layer (top by peakedness)
        self.peek_layer_min = 0      # plan only at layers in [min, max] (inclusive); others = topk
        self.peek_layer_max = 999
        # Peek scoring metric. Three families on different signals:
        #   "peakedness"           = max - median of head-averaged L+1 block
        #                            scores (original; degenerate at depth>=1
        #                            because q_next is nearly candidate-invariant)
        #   "cross_head_agreement" = # blocks chosen by >= peek_xh_threshold * H_q
        #                            heads' independent top-k of L+1 block scores
        #                            (same signal as peakedness; also degenerate)
        #   "residual_norm"        = ||o_proj(attn_out_last) under candidate c||_2
        #                            -- direct signal of THIS layer's attention
        #                            contribution, with guaranteed first-order
        #                            candidate dependence. Skips the MLP/q_proj
        #                            downstream pipeline that washes out variation.
        self.peek_metric = "peakedness"
        self.peek_xh_threshold = 0.5     # cross_head_agreement only
        self.model_ref = None        # set by install_rollout_hooks(model)
        self.hidden_at = {}          # layer_idx -> [B, n, d_model] captured by hook
        self.k_pool_at = {}          # layer_idx -> [B, H_q, n_blocks, D] cached K̄
        self.log = False
        self.reset_log()

    def reset_log(self):
        # layer_idx -> bool [n_blocks]: blocks kept for the LAST query position,
        # unioned over heads. Plus realised per-query budget samples. Plus the
        # rollout policy's per-example selection-history EMA (lazily init'd in
        # _block_select_mask once n_blocks is known).
        self.kept_blocks = {}
        self.budget_samples = []
        self.selection_history = None    # [H, n_blocks] float32, EMA of last-row keep
        self.k_pool_at = {}              # cleared per example; filled by prefill
        # router diagnostics, appended per-layer (per forward-pass call)
        self.router_trigger_rate = []    # fraction of (h, q) cells triggered
        self.router_eff_budget = []      # mean kept blocks per (h, q) post-router
        self.router_margin_mean = []     # mean σ across (h, q)


SEL = Selector()


class _Profiler:
    """Cuda-event profiler for block_sparse_attention internals.

    Opt-in (`PROF.enabled = True`). When on, prefill-only attention calls are
    bracketed by name; `summary()` synchronizes and returns per-bucket stats.
    Skips Q==1 calls (decode) so prefill is measured cleanly.
    """

    def __init__(self):
        self.enabled = False
        self.events = {}    # name -> list of (start_event, end_event)

    def reset(self):
        self.events = {}

    def start(self, name):
        if not self.enabled or not torch.cuda.is_available():
            return None
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()
        self.events.setdefault(name, []).append((ev_s, ev_e))
        return ev_e

    @staticmethod
    def stop(ev_e):
        if ev_e is not None:
            ev_e.record()

    def summary(self):
        if not self.events:
            return {}
        torch.cuda.synchronize()
        out = {}
        for name, ev_list in self.events.items():
            ms = sorted(s.elapsed_time(e) for s, e in ev_list)
            out[name] = {
                "calls": len(ms),
                "total_ms": sum(ms),
                "mean_ms": sum(ms) / len(ms),
                "p50_ms": ms[len(ms) // 2],
            }
        return out


PROF = _Profiler()


def install_rollout_hooks(model):
    """Register pre-hooks on each DecoderLayer to capture its input hidden state
    at the LAST query position, and stash a model reference. Required before
    using the `rollout_peek` policy."""
    SEL.model_ref = model
    SEL.hidden_at = {}

    def make_hook(idx):
        def hook(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            if hs is not None and hs.shape[-2] > 1:        # prefill only
                SEL.hidden_at[idx] = hs.detach()           # full [B, n, d]
            return None
        return hook

    for idx, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(make_hook(idx), with_kwargs=True)


def _block_select_mask(query, key_states, scaling, layer_idx):
    """
    Return a boolean [B, H, Q, K] attention mask (True = attend) implementing
    the current block-selection policy, including token-level causality.
    """
    B, H, Q, D = query.shape
    K = key_states.shape[-2]
    n_blocks = (K + BLOCK - 1) // BLOCK
    pad = n_blocks * BLOCK - K

    # Block summary: mean key per block (topk/router) or min+max (quest).
    kp = key_states
    if pad:
        kp = F.pad(kp, (0, 0, 0, pad))
    kp_blocks = kp.view(B, H, n_blocks, BLOCK, D)
    use_quest = (SEL.mode == "quest"
                 or (SEL.mode == "router" and SEL.router_score == "quest"))
    if use_quest:
        K_min = kp_blocks.amin(dim=3)                              # [B,H,nb,D]
        K_max = kp_blocks.amax(dim=3)                              # [B,H,nb,D]
        q_pos = query.clamp(min=0)
        q_neg = query.clamp(max=0)
        block_score = (torch.matmul(q_pos, K_max.transpose(2, 3))
                       + torch.matmul(q_neg, K_min.transpose(2, 3))
                       ) * scaling                                 # [B,H,Q,nb]
    else:
        kp = kp_blocks.mean(dim=3)                                 # [B,H,nb,D]
        if Q > 1:                          # prefill -- cache for any future peek
            SEL.k_pool_at[layer_idx] = kp.detach()
        block_score = torch.matmul(query, kp.transpose(2, 3)) * scaling  # [B,H,Q,nb]

    # Block-level causality: block b is visible to a query at global position p
    # iff the block starts at or before p.
    q_global = torch.arange(K - Q, K, device=query.device)
    blk_start = torch.arange(n_blocks, device=query.device) * BLOCK
    visible = blk_start.view(1, n_blocks) <= q_global.view(Q, 1)      # [Q, nb]
    block_score = block_score.masked_fill(~visible, float("-inf"))

    # Score adjustment + budget, per policy.
    if SEL.mode == "rollout":
        # Lazy init of the per-example selection-history EMA.
        if (SEL.selection_history is None
                or SEL.selection_history.shape != (H, n_blocks)):
            SEL.selection_history = torch.zeros(
                H, n_blocks, device=query.device, dtype=torch.float32)
        # Penalise the LAST query row's scores by accumulated history; other
        # rows keep raw (topk-equivalent) scoring. Bias is in the score scale.
        score_for_sel = block_score.clone()
        penalty = (SEL.explore_bonus
                   * SEL.selection_history.unsqueeze(0).to(query.dtype))
        score_for_sel[:, :, -1, :] = score_for_sel[:, :, -1, :] - penalty
        budget = torch.full((B, H, Q), float(SEL.budget_blocks),
                            device=query.device)
    elif SEL.mode in ("topk", "router", "quest"):
        score_for_sel = block_score
        budget = torch.full((B, H, Q), float(SEL.budget_blocks),
                            device=query.device)
    else:  # belief: smallest block set covering top_p of the score mass
        score_for_sel = block_score
        probs = torch.softmax(block_score.float(), dim=-1)
        srt, _ = probs.sort(dim=-1, descending=True)
        budget = (srt.cumsum(dim=-1) < SEL.top_p).sum(dim=-1).float() + 1.0
    budget = budget.clamp(SEL.k_min, SEL.k_max)

    # Uniform-budget modes (topk/router/rollout) use topk: O(N·k·log k) and
    # ~10x faster than the rank path at long context. Belief is the only
    # variable-per-cell-budget mode and keeps the rank-based selection.
    if SEL.mode in ("topk", "router", "rollout", "quest"):
        k_bud = int(max(min(SEL.k_max, SEL.budget_blocks), SEL.k_min))
        k_bud = min(k_bud, n_blocks)
        # -inf scores at non-visible positions sort to the bottom -> visible
        # blocks fill the topk preferentially; identical kept set vs rank path.
        top_idx = score_for_sel.topk(k_bud, dim=-1).indices    # [B,H,Q,k_bud]
        keep = torch.zeros(B, H, Q, n_blocks, dtype=torch.bool,
                           device=query.device)
        keep.scatter_(-1, top_idx, True)
    else:
        rank = score_for_sel.argsort(dim=-1, descending=True).argsort(dim=-1)
        keep = rank < budget.unsqueeze(-1)              # [B, H, Q, n_blocks]
    keep[..., 0] = True                                 # attention sink
    keep.scatter_(-1, (q_global // BLOCK).view(1, 1, Q, 1).expand(B, H, Q, 1),
                  True)                                 # query's own block

    if SEL.mode == "rollout":
        # Update the EMA with this layer's last-row selection (post-force).
        last_keep = keep[0, :, -1, :].float()
        SEL.selection_history = ((1.0 - SEL.ema_alpha) * SEL.selection_history
                                 + SEL.ema_alpha * last_keep)

    if SEL.mode == "router":
        # Per-(head, row) normalized margin at the budget cutoff. Cells with a
        # tight gap between the k-th and (k+1)-th block are "uncertain" and
        # promoted to per-row dense (all visible blocks). This is the upper
        # bound on what any planner could do for those cells.
        k = int(SEL.budget_blocks)
        n_blocks_dim = score_for_sel.shape[-1]
        k_eff = max(1, min(k, n_blocks_dim - 1))
        # Only need top-(k_eff+1) sorted values for the margin; topk is much
        # cheaper than a full sort at long context.
        n_take = min(k_eff + 1, n_blocks_dim)
        top_vals, _ = score_for_sel.topk(n_take, dim=-1)             # [B,H,Q,n_take]
        s_top = top_vals[..., 0]
        s_km1 = top_vals[..., k_eff - 1]
        s_k   = top_vals[..., k_eff] if k_eff < n_take else top_vals[..., -1]
        # Normalize: spread = top - k-th; σ = (k-1-th - k-th) / spread.
        # σ -> 0 means the cutoff is undefined; σ -> 1 means the k-th block
        # is right at the top of the kept set and far above the rejected tail.
        spread = (s_top - s_k).clamp(min=1e-6)
        margin_rel = ((s_km1 - s_k) / spread).float()                # [B,H,Q]
        # Only meaningful when there are >k visible blocks for that row.
        n_visible = visible.sum(dim=-1)                              # [Q]
        eligible_q = (n_visible > k_eff)                             # [Q]
        # In "row" grain we aggregate σ over heads -> one decision per row,
        # broadcast back to all heads. In "cell" grain each (h, q) decides.
        if SEL.router_grain == "row":
            margin_score = margin_rel.mean(dim=1, keepdim=True)      # [B,1,Q]
            eligible_score = eligible_q.view(1, 1, Q)                # [1,1,Q]
        else:
            margin_score = margin_rel                                # [B,H,Q]
            eligible_score = eligible_q.view(1, 1, Q).expand_as(margin_rel)
        if SEL.router_mode == "quantile":
            # Per-layer threshold: σ at the router_quantile percentile across
            # eligible scoring cells. Mask ineligible/nan to +inf so they're
            # never picked. kthvalue returns the k-th smallest -> trigger when
            # margin <= that value, yielding ~router_quantile trigger fraction.
            q_frac = float(SEL.router_quantile)
            m_safe = torch.where(eligible_score, margin_score,
                                 torch.full_like(margin_score, float("inf")))
            m_safe = torch.where(torch.isnan(m_safe),
                                 torch.full_like(m_safe, float("inf")),
                                 m_safe)
            n_elig = int(eligible_score.sum().item())
            k_quant = max(1, int(q_frac * n_elig)) if n_elig > 0 else 0
            if k_quant > 0:
                thresh = m_safe.flatten().kthvalue(k_quant).values
                uncertain_score = (m_safe <= thresh) & eligible_score
            else:
                uncertain_score = torch.zeros_like(margin_score,
                                                    dtype=torch.bool)
        else:  # "abs"
            uncertain_score = ((margin_score < float(SEL.router_tau))
                               & eligible_score)
        # Broadcast row decisions to all heads (no-op for "cell" grain).
        uncertain_hq = uncertain_score.expand(B, H, Q)
        # Dense fallback for uncertain (h, q) cells: keep all visible blocks.
        visible_keep = visible.view(1, 1, Q, n_blocks).expand(B, H, Q, n_blocks)
        keep = torch.where(uncertain_hq.unsqueeze(-1), visible_keep, keep)
        if SEL.log:
            trig = float(uncertain_hq.float().mean().item())
            kept_per_cell = float(keep.float().sum(dim=-1).mean().item())
            SEL.router_trigger_rate.append(trig)
            SEL.router_eff_budget.append(kept_per_cell)
            SEL.router_margin_mean.append(float(margin_rel.mean().item()))
            # Per-example summary on the last prefill-layer call. Q > 1 guards
            # against decode steps; layer_idx == n_layers-1 guards against
            # mid-stack noise.
            n_layers = (len(SEL.model_ref.model.layers)
                        if SEL.model_ref is not None else 0)
            if Q > 1 and n_layers and layer_idx == n_layers - 1:
                tr = sum(SEL.router_trigger_rate) / len(SEL.router_trigger_rate)
                eb = sum(SEL.router_eff_budget) / len(SEL.router_eff_budget)
                mg = sum(SEL.router_margin_mean) / len(SEL.router_margin_mean)
                knob = (f"q={SEL.router_quantile:.2f}"
                        if SEL.router_mode == "quantile"
                        else f"tau={SEL.router_tau:.3f}")
                print(f"  router[{SEL.router_mode}/{SEL.router_grain}]: "
                      f"trigger={tr:.3f} eff_budget={eb:.1f} "
                      f"margin={mg:.3f} {knob}", flush=True)

    if SEL.log:
        SEL.kept_blocks[layer_idx] = keep[0, :, -1, :].any(dim=0).cpu()
        SEL.budget_samples.append(
            keep[0, :, -1, :].sum(dim=-1).float().mean().item())

    return keep


# ---------------------------------------------------------------------------
# Phase-B rollout_peek -- real depth-1 forward peek
# ---------------------------------------------------------------------------

def _peek_score(hidden_post_attn_last, layer_idx, position_embeddings,
                key_pool_L, last_pos):
    """
    Score a candidate selection by the peakedness of what layer L+1 WOULD see.
    Continues layer L's forward (MLP + residual) on the last row to get the
    input to layer L+1, runs L+1's input_layernorm + q_proj + RoPE, then
    block-scores against layer L's key-pool (cheap proxy for L+1's K, which
    would otherwise need a full re-projection over all positions).

    hidden_post_attn_last : [B, 1, d_model]
    key_pool_L            : [B, H_q, n_blocks, head_dim]  -- already repeat_kv'd
    last_pos              : int, the global position of the last query token
    Returns               : scalar float (higher = healthier next-layer signal)
    """
    if SEL.model_ref is None or position_embeddings is None:
        return 0.0
    layers = SEL.model_ref.model.layers
    if layer_idx + 1 >= len(layers):
        return 0.0
    cur = layers[layer_idx]
    nxt = layers[layer_idx + 1]
    # Finish layer L: post-attn LN + MLP + residual.
    mlp_in = cur.post_attention_layernorm(hidden_post_attn_last)
    hidden_next = hidden_post_attn_last + cur.mlp(mlp_in)
    # Layer L+1: input LN + q_proj.
    normed = nxt.input_layernorm(hidden_next)
    H_q = nxt.self_attn.config.num_attention_heads
    head_dim = nxt.self_attn.head_dim
    q_next = nxt.self_attn.q_proj(normed).view(1, 1, H_q, head_dim).transpose(1, 2)
    # RoPE at the last position. cos/sin shape: [B, n_kv, head_dim].
    cos, sin = position_embeddings
    cos_last = cos[:, last_pos:last_pos + 1, :]
    sin_last = sin[:, last_pos:last_pos + 1, :]
    q_next, _ = apply_rotary_pos_emb(q_next, q_next, cos_last, sin_last)
    # Next-layer block scores at the planning row.
    block_score_next = (torch.matmul(q_next, key_pool_L.transpose(-2, -1))
                        * nxt.self_attn.scaling)        # [B, H_q, 1, n_blocks]
    bs_per_head = block_score_next.squeeze(2).squeeze(0).float()  # [H_q, n_blocks]
    if SEL.peek_metric == "peakedness":
        bs = bs_per_head.mean(dim=0)                    # [n_blocks]
        return float((bs.max() - bs.median()).item())
    elif SEL.peek_metric == "cross_head_agreement":
        return _peek_score_cross_head(bs_per_head, SEL.budget_blocks)
    else:
        raise ValueError(f"unknown peek_metric: {SEL.peek_metric}")


def _peek_score_cross_head(bs_per_head, budget):
    """A1 family -- different signal from peakedness. Each head independently
    picks its top-`budget` blocks; we count the kv-blocks that >= threshold
    fraction of heads agree on. Higher = stronger head consensus about where
    the next layer would attend. Doesn't measure sharpness of any single
    distribution; measures multi-head agreement on the next-layer selection.
    """
    H_q, n_blocks = bs_per_head.shape
    budget = min(budget, n_blocks)
    top_per_head = bs_per_head.topk(budget, dim=-1).indices       # [H_q, bud]
    counts = torch.zeros(n_blocks, device=bs_per_head.device, dtype=torch.float32)
    counts.scatter_add_(
        0,
        top_per_head.flatten(),
        torch.ones_like(top_per_head, dtype=torch.float32).flatten(),
    )
    threshold = SEL.peek_xh_threshold * H_q
    return float((counts >= threshold).sum().item())


def _block_select_mask_peek(module, query, key_states, value_states, scaling,
                            position_embeddings):
    """
    rollout_peek: per layer, override the *top peakedness* query rows'
    selections with candidates chosen by depth-1 forward peek (which uses
    cached K̄_{L+1} from a topk pre-pass). Non-planned rows keep plain topk.

    Multi-row planning rationale: planning the last row alone was empirically
    null (its selection is not the binding constraint on PCH; by the answer
    row, earlier rows have already propagated what's needed). The binding
    constraint is the *intermediate* rows that read chain links and need to
    attend back to prior chain entries. Peakedness (top-1 minus top-(K+1))
    of a row's block-score distribution is a cheap proxy for "this row is
    looking for something specific" -- the rows likely doing real retrieval.
    """
    B, H, Q, D = query.shape
    K = key_states.shape[-2]
    n_blocks = (K + BLOCK - 1) // BLOCK
    pad = n_blocks * BLOCK - K
    layer_idx = module.layer_idx

    # Block summary, scores, block-level causality -- same as topk.
    kp = key_states
    if pad:
        kp = F.pad(kp, (0, 0, 0, pad))
    kp = kp.view(B, H, n_blocks, BLOCK, D).mean(dim=3)        # [B,H,nb,D]
    if Q > 1:
        SEL.k_pool_at[layer_idx] = kp.detach()
    block_score = torch.matmul(query, kp.transpose(2, 3)) * scaling
    q_global = torch.arange(K - Q, K, device=query.device)
    blk_start = torch.arange(n_blocks, device=query.device) * BLOCK
    visible = blk_start.view(1, n_blocks) <= q_global.view(Q, 1)    # [Q, nb]
    block_score = block_score.masked_fill(~visible, float("-inf"))

    # Baseline per-head topk for ALL rows (we only override the last one).
    budget = torch.full((B, H, Q), float(SEL.budget_blocks),
                        device=query.device).clamp(SEL.k_min, SEL.k_max)
    rank = block_score.argsort(dim=-1, descending=True).argsort(dim=-1)
    keep = rank < budget.unsqueeze(-1)
    keep[..., 0] = True
    keep.scatter_(-1, (q_global // BLOCK).view(1, 1, Q, 1).expand(B, H, Q, 1),
                  True)

    # Identify planning rows: top-K rows by peakedness, restricted to rows
    # that have enough visible blocks to enumerate candidates.
    bud = int(SEL.budget_blocks)
    w = SEL.peek_swap_w
    k_need = bud + w
    n_valid_per_row = visible.sum(dim=-1)                         # [Q]
    enumerable = n_valid_per_row >= k_need                        # [Q] bool
    top_vals, _ = torch.topk(block_score[0], k=min(bud + 1, n_blocks), dim=-1)
    row_peak = (top_vals[..., 0] - top_vals[..., -1]).float().mean(dim=0)  # [Q]
    row_peak = torch.where(enumerable, row_peak,
                           torch.full_like(row_peak, -float("inf")))

    n_enum = int(enumerable.sum().item())
    n_plan = min(SEL.peek_rows_per_layer, n_enum)
    hidden_full = SEL.hidden_at.get(layer_idx)
    in_range = SEL.peek_layer_min <= layer_idx <= SEL.peek_layer_max
    if (not in_range or n_plan == 0 or hidden_full is None
            or hidden_full.shape[-2] != K):
        planning_rows = []      # outside planning range -> baseline topk for this layer
    else:
        planning_rows = row_peak.argsort(descending=True)[:n_plan].tolist()

    # Per-row peek: override `keep` for each planning row with the best candidate.
    next_kp = SEL.k_pool_at.get(layer_idx + 1, kp)
    for row_idx in planning_rows:
        row_pos = int(q_global[row_idx].item())
        row_score = block_score[0, :, row_idx, :].mean(dim=0)     # [n_blocks]
        order = torch.argsort(row_score, descending=True).tolist()
        n_v = int(n_valid_per_row[row_idx].item())
        bud_eff = min(bud, max(1, n_v))
        safe_end = max(0, bud_eff - w)
        band_end = min(n_v, bud_eff + w)
        safe = order[:safe_end]
        band = order[safe_end:band_end]
        need = bud_eff - len(safe)
        if need <= 0 or need > len(band):
            continue

        candidates = list(itertools.combinations(band, need))
        best_score, best_blocks = -float("inf"), None
        hidden_row = hidden_full[:, row_idx:row_idx + 1, :]       # [B, 1, d]
        causal_row = (torch.arange(K, device=query.device) <= row_pos)  # [K]

        for combo in candidates:
            blocks = safe + list(combo)
            cand = torch.zeros(n_blocks, dtype=torch.bool, device=query.device)
            for b in blocks:
                cand[b] = True
            ck = (cand.unsqueeze(-1).expand(n_blocks, BLOCK)
                       .reshape(n_blocks * BLOCK)[:K]) & causal_row
            q_row = query[:, :, row_idx:row_idx + 1, :]
            s_row = (torch.matmul(q_row, key_states.transpose(2, 3))
                     * scaling)
            s_row = s_row.masked_fill(~ck.view(1, 1, 1, K), float("-inf"))
            a_row = F.softmax(s_row, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_out = torch.matmul(a_row, value_states)
            attn_out = attn_out.transpose(1, 2).reshape(B, 1, H * D)
            o_proj_out = module.o_proj(attn_out)
            if SEL.peek_metric == "residual_norm":
                # Direct signal: candidate's contribution to the residual at
                # layer L. First-order candidate-dependent by construction,
                # so the argmax has real discrimination (unlike peakedness /
                # cross_head_agreement, whose signal is washed out by the
                # unchanged residual dominating q_next at L+1).
                score = float(o_proj_out.float().norm().item())
            else:
                hidden_post = hidden_row + o_proj_out
                score = _peek_score(hidden_post, layer_idx,
                                    position_embeddings, next_kp, row_pos)
            if score > best_score:
                best_score, best_blocks = score, blocks

        if best_blocks is not None:
            row_keep = torch.zeros(n_blocks, dtype=torch.bool,
                                   device=query.device)
            for b in best_blocks:
                row_keep[b] = True
            row_keep[0] = True                                    # sink
            row_keep[row_pos // BLOCK] = True                     # self block
            keep[0, :, row_idx, :] = row_keep.unsqueeze(0).expand(H, n_blocks)

    if SEL.log:
        SEL.kept_blocks[layer_idx] = keep[0, :, -1, :].any(dim=0).cpu()
        SEL.budget_samples.append(
            keep[0, :, -1, :].sum(dim=-1).float().mean().item())

    return keep


def _block_select_kv_idx_v2(query, key_states, scaling, block_m=64):
    """v2 selection: build kv_idx [B, H, Q_tiles, k_bud+2] int32 DIRECTLY,
    no keep tensor, no _compute_kv_idx sort.

    Selection at Q-tile granularity (BLOCK_M=64 rows share a kept set,
    SSA-style):
      1. mean-pool K within each block  -> K_pool [B, H, NB, D]   (same as v1)
      2. per-row block_score = Q @ K_pool^T -> [B, H, M, NB]      (same as v1)
      3. per-tile MAX over rows         -> tile_score [B, H, Qt, NB]
         (this is the v1 "union of per-row top-k" intent expressed as a
         per-tile ranking; mean-pool Q in an earlier iteration tanked
         quality because it diluted single-row signal)
      4. causal mask at tile granularity
      5. topk(k_bud) -> indices = kv_idx
      6. concat block 0 (sink) + tile's self-block as extra columns

    Kills the v1 hot spots at long context: keep-tensor scatter (2.7 GB/layer
    at 64K) and _compute_kv_idx's sort on the [B, H, Q_tiles, NB] bool any-keep
    tensor. The trade-off: per-tile-uniform attention instead of per-row, so
    each row in a tile attends to the same set; never less attention than v1,
    sometimes more.
    """
    B, H, Q, D = query.shape
    K = key_states.shape[-2]
    n_blocks = (K + BLOCK - 1) // BLOCK
    pad = n_blocks * BLOCK - K
    Q_tiles = (Q + block_m - 1) // block_m
    qpad = Q_tiles * block_m - Q

    # Block summary: mean (topk/router) or elementwise min+max (quest).
    kp = key_states
    if pad:
        kp = F.pad(kp, (0, 0, 0, pad))
    kp_blocks = kp.view(B, H, n_blocks, BLOCK, D)                 # [B,H,NB,BS,D]

    q_padded = query
    if qpad:
        q_padded = F.pad(query, (0, 0, 0, qpad))                  # [B,H,M_p,D]

    use_quest = (SEL.mode == "quest"
                 or (SEL.mode == "router" and SEL.router_score == "quest"))
    if use_quest:
        # Quest (MLSys '24) upper-bound score: per-block elementwise min/max
        # of K, score = sum_d max(q[d]*K_min[d], q[d]*K_max[d]) — an upper
        # bound on max_{k in block} q·k. Implemented in two matmuls:
        # q_pos = q.clamp(min=0), q_neg = q.clamp(max=0)
        # score = q_pos @ K_max.T + q_neg @ K_min.T
        # because for q[d] >= 0, max(q*K_min, q*K_max) = q*K_max, and vice versa.
        K_min = kp_blocks.amin(dim=3)                             # [B,H,NB,D]
        K_max = kp_blocks.amax(dim=3)                             # [B,H,NB,D]
        q_pos = q_padded.clamp(min=0)
        q_neg = q_padded.clamp(max=0)
        block_score = (torch.matmul(q_pos, K_max.transpose(-1, -2))
                       + torch.matmul(q_neg, K_min.transpose(-1, -2))
                       ) * scaling                                # [B,H,M_p,NB]
    else:
        # Per-row block scores (same as v1's _block_select_mask matmul). Memory
        # at 64K is ~5 GB bf16 -- same as v1, so feasible where v1 was.
        kp_mean = kp_blocks.mean(dim=3)                           # [B,H,NB,D]
        block_score = (torch.matmul(q_padded, kp_mean.transpose(-1, -2))
                       * scaling)                                 # [B,H,M_p,NB]

    # Per-tile max over rows: tile_score[t, b] = max_{rows in tile t} score.
    # This is equivalent in intent to "block b is selected iff any row in
    # tile t would have selected it" (i.e., v1's per-tile union of per-row
    # top-k), just routed through a tile-level ranking instead of a bool OR
    # plus sort.
    tile_score = (block_score.view(B, H, Q_tiles, block_m, n_blocks)
                              .amax(dim=-2))                      # [B,H,Qt,NB]

    # Causal at tile granularity: block b is visible to Q-tile t iff block
    # starts at or before t's last row.
    q_tile_last = (torch.arange(Q_tiles, device=query.device) * block_m
                   + block_m - 1).clamp(max=K - 1)
    blk_start = torch.arange(n_blocks, device=query.device) * BLOCK
    visible_tile = (blk_start.view(1, n_blocks)
                    <= q_tile_last.view(Q_tiles, 1))              # [Qt, NB]
    tile_score = tile_score.masked_fill(
        ~visible_tile.view(1, 1, Q_tiles, n_blocks), float("-inf"))

    # Force attention sink (block 0) and each tile's self-block into the
    # top-k by setting their score to +inf. Doing this BEFORE topk
    # guarantees they're picked without producing duplicates (concat-after
    # would double-count when topk already had them, which breaks the
    # kernel's online-softmax normalisation). Sink-forcing is critical for
    # row 0 of tile 0: it can only attend to block 0 under causality, so
    # missing it leaves the row with no valid attention -> garbage output.
    INF = torch.finfo(tile_score.dtype).max
    tile_score[..., 0] = INF
    self_blk_per_tile = (q_tile_last // BLOCK).to(torch.long).view(
        1, 1, Q_tiles, 1).expand(B, H, Q_tiles, 1)
    tile_score.scatter_(-1, self_blk_per_tile,
                        torch.full_like(self_blk_per_tile, INF,
                                        dtype=tile_score.dtype))

    # v2 effective budget: tile-max-pool ranks blocks by their PEAK row score,
    # which biases against needles that only one row strongly wants. To match
    # v1's per-tile-union size (~38 at our scale; v1's per-row top-29 unions
    # to ~38 unique blocks across 64 rows), use 2x the v1 budget as a quality
    # baseline. Same asymptotic sparsity (a few percent of NB at long context).
    k_bud_row = int(max(min(SEL.k_max, SEL.budget_blocks), SEL.k_min))
    k_bud_base = min(2 * k_bud_row, n_blocks)

    is_router = (SEL.mode == "router")
    # Router-v2: certain tiles use k_bud_base; uncertain tiles use 2x base.
    # All tiles share the same kv_idx shape (k_bud_max slots); certain tiles
    # pad the extra slots with the kernel's NB sentinel (cheap skip).
    k_bud_max = min(2 * k_bud_base, n_blocks) if is_router else k_bud_base

    top_vals, top_idx = tile_score.topk(k_bud_max, dim=-1)        # both [B,H,Qt,k_bud_max]
    kv_idx_i32 = top_idx.to(torch.int32)

    if is_router and k_bud_base < k_bud_max:
        # Per-tile margin at the BASE cutoff.
        # σ = (s[base-1] - s[base]) / (s[0] - s[base]), per (B, H, Qt).
        s_top = top_vals[..., 0]
        s_km1 = top_vals[..., k_bud_base - 1]
        s_k   = top_vals[..., k_bud_base]
        spread = (s_top - s_k).clamp(min=1e-6)
        margin = ((s_km1 - s_k) / spread).float()                 # [B,H,Qt]
        if SEL.router_grain == "row":
            # Aggregate over heads (the Phase-D row-grain sharper signal).
            margin = margin.mean(dim=1, keepdim=True).expand_as(margin)
        if SEL.router_mode == "quantile":
            q_frac = float(SEL.router_quantile)
            flat = margin.flatten()
            k_quant = max(1, int(q_frac * flat.numel()))
            thresh = flat.kthvalue(k_quant).values
            uncertain = margin <= thresh                          # [B,H,Qt]
        else:
            uncertain = margin < float(SEL.router_tau)
        # Build final kv_idx: for certain tiles, sentinel out the extra slots.
        slot_idx = torch.arange(k_bud_max, device=query.device)
        slot_is_extra = (slot_idx >= k_bud_base).view(1, 1, 1, k_bud_max)
        uncertain_4d = uncertain.unsqueeze(-1).expand_as(kv_idx_i32)
        sentinel_val = torch.full_like(kv_idx_i32, n_blocks)      # NB sentinel
        kv_idx = torch.where(slot_is_extra & ~uncertain_4d,
                             sentinel_val, kv_idx_i32)
        if SEL.log:
            trig = float(uncertain.float().mean().item())
            SEL.router_trigger_rate.append(trig)
    else:
        kv_idx = kv_idx_i32                                       # [B,H,Qt,k_bud_max]

    if SEL.log:
        # Diagnostics: kept blocks for the LAST Q-tile, unioned over heads.
        # This is what poc_core.needle_hit_rate reads; matches v1's reporting.
        last_tile_kept = torch.zeros(n_blocks, dtype=torch.bool,
                                     device=query.device)
        last_tile_blocks = kv_idx[0, :, -1, :].long().flatten()
        # Filter out NB sentinel before scatter (would IndexError into bool of
        # size n_blocks).
        last_tile_blocks = last_tile_blocks[last_tile_blocks < n_blocks]
        last_tile_kept.scatter_(0, last_tile_blocks, True)
        layer_idx = len(SEL.kept_blocks)
        SEL.kept_blocks[layer_idx] = last_tile_kept.cpu()
        SEL.budget_samples.append(float(k_bud_max))
        if is_router and layer_idx == 0 and SEL.model_ref is None:
            pass  # quiet; per-call summary done elsewhere

    return kv_idx.contiguous()


def block_sparse_attention(module, query, key, value, attention_mask,
                           scaling, dropout=0.0, **kwargs):
    """Custom attention: dense path through SDPA's FlashAttention; sparse path
    through our Triton block-sparse kernel (triton_block_attn).

    Causality is handled here (SDPA is_causal / kernel-internal causal): the
    transformers wrapper does not pass a usable mask for an unrecognised
    custom attention impl. Assumes one-shot prefill (Q == K) then single-token
    decode (Q == 1), which is what model.generate() does.
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    Q = query.shape[-2]
    is_prefill = Q > 1   # gate the profiler so decode (Q==1) doesn't pollute

    if SEL.mode == "dense" or Q == 1:
        ev = PROF.start("dense_sdpa") if is_prefill else None
        out = F.scaled_dot_product_attention(
            query, key_states, value_states,
            is_causal=(Q > 1), scale=scaling)
        PROF.stop(ev)
        return out.transpose(1, 2).contiguous(), None

    # v2 fast path: skip the [B, H, Q, NB] keep tensor entirely. Handles
    # topk, router, and quest; belief/rollout/rollout_peek fall through to v1.
    if SEL.kernel_v2 and SEL.mode in ("topk", "router", "quest"):
        ev = PROF.start("v2_select") if is_prefill else None
        kv_idx = _block_select_kv_idx_v2(query, key_states, scaling)
        PROF.stop(ev)
        from triton_block_attn import block_sparse_attn_v2
        ev = PROF.start("v2_kernel") if is_prefill else None
        out = block_sparse_attn_v2(query.contiguous(),
                                   key_states.contiguous(),
                                   value_states.contiguous(),
                                   kv_idx,
                                   scale=scaling, block_n=BLOCK)
        PROF.stop(ev)
        return out.transpose(1, 2).contiguous(), None

    if SEL.mode == "rollout_peek":
        keep = _block_select_mask_peek(
            module, query, key_states, value_states, scaling,
            kwargs.get("position_embeddings"))
    else:
        keep = _block_select_mask(query, key_states, scaling, module.layer_idx)

    from triton_block_attn import block_sparse_attn
    out = block_sparse_attn(query.contiguous(),
                            key_states.contiguous(),
                            value_states.contiguous(),
                            keep.contiguous(),
                            scale=scaling, block_n=BLOCK)
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register("block_sparse", block_sparse_attention)
