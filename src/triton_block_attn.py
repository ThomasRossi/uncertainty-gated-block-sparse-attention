"""
triton_block_attn.py
====================

Block-sparse causal attention forward (Triton).

Math:
    S    = (Q . K^T) * scale
    S    = S  if (kv_pos in a kept block AND kv_pos <= q_pos)  else -inf
    out  = softmax(S) . V

Inputs (caller has done repeat_kv so Q and KV share H):
    q     : [B, H, M, D]   bf16
    k     : [B, H, N, D]   bf16
    v     : [B, H, N, D]   bf16
    keep  : [B, H, M, NB]  bool        True = this query attends to this kv block
    scale : float
    block_n : kv block size (NB = ceil(N / block_n))

Output:
    out   : [B, H, M, D]   bf16

Implementation: NSA-style precomputed kv index list. Per Q-tile (group of
BLOCK_M contiguous rows) we OR `keep` across the tile -- giving the set of
kv blocks ANY row in the tile attends to -- and pack the kept indices into
`kv_idx[B, H, Q_tiles, MAX_KEPT]` (int32, sentinel = NB for padding). The
kernel then iterates `MAX_KEPT` slots instead of `NB`. At BUD=33, MAX_KEPT
~ 40-70; at 64K context (NB=1024) that's a ~15-25x cut to the inner loop.

Per-row precision is preserved by an inner mask: within a kept tile, rows
that did not select this block are masked to -inf.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, KEEP_ptr, IDX_ptr, OUT_ptr,
    sqb, sqh, sqm, sqd,
    skb, skh, skn, skd,
    svb, svh, svn, svd,
    sKpb, sKph, sKpm, sKpblk,
    sIb, sIh, sIt, sIk,
    sob, soh, som, sod,
    scale,
    M, N, H, NB,
    MAX_KEPT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh %  H

    q_start = pid_m * BLOCK_M
    offs_m  = q_start + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, D)

    q_base   = Q_ptr   + b * sqb  + h * sqh
    k_base   = K_ptr   + b * skb  + h * skh
    v_base   = V_ptr   + b * svb  + h * svh
    o_base   = OUT_ptr + b * sob  + h * soh
    keep_base = KEEP_ptr + b * sKpb + h * sKph
    idx_base  = IDX_ptr  + b * sIb  + h * sIh  + pid_m * sIt

    q_mask = offs_m[:, None] < M
    q_tile = tl.load(q_base + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                     mask=q_mask, other=0.0)

    NEG = -1.0e30
    m_i = tl.full([BLOCK_M], NEG, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q_max_in_tile = q_start + BLOCK_M - 1

    for i in range(0, MAX_KEPT):
        kv_block = tl.load(idx_base + i * sIk)            # int32; NB == sentinel
        if kv_block < NB:
            kv_start = kv_block * BLOCK_N
            if kv_start <= q_max_in_tile:
                row_keep = tl.load(
                    keep_base + offs_m * sKpm + kv_block * sKpblk,
                    mask=offs_m < M, other=0).to(tl.int32)

                offs_n = kv_start + tl.arange(0, BLOCK_N)
                n_in   = offs_n < N

                k_tile = tl.load(
                    k_base + offs_n[:, None] * skn + offs_d[None, :] * skd,
                    mask=n_in[:, None], other=0.0)

                s = tl.dot(q_tile, tl.trans(k_tile)) * scale

                keep_2d = row_keep[:, None] > 0
                causal  = offs_m[:, None] >= offs_n[None, :]
                inb     = n_in[None, :] & (offs_m[:, None] < M)
                ok      = keep_2d & causal & inb
                s = tl.where(ok, s, NEG)

                m_ij  = tl.max(s, axis=1)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                p     = tl.exp(s - m_new[:, None])

                v_tile = tl.load(
                    v_base + offs_n[:, None] * svn + offs_d[None, :] * svd,
                    mask=n_in[:, None], other=0.0)

                acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    tl.store(o_base + offs_m[:, None] * som + offs_d[None, :] * sod,
             out.to(OUT_ptr.dtype.element_ty), mask=q_mask)


def _compute_kv_idx(keep, block_m):
    """
    keep    : [B, H, M, NB] bool. Returns:
      kv_idx  : [B, H, Q_tiles, MAX_KEPT] int32, sorted ascending,
                padded with sentinel = NB.
      MAX_KEPT: int. Max number of kept kv-blocks across any Q-tile.
    """
    B, H, M, NB = keep.shape
    Q_tiles = (M + block_m - 1) // block_m
    pad = Q_tiles * block_m - M
    if pad:
        keep = torch.nn.functional.pad(keep, (0, 0, 0, pad))
    any_keep = keep.view(B, H, Q_tiles, block_m, NB).any(dim=-2)  # [B,H,Qt,NB]
    block_idx = torch.arange(NB, device=keep.device, dtype=torch.int32)
    sentinel = torch.full((), NB, device=keep.device, dtype=torch.int32)
    masked = torch.where(any_keep, block_idx.view(1, 1, 1, NB), sentinel)
    sorted_idx, _ = masked.sort(dim=-1)                          # NB-padded at end
    counts = (sorted_idx < NB).sum(dim=-1)                       # [B,H,Qt]
    max_kept = int(counts.max().item())
    if max_kept == 0:
        max_kept = 1
    return sorted_idx[..., :max_kept].contiguous(), max_kept


def block_sparse_attn(q, k, v, keep, scale, block_n=64, block_m=64,
                      num_warps=4, num_stages=1):
    """
    q, k, v : [B, H, M, D]   bf16, contiguous, CUDA
    keep    : [B, H, M, NB]  bool, contiguous, CUDA
    Returns : [B, H, M, D]
    """
    assert q.shape == k.shape == v.shape
    assert q.is_cuda and keep.dtype == torch.bool
    B, H, M, D = q.shape
    N = k.shape[-2]
    NB = keep.shape[-1]

    kv_idx, max_kept = _compute_kv_idx(keep, block_m)            # [B,H,Qt,MAX_KEPT]
    keep_u8 = keep.to(torch.uint8).contiguous()

    out = torch.empty_like(q)
    grid = (B * H, triton.cdiv(M, block_m))
    _fwd_kernel[grid](
        q, k, v, keep_u8, kv_idx, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        keep_u8.stride(0), keep_u8.stride(1), keep_u8.stride(2), keep_u8.stride(3),
        kv_idx.stride(0), kv_idx.stride(1), kv_idx.stride(2), kv_idx.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale,
        M, N, H, NB,
        max_kept,
        BLOCK_M=block_m, BLOCK_N=block_n, D=D,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


@triton.jit
def _fwd_kernel_v2(
    Q_ptr, K_ptr, V_ptr, IDX_ptr, OUT_ptr,
    sqb, sqh, sqm, sqd,
    skb, skh, skn, skd,
    svb, svh, svn, svd,
    sIb, sIh, sIt, sIk,
    sob, soh, som, sod,
    scale,
    M, N, H, NB,
    MAX_KEPT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """Per-tile-selection variant: same online softmax as _fwd_kernel but
    drops the per-row keep-mask load. All rows in a Q-tile attend to all
    blocks in this tile's kv_idx. Selection happens upstream at tile
    granularity (SSA-style); the keep tensor is not materialized."""
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh %  H

    q_start = pid_m * BLOCK_M
    offs_m  = q_start + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, D)

    q_base   = Q_ptr   + b * sqb  + h * sqh
    k_base   = K_ptr   + b * skb  + h * skh
    v_base   = V_ptr   + b * svb  + h * svh
    o_base   = OUT_ptr + b * sob  + h * soh
    idx_base = IDX_ptr  + b * sIb + h * sIh + pid_m * sIt

    q_mask = offs_m[:, None] < M
    q_tile = tl.load(q_base + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                     mask=q_mask, other=0.0)

    NEG = -1.0e30
    m_i = tl.full([BLOCK_M], NEG, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q_max_in_tile = q_start + BLOCK_M - 1

    for i in range(0, MAX_KEPT):
        kv_block = tl.load(idx_base + i * sIk)
        if kv_block < NB:
            kv_start = kv_block * BLOCK_N
            if kv_start <= q_max_in_tile:
                offs_n = kv_start + tl.arange(0, BLOCK_N)
                n_in   = offs_n < N

                k_tile = tl.load(
                    k_base + offs_n[:, None] * skn + offs_d[None, :] * skd,
                    mask=n_in[:, None], other=0.0)

                s = tl.dot(q_tile, tl.trans(k_tile)) * scale

                causal  = offs_m[:, None] >= offs_n[None, :]
                inb     = n_in[None, :] & (offs_m[:, None] < M)
                ok      = causal & inb
                s = tl.where(ok, s, NEG)

                m_ij  = tl.max(s, axis=1)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                p     = tl.exp(s - m_new[:, None])

                v_tile = tl.load(
                    v_base + offs_n[:, None] * svn + offs_d[None, :] * svd,
                    mask=n_in[:, None], other=0.0)

                acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    tl.store(o_base + offs_m[:, None] * som + offs_d[None, :] * sod,
             out.to(OUT_ptr.dtype.element_ty), mask=q_mask)


def block_sparse_attn_v2(q, k, v, kv_idx, scale, block_n=64, block_m=64,
                          num_warps=4, num_stages=1):
    """v2: per-tile-selection block-sparse attention.

    q, k, v : [B, H, M, D]   bf16, contiguous, CUDA
    kv_idx  : [B, H, Q_tiles, MAX_KEPT] int32, the kept kv-block indices per
              Q-tile (pad with sentinel = NB). Caller produces this directly
              via tile-pooled selection (no materialized keep tensor).
    """
    assert q.shape == k.shape == v.shape
    assert q.is_cuda and kv_idx.dtype == torch.int32
    B, H, M, D = q.shape
    N = k.shape[-2]
    NB = (N + block_n - 1) // block_n
    MAX_KEPT = kv_idx.shape[-1]

    out = torch.empty_like(q)
    grid = (B * H, triton.cdiv(M, block_m))
    _fwd_kernel_v2[grid](
        q, k, v, kv_idx, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        kv_idx.stride(0), kv_idx.stride(1), kv_idx.stride(2), kv_idx.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale,
        M, N, H, NB,
        MAX_KEPT,
        BLOCK_M=block_m, BLOCK_N=block_n, D=D,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def reference_attn(q, k, v, keep, scale, block_n=64):
    """SDPA reference (materialises [B,H,M,N] mask). Used to verify."""
    import torch.nn.functional as F
    B, H, M, D = q.shape
    N = k.shape[-2]
    NB = keep.shape[-1]
    kk = (keep.unsqueeze(-1).expand(B, H, M, NB, block_n)
              .reshape(B, H, M, NB * block_n)[..., :N])
    causal = (torch.arange(N, device=q.device).view(1, N)
              <= torch.arange(M, device=q.device).view(M, 1))
    mask = kk & causal
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
