"""
modal_app_qwen36_kernel_test.py
===============================

Standalone correctness test for our v2 block-sparse kernel at Qwen3.6's
head_dim = 256. No model load, just random Q/K/V at Qwen3.6-35B-A3B's
attention shapes (post-GQA-expansion).

Three cases:

  A: keep=all-blocks (dense baseline). v2 output must match SDPA dense to
     bf16 precision (~1e-2 max|diff|, dominated by reduction-order noise).
  B: random ~50% sparse kv_idx per Q-tile (sink + self-block forced).
     v2 output must match reference_attn (SDPA with the same materialised
     mask) to bf16 precision.
  C: BLOCK_M=32 fallback path (in case 64 blows SMEM on A100).

Tested on H100-80GB (our intended smoke GPU). If BLOCK_M=64 works there,
fine. If not, the test reports clearly and the fallback BLOCK_M=32 is
tried.

Cost: ~$0.30, ~3 min.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "triton")
    .add_local_python_source("triton_block_attn")
)
app = modal.App("voi-router-qwen36-kernel-test")


@app.function(image=image, gpu="H100:1", timeout=1800)
def test():
    import torch
    from triton_block_attn import (
        block_sparse_attn, block_sparse_attn_v2, _compute_kv_idx,
        reference_attn,
    )

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    # Qwen3.6-35B-A3B full-attn shapes (post GQA expansion: Q=16 heads, KV
    # repeated to 16 via repeat_kv in block_sparse_attention).
    B, H, M, N = 1, 16, 256, 256
    D = 256
    BLOCK_N = 64
    NB = (N + BLOCK_N - 1) // BLOCK_N
    scale = D ** -0.5   # 1/16 = 0.0625, matches Qwen3.6's scaling

    print(f"=== shapes: B={B} H={H} M={M} N={N} D={D} NB={NB} BLOCK_N={BLOCK_N}",
          flush=True)
    print(f"=== scale = {scale}", flush=True)

    q = torch.randn(B, H, M, D, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, H, N, D, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, H, N, D, dtype=dtype, device=device) * 0.1

    import torch.nn.functional as F
    def sdpa_dense():
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale
        )

    def report(name, out_kernel, out_ref):
        diff = (out_kernel - out_ref).abs().float()
        denom = out_ref.abs().float().mean().clamp_min(1e-6)
        max_d, mean_d = float(diff.max()), float(diff.mean())
        rel = max_d / float(denom)
        verdict = "OK" if rel < 0.5 else "BAD"
        print(f"  [{name}] max|diff|={max_d:.5f}  mean|diff|={mean_d:.5f}  "
              f"ref_mean|val|={float(denom):.5f}  rel_max={rel:.3f}  -> {verdict}",
              flush=True)
        return verdict == "OK"

    results = {}

    # ---- Case A: dense via v2 with keep=all blocks.
    print(f"\n--- Case A: dense via v2 (all blocks kept) ---", flush=True)
    keep_all = torch.ones(B, H, M, NB, dtype=torch.bool, device=device)
    kv_idx_all, max_kept_all = _compute_kv_idx(keep_all, block_m=64)
    print(f"  kv_idx_all shape={tuple(kv_idx_all.shape)}  MAX_KEPT={max_kept_all}",
          flush=True)
    try:
        out_v2 = block_sparse_attn_v2(
            q.contiguous(), k.contiguous(), v.contiguous(),
            kv_idx_all, scale=scale, block_n=BLOCK_N, block_m=64,
        )
        out_sdpa = sdpa_dense()
        results["A_v2_BM64"] = report("A v2 BM=64", out_v2, out_sdpa)
    except Exception as e:
        print(f"  [A v2 BM=64] EXCEPTION: {type(e).__name__}: {e}", flush=True)
        results["A_v2_BM64"] = False

    # ---- Case B: random ~50% sparse, force sink (block 0) and self-block.
    print(f"\n--- Case B: random ~50% sparse ---", flush=True)
    Q_tiles_64 = (M + 63) // 64
    keep_rand = (torch.rand(B, H, M, NB, device=device) < 0.5)
    # Force block 0 (sink) and the per-row "self block" (block containing q
    # position i): i // BLOCK_N.
    keep_rand[..., 0] = True
    row_idx = torch.arange(M, device=device)
    self_blk = (row_idx // BLOCK_N).clamp(max=NB - 1)
    keep_rand[0, :, row_idx, self_blk] = True
    # Also force causal: a row can't keep a block strictly in the future.
    block_starts = torch.arange(NB, device=device) * BLOCK_N
    causal_ok = block_starts[None, :] <= row_idx[:, None]   # [M, NB]
    keep_rand = keep_rand & causal_ok[None, None, :, :]

    kv_idx_rand, max_kept_rand = _compute_kv_idx(keep_rand, block_m=64)
    print(f"  kv_idx_rand shape={tuple(kv_idx_rand.shape)}  MAX_KEPT={max_kept_rand}",
          flush=True)
    print(f"  mean kept blocks per row: "
          f"{float(keep_rand.float().sum(-1).mean()):.2f} / {NB}", flush=True)
    try:
        out_v2 = block_sparse_attn_v2(
            q.contiguous(), k.contiguous(), v.contiguous(),
            kv_idx_rand, scale=scale, block_n=BLOCK_N, block_m=64,
        )
        out_ref = reference_attn(q, k, v, keep_rand, scale=scale, block_n=BLOCK_N)
        results["B_v2_BM64_sparse"] = report("B v2 BM=64 sparse", out_v2, out_ref)
    except Exception as e:
        print(f"  [B v2 BM=64 sparse] EXCEPTION: {type(e).__name__}: {e}",
              flush=True)
        results["B_v2_BM64_sparse"] = False

    # ---- Case C: BLOCK_M=32 fallback.
    print(f"\n--- Case C: BLOCK_M=32 fallback ---", flush=True)
    kv_idx_all_32, _ = _compute_kv_idx(keep_all, block_m=32)
    try:
        out_v2_32 = block_sparse_attn_v2(
            q.contiguous(), k.contiguous(), v.contiguous(),
            kv_idx_all_32, scale=scale, block_n=BLOCK_N, block_m=32,
        )
        out_sdpa = sdpa_dense()
        results["C_v2_BM32"] = report("C v2 BM=32", out_v2_32, out_sdpa)
    except Exception as e:
        print(f"  [C v2 BM=32] EXCEPTION: {type(e).__name__}: {e}", flush=True)
        results["C_v2_BM32"] = False

    print(f"\n=== summary: {results}", flush=True)
    return results


@app.local_entrypoint()
def main():
    out = test.remote()
    print(f"\n--- result ---")
    print(out)
