"""analyze_block_scores.py
============================

Post-processing for ``dump_block_scores.py`` shards. Computes the empirical
leading factor of Proposition (BAI) -- ``CV(r)``, ``r_q``, ``g_(q)``,
``sigma_(q)`` -- per (layer, scoring backbone) and tabulates the
``Phi(-g_q/(sqrt2 tau)) * r_q/tau`` factor for a small grid of plausible
``tau`` values.

Why ``tau`` is reported as a grid rather than a point estimate: ``tau`` cannot
be recovered from the per-tile order-statistic gaps (g, r, g_rho, sigma)
alone. The shape of the bound (B.5) in tau is monotone but not separable,
so we display the bound for ``tau`` ranging over the empirical g_q itself --
this brackets the regime where signal and noise are comparable.

Run::

    python3 analyze_block_scores.py /tmp/block_scores_qwen14b_niah_n10.pkl \
        --q 0.40 --n-layers 48
"""

import argparse
import math
import pickle
import sys
from collections import defaultdict

import numpy as np

MODE_LABELS = {0: "topk", 1: "router_mean", 2: "router_quest",
               3: "quest", 4: "other"}


def phi_neg(x):
    """Phi(-x) = 0.5 * erfc(x / sqrt(2))."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def collect_valid(shard):
    """Return four 1-D float32 arrays (g, r, sigma, g_rho) flattened over
    (B, H, Qt) but restricted to tiles where shard['valid'] is True. (The
    valid mask is [Qt]; we broadcast over B, H.)"""
    v = shard["valid"]                              # [Qt]
    if not v.any():
        return None
    mask = np.broadcast_to(v, shard["g"].shape)
    return (shard["g"][mask], shard["r"][mask],
            shard["sigma"][mask], shard["g_rho"][mask])


def analyze(shards, q, n_layers, tau_grid):
    by_layer_mode = defaultdict(list)
    for s in shards:
        layer = s["call_idx"] % n_layers if n_layers > 1 else 0
        by_layer_mode[(s["mode_id"], layer)].append(s)

    rows = []                            # (mode, layer, cv_r, r_q, g_q, sig_q, lead, n_tiles)
    for (mode, layer), group in sorted(by_layer_mode.items()):
        all_g, all_r, all_sig, all_grho = [], [], [], []
        for s in group:
            res = collect_valid(s)
            if res is None:
                continue
            g, r, sig, grho = res
            all_g.append(g); all_r.append(r)
            all_sig.append(sig); all_grho.append(grho)
        if not all_r:
            continue
        g_all = np.concatenate(all_g)
        r_all = np.concatenate(all_r)
        sig_all = np.concatenate(all_sig)
        grho_all = np.concatenate(all_grho)
        # Eq. (B.5) leading factor: 2 * CV(r) * q*(1-q) * r_q (tau-free).
        mean_r = float(np.mean(r_all))
        cv_r = float(np.std(r_all) / (mean_r + 1e-12))
        r_q = float(np.quantile(r_all, q))
        g_q = float(np.quantile(g_all, q))
        sig_q = float(np.quantile(sig_all, q))
        lead = 2.0 * cv_r * q * (1 - q) * r_q
        # Lemma 1 empirical: mean(g_rho - g) >= 0 ?
        delta_rho = float(np.mean(grho_all - g_all))
        rows.append((mode, layer, cv_r, r_q, g_q, sig_q, lead,
                     delta_rho, g_all.size))

    if not rows:
        print("[error] no valid tiles in dump", file=sys.stderr)
        return

    # Header: per (mode, layer) table.
    hdr = ("mode", "L", "n_tiles", "CV(r)", "r_q", "g_q", "sig_q",
           "lead(t-free)", "<grho-g>")
    fmt = " {:>13s} {:>3s} {:>8s} {:>7s} {:>7s} {:>8s} {:>8s} {:>13s} {:>9s}"
    print(fmt.format(*hdr))
    for mode, layer, cv_r, r_q, g_q, sig_q, lead, drho, n in rows:
        print(fmt.format(MODE_LABELS.get(mode, str(mode)), str(layer),
                          str(n), f"{cv_r:.3g}",
                          f"{r_q:.3g}", f"{g_q:.3g}", f"{sig_q:.3g}",
                          f"{lead:.3g}", f"{drho:+.3g}"))

    # Aggregate over layers per mode.
    print("\n# aggregated over layers (mean +/- std)")
    by_mode = defaultdict(list)
    for mode, layer, cv_r, r_q, g_q, sig_q, lead, drho, n in rows:
        by_mode[mode].append((cv_r, r_q, g_q, sig_q, lead, drho))
    agg_hdr = ("mode", "n_layers", "mean(CV(r))", "mean(r_q)", "mean(g_q)",
               "mean(sig_q)", "mean(lead)", "mean(<grho-g>)")
    fmtA = " {:>13s} {:>8s} {:>12s} {:>10s} {:>10s} {:>11s} {:>10s} {:>14s}"
    print(fmtA.format(*agg_hdr))
    for mode, vals in by_mode.items():
        a = np.array(vals)
        print(fmtA.format(
            MODE_LABELS.get(mode, str(mode)), str(len(vals)),
            f"{a[:,0].mean():.3g}", f"{a[:,1].mean():.3g}",
            f"{a[:,2].mean():.3g}", f"{a[:,3].mean():.3g}",
            f"{a[:,4].mean():.3g}", f"{a[:,5].mean():+.3g}"))

    # Phi-suppression factor as a function of an assumed tau, evaluated at
    # the layer-averaged g_q. Two points on the grid: tau = g_q (signal-noise
    # comparable, the bound's "worst-case" regime), and tau = g_q/4 (clean
    # signal, the regime where the bound predicts near-zero regret).
    print(f"\n# bound (B.5) as a function of assumed tau, averaged over layers, q={q}")
    bnd_hdr = ("mode", "tau/g_q", "Phi(-g_q/sqrt2/tau)", "bound = lead*Phi*r_q/tau")
    fmtB = " {:>13s} {:>10s} {:>22s} {:>26s}"
    print(fmtB.format(*bnd_hdr))
    for mode, vals in by_mode.items():
        a = np.array(vals)
        mean_g_q = a[:, 2].mean()
        mean_r_q = a[:, 1].mean()
        mean_lead = a[:, 4].mean()
        for ratio in tau_grid:
            tau = mean_g_q * ratio if mean_g_q > 0 else float("nan")
            if not (tau > 0 and math.isfinite(tau)):
                continue
            phi = phi_neg(mean_g_q / (math.sqrt(2) * tau))
            bnd = mean_lead * phi * (mean_r_q / tau) / mean_r_q  # cancels r_q in lead
            # The lead already contains r_q, so the full bound is
            #   bound = 2 * CV(r) * q(1-q) * Phi(-g_q/sqrt2/tau) * r_q/tau
            #        = lead * Phi * (1/tau)
            # Express it numerically in the next column.
            bnd_full = mean_lead * phi / tau
            print(fmtB.format(MODE_LABELS.get(mode, str(mode)),
                              f"{ratio:.3g}", f"{phi:.3g}",
                              f"{bnd_full:.3g}"))

    # Headline summary.
    print("\n# headline:")
    for mode, vals in by_mode.items():
        a = np.array(vals)
        ml = MODE_LABELS.get(mode, str(mode))
        print(f"  {ml:>14s}: "
              f"CV(r)={a[:,0].mean():.2f}  "
              f"sig_q={a[:,3].mean():.3g}  "
              f"<grho-g>={a[:,5].mean():+.3g}  "
              f"lead(tau-free)={a[:,4].mean():.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl_path")
    ap.add_argument("--q", type=float, default=0.40)
    ap.add_argument("--n-layers", type=int, required=True,
                    help="transformer layer count (Qwen-14B=48, Qwen-7B-1M=28, "
                         "Mistral-Nemo=40)")
    ap.add_argument("--tau-grid", type=str, default="0.1,0.25,0.5,1,2,4",
                    help="comma-separated multipliers of g_q to use as "
                         "assumed tau")
    args = ap.parse_args()
    with open(args.pkl_path, "rb") as f:
        shards = pickle.load(f)
    print(f"# loaded {len(shards)} shards from {args.pkl_path}")
    tau_grid = [float(x) for x in args.tau_grid.split(",")]
    analyze(shards, args.q, args.n_layers, tau_grid)


if __name__ == "__main__":
    main()
