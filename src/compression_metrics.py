#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compressione nominale del waist, calcolata sul CSV completo.

Superata da compression_realized.py. La metrica implementata qui e' dominata
da gruppi in cui pochissimi neuroni sono attivi: LO.PVLP_L2 raggiunge 2431
con 2 neuroni attivi su 21, mentre AL.MB_CA_L2 resta a 40. Il file e'
conservato perche' quel confronto e' un risultato riportato in tesi.

Uso:
    python src/compression_metrics.py --input <csv dei motif>
"""

from thesis_paths import data, results

import argparse
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from directional_signature import add_signature, interpret


def log(msg):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_group_sizes(neurons_file, levels_file):
    """(group, level) -> numero di neuroni. Stessa definizione di hourglass_areas.py."""
    dn = pd.read_csv(neurons_file, usecols=["root_id", "group"])
    dn = dn.dropna(subset=["group"])
    dn = dn[dn["group"] != "NO_CONS"]
    dl = pd.read_csv(levels_file, usecols=["root_id", "y_level"])
    m = dn.merge(dl, on="root_id", how="inner")
    m["y_level"] = m["y_level"].astype(int)
    sizes = m.groupby(["group", "y_level"]).size()
    return {f"{g}L{l}": int(v) for (g, l), v in sizes.items()}


def add_metrics(df, k_in, k_out, sizes):
    """Aggiunge compression, density, count_per_waist e le dimensioni dei gruppi."""
    def sz(area_col, lev_col):
        key = df[area_col].astype(str) + "L" + df[lev_col].astype(str)
        return key.map(sizes).to_numpy(dtype=np.float64)

    n_b = sz("bottleneck_area", "bottleneck_level")
    in_sizes = [sz(f"fan_in_{i}_area", f"fan_in_{i}_level") for i in range(k_in)]
    out_sizes = [sz(f"fan_out_{i}_area", f"fan_out_{i}_level") for i in range(k_out)]

    periphery = sum(in_sizes) + sum(out_sizes)
    prod_periphery = np.ones(len(df))
    for a in in_sizes + out_sizes:
        prod_periphery = prod_periphery * a

    counts = df["motif_count"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["n_waist"] = n_b
        df["n_periphery"] = periphery
        df["compression"] = periphery / n_b
        df["density"] = counts / (prod_periphery * n_b)
        df["count_per_waist"] = counts / n_b
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Compressione del waist e normalizzazioni del conteggio.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default=results("compression_results"))
    ap.add_argument("--min_waist", type=int, default=20,
                    help="Dimensione minima del waist perche' il pattern entri "
                         "nelle classifiche per compressione (default 20). "
                         "Serve a escludere waist minuscoli, che producono "
                         "compressioni enormi ma prive di significato.")
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--neurons_file", default=data("neurons.csv"))
    ap.add_argument("--levels_file", default=data("COORDINATE_XY_with_levels_tree.csv"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    print("=" * 78)
    print("  COMPRESSIONE DEL WAIST E NORMALIZZAZIONI DEL CONTEGGIO")
    print("=" * 78)
    log(f"Input: {args.input}")
    sizes = load_group_sizes(args.neurons_file, args.levels_file)
    log(f"Dimensioni note per {len(sizes):,} metanodi (gruppo, livello)")
    log(f"Soglia --min_waist = {args.min_waist} neuroni")

    COLS = ["structure_str", "motif_count", "hourglass_type", "signature",
            "bottleneck_area", "bottleneck_level", "n_waist", "n_periphery",
            "compression", "density", "count_per_waist"]

    top_count, top_comp, top_dens = None, None, None
    per_sig = {}
    waist_stats = {}
    rows = n_kept = 0
    k_in = k_out = None
    hist = np.zeros(0)
    bins = np.array([0, 1, 2, 5, 10, 20, 50, 100, 1e9])

    for chunk in pd.read_csv(args.input, chunksize=args.chunksize):
        if k_in is None:
            k_in, k_out = int(chunk["k_in"].iloc[0]), int(chunk["k_out"].iloc[0])
            log(f"k_in = {k_in}, k_out = {k_out}")
            hist = np.zeros(len(bins) - 1)
        rows += len(chunk)
        add_signature(chunk, k_in, k_out)
        add_metrics(chunk, k_in, k_out, sizes)
        chunk = chunk[chunk["compression"].notna()]

        hist += np.histogram(chunk["compression"], bins=bins)[0]

        top_count = (chunk.nlargest(args.top_n, "motif_count")[COLS]
                     if top_count is None else
                     pd.concat([top_count, chunk.nlargest(args.top_n, "motif_count")[COLS]])
                     .nlargest(args.top_n, "motif_count"))

        elig = chunk[chunk["n_waist"] >= args.min_waist]
        n_kept += len(elig)
        if len(elig):
            top_comp = (elig.nlargest(args.top_n, "compression")[COLS]
                        if top_comp is None else
                        pd.concat([top_comp, elig.nlargest(args.top_n, "compression")[COLS]])
                        .nlargest(args.top_n, "compression"))
            top_dens = (elig.nlargest(args.top_n, "density")[COLS]
                        if top_dens is None else
                        pd.concat([top_dens, elig.nlargest(args.top_n, "density")[COLS]])
                        .nlargest(args.top_n, "density"))
            for s, sub in elig.groupby("signature"):
                t = sub.nlargest(args.top_n, "compression")[COLS]
                per_sig[s] = t if s not in per_sig else \
                    pd.concat([per_sig[s], t]).nlargest(args.top_n, "compression")
            w = elig["bottleneck_area"].astype(str) + "L" + \
                elig["bottleneck_level"].astype(str)
            for key, v in elig.groupby(w)["compression"].agg(["size", "max"]).iterrows():
                p = waist_stats.get(key, (0, 0.0))
                waist_stats[key] = (p[0] + int(v["size"]), max(p[1], float(v["max"])))

    log(f"{rows:,} righe elaborate, {n_kept:,} con waist >= {args.min_waist}")

    print("\n  [DISTRIBUZIONE DELLA COMPRESSIONE] (tutti i pattern)")
    labels = ["<1", "1-2", "2-5", "5-10", "10-20", "20-50", "50-100", ">100"]
    for lb, h in zip(labels, hist):
        print(f"    {lb:>8}: {int(h):>12,}  ({100*h/hist.sum():5.2f}%)")
    print(f"    Mediana della classe: la maggior parte dei pattern ha "
          f"compressione modesta.")

    fmt = {"compression": "{:.1f}", "density": "{:.3e}",
           "count_per_waist": "{:.3e}", "motif_count": "{:.4e}"}

    def show(df, title, sort):
        print(f"\n  [{title}]")
        d = df.sort_values(sort, ascending=False).head(15).copy()
        for c, f in fmt.items():
            d[c] = d[c].map(lambda x: f.format(x))
        with pd.option_context("display.width", 250, "display.max_colwidth", 52):
            print(d[["structure_str", "motif_count", "compression", "density",
                     "n_waist", "signature"]].to_string(index=False))

    show(top_count, "TOP PER motif_count GREZZO (il ranking attuale)", "motif_count")
    show(top_comp, f"TOP PER COMPRESSIONE (waist >= {args.min_waist})", "compression")
    show(top_dens, f"TOP PER DENSITA' (waist >= {args.min_waist})", "density")

    # confronto dei ranking
    s_count = set(top_count["structure_str"])
    s_comp = set(top_comp["structure_str"]) if top_comp is not None else set()
    print(f"\n  [CONFRONTO DEI RANKING]  top-{args.top_n} per conteggio vs per "
          f"compressione: {len(s_count & s_comp)} pattern in comune")

    def waists_of(df):
        return (df["bottleneck_area"].astype(str) + "L" +
                df["bottleneck_level"].astype(str)).value_counts()
    print("\n    waist dominanti nel top per CONTEGGIO:")
    for k, v in waists_of(top_count).head(5).items():
        print(f"      {k:<24} {v}")
    if top_comp is not None:
        print("    waist dominanti nel top per COMPRESSIONE:")
        for k, v in waists_of(top_comp).head(5).items():
            print(f"      {k:<24} {v}")

    # per firma direzionale
    print("\n  [MASSIMA COMPRESSIONE PER FIRMA DIREZIONALE]")
    rows_sig = []
    for s in sorted(per_sig, key=lambda x: -per_sig[x]["compression"].max()):
        t = per_sig[s].sort_values("compression", ascending=False)
        best = t.iloc[0]
        rows_sig.append({"signature": s, "compressione_max": best["compression"],
                         "waist": f"{best['bottleneck_area']}L{best['bottleneck_level']}",
                         "n_waist": int(best["n_waist"]),
                         "struttura": best["structure_str"],
                         "interpretazione": interpret(s)})
    dsig = pd.DataFrame(rows_sig)
    with pd.option_context("display.width", 250, "display.max_colwidth", 46):
        print(dsig.round(2).to_string(index=False))

    # salvataggi
    top_count.to_csv(os.path.join(args.outdir, "top_by_count.csv"), index=False)
    if top_comp is not None:
        top_comp.to_csv(os.path.join(args.outdir, "top_by_compression.csv"), index=False)
        top_dens.to_csv(os.path.join(args.outdir, "top_by_density.csv"), index=False)
    dsig.to_csv(os.path.join(args.outdir, "compression_by_signature.csv"), index=False)
    ws = pd.DataFrame([{"waist": k, "n_pattern": v[0], "compressione_max": v[1]}
                       for k, v in waist_stats.items()])
    ws.sort_values("compressione_max", ascending=False).to_csv(
        os.path.join(args.outdir, "compression_by_waist.csv"), index=False)
    d = os.path.join(args.outdir, "top_compression_by_signature")
    os.makedirs(d, exist_ok=True)
    for s, t in per_sig.items():
        t.sort_values("compression", ascending=False).to_csv(
            os.path.join(d, f"{s.replace('->', '_to_')}.csv"), index=False)

    print(f"\n  [OUTPUT] {args.outdir}/  (top_by_count, top_by_compression, "
          f"top_by_density, compression_by_signature, compression_by_waist)")
    print(f"  Tempo totale: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
