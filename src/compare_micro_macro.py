#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confronto fra l'analisi micro (bow-tie motif) e quella macro (tau-core).

Le due misurano la stessa intuizione architetturale con strumenti
indipendenti: dove concordano il risultato e' robusto, dove divergono la
divergenza stessa e' informativa. Si riportano la correlazione di rango su
tutti i gruppi e la sovrapposizione delle rispettive top-N, con il p-value
ipergeometrico.

Il confronto e' ristretto ai livelli comuni alle due analisi.

Uso:
    python src/compare_micro_macro.py --window 3-4-5 --macro <core_by_group.csv>
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from motif_distill import load as load_distilled


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"  [{timestamp()}] {msg}", flush=True)


def aggregate_micro(path, chunksize=500_000):
    """
    Aggrega il CSV dei motif per waist (bottleneck_area, bottleneck_level).

    Due metriche, deliberatamente diverse:
      * total_motif_count : somma dei conteggi. Fortemente confusa con la
        dimensione dei gruppi e col prodotto dei gradi.
      * n_patterns        : numero di pattern distinti in cui il gruppo fa da
        waist. Piu' robusta alla taglia, misura la VERSATILITA' del waist
        (quante combinazioni diverse di input/output esso media) ed e'
        concettualmente piu' vicina alla path centrality.
    """
    log(f"Aggregazione dell'analisi micro: {path}")
    parts = []
    n_rows = 0
    for chunk in pd.read_csv(
        path, usecols=["bottleneck_area", "bottleneck_level", "motif_count"],
        chunksize=chunksize,
    ):
        n_rows += len(chunk)
        parts.append(
            chunk.groupby(["bottleneck_area", "bottleneck_level"], sort=False)
            .agg(total_motif_count=("motif_count", "sum"),
                 n_patterns=("motif_count", "size"))
        )
    df = (pd.concat(parts)
          .groupby(level=[0, 1], sort=False)
          .sum()
          .reset_index()
          .rename(columns={"bottleneck_area": "group",
                           "bottleneck_level": "y_level"}))
    df["y_level"] = df["y_level"].astype(int)
    df["total_motif_count"] = df["total_motif_count"].astype(float)
    log(f"  righe lette: {n_rows:,}  waist distinti: {len(df):,}")
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Confronto tra waist dei bow-tie motif (micro) e "
                    "tau-core globale (macro).")
    ap.add_argument("--window", default="1-2-3",
                    help="Finestra da cui leggere il distillato")
    ap.add_argument("--micro", default=None,
                    help="CSV riga-per-riga (legacy).")
    ap.add_argument("--macro", required=True,
                    help="core_by_group_*.csv (output di hourglass_core.py)")
    ap.add_argument("--top_k", type=int, default=20,
                    help="Dimensione delle liste di testa da confrontare")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("  VALIDAZIONE INCROCIATA  micro (bow-tie motif)  <->  macro (tau-core)")
    print("=" * 78)

    if args.micro:
        micro = aggregate_micro(args.micro)
    else:
        # aggregati_waist contiene gia' n_pattern e massa per waist
        aw = load_distilled(args.window, "aggregati_waist")
        micro = pd.DataFrame({
            "group": aw["waist"].str.rsplit("L", n=1).str[0],
            "y_level": aw["waist"].str.rsplit("L", n=1).str[1].astype(int),
            "total_motif_count": aw["massa"].astype(float),
            "n_patterns": aw["n_pattern"].astype(int)})
        log(f"distillato {args.window}: {len(micro)} waist")
    macro = pd.read_csv(args.macro)
    macro["y_level"] = macro["y_level"].astype(int)
    log(f"tau-core: {len(macro)} gruppi, "
        f"{int(macro['n_core_neurons'].sum())} neuroni")

    # --- restrizione ai livelli comuni
    lv_micro = set(micro["y_level"])
    lv_macro = set(macro["y_level"])
    common_levels = sorted(lv_micro & lv_macro)
    log(f"Livelli in comune: {common_levels}  "
        f"(micro: {sorted(lv_micro)}, macro: {sorted(lv_macro)})")
    micro_r = micro[micro["y_level"].isin(common_levels)].copy()
    macro_r = macro[macro["y_level"].isin(common_levels)].copy()
    log(f"Dopo restrizione: micro={len(micro_r)} gruppi, macro={len(macro_r)} gruppi")

    # --- merge
    merged = micro_r.merge(macro_r, on=["group", "y_level"], how="outer")
    merged["total_motif_count"] = merged["total_motif_count"].fillna(0.0)
    merged["n_patterns"] = merged["n_patterns"].fillna(0).astype(int)
    merged["coverage"] = merged["coverage"].fillna(0.0)
    merged["n_core_neurons"] = merged["n_core_neurons"].fillna(0).astype(int)
    merged["in_core"] = merged["n_core_neurons"] > 0

    # --- sovrapposizione delle liste di testa
    K = args.top_k
    top_micro_cnt = set(map(tuple, micro_r.nlargest(K, "total_motif_count")
                            [["group", "y_level"]].values))
    top_micro_pat = set(map(tuple, micro_r.nlargest(K, "n_patterns")
                            [["group", "y_level"]].values))
    top_macro = set(map(tuple, macro_r.nlargest(K, "coverage")
                        [["group", "y_level"]].values))

    print()
    print(f"  [SOVRAPPOSIZIONE DELLE TOP-{K}]")
    for name, s in [("per total_motif_count", top_micro_cnt),
                    ("per n_patterns       ", top_micro_pat)]:
        inter = s & top_macro
        print(f"    micro {name} vs macro : {len(inter)}/{K} in comune")
        if inter:
            for g, l in sorted(inter):
                print(f"        - {g} L{l}")

    # --- ipergeometrico: la sovrapposizione e' piu' di quanto atteso a caso?
    N = len(merged)
    try:
        from scipy.stats import hypergeom
        for name, s in [("total_motif_count", top_micro_cnt),
                        ("n_patterns", top_micro_pat)]:
            k = len(s & top_macro)
            p = hypergeom.sf(k - 1, N, K, K) if k > 0 else 1.0
            exp = K * K / N
            print(f"    [{name}] attesi a caso: {exp:.2f}, osservati: {k}, "
                  f"p(iperg.) = {p:.3g}")
    except Exception as e:  # pragma: no cover
        print(f"    [WARN] test ipergeometrico non calcolato: {e}")

    # --- correlazione di rango su tutti i gruppi
    print()
    print("  [CORRELAZIONE DI RANGO SU TUTTI I GRUPPI]")
    for col in ("total_motif_count", "n_patterns"):
        rho, pv = spearmanr(merged[col], merged["coverage"])
        print(f"    Spearman({col:>18}, coverage) = {rho:+.3f}   p = {pv:.3g}")

    # --- i gruppi del core: dove si collocano nel ranking micro?
    print()
    print(f"  [I GRUPPI DEL tau-CORE, con la loro posizione nell'analisi micro]")
    merged["rank_micro_count"] = merged["total_motif_count"].rank(
        ascending=False, method="min").astype(int)
    merged["rank_micro_patterns"] = merged["n_patterns"].rank(
        ascending=False, method="min").astype(int)
    view = (merged[merged["in_core"]]
            .sort_values("coverage", ascending=False)
            [["group", "y_level", "n_core_neurons", "coverage",
              "rank_micro_count", "rank_micro_patterns", "n_patterns"]])
    with pd.option_context("display.width", 200):
        print(view.to_string(index=False))
    print(f"\n    (su {N} gruppi totali nei livelli {common_levels})")

    out = args.out or "micro_macro_comparison.csv"
    merged.sort_values("coverage", ascending=False).to_csv(out, index=False)
    print(f"\n  [OUTPUT] {out}")
    print(f"  Tempo totale: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
