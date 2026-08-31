#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conteggi per livello gerarchico e per transizione fra livelli.

Tre viste. La prima e' la distribuzione dei neuroni per livello, incrociata
con la classe funzionale. La seconda sono le transizioni del connettoma,
riportate anche come densita' (archi per 10.000 coppie possibili), perche' il
conteggio grezzo e' dominato dalla taglia dei livelli. La terza e' la massa
dei motif per catena (livello_in, livello_waist, livello_out).

La diagonale delle matrici e' il caso laterale.

Uso:
    python src/level_transitions.py --window 3-4-5
"""

from thesis_paths import data, results

import argparse
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from hourglass_areas import timestamp, format_elapsed
from motif_distill import load as load_distilled


def log(msg):
    print(f"  [{timestamp()}] {msg}", flush=True)


def fmt_matrix(M, levels, title, pct_by_row=False, fmt="{:,.0f}"):
    df = pd.DataFrame(M, index=[f"L{l}" for l in levels],
                      columns=[f"L{l}" for l in levels])
    print(f"\n  [{title}]")
    if pct_by_row:
        tot = df.sum(axis=1).replace(0, np.nan)
        df = (df.div(tot, axis=0) * 100).round(1)
        print(df.fillna(0).to_string())
    else:
        print(df.map(lambda x: fmt.format(x)).to_string())
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Punto 2: conteggi per livello e per transizione.")
    ap.add_argument("--motifs", default=None,
                    help="CSV riga-per-riga (legacy). Senza, usa il "
                         "distillato della finestra.")
    ap.add_argument("--window", default="1-2-3")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--outdir", default=results("level_transitions"))
    ap.add_argument("--conn_file", default=data("connections.csv"))
    ap.add_argument("--levels_file",
                    default=data("COORDINATE_XY_with_levels_tree.csv"))
    ap.add_argument("--class_file", default=data("classification.csv"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()

    print("=" * 78)
    print("  PUNTO 2 - CONTEGGI PER LIVELLO E PER TRANSIZIONE")
    print("=" * 78)

    # ---------- A. neuroni per livello -------------------------------------
    log("Caricamento livelli e classi...")
    dl = pd.read_csv(a.levels_file, usecols=["root_id", "y_level"])
    dl["y_level"] = dl["y_level"].astype(int)
    dc = pd.read_csv(a.class_file, usecols=["root_id", "super_class"])
    m = dl.merge(dc, on="root_id", how="left")
    levels = sorted(dl["y_level"].unique())
    n_lv = len(levels)

    per_lv = dl.groupby("y_level").size()
    print("\n  [A. NEURONI PER LIVELLO]")
    tot_n = int(per_lv.sum())
    for l in levels:
        n = int(per_lv.get(l, 0))
        print(f"    L{l}: {n:>7,}  ({100*n/tot_n:5.1f}%)")
    print(f"    {'tot':>3}: {tot_n:>7,}")

    ct = pd.crosstab(m["y_level"], m["super_class"].fillna("n/d"))
    ct.to_csv(os.path.join(a.outdir, "neuroni_per_livello_classe.csv"))
    print("\n  [A-bis. COMPOSIZIONE FUNZIONALE DI CIASCUN LIVELLO, %]")
    keep = [c for c in ct.columns if ct[c].sum() >= 500]
    print((ct[keep].div(ct[keep].sum(axis=1), axis=0) * 100).round(1).to_string())

    # ---------- B. transizioni del connettoma ------------------------------
    log("Matrice di transizione del connettoma...")
    lev = dict(zip(dl["root_id"].to_numpy(), dl["y_level"].to_numpy()))
    conn = pd.read_csv(a.conn_file,
                       usecols=["pre_root_id", "post_root_id", "syn_count"])
    conn = conn.groupby(["pre_root_id", "post_root_id"],
                        sort=False)["syn_count"].sum().reset_index()
    lp = np.array([lev.get(x, -1) for x in conn["pre_root_id"]])
    lq = np.array([lev.get(x, -1) for x in conn["post_root_id"]])
    ok = (lp > 0) & (lq > 0)
    lp, lq = lp[ok], lq[ok]
    syn = conn["syn_count"].to_numpy()[ok]

    E = np.zeros((n_lv, n_lv))
    S = np.zeros((n_lv, n_lv))
    idx = {l: i for i, l in enumerate(levels)}
    np.add.at(E, (np.array([idx[x] for x in lp]),
                  np.array([idx[x] for x in lq])), 1)
    np.add.at(S, (np.array([idx[x] for x in lp]),
                  np.array([idx[x] for x in lq])), syn)

    dfE = fmt_matrix(E, levels, "B. ARCHI DAL LIVELLO i AL LIVELLO j")
    fmt_matrix(E, levels, "B-bis. STESSO, in % per riga", pct_by_row=True)

    # densita': archi osservati / coppie possibili
    sizes = np.array([per_lv.get(l, 0) for l in levels], dtype=float)
    poss = np.outer(sizes, sizes)
    np.fill_diagonal(poss, sizes * (sizes - 1))
    dens = np.divide(E, poss, out=np.zeros_like(E), where=poss > 0)
    fmt_matrix(dens * 1e4, levels,
               "B-ter. DENSITA' (archi per 10.000 coppie possibili)",
               fmt="{:.2f}")
    print("    La densita' toglie l'effetto taglia: il conteggio grezzo e'")
    print("    dominato dai livelli grandi (L2 ha 47.049 neuroni, L1 13.601).")

    pd.DataFrame(E, index=[f"L{l}" for l in levels],
                 columns=[f"L{l}" for l in levels]).to_csv(
        os.path.join(a.outdir, "transizioni_connettoma_archi.csv"))
    pd.DataFrame(S, index=[f"L{l}" for l in levels],
                 columns=[f"L{l}" for l in levels]).to_csv(
        os.path.join(a.outdir, "transizioni_connettoma_sinapsi.csv"))

    # ---------- C. transizioni dei motif -----------------------------------
    if a.motifs is None:
        # le catene (livello_in, waist, livello_out) sono gia' nel distillato
        dch = load_distilled(a.window, "aggregati_catene")
        dch = dch.rename(columns={"massa": "massa"})
        dch["quota"] = dch["massa"] / dch["massa"].sum()
        dch = dch.sort_values("massa", ascending=False)
        dch.to_csv(os.path.join(a.outdir, "catene_livello.csv"), index=False)
        print("\n  [C. CATENE  fan-in -> waist -> fan-out]  (dal distillato)")
        v = dch.head(14).copy()
        v["massa"] = v["massa"].map("{:.4e}".format)
        v["quota"] = (dch.head(14)["quota"] * 100).round(2).astype(str) + "%"
        print(v.to_string(index=False))
        def direz(r):
            a_, b_, c_ = (r.livello_fan_in, r.livello_waist, r.livello_fan_out)
            d1 = "FF" if a_ < b_ else ("FB" if a_ > b_ else "lat")
            d2 = "FF" if b_ < c_ else ("FB" if b_ > c_ else "lat")
            return f"{d1}->{d2}"
        dch["direzione"] = dch.apply(direz, axis=1)
        g = dch.groupby("direzione")["massa"].sum().sort_values(ascending=False)
        print("\n  [SINTESI PER DIREZIONE DELLA CATENA]")
        for k_, vv in g.items():
            print(f"    {k_:<10} {vv:.4e}  ({100*vv/g.sum():5.2f}%)")
        print(f"\n  [OUTPUT] {a.outdir}/")
        print(f"  Tempo totale: {format_elapsed(time.time()-t0)}")
        return

    if not os.path.exists(a.motifs):
        print(f"\n  [WARN] CSV dei motif non trovato: {a.motifs}")
        print("  Sezione C saltata.")
        return

    log(f"Transizioni dei motif da {os.path.basename(a.motifs)} ...")
    Min = np.zeros((n_lv, n_lv))     # fan-in level -> waist level
    Mout = np.zeros((n_lv, n_lv))    # waist level -> fan-out level
    Nin = np.zeros((n_lv, n_lv))     # stesso, contando i pattern non la massa
    chain = defaultdict(float)       # (l_in, l_waist, l_out) -> massa
    rows = 0
    k_in = k_out = None

    for ch in pd.read_csv(a.motifs, chunksize=a.chunksize):
        if k_in is None:
            k_in, k_out = int(ch["k_in"].iloc[0]), int(ch["k_out"].iloc[0])
        rows += len(ch)
        cnt = ch["motif_count"].to_numpy(dtype=float)
        bl = ch["bottleneck_level"].to_numpy(dtype=int)
        bi = np.array([idx[x] for x in bl])
        il = [np.array([idx[x] for x in
                        ch[f"fan_in_{i}_level"].to_numpy(dtype=int)])
              for i in range(k_in)]
        ol = [np.array([idx[x] for x in
                        ch[f"fan_out_{i}_level"].to_numpy(dtype=int)])
              for i in range(k_out)]
        for v in il:
            np.add.at(Min, (v, bi), cnt)
            np.add.at(Nin, (v, bi), 1)
        for v in ol:
            np.add.at(Mout, (bi, v), cnt)
        # la catena si aggrega in modo vettorizzato: si codifica la terna
        # (fan_in, waist, fan_out) in un intero e si usa bincount
        for vi in il:
            for vo in ol:
                key = (vi * n_lv + bi) * n_lv + vo
                bc = np.bincount(key, weights=cnt, minlength=n_lv ** 3)
                for kk in np.flatnonzero(bc):
                    chain[(kk // (n_lv * n_lv), (kk // n_lv) % n_lv,
                           kk % n_lv)] += bc[kk]
        log(f"  {rows:,} righe")

    log(f"{rows:,} pattern elaborati (k_in={k_in}, k_out={k_out})")

    fmt_matrix(Min, levels,
               "C. MASSA DEI MOTIF: archi di FAN-IN, livello i -> waist j",
               fmt="{:.3e}")
    fmt_matrix(Min, levels, "C-bis. STESSO, in % per riga", pct_by_row=True)
    fmt_matrix(Nin, levels,
               "C-ter. NUMERO DI PATTERN (non la massa) con un fan-in i -> j")
    fmt_matrix(Mout, levels,
               "C-quater. MASSA: waist i -> fan-out j", fmt="{:.3e}")

    # vista a catena: e' letteralmente "quanti pattern da L1 a L2 a L3"
    ch_rows = [{"livello_fan_in": levels[i], "livello_waist": levels[j],
                "livello_fan_out": levels[k], "massa": v}
               for (i, j, k), v in chain.items() if v > 0]
    dch = pd.DataFrame(ch_rows).sort_values("massa", ascending=False)
    dch["quota"] = dch["massa"] / dch["massa"].sum()
    dch.to_csv(os.path.join(a.outdir, "catene_livello.csv"), index=False)

    print("\n  [C-quinquies. CATENE  fan-in -> waist -> fan-out]")
    print("    (e' la lettura diretta di 'quanti pattern partono da L1 a L2')")
    v = dch.head(14).copy()
    v["massa"] = v["massa"].map("{:.4e}".format)
    v["quota"] = (dch.head(14)["quota"] * 100).round(2).astype(str) + "%"
    print(v.to_string(index=False))

    print("\n  [SINTESI PER DIREZIONE DELLA CATENA]")
    def direz(r):
        a_, b_, c_ = r.livello_fan_in, r.livello_waist, r.livello_fan_out
        d1 = "FF" if a_ < b_ else ("FB" if a_ > b_ else "lat")
        d2 = "FF" if b_ < c_ else ("FB" if b_ > c_ else "lat")
        return f"{d1}->{d2}"
    dch["direzione"] = dch.apply(direz, axis=1)
    g = dch.groupby("direzione")["massa"].sum().sort_values(ascending=False)
    for k, vv in g.items():
        print(f"    {k:<10} {vv:.4e}  ({100*vv/g.sum():5.2f}%)")

    print(f"\n  [OUTPUT] {a.outdir}/")
    print(f"  Tempo totale: {format_elapsed(time.time()-t0)}")


if __name__ == "__main__":
    main()
