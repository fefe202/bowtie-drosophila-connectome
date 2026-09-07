#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figura del metagrafo: i neuroni raggruppati in metanodi (categoria, livello) e
gli archi fra metanodi su cui si cerca il motif.

Produce due pannelli. A sinistra il metagrafo alla granularita' delle
superclassi funzionali, 24 metanodi, disegnabile per intero. A destra la
distribuzione delle dimensioni dei metanodi alle due granularita', che mostra
perche' quella fine non e' disegnabile.

Uso:
    python src/make_metagraph_figure.py --outdir thesis/figures
"""

from thesis_paths import data, results

import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 300
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

C_EDGE = "#9aa5b1"
PAL = {
    "sensory": "#4c9f70", "optic": "#4878a8", "central": "#c44e52",
    "visual_projection": "#dd8452", "visual_centrifugal": "#8172b3",
    "descending": "#937860", "ascending": "#da8bc3", "motor": "#8c8c8c",
    "endocrine": "#ccb974",
}


def build(levels_file, class_file, neurons_file, conn_file, window, max_jump):
    lv = pd.read_csv(levels_file, usecols=["root_id", "y_level"])
    cl = pd.read_csv(class_file, usecols=["root_id", "super_class"]).dropna()
    nr = pd.read_csv(neurons_file, usecols=["root_id", "group"]).dropna()
    nr = nr[nr["group"] != "NO_CONS"]

    d = lv.merge(cl, on="root_id").merge(nr, on="root_id")
    d = d[d["y_level"].isin(window)]

    sc = d.groupby(["super_class", "y_level"]).size()
    gr = d.groupby(["group", "y_level"]).size()

    lev = dict(zip(d["root_id"], d["y_level"]))
    cls = dict(zip(d["root_id"], d["super_class"]))
    conn = pd.read_csv(conn_file, usecols=["pre_root_id", "post_root_id"])
    conn = conn[conn["pre_root_id"].isin(lev) & conn["post_root_id"].isin(lev)]

    edges = defaultdict(int)
    for a, b in zip(conn["pre_root_id"].to_numpy(), conn["post_root_id"].to_numpy()):
        la, lb = lev[a], lev[b]
        if abs(la - lb) > max_jump:
            continue
        s, t = (cls[a], la), (cls[b], lb)
        if s != t:
            edges[(s, t)] += 1
    return sc, gr, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=results("figures"))
    ap.add_argument("--window", default="1-2-3")
    ap.add_argument("--max_jump", type=int, default=1)
    ap.add_argument("--levels_file",
                    default=data("COORDINATE_XY_with_levels_tree.csv"))
    ap.add_argument("--class_file", default=data("classification.csv"))
    ap.add_argument("--neurons_file", default=data("neurons.csv"))
    ap.add_argument("--conn_file", default=data("connections.csv"))
    a = ap.parse_args()
    window = [int(x) for x in a.window.split("-")]

    sc, gr, edges = build(a.levels_file, a.class_file, a.neurons_file,
                          a.conn_file, window, a.max_jump)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.4),
                                   gridspec_kw={"width_ratios": [1.45, 1]})

    # ---- pannello sinistro: il metagrafo alle superclassi ----------------
    classes = sorted({k[0] for k in sc.index})
    ypos = {c: i for i, c in enumerate(classes)}
    pos = {(c, l): (l, ypos[c]) for c, l in sc.index}

    # Si disegnano solo gli archi piu' forti: con tutti e 293 il pannello
    # diventa illeggibile e non aggiunge informazione.
    mx = max(edges.values()) if edges else 1
    keep = sorted(edges.items(), key=lambda kv: kv[1], reverse=True)[:60]
    for (s, t), w in sorted(keep, key=lambda kv: kv[1]):
        if s not in pos or t not in pos:
            continue
        x1, y1 = pos[s]
        x2, y2 = pos[t]
        lw = 0.35 + 2.2 * (w / mx) ** 0.5
        axL.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=C_EDGE, lw=lw,
                                     alpha=0.5, shrinkA=16, shrinkB=18,
                                     connectionstyle="arc3,rad=0.18"))

    sizes = np.array([sc[k] for k in sc.index], dtype=float)
    for k, n in zip(sc.index, sizes):
        x, y = pos[k]
        axL.scatter([x], [y], s=70 + 620 * (n / sizes.max()) ** 0.5,
                    c=[PAL.get(k[0], "#777777")], zorder=3,
                    edgecolors="white", linewidths=1.4)
        # il conteggio va fuori dal nodo, altrimenti viene tagliato
        axL.annotate(f"{int(n):,}", (x, y), textcoords="offset points",
                     xytext=(0, -17), ha="center", fontsize=6.8,
                     color="#333333", zorder=5)

    axL.set_xticks(window, [f"level {l}" for l in window])
    axL.set_yticks(range(len(classes)), classes)
    axL.set_xlim(min(window) - 0.5, max(window) + 0.5)
    axL.set_ylim(-1.0, len(classes) - 0.15)
    axL.set_title(f"Metagraph at super-class granularity: {len(sc)} metanodes,\n"
                  f"{len(edges)} edges (60 strongest shown)",
                  fontsize=10, fontweight="bold")
    axL.tick_params(length=0)
    for sp in ("left", "bottom"):
        axL.spines[sp].set_visible(False)

    # ---- pannello destro: dimensioni dei metanodi -----------------------
    bins = np.logspace(0, np.log10(max(gr.max(), sc.max())) + 0.05, 26)
    axR.hist(gr.values, bins=bins, color="#4878a8", alpha=0.85,
             label=f"anatomical group: {len(gr)} metanodes")
    axR.hist(sc.values, bins=bins, color="#c44e52", alpha=0.85,
             label=f"super-class: {len(sc)} metanodes")
    axR.set_xscale("log")
    axR.set_xlabel("neurons per metanode")
    axR.set_ylabel("number of metanodes")
    axR.axvline(np.median(gr.values), color="#4878a8", ls="--", lw=1.2)
    axR.axvline(np.median(sc.values), color="#c44e52", ls="--", lw=1.2)
    axR.legend(fontsize=8, loc="upper right")
    axR.set_title("The two granularities differ by a factor of 35\n"
                  "in the number of metanodes", fontsize=10, fontweight="bold")

    os.makedirs(a.outdir, exist_ok=True)
    p = os.path.join(a.outdir, "metagraph.png")
    fig.tight_layout()
    fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {p}")
    print(f"  metanodi: super-class {len(sc)}, gruppo anatomico {len(gr)}")
    print(f"  mediana neuroni: super-class {int(np.median(sc.values))}, "
          f"gruppo {int(np.median(gr.values))}")
    print(f"  archi del metagrafo (superclassi): {len(edges)}")


if __name__ == "__main__":
    main()
