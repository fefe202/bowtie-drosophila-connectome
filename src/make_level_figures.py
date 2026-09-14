#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Le due figure sui livelli gerarchici.

    level_construction.png  come i livelli sono ottenuti: l'albero di
                            decisione e le quattro soglie sull'asse del
                            flow score
    level_composition.png   cosa contengono: composizione di ogni livello
                            per superclasse funzionale

Sono due domande diverse e vanno tenute separate. La prima e' costruzione e
sta nel capitolo di background; la seconda e' un risultato, perche' verifica
che i livelli separino davvero sensoriale, elaborazione e uscita, e sta nel
capitolo dei risultati.

Le soglie non sono cablate: si ricavano dai dati come punto di mezzo fra il
punteggio minimo di un livello e quello massimo del livello successivo.

Uso:
    python src/make_level_figures.py --outdir thesis/figures
"""

from thesis_paths import data

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

DPI = 300
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LEVELS = [1, 2, 3, 4, 5]

# le superclassi in ordine sensoriale -> centrale -> motorio, cosi' la barra
# impilata si legge come un gradiente e non come un elenco
SUPERCLASS_ORDER = ["sensory", "optic", "visual_projection", "visual_centrifugal",
                    "ascending", "central", "endocrine", "descending", "motor"]
SUPERCLASS_COLOR = {
    "sensory": "#4fb3bf", "optic": "#2f6f9f", "visual_projection": "#dd8452",
    "visual_centrifugal": "#9b7cb8", "ascending": "#c44e52",
    "central": "#4c9a5a", "endocrine": "#9c9c9c", "descending": "#d98cb3",
    "motor": "#8c6d4f",
}


def save(fig, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {p}")


def load():
    lv = pd.read_csv(data("COORDINATE_XY_with_levels_tree.csv"),
                     usecols=["root_id", "score", "y_level"])
    cl = pd.read_csv(data("classification.csv"),
                     usecols=["root_id", "super_class"])
    return lv.merge(cl, on="root_id", how="left")


def thresholds(df):
    """Le quattro soglie, ricavate dai dati invece che cablate."""
    out = []
    for k in LEVELS[:-1]:
        hi = df[df.y_level == k]["score"].min()
        lo = df[df.y_level == k + 1]["score"].max()
        out.append((hi + lo) / 2.0)
    return out


# --- 1. come i livelli sono costruiti ---------------------------------------

def fig_construction(df, outdir):
    cuts = thresholds(df)
    counts = df.y_level.value_counts().reindex(LEVELS)
    share = counts / counts.sum()

    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(8.4, 7.4), gridspec_kw={"height_ratios": [1.0, 1.15]})

    # ---- pannello alto: l'albero di decisione ----------------------------
    # I quattro tagli stanno sullo stesso asse, quindi l'albero e' un pettine:
    # ogni nodo interno stacca un livello e passa il resto al successivo.
    axT.set_axis_off()
    axT.set_xlim(0, 9.6)
    axT.set_ylim(-0.95, 5.0)

    def box(x, y, text, leaf=False):
        w, h = (1.42, 0.60) if leaf else (2.15, 0.60)
        axT.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.05", linewidth=1.1,
            facecolor="#ffffff" if leaf else "#eef2f6",
            edgecolor="#5a6b7a", zorder=3))
        axT.text(x, y, text, ha="center", va="center", fontsize=8.2,
                 linespacing=1.35, zorder=4)

    def link(x1, y1, x2, y2, label):
        axT.annotate("", xy=(x2, y2 + 0.32), xytext=(x1, y1 - 0.32),
                     arrowprops=dict(arrowstyle="-|>", color="#5a6b7a",
                                     lw=1.0, shrinkA=0, shrinkB=0), zorder=2)
        axT.text((x1 + x2) / 2 + (0.16 if x2 > x1 else -0.16),
                 (y1 + y2) / 2, label, fontsize=7, color="#7a8894",
                 ha="left" if x2 > x1 else "right", va="center", zorder=4)

    xs_node = [1.9, 3.7, 5.5, 7.3]
    ys = [4.3, 3.2, 2.1, 1.0]
    for i, c in enumerate(cuts):
        box(xs_node[i], ys[i], f"score $>$ {c:.1f}?")

    for i in range(4):
        leaf_x = xs_node[i] - 1.05
        leaf_y = ys[i] - 1.1
        lab = f"$L_{i+1}$\n%.1f%%" % (100 * share[i + 1])
        box(leaf_x, leaf_y, lab, leaf=True)
        link(xs_node[i], ys[i], leaf_x, leaf_y, "yes")
        if i < 3:
            link(xs_node[i], ys[i], xs_node[i + 1], ys[i + 1], "no")

    box(xs_node[3] + 1.05, ys[3] - 1.1,
        "$L_5$\n%.1f%%" % (100 * share[5]), leaf=True)
    link(xs_node[3], ys[3], xs_node[3] + 1.05, ys[3] - 1.1, "no")

    axT.set_title("The five levels are four cuts on one continuous score",
                  fontsize=11, fontweight="bold", loc="left")

    # ---- pannello basso: la distribuzione del punteggio ------------------
    lo, hi = -70, 70
    clipped = df["score"].clip(lo, hi)
    axB.hist(clipped, bins=200, color="#8fa9bf", edgecolor="none")

    # spazio in testa per le etichette, altrimenti finiscono sul picco di L3
    ymax = axB.get_ylim()[1]
    axB.set_ylim(0, ymax * 1.26)
    band = [hi] + cuts + [lo]
    for k in LEVELS:
        left, right = band[k], band[k - 1]
        axB.axvspan(left, right, color="#f2f2f2" if k % 2 else "#fafafa",
                    zorder=0)
        axB.text((left + right) / 2, ymax * 1.20, f"L{k}",
                 ha="center", va="top", fontsize=10, fontweight="bold",
                 color="#44525e")
        axB.text((left + right) / 2, ymax * 1.09,
                 f"{counts[k]:,}", ha="center", va="top", fontsize=8,
                 color="#6a7783")
    for c in cuts:
        axB.axvline(c, color="#c44e52", lw=1.3, ls="--", zorder=3)
        # il valore della soglia sta dentro il grafico e ruotato: sotto
        # l'asse finirebbe sopra le etichette dei tick
        axB.text(c, ymax * 0.62, f"{c:.1f}", ha="right", va="center",
                 fontsize=7.5, color="#c44e52", rotation=90, zorder=4)

    axB.set_xlim(hi, lo)          # punteggio alto a sinistra: L1 in testa
    axB.set_xlabel("flow score  (high = receives from few, projects to many)")
    axB.set_ylabel("neurons")
    axB.set_title("Where the cuts fall on the population",
                  fontsize=11, fontweight="bold", loc="left")

    fig.tight_layout()
    save(fig, "level_construction.png", outdir)


# --- 2. cosa contengono i livelli -------------------------------------------

def fig_composition(df, outdir):
    d = df.dropna(subset=["super_class"])
    tab = pd.crosstab(d.y_level, d.super_class).reindex(
        index=LEVELS,
        columns=[c for c in SUPERCLASS_ORDER if c in d.super_class.unique()],
        fill_value=0)
    totals = tab.sum(axis=1)
    pct = tab.div(totals, axis=0) * 100

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1]})

    for ax, frame, ylab, title in (
            (axL, tab, "neurons",
             "How many neurons each level holds"),
            (axR, pct, "% of the level",
             "What each level is made of")):
        bottom = np.zeros(len(LEVELS))
        for c in frame.columns:
            v = frame[c].to_numpy(dtype=float)
            ax.bar([f"L{l}" for l in LEVELS], v, bottom=bottom, width=0.66,
                   color=SUPERCLASS_COLOR.get(c, "#bbbbbb"),
                   label=c.replace("_", " "))
            bottom += v
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, fontweight="bold")

    for i, l in enumerate(LEVELS):
        axL.text(i, totals[l] * 1.02, f"{totals[l]:,}", ha="center",
                 va="bottom", fontsize=8.5, fontweight="bold")
    axL.set_ylim(0, totals.max() * 1.12)
    axR.set_ylim(0, 100)

    handles, labels = axR.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="center left",
               bbox_to_anchor=(0.995, 0.5), fontsize=8.5, frameon=False,
               title="super-class", title_fontsize=9)
    fig.tight_layout()
    save(fig, "level_composition.png", outdir)


def main():
    ap = argparse.ArgumentParser(description="Figure sui livelli gerarchici.")
    ap.add_argument("--outdir", default="thesis/figures")
    a = ap.parse_args()

    df = load()
    print(f"  {len(df):,} neuroni con punteggio e livello")
    print("  soglie ricavate dai dati: " +
          ", ".join(f"{c:.3f}" for c in thresholds(df)))
    fig_construction(df, a.outdir)
    fig_composition(df, a.outdir)


if __name__ == "__main__":
    main()
