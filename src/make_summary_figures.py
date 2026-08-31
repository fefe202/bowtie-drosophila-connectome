#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure di sintesi per il capitolo dei risultati. Legge gli output gia'
prodotti dagli altri script, non ricalcola nulla.

    hscore_curve.png       curva H(tau)
    coverage_curve.png     copertura cumulativa dei cammini
    null_hscore.png        distribuzione nulla dell'H-score
    signature_distrib.png  firme direzionali, pattern e massa
    nt_by_direction.png    composizione E/I per direzione dell'arco
    compression.png        compressione nominale contro realizzata

Le etichette si generano in italiano o in inglese con --lang.

Uso:
    python src/make_summary_figures.py --lang en --outdir thesis/figures
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis_paths import results


# Etichette in italiano o inglese, scelte con --lang.
LANG = "it"

STRINGS = {
    "it": {
        "dros_levels": "Drosophila (routing per livelli)",
        "tau_x": r"soglia di copertura dei cammini  $\tau$",
        "h_y": r"H-score  $= 1 - C(\tau)/C_f(\tau)$",
        "h_title": "Effetto hourglass: piu' debole che in C. elegans",
        "cov_x": "neuroni nel core, in ordine di path centrality",
        "cov_y": "frazione di cammini sensoriale-motorio coperti",
        "cov_title": r"Copertura cumulativa  ({tot:.3g} cammini $S\rightarrow T$)",
        "cov_note": "{c} neuroni\ncoprono il 90%",
        "null_lab": "{n} reti randomizzate",
        "null_real": "rete reale: H = {h:.3f}",
        "null_x": r"H-score  ($\tau = 0.5$)",
        "null_y": "numero di reti",
        "null_title": "L'effetto non e' un artefatto  ($p = {p:.4f}$)",
        "sig_x0": "% dei pattern",
        "sig_t0": "Quanti pattern",
        "sig_x1": "% della massa di conteggio",
        "sig_t1": "Quanto pesano",
        "sig_sup": "Firme direzionali: il calderone 'mixed' si scompone in 13 classi",
        "nt_x": "% degli archi",
        "nt_title": "Composizione E/I per direzione\n(il divario e' quasi tutto medulla)",
        "nt_exc": "eccitatorio (ACh)",
        "nt_inh": "inibitorio (GABA, Glu)",
        "nt_mod": "modulatorio",
        "nt_unk": "sconosciuto",
        "cmp_x": r"$n^{\mathrm{eff}}$ del waist (participation ratio)",
        "cmp_y": "compressione realizzata",
        "cmp_title": "Un vero collo di bottiglia ha molti neuroni efficaci\ne alta compressione",
    },
    "en": {
        "dros_levels": "Drosophila (level-based routing)",
        "tau_x": r"path coverage threshold  $\tau$",
        "h_y": r"H-score  $= 1 - C(\tau)/C_f(\tau)$",
        "h_title": "Hourglass effect across coverage thresholds",
        "cov_x": "neurons in the core, ordered by path centrality",
        "cov_y": "fraction of sensory-to-motor paths covered",
        "cov_title": r"Cumulative coverage  ({tot:.3g} paths $S\rightarrow T$)",
        "cov_note": "{c} neurons\ncover 90% of paths",
        "null_lab": "{n} randomised networks",
        "null_real": "real network: H = {h:.3f}",
        "null_x": r"H-score  ($\tau = 0.5$)",
        "null_y": "number of networks",
        "null_title": "The effect is not an artefact  ($p = {p:.4f}$)",
        "sig_x0": "% of patterns",
        "sig_t0": "How many patterns",
        "sig_x1": "% of count mass",
        "sig_t1": "How much they weigh",
        "sig_sup": "Directional signatures: the 'mixed' bucket splits into 13 classes",
        "nt_x": "% of edges",
        "nt_title": "E/I composition by edge direction\n(the gap is almost entirely medulla)",
        "nt_exc": "excitatory (ACh)",
        "nt_inh": "inhibitory (GABA, Glu)",
        "nt_mod": "modulatory",
        "nt_unk": "unknown",
        "cmp_x": r"waist $n^{\mathrm{eff}}$ (participation ratio)",
        "cmp_y": "realised compression",
        "cmp_title": "A genuine bottleneck has many effective neurons\nand high compression",
    },
}


def T(key, **kw):
    v = STRINGS[LANG][key]
    return v.format(**kw) if kw else v


OUT = results(os.path.join("figures", "sintesi"))
DPI = 300
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.autolayout": True})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"    salvata: {p}")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- 1-3. analisi macro -----------------------------------------------------

def macro_figures():
    d = results("hourglass_core_results")
    f = os.path.join(d, "hscore_summary_tau0.9_syn5_levels_h5.json")
    if not os.path.exists(f):
        print("  [skip] manca il sommario dell'analisi macro")
        return
    s = read_json(f)

    core = pd.read_csv(os.path.join(
        d, "core_neurons_tau0.9_syn5_levels_h5.csv"))

    # La curva H(tau) viene LETTA dall'output di hourglass_core.py, non
    # scritta a mano: cosi' resta corretta se l'analisi viene rilanciata
    # con parametri diversi.
    cf = os.path.join(d, "hscore_curve_tau0.9_syn5_levels_h5.csv")
    if not os.path.exists(cf):
        print("  [skip] manca hscore_curve_*.csv: rilanciare hourglass_core.py")
        return
    cur = pd.read_csv(cf).dropna(subset=["H"])

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(cur["tau"], cur["H"], "o-", color="#1f77b4", lw=2, ms=6,
            label=T("dros_levels"))
    ax.axhspan(0.74, 0.87, color="#ff7f0e", alpha=0.18)
    ax.text(0.52, 0.805, "C. elegans\n(Sabrin & Dovrolis 2020)",
            fontsize=8, color="#b35900", va="center")
    ax.set_xlabel(T("tau_x"))
    ax.set_ylabel(T("h_y"))
    ax.set_ylim(0, 1)
    ax.set_title(T("h_title"),
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "hscore_curve.png")

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(core["rank"], core["cumulative_coverage"], color="#2ca02c", lw=2)
    ax.axhline(0.9, ls="--", c="gray", lw=1)
    ax.axvline(s["C"], ls="--", c="gray", lw=1)
    ax.annotate(T("cov_note", c=s["C"]),
                xy=(s["C"], 0.9), xytext=(s["C"] * 0.35, 0.55), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.set_xlabel(T("cov_x"))
    ax.set_ylabel(T("cov_y"))
    ax.set_ylim(0, 1)
    ax.set_title(T("cov_title", tot=s["total_paths"]),
                 fontsize=11, fontweight="bold")
    save(fig, "coverage_curve.png")

    # I valori nulli si leggono dal CSV dedicato, non dal JSON di sommario:
    # quest'ultimo viene sovrascritto da qualunque rilancio senza --n_random,
    # e i 100 valori andrebbero persi silenziosamente.
    rf = os.path.join(d, "hscore_random_tau0.9_syn5_levels_h5.csv")
    if os.path.exists(rf):
        hr = pd.read_csv(rf)["H_random"].to_numpy()
        h_real = float(cur.loc[cur["tau"] == 0.5, "H"].iloc[0])
        p_emp = (1 + int((hr >= h_real).sum())) / (len(hr) + 1)
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.hist(hr, bins=22, color="#999999", edgecolor="white",
                label=T("null_lab", n=len(hr)))
        ax.axvline(h_real, color="#d62728", lw=2.5,
                   label=T("null_real", h=h_real))
        ax.set_xlabel(T("null_x"))
        ax.set_ylabel(T("null_y"))
        ax.set_title(T("null_title", p=p_emp),
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        save(fig, "null_hscore.png")


# --- 4. firme direzionali ---------------------------------------------------

def signature_figure():
    f = results(os.path.join("signature_results", "signature_summary.csv"))
    if not os.path.exists(f):
        print("  [skip] manca il sommario delle firme")
        return
    d = pd.read_csv(f).sort_values("pct_pattern", ascending=True)
    highlight = {"FF->FB": "#d62728", "FB->FF": "#ff7f0e",
                 "FB->FB": "#9467bd", "FF->FF": "#2ca02c"}
    colors = [highlight.get(s, "#bbbbbb") for s in d["signature"]]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5), sharey=True)
    axes[0].barh(d["signature"], d["pct_pattern"], color=colors)
    axes[0].set_xlabel(T("sig_x0"))
    axes[0].set_title(T("sig_t0"), fontsize=10, fontweight="bold")
    axes[1].barh(d["signature"], d["pct_count"], color=colors)
    axes[1].set_xlabel(T("sig_x1"))
    axes[1].set_title(T("sig_t1"), fontsize=10, fontweight="bold")
    fig.suptitle(T("sig_sup"),
                 fontsize=11, fontweight="bold")
    save(fig, "signature_distrib.png")


# --- 5. neurotrasmettitori --------------------------------------------------

def nt_figure():
    f = results(os.path.join("nt_results", "nt_by_direction_score0.5.csv"))
    if not os.path.exists(f):
        print("  [skip] mancano i risultati sui neurotrasmettitori")
        return
    d = pd.read_csv(f, index_col=0).reindex(["FF", "lateral", "FB"])
    cols = [c for c in ["E", "I", "M", "U"] if c in d.columns]
    pct = d[cols].div(d[cols].sum(axis=1), axis=0) * 100
    palette = {"E": "#d62728", "I": "#1f77b4", "M": "#9467bd", "U": "#cccccc"}
    label = {"E": T("nt_exc"), "I": T("nt_inh"),
             "M": T("nt_mod"), "U": T("nt_unk")}

    fig, ax = plt.subplots(figsize=(6, 3.4))
    left = np.zeros(len(pct))
    for c in cols:
        ax.barh(pct.index, pct[c], left=left, color=palette[c], label=label[c])
        left += pct[c].to_numpy()
    ax.set_xlabel(T("nt_x"))
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.42))
    ax.set_title(T("nt_title"),
                 fontsize=10, fontweight="bold")
    save(fig, "nt_by_direction.png")


# --- 6. compressione --------------------------------------------------------

def compression_figure():
    f = results(os.path.join("compression_results", "compression_realized.csv"))
    if not os.path.exists(f):
        print("  [skip] mancano i risultati sulla compressione")
        return
    d = pd.read_csv(f)
    d = d[d["n_waist_efficace"] >= 5]
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.scatter(d["n_waist_efficace"], d["compressione_realizzata"],
               s=14, alpha=0.35, color="#7f7f7f", edgecolors="none")
    for w, col in (("AL.MB_CAL2", "#d62728"), ("AL.LHL2", "#ff7f0e"),
                   ("ME.LOL3", "#1f77b4")):
        s = d[d["waist"] == w]
        if len(s):
            ax.scatter(s["n_waist_efficace"], s["compressione_realizzata"],
                       s=42, color=col, label=w, zorder=5, edgecolors="white",
                       linewidths=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(T("cmp_x"))
    ax.set_ylabel(T("cmp_y"))
    ax.set_title(T("cmp_title"),
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    save(fig, "compression.png")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--lang", choices=["it", "en"], default="it",
                     help="lingua delle etichette delle figure")
    _ap.add_argument("--outdir", default=None,
                     help="cartella di destinazione")
    _a = _ap.parse_args()
    LANG = _a.lang
    if _a.outdir:
        OUT = _a.outdir
    print("Generazione delle figure di sintesi...")
    macro_figures()
    signature_figure()
    nt_figure()
    compression_figure()
    print(f"\nFatto. Output in {OUT}")
