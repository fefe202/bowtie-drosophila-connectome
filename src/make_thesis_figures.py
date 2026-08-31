#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure per la tesi non coperte da make_summary_figures.py. Legge gli output
gia' prodotti, non ricalcola.

    level_density.png      densita' di connessione livello -> livello
    signature_shift.png    firme direzionali, finestra superficiale contro
                           profonda
    window_growth.png      motif e neuroni per finestra
    nt_across_windows.png  confronto E/I dei waist sulle quattro finestre
    micro_macro.png        accordo fra le due scale
    level_chains.png       massa per catena di livelli

Uso:
    python src/make_thesis_figures.py --outdir thesis/figures
"""

from thesis_paths import results

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 300
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
})

# palette comune a tutte le figure
C_SHALLOW = "#4878a8"
C_DEEP = "#c44e52"
C_NEUTRAL = "#8c8c8c"
C_ACCENT = "#dd8452"


def save(fig, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {p}")


# --- 1. densita' livello -> livello ----------------------------------------

def fig_level_density(outdir):
    f = results(os.path.join("level_transitions",
                             "transizioni_connettoma_archi.csv"))
    if not os.path.exists(f):
        print("  [skip] level_density: manca la matrice delle transizioni")
        return
    E = pd.read_csv(f, index_col=0)
    sizes = np.array([13601, 47049, 44265, 17124, 12142], dtype=float)
    poss = np.outer(sizes, sizes)
    np.fill_diagonal(poss, sizes * (sizes - 1))
    dens = E.to_numpy() / poss * 1e4

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(dens, cmap="YlOrRd", aspect="equal")
    lv = [f"L{i}" for i in range(1, 6)]
    ax.set_xticks(range(5), lv)
    ax.set_yticks(range(5), lv)
    ax.set_xlabel("post-synaptic level")
    ax.set_ylabel("pre-synaptic level")
    for i in range(5):
        for j in range(5):
            v = dens[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > dens.max() * 0.55 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("edges per 10,000 possible pairs")
    ax.set_title("Connection density grows by an order of magnitude\n"
                 "with hierarchical depth", fontsize=11, fontweight="bold")
    save(fig, "level_density.png", outdir)


# --- 2. spostamento delle firme --------------------------------------------

def fig_signature_shift(outdir, load):
    try:
        a = load("1-2-3", "aggregati_firma")
        b = load("3-4-5", "aggregati_firma")
    except Exception as e:
        print(f"  [skip] signature_shift: {e}")
        return
    for d in (a, b):
        d["pct"] = 100 * d["massa"] / d["massa"].sum()
    m = (a[["signature", "pct"]].rename(columns={"pct": "shallow"})
         .merge(b[["signature", "pct"]].rename(columns={"pct": "deep"}),
                on="signature", how="outer").fillna(0))
    m["shift"] = m["deep"] - m["shallow"]
    m = m.sort_values("shift")

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    y = np.arange(len(m))
    ax.barh(y - 0.2, m["shallow"], height=0.38, color=C_SHALLOW,
            label=r"shallow window $\{1,2,3\}$")
    ax.barh(y + 0.2, m["deep"], height=0.38, color=C_DEEP,
            label=r"deep window $\{3,4,5\}$")
    ax.set_yticks(y, m["signature"])
    ax.set_xlabel("% of total count mass")
    ax.legend(fontsize=9, loc="center right", framealpha=0.95)
    ax.set_title("The deep brain is laterally dominated", fontsize=11,
                 fontweight="bold")
    ax.set_xlim(0, 46)
    # evidenzia le firme con un ramo puramente laterale
    for i, s in enumerate(m["signature"]):
        if "lat" in s:
            ax.axhspan(i - 0.45, i + 0.45, color="#f5f0e6", zorder=0)
    save(fig, "signature_shift.png", outdir)


# --- 3. crescita per finestra ----------------------------------------------

def fig_window_growth(outdir):
    w = ["{1,2,3}", "{2,3,4}", "{3,4,5}", "{1,..,5}"]
    motifs = np.array([31184800, 119026375, 433546629, 512969374], dtype=float)
    neurons = np.array([104915, 108438, 73531, 134181], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(w))
    bars = ax.bar(x, motifs / 1e6, color=[C_SHALLOW, C_NEUTRAL, C_DEEP,
                                          C_ACCENT], width=0.62)
    ax.set_xticks(x, w)
    ax.set_ylabel("motifs found (millions)")
    ax.set_xlabel("level window")
    for b, v in zip(bars, motifs):
        ax.text(b.get_x() + b.get_width() / 2, v / 1e6 + 12,
                f"{v/1e6:.0f}M", ha="center", fontsize=9, fontweight="bold")

    ax2 = ax.twinx()
    ax2.plot(x, neurons / 1000, "o--", color="black", lw=1.4, ms=6,
             label="neurons in window")
    ax2.set_ylabel("neurons (thousands)")
    ax2.set_ylim(0, 160)
    ax2.spines["top"].set_visible(False)
    ax2.legend(fontsize=9, loc="upper left")
    ax.set_title("Motif count grows with depth, not with size:\n"
                 r"$\{3,4,5\}$ has fewer neurons than $\{1,2,3\}$ "
                 "yet 14 times more motifs",
                 fontsize=11, fontweight="bold")
    save(fig, "window_growth.png", outdir)


# --- 4. neurotrasmettitori sulle quattro finestre ---------------------------

def fig_nt_across_windows(outdir):
    src = {
        r"$\{1,2,3\}$": "nt_results_1-2-3_dist",
        r"$\{2,3,4\}$": "nt_results_2-3-4",
        r"$\{3,4,5\}$": "nt_results_3-4-5",
        r"$\{1,2,3,4,5\}$": "nt_results_1-2-3-4-5",
    }
    rows = []
    for lab, d in src.items():
        f = results(os.path.join(d, "waist_paired_score0.5.csv"))
        if not os.path.exists(f):
            print(f"  [skip] nt_across_windows: manca {f}")
            return
        t = pd.read_csv(f)
        r = t[(t.tipo_A == "pure_FB") & (t.tipo_B == "pure_FF")]
        if not len(r):
            print(f"  [skip] nt_across_windows: nessun confronto FB/FF in {d}")
            return
        r = r.iloc[0]
        rows.append({"win": lab, "n": int(r.n_waist),
                     "gap": float(r.media_B) - float(r.media_A),
                     "p": float(r.p_wilcoxon)})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    y = np.arange(len(df))[::-1]
    cols = [C_DEEP if g < 0 else C_SHALLOW for g in df.gap]
    ax.barh(y, df.gap, color=cols, height=0.55)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_yticks(y, df.win)
    ax.set_xlabel("excitatory-fraction gap  (pure FF minus pure FB), "
                  "at parity of waist")
    # annotazioni allineate a destra: in fondo alla barra quelle negative
    # collidono con le etichette dell'asse
    for yy, (_, r) in zip(y, df.iterrows()):
        sig = "" if r.p >= 0.05 else "*"
        ax.text(0.145, yy, f"n = {r.n}    p = {r.p:.3f}{sig}",
                va="center", ha="left", fontsize=9,
                color="black" if r.p < 0.05 else "#666666")
    ax.set_xlim(-0.10, 0.235)
    ax.set_title("A result that does not replicate:\n"
                 "the sign reverses, and on the full window the effect "
                 "vanishes", fontsize=11, fontweight="bold")
    save(fig, "nt_across_windows.png", outdir)


# --- 5. accordo micro / macro ----------------------------------------------

def fig_micro_macro(outdir):
    f = results("micro_macro_comparison_postfix.csv")
    if not os.path.exists(f):
        print("  [skip] micro_macro: manca il confronto")
        return
    d = pd.read_csv(f)
    d = d[(d.total_motif_count > 0) & (d.coverage > 0)]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(d.total_motif_count, d.coverage, s=16, alpha=0.35,
               color=C_NEUTRAL, edgecolors="none")
    hi = {"AL.MB_CA": 2, "AL.LH": 2, "AL": 2, "AL.PLP": 2}
    for g, lv in hi.items():
        r = d[(d.group == g) & (d.y_level == lv)]
        if len(r):
            ax.scatter(r.total_motif_count, r.coverage, s=64, color=C_DEEP,
                       zorder=5, edgecolors="white", linewidths=1.0)
            ax.annotate(f"{g} L{lv}",
                        (r.total_motif_count.iloc[0], r.coverage.iloc[0]),
                        textcoords="offset points", xytext=(7, 4), fontsize=8.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("micro: waist motif mass")
    ax.set_ylabel(r"macro: $\tau$-core coverage")
    ax.set_title(r"The two scales agree weakly overall ($\rho = +0.121$)"
                 "\nand clearly on the olfactory mid-hierarchy",
                 fontsize=11, fontweight="bold")
    save(fig, "micro_macro.png", outdir)


# --- 6. catene di livello ---------------------------------------------------

def fig_level_chains(outdir):
    f = results(os.path.join("level_transitions_3-4-5", "catene_livello.csv"))
    if not os.path.exists(f):
        print("  [skip] level_chains: mancano le catene")
        return
    d = pd.read_csv(f).sort_values("massa", ascending=False).head(8)
    lab = [f"{int(a)}$\\to${int(b)}$\\to${int(c)}"
           for a, b, c in zip(d.livello_fan_in, d.livello_waist,
                              d.livello_fan_out)]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    y = np.arange(len(d))[::-1]
    cols = [C_DEEP if b == 3 else C_NEUTRAL for b in d.livello_waist]
    ax.barh(y, 100 * d.quota, color=cols, height=0.62)
    ax.set_yticks(y, lab)
    ax.set_xlabel("% of count mass")
    ax.set_ylabel("input level $\\to$ waist level $\\to$ output level")
    for yy, q in zip(y, d.quota):
        ax.text(100 * q + 0.6, yy, f"{100*q:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 55)
    ax.set_title("In the deep window the waist sits at L3\n"
                 "in about 90% of the mass (highlighted)",
                 fontsize=11, fontweight="bold")
    save(fig, "level_chains.png", outdir)


def main():
    ap = argparse.ArgumentParser(
        description="Figure aggiuntive per la tesi, etichette in inglese.")
    ap.add_argument("--outdir", default=results(os.path.join("figures",
                                                             "tesi")))
    a = ap.parse_args()

    from motif_distill import load

    print("Generating additional thesis figures...")
    fig_level_density(a.outdir)
    fig_signature_shift(a.outdir, load)
    fig_window_growth(a.outdir)
    fig_nt_across_windows(a.outdir)
    fig_micro_macro(a.outdir)
    fig_level_chains(a.outdir)
    print(f"\nDone. Output in {a.outdir}")


if __name__ == "__main__":
    main()
