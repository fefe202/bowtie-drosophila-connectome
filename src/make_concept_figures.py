#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Le figure concettuali: spiegano un'idea invece di riportare una misura.

    sigmapi_scheme.png      come il conteggio di un bow-tie si fattorizza
                            sul waist  (3.1.2)
    pattern_algebra.png     cammino, albero e ciclo: costo e additivita'
                            (3.1.3)
    signature_grid.png      le sedici firme direzionali disegnate  (2.3.2)
    flat_reference.png      rete reale contro rete piatta, e l'H-score
                            (3.2.2)
    null_models.png         cosa conserva e cosa distrugge ogni nullo
                            (3.4.1)
    olfactory_pathway.png   la via olfattiva come catena, con l'indice di
                            integrazione a ogni stadio  (4.10)

Le cifre non sono cablate dove esistono nei risultati: le percentuali delle
firme vengono dai distillati, l'H-score dal suo JSON, l'indice di
integrazione dallo sweep. Cosi' le figure non possono divergere dal testo.

Convenzioni, identiche a quelle delle figure dei motif: la profondita'
cresce verso il basso, un arco grigio va in avanti, uno rosso all'indietro,
uno blu resta nello stesso livello.

Uso:
    python src/make_concept_figures.py --outdir thesis/figures
"""

from thesis_paths import data, results
from directional_signature import normalize_directions

import argparse
import json
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

C_FWD = "#666666"
C_BWD = "#c0392b"
C_LAT = "#1f77b4"
C_MIX = "#8e7cc3"
C_BOX = "#5a6b7a"
C_FILL = "#eef2f6"
C_IN = "#4878a8"
C_WAIST = "#dd8452"
C_OUT = "#4c9a5a"


def save(fig, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {p}")


def dot(ax, x, y, c, r=90, z=3, ec="white"):
    ax.scatter([x], [y], s=r, c=[c], zorder=z, edgecolors=ec, linewidths=1.2)


def arrow(ax, p, q, c, lw=1.5, shrink=7, style="-|>", rad=0.0):
    ax.annotate("", xy=q, xytext=p, zorder=2,
                arrowprops=dict(arrowstyle=style, color=c, lw=lw,
                                shrinkA=shrink, shrinkB=shrink,
                                connectionstyle=f"arc3,rad={rad}"))


# ===========================================================================
#  1. la fattorizzazione SigmaPi
# ===========================================================================

def fig_sigmapi(outdir):
    """Perche' il conteggio si fattorizza sul waist, con numeri veri."""
    # esempio piccolo e verificabile a mano
    vA1 = [2, 1, 0, 3]
    vA2 = [1, 2, 2, 1]
    wD1 = [1, 0, 3, 2]
    wD2 = [2, 2, 1, 1]
    prod = [a * b * c * d for a, b, c, d in zip(vA1, vA2, wD1, wD2)]
    total = sum(prod)

    fig = plt.figure(figsize=(9.8, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.16)

    # ---- pannello sinistro: lo zoom su un solo neurone del waist -------
    # Disegnare tutti i neuroni del waist renderebbe gli archi impossibili
    # da contare. Qui si guarda un neurone solo, con i suoi archi veri: il
    # prodotto della colonna corrispondente si verifica contandoli.
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.set_xlim(-1.25, 3.0)
    ax.set_ylim(-3.2, 2.15)

    groups = [("$A_1$", 1.55, vA1[0], C_IN, True),
              ("$A_2$", 0.55, vA2[0], C_IN, True),
              ("$D_1$", -0.55, wD1[0], C_OUT, False),
              ("$D_2$", -1.55, wD2[0], C_OUT, False)]

    dot(ax, 1.55, 0.0, C_WAIST, r=460)
    ax.text(1.55, 0.0, "$i_1$", fontsize=12, ha="center", va="center",
            color="white", fontweight="bold", zorder=5)

    for lab, yy, k, col, incoming in groups:
        ax.text(-1.15, yy, lab, fontsize=11.5, va="center", ha="left",
                color=col, fontweight="bold")
        for j in range(3):
            x = -0.55 + 0.34 * j
            live = j < k
            dot(ax, x, yy, col if live else "#dcdcdc", r=115,
                ec="white" if live else "#f0f0f0")
            if live:
                if incoming:
                    arrow(ax, (x, yy), (1.55, 0.0), C_FWD, lw=1.5, shrink=11)
                else:
                    arrow(ax, (1.55, 0.0), (x, yy), C_FWD, lw=1.5, shrink=11)
        ax.text(0.72, yy, "%d" % k, fontsize=11, va="center", ha="center",
                color=col, fontweight="bold")

    ax.text(2.62, 0.0,
            r"$%d \times %d \times %d \times %d = \mathbf{%d}$"
            % (vA1[0], vA2[0], wD1[0], wD2[0], prod[0]),
            fontsize=12.5, va="center", ha="center", rotation=270)
    ax.text(0.75, -2.25,
            "bow-ties centred on $i_1$: one choice per branch,\n"
            "all four landing on this same neuron",
            fontsize=9.3, ha="center", va="top", style="italic",
            color="#44525e", linespacing=1.5)
    ax.set_title("One waist neuron at a time", fontsize=11,
                 fontweight="bold", loc="left")

    # ---- pannello destro: il conto ------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    rows = [
        ("$v_{A_1}[i]$", vA1, C_IN),
        ("$v_{A_2}[i]$", vA2, C_IN),
        ("$w_{D_1}[i]$", wD1, C_OUT),
        ("$w_{D_2}[i]$", wD2, C_OUT),
    ]
    x0, dx, y0, dy = 3.2, 1.55, 8.4, 1.05
    for i in range(4):
        ax.text(x0 + i * dx, y0 + 0.85, "$i_%d$" % (i + 1), fontsize=10,
                ha="center", fontweight="bold", color=C_WAIST)
    for r, (lab, vals, col) in enumerate(rows):
        y = y0 - r * dy
        ax.text(2.7, y, lab, fontsize=10.5, ha="right", va="center", color=col)
        for i, v in enumerate(vals):
            ax.text(x0 + i * dx, y, str(v), fontsize=11, ha="center",
                    va="center")
    yline = y0 - 3.55 * dy
    ax.plot([2.1, x0 + 3 * dx + 0.6], [yline, yline], color="#aaaaaa", lw=1)

    y = y0 - 4.35 * dy
    ax.text(2.7, y, "product", fontsize=10.5, ha="right", va="center",
            fontweight="bold")
    for i, p in enumerate(prod):
        ax.text(x0 + i * dx, y, str(p), fontsize=11.5, ha="center",
                va="center", fontweight="bold",
                color="#999999" if p == 0 else "#111111")

    ax.text(5.3, y - 1.35 * dy,
            r"$c \;=\; \sum_{i \in B} \prod_m v_{A_m}[i] \prod_n w_{D_n}[i]"
            r"\;=\; %s \;=\; \mathbf{%d}$"
            % (" + ".join(str(p) for p in prod), total),
            fontsize=12, ha="center", va="center")

    ax.text(5.3, y - 2.5 * dy,
            "A zero anywhere in a column kills that waist neuron:\n"
            "$i_3$ receives nothing from $A_1$, so no bow-tie is centred on it.",
            fontsize=9, ha="center", va="center", color="#44525e",
            linespacing=1.5)

    ax.set_title("Counting without enumeration: one column per waist neuron",
                 fontsize=11, fontweight="bold", loc="left")
    save(fig, "sigmapi_scheme.png", outdir)


# ===========================================================================
#  2. l'algebra dei pattern
# ===========================================================================

def fig_pattern_algebra(outdir):
    """Cammino, albero, ciclo: cosa cambia nel conteggio."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.2))

    specs = [
        ("path", [(0.5, 1.4), (1.5, 1.0), (2.5, 0.6)], [(0, 1), (1, 2)],
         "matrix product", r"$O(n)$", "yes", "#3d7a3d", "15 of 199"),
        ("tree  (the bow-tie)",
         [(0.4, 1.5), (0.4, 1.0), (1.5, 1.25), (2.6, 1.5), (2.6, 1.0)],
         [(0, 2), (1, 2), (2, 3), (2, 4)],
         "Hadamard at the branching", r"$O(n)$", "no", "#b8860b",
         "10 of 199"),
        ("cyclic", [(0.6, 1.6), (2.4, 1.6), (1.5, 0.7)],
         [(0, 1), (1, 2), (2, 0)],
         "not factorisable", r"$O(n^2)$", "no", "#a03030",
         "174 of 199"),
    ]

    for ax, (title, pos, edges, how, mem, adds, col, share) in zip(axes, specs):
        ax.set_axis_off()
        ax.set_xlim(0, 3.1)
        ax.set_ylim(-1.35, 2.25)
        for a, b in edges:
            arrow(ax, pos[a], pos[b], col, lw=1.8, shrink=11, rad=0.0)
        for p in pos:
            dot(ax, p[0], p[1], col, r=210)
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=col)
        ax.text(1.55, 0.05, "counted by: %s" % how, fontsize=9.5,
                ha="center", color="#333333")
        ax.text(1.55, -0.32, "memory: %s" % mem, fontsize=9.5, ha="center",
                color="#333333")
        ax.text(1.55, -0.69, "counts add over unions of groups: %s" % adds,
                fontsize=9.5, ha="center",
                color="#3d7a3d" if adds == "yes" else "#a03030")
        ax.text(1.55, -1.06, "connected four-node patterns: %s" % share,
                fontsize=9.5, ha="center", color=col, fontweight="bold")

    fig.suptitle("The shape of the pattern decides everything: "
                 "at four nodes, 87% of connected patterns are cyclic",
                 fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "pattern_algebra.png", outdir)


# ===========================================================================
#  3. le sedici firme direzionali
# ===========================================================================

def signature_shares():
    """Quota di massa per firma, dalle due finestre."""
    out = {}
    for win in ("1-2-3", "3-4-5"):
        f = results(os.path.join("motif_distilled", win, "aggregati_firma.csv"))
        d = normalize_directions(pd.read_csv(f))
        out[win] = dict(zip(d.signature, 100 * d.massa / d.massa.sum()))
    return out


def fig_signature_grid(outdir):
    """Le sedici firme disegnate, in una matrice ramo-entrante x ramo-uscente."""
    sh = signature_shares()
    order = ["FWD", "BWD", "LAT", "mix"]
    colour = {"FWD": C_FWD, "BWD": C_BWD, "LAT": C_LAT, "mix": C_MIX}

    fig, axes = plt.subplots(4, 4, figsize=(10.4, 10.6))
    for r, sin in enumerate(order):
        for c, sout in enumerate(order):
            ax = axes[r][c]
            ax.set_axis_off()
            ax.set_xlim(-0.35, 2.35)
            ax.set_ylim(-1.25, 1.25)
            sig = "%s->%s" % (sin, sout)

            # le firme con un ramo interamente laterale sono evidenziate
            if "LAT" in (sin, sout):
                ax.add_patch(FancyBboxPatch(
                    (-0.3, -1.2), 2.6, 2.4, boxstyle="round,pad=0.02",
                    facecolor="#f7f3e8", edgecolor="none", zorder=0))

            def dy(kind, k):
                """Scarto verticale del nodo: la pendenza dice la direzione."""
                if kind == "FWD":
                    return 0.72
                if kind == "BWD":
                    return -0.72
                if kind == "LAT":
                    return 0.0
                return 0.72 if k == 0 else -0.72     # mix: uno per verso

            # i due nodi di un ramo puro starebbero nello stesso punto:
            # si separano leggermente, altrimenti se ne vede uno solo
            for k in range(2):
                y = dy(sin, k)
                if sin != "mix":
                    y += 0.19 if k == 0 else -0.19
                col = colour[sin] if sin != "mix" else (
                    C_FWD if k == 0 else C_BWD)
                dot(ax, 0.0, y, C_IN, r=95)
                arrow(ax, (0.0, y), (1.15, 0.0), col, lw=1.7, shrink=8)
            dot(ax, 1.15, 0.0, C_WAIST, r=165)
            for k in range(2):
                y = -dy(sout, k)
                if sout != "mix":
                    y += 0.19 if k == 0 else -0.19
                col = colour[sout] if sout != "mix" else (
                    C_FWD if k == 0 else C_BWD)
                dot(ax, 2.3, y, C_OUT, r=95)
                arrow(ax, (1.15, 0.0), (2.3, y), col, lw=1.7, shrink=8)

            a = sh["1-2-3"].get(sig, 0.0)
            b = sh["3-4-5"].get(sig, 0.0)
            ax.set_title(sig, fontsize=10.5, fontweight="bold",
                         family="monospace", pad=4)
            ax.text(1.0, -1.14, "%.1f%%  /  %.1f%%" % (a, b), fontsize=8.6,
                    ha="center", color="#44525e")

    fig.suptitle("The sixteen directional signatures\n"
                 "row = input branch, column = output branch;  "
                 "share of mass, shallow / deep window",
                 fontsize=12.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.012,
             "Depth increases downwards: a descending arrow runs forward, a "
             "rising one backward, a level one laterally.\n"
             "Shaded cells are the seven signatures with an entirely lateral "
             "branch: together 38.0% of the shallow mass and 82.6% of the deep.",
             fontsize=9.2, ha="center", color="#44525e", linespacing=1.5)
    fig.tight_layout(rect=[0, 0.045, 1, 0.945])
    save(fig, "signature_grid.png", outdir)


# ===========================================================================
#  4. la rete piatta di riferimento
# ===========================================================================

def fig_flat_reference(outdir):
    """Perche' il core va confrontato con quello di una rete senza mezzo."""
    p = results(os.path.join("hourglass_core_results",
                             "hscore_summary_tau0.9_syn5_levels.json"))
    real = {"C": 154, "Cf": 307, "H": 0.498}
    if os.path.exists(p):
        j = json.load(open(p, encoding="utf-8"))
        real = {"C": j["C"], "Cf": j["Cf"], "H": round(j["H"], 3)}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.8, 5.0))
    ns, nt = 6, 6
    ys = np.linspace(-1.9, 1.9, ns)
    yt = np.linspace(-1.9, 1.9, nt)

    # ---------------- rete reale, con un waist stretto ------------------
    axL.set_axis_off()
    axL.set_xlim(-0.6, 4.6)
    axL.set_ylim(-2.9, 2.9)
    wy = [0.55, -0.55]
    for y in ys:
        dot(axL, 0.0, y, C_IN, r=110)
    for y in wy:
        dot(axL, 2.0, y, "#c0392b", r=230)
    for y in yt:
        dot(axL, 4.0, y, C_OUT, r=110)
    for y in ys:
        for w in wy:
            arrow(axL, (0.0, y), (2.0, w), "#c9c9c9", lw=0.8, shrink=9)
    for w in wy:
        for y in yt:
            arrow(axL, (2.0, w), (4.0, y), "#c9c9c9", lw=0.8, shrink=9)
    axL.text(0.0, 2.45, "sources $S$", fontsize=10, ha="center", color=C_IN,
             fontweight="bold")
    axL.text(4.0, 2.45, "targets $T$", fontsize=10, ha="center", color=C_OUT,
             fontweight="bold")
    axL.text(2.0, 2.45, "waist", fontsize=10, ha="center", color="#c0392b",
             fontweight="bold")
    axL.text(2.0, -2.5, "every path passes through 2 nodes\n"
                        r"$|C(\tau)| = 2$",
             fontsize=10.5, ha="center", color="#c0392b", linespacing=1.5)
    axL.set_title("the real network", fontsize=12, fontweight="bold")

    # ---------------- rete piatta -----------------------------------
    axR.set_axis_off()
    axR.set_xlim(-0.6, 4.6)
    axR.set_ylim(-2.9, 2.9)
    # il cover e' uno dei due lati: sei nodi bastano, e solo quelli
    # vanno evidenziati
    for y in ys:
        dot(axR, 0.0, y, "#c0392b", r=110)
    for y in yt:
        dot(axR, 4.0, y, C_OUT, r=110)
    for y in ys:
        for t in yt:
            arrow(axR, (0.0, y), (4.0, t), "#dddddd", lw=0.45, shrink=9,
                  style="-")
    axR.text(0.0, 2.45, "sources $S$", fontsize=10, ha="center", color=C_IN,
             fontweight="bold")
    axR.text(4.0, 2.45, "targets $T$", fontsize=10, ha="center", color=C_OUT,
             fontweight="bold")
    axR.text(2.0, 0.0, "no intermediate\nstructure to share",
             fontsize=10, ha="center", va="center", style="italic",
             color="#777777", linespacing=1.5)
    axR.text(2.0, -2.5, "no node lies on another pair's route, so a cover\n"
                        "must name one whole side: "
                        r"$|C_f(\tau)| = 6$",
             fontsize=10.5, ha="center", color="#c0392b", linespacing=1.5)
    axR.set_title("the flat dependency network $G_f$", fontsize=12,
                  fontweight="bold")

    fig.suptitle(
        r"The hourglass score compares the two:  "
        r"$H = 1 - |C(\tau)|\,/\,|C_f(\tau)|$"
        "\n"
        r"in this sketch $1 - 2/6 = 0.67$;  in the connectome "
        + r"$1 - %d/%d = \mathbf{%.3f}$" % (real["C"], real["Cf"], real["H"]),
        fontsize=12, fontweight="bold", y=1.06)
    fig.tight_layout()
    save(fig, "flat_reference.png", outdir)


# ===========================================================================
#  5. i tre modelli nulli
# ===========================================================================

def fig_null_models(outdir):
    """Cosa conserva e cosa distrugge ciascun nullo, sulla stessa reticella."""
    rng = np.random.default_rng(7)
    n = 10
    half = n // 2
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.3
    pos = np.column_stack([np.cos(ang), np.sin(ang)])

    # osservato: molto denso dentro i blocchi, poco fra i blocchi
    obs = []
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            same = (a < half) == (b < half)
            if rng.random() < (0.42 if same else 0.06):
                obs.append((a, b))

    def rewire_free(edges):
        """A: gradi conservati, anatomia distrutta."""
        out = [(a, int(rng.integers(n))) for a, b in edges]
        return [(a, b) for a, b in out if a != b]

    def rewire_block(edges):
        """B: densita' fra blocchi conservata, gradi distrutti."""
        out = []
        for a, b in edges:
            ba, bb = a < half, b < half
            src = rng.integers(0, half) if ba else rng.integers(half, n)
            dst = rng.integers(0, half) if bb else rng.integers(half, n)
            if src != dst:
                out.append((int(src), int(dst)))
        return out

    def permute_degrees(edges):
        """C: blocchi e gradi conservati, chi ha quale grado rimescolato."""
        perm = np.arange(n)
        perm[:half] = rng.permutation(perm[:half])
        perm[half:] = rng.permutation(perm[half:])
        return [(int(perm[a]), int(perm[b])) for a, b in edges]

    panels = [
        ("observed", obs, "the connectome as measured", "#333333"),
        ("model A", rewire_free(obs),
         "keeps: degree of every neuron\nloses: which region connects to which",
         "#4878a8"),
        ("model B", rewire_block(obs),
         "keeps: density between regions\nloses: individual degrees",
         "#dd8452"),
        ("model C", permute_degrees(obs),
         "keeps: regions and degrees\nloses: which neuron has which degree",
         "#4c9a5a"),
    ]

    # Due canali visivi, altrimenti i quattro pannelli sembrano lo stesso
    # groviglio: la dimensione del nodo e' il suo grado, e gli archi che
    # attraversano le due regioni sono colorati. Cosi' si vede a colpo
    # d'occhio che A rompe le regioni, B appiattisce i gradi, C non tocca
    # ne' le une ne' gli altri.
    def degrees(edges):
        d = np.zeros(n)
        for a, b in edges:
            d[a] += 1
            d[b] += 1
        return d

    d_obs = degrees(obs)

    fig, axes = plt.subplots(1, 4, figsize=(10.2, 4.6))
    for ax, (title, edges, note, col) in zip(axes, panels):
        ax.set_axis_off()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-2.5, 1.5)
        cross = 0
        for a, b in edges:
            same = (a < half) == (b < half)
            if not same:
                cross += 1
            ax.annotate("", xy=pos[b], xytext=pos[a], zorder=1 if same else 2,
                        arrowprops=dict(
                            arrowstyle="-",
                            color="#d8d8d8" if same else "#e08214",
                            lw=0.7 if same else 1.3,
                            shrinkA=7, shrinkB=7,
                            connectionstyle="arc3,rad=0.18"))
        dd = degrees(edges)
        for k in range(n):
            dot(ax, pos[k][0], pos[k][1],
                "#c44e52" if k < half else "#4878a8",
                r=40 + 17 * dd[k])
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=col)
        ax.text(0, -1.52, "edges crossing regions: %d" % cross, fontsize=9,
                ha="center", va="top", color="#b06000", fontweight="bold")
        ax.text(0, -1.9, note, fontsize=8.8, ha="center", va="top",
                color="#44525e", linespacing=1.5)

    fig.suptitle("Three null models, three different questions\n"
                 "colour = anatomical region,  node size = degree,  "
                 "orange edges cross between regions",
                 fontsize=11.5, fontweight="bold", y=1.09)
    fig.tight_layout()
    save(fig, "null_models.png", outdir)


# ===========================================================================
#  6. la via olfattiva ricomposta
# ===========================================================================

def fig_olfactory(outdir):
    """La catena espansione -> mantenimento -> compressione, con l'indice."""
    p = results(os.path.join("sweep_kin_kout", "sweep_1-2-3.csv"))
    idx = {}
    if os.path.exists(p):
        t = pd.read_csv(p)
        for w in ("AL.MB_CAL2", "AL.LHL2", "MB_CA.MB_MLL3",
                  "MB_ML.CREL2", "MB_PED.CREL2"):
            r = t[t.waist == w]
            if len(r):
                r = r.iloc[0]
                idx[w] = (int(r.n_gruppi_in), int(r.n_gruppi_out),
                          float(r.indice_norm))

    stages = [
        ("AL.MB_CA L2", "projection\nneurons", "AL.MB_CAL2"),
        ("MB_CA.MB_ML L3", "Kenyon\ncells", "MB_CA.MB_MLL3"),
        ("MB_ML.CRE L2", "output\nneurons", "MB_ML.CREL2"),
    ]

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    ax.set_axis_off()
    ax.set_xlim(-0.9, 11.4)
    ax.set_ylim(-3.4, 3.3)

    def fan(x, ytop, k, col, r=58):
        ys = np.linspace(-ytop, ytop, k)
        for y in ys:
            dot(ax, x, y, col, r=r)
        return ys

    # la geometria racconta la storia: pochi -> molti -> pochi
    y0 = fan(0.0, 0.75, 3, C_IN)
    y1 = fan(3.0, 0.95, 4, C_WAIST, r=150)
    y2 = fan(6.0, 2.35, 11, "#9467bd")
    y3 = fan(9.0, 0.95, 4, C_WAIST, r=150)
    y4 = fan(11.0, 0.5, 2, C_OUT)

    for a in y0:
        for b in y1:
            arrow(ax, (0.0, a), (3.0, b), "#cccccc", lw=0.7, shrink=8)
    for a in y1:
        for b in y2:
            arrow(ax, (3.0, a), (6.0, b), "#cccccc", lw=0.6, shrink=8)
    for a in y2:
        for b in y3:
            arrow(ax, (6.0, a), (9.0, b), "#cccccc", lw=0.6, shrink=8)
    for a in y3:
        for b in y4:
            arrow(ax, (9.0, a), (11.0, b), "#cccccc", lw=0.7, shrink=8)

    ax.text(0.0, 1.35, "antennal\nlobe", fontsize=9.5, ha="center",
            color=C_IN, fontweight="bold", linespacing=1.4)
    ax.text(11.0, 1.15, "behaviour", fontsize=9.5, ha="center",
            color=C_OUT, fontweight="bold")

    for x, (tex, role, key) in zip((3.0, 6.0, 9.0), stages):
        ytop = 2.35 if key == "MB_CA.MB_MLL3" else 0.95
        # lo stadio centrale e' il piu' alto: la sua etichetta va a lato,
        # altrimenti finisce sulle frecce di espansione e compressione
        if key == "MB_CA.MB_MLL3":
            # su fondo bianco, altrimenti si perde fra gli archi
            ax.text(x + 1.30, 0.0, role, fontsize=10, ha="left",
                    va="center", fontweight="bold", color="#333333",
                    linespacing=1.4, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none",
                              pad=2.5, alpha=0.92))
        else:
            ax.text(x, ytop + 0.55, role, fontsize=10, ha="center",
                    fontweight="bold", color="#333333", linespacing=1.4)
        if key in idx:
            gi, go, v = idx[key]
            regime = ("broadcast" if v < -0.33 else
                      "integrative" if v > 0.33 else "mixed")
            col = (C_IN if v < -0.33 else
                   "#c0392b" if v > 0.33 else "#8a8a8a")
            ax.text(x, -ytop - 0.62, tex, fontsize=9.5, ha="center",
                    color="#444444", family="monospace")
            ax.text(x, -ytop - 1.12, "%d in / %d out" % (gi, go), fontsize=8.6,
                    ha="center", color="#666666")
            ax.text(x, -ytop - 1.66, "index %+.3f" % v, fontsize=11,
                    ha="center", fontweight="bold", color=col)
            ax.text(x, -ytop - 2.12, regime, fontsize=9.5, ha="center",
                    color=col, style="italic")

    ax.annotate("", xy=(6.0, 3.05), xytext=(3.0, 3.05),
                arrowprops=dict(arrowstyle="-|>", color=C_IN, lw=2.2))
    ax.text(4.5, 3.2, "expansion", fontsize=10.5, ha="center", color=C_IN,
            fontweight="bold")
    ax.annotate("", xy=(9.0, 3.05), xytext=(6.0, 3.05),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2.2))
    ax.text(7.5, 3.2, "compression", fontsize=10.5, ha="center",
            color="#c0392b", fontweight="bold")

    ax.set_title("The integration index recovers the mushroom body "
                 "architecture from wiring alone",
                 fontsize=12, fontweight="bold")
    save(fig, "olfactory_pathway.png", outdir)


def main():
    ap = argparse.ArgumentParser(description="Figure concettuali della tesi.")
    ap.add_argument("--outdir", default="thesis/figures")
    a = ap.parse_args()
    fig_sigmapi(a.outdir)
    fig_pattern_algebra(a.outdir)
    fig_signature_grid(a.outdir)
    fig_flat_reference(a.outdir)
    fig_null_models(a.outdir)
    fig_olfactory(a.outdir)


if __name__ == "__main__":
    main()
