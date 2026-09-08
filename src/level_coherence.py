#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coerenza fra categoria anatomica e livello gerarchico.

I livelli sono stati ottenuti da un embedding spettrale discretizzato da un
albero di decisione addestrato a separare le superclassi funzionali. Se ne
puo' quindi chiedere se restino informativi quando i neuroni si raggruppano
per gruppo anatomico, che e' l'abbinamento usato in questa tesi.

Lo script misura, per ciascuna delle tre granularita' disponibili:

    purezza      quota di neuroni della categoria nel suo livello modale
    entropia     dispersione della categoria sui cinque livelli
    NMI          informazione mutua normalizzata fra categoria e livello
    contiguita'  se i livelli occupati da una categoria sono consecutivi

e confronta ogni valore con un'assegnazione casuale dei livelli che
conserva le dimensioni delle categorie e la distribuzione marginale dei
livelli. Produce le tabelle e la figura usate nella tesi.

Uso:
    python src/level_coherence.py
    python src/level_coherence.py --figdir thesis/figures
"""

from thesis_paths import data, results
from neuropil_taxonomy import superregion_of_group

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEVELS = [1, 2, 3, 4, 5]
MASS_THRESHOLD = 0.05     # quota minima perche' un livello conti come occupato


def load_neurons():
    """Neuroni con gruppo anatomico, super-regione, superclasse e livello."""
    neu = pd.read_csv(data("neurons.csv"), usecols=["root_id", "group"])
    lev = pd.read_csv(data("COORDINATE_XY_with_levels_tree.csv"),
                      usecols=["root_id", "y_level"])
    cls = pd.read_csv(data("classification.csv"),
                      usecols=["root_id", "super_class"])
    df = neu.merge(lev, on="root_id").merge(cls, on="root_id", how="left")
    df = df.dropna(subset=["group"])
    df = df[df["group"] != "NO_CONS"].copy()
    df["superregion"] = df["group"].map(superregion_of_group)
    df = df.rename(columns={"y_level": "level"})
    return df


def per_category(df, col):
    """Purezza, entropia e contiguita' di ogni categoria della colonna."""
    rows = []
    for name, sub in df.groupby(col):
        c = sub["level"].value_counts().reindex(LEVELS, fill_value=0)
        p = c / c.sum()
        nz = p[p > 0]
        occupied = [lv for lv in LEVELS if p[lv] >= MASS_THRESHOLD]
        contiguous = (len(occupied) <= 1 or
                      occupied == list(range(min(occupied),
                                             max(occupied) + 1)))
        rows.append({
            col: name,
            "n": int(c.sum()),
            "mode_level": int(p.idxmax()),
            "purity": float(p.max()),
            "entropy": float(-(nz * np.log2(nz)).sum()),
            "n_levels": int((c > 0).sum()),
            "n_levels_5pct": len(occupied),
            "contiguous": bool(contiguous),
            "spread": float(np.sqrt(((np.array(LEVELS) - float(
                (p * LEVELS).sum())) ** 2 * p).sum())),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def mutual_information(df, col):
    """I(categoria; livello) e la sua versione normalizzata su H(livello)."""
    tab = pd.crosstab(df[col], df["level"]).to_numpy(dtype=float)
    p = tab / tab.sum()
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = p * np.log2(p / (px * py))
    mi = float(np.nansum(term))
    hy = float(-(py[py > 0] * np.log2(py[py > 0])).sum())
    return mi, mi / hy if hy else float("nan")


def null_reference(df, col, n_rep, rng):
    """
    Riferimento casuale: i livelli vengono rimescolati fra i neuroni.

    Conserva la dimensione di ogni categoria e la distribuzione marginale
    dei livelli, e distrugge solo l'associazione fra le due, cosi' il
    confronto isola quanto della coerenza osservata sia strutturale.
    """
    pur, nmi = [], []
    lv = df["level"].to_numpy()
    tmp = df[[col]].copy()
    for _ in range(n_rep):
        tmp["level"] = rng.permutation(lv)
        t = per_category(tmp, col)
        pur.append(float(np.average(t["purity"], weights=t["n"])))
        nmi.append(mutual_information(tmp, col)[1])
    return float(np.mean(pur)), float(np.mean(nmi))


def summarise(df, n_rep, rng):
    """Tabella di confronto fra le tre granularita'."""
    rows = []
    for col, label in (("super_class", "super-class"),
                       ("superregion", "super-region"),
                       ("group", "anatomical group")):
        d = df.dropna(subset=[col])
        t = per_category(d, col)
        w = float(np.average(t["purity"], weights=t["n"]))
        ent = float(np.average(t["entropy"], weights=t["n"]))
        mi, nmi = mutual_information(d, col)
        p_null, nmi_null = null_reference(d, col, n_rep, rng)
        contig = float(np.average(t["contiguous"].astype(float),
                                  weights=t["n"]))
        rows.append({
            "granularity": label,
            "categories": int(t.shape[0]),
            "metanodes": int(d.groupby([col, "level"]).ngroups),
            "median_size": float(t["n"].median()),
            "purity": w,
            "purity_null": p_null,
            "entropy": ent,
            "nmi": nmi,
            "nmi_null": nmi_null,
            "contiguous_share": contig,
        })
    return pd.DataFrame(rows)


def figure(df, tab_group, summary, outdir):
    """Tre pannelli: distribuzione per gruppo, purezza, informazione mutua."""
    fig = plt.figure(figsize=(9.0, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0],
                          hspace=0.45, wspace=0.42)

    # --- 1. i venti gruppi piu' popolosi, riga per riga ---------------------
    ax = fig.add_subplot(gs[:, 0])
    top = tab_group.head(20)["group"].tolist()
    m = (pd.crosstab(df[df["group"].isin(top)]["group"], df["level"])
         .reindex(index=top, columns=LEVELS, fill_value=0))
    m = m.div(m.sum(axis=1), axis=0)
    im = ax.imshow(m.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(LEVELS)), [f"L{l}" for l in LEVELS], fontsize=9)
    ax.set_yticks(range(len(top)), top, fontsize=8.5)
    ax.set_title("Level profile of the twenty\nlargest anatomical groups",
                 fontsize=10.5, fontweight="bold")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iloc[i, j]
            if v >= 0.10:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=7.5, color="white" if v > 0.55 else "#333333")
    # barra dei colori orizzontale: in verticale finirebbe sopra le
    # etichette del pannello di destra
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.032,
                      pad=0.07)
    cb.set_label("share of the group's neurons", fontsize=8.5, labelpad=2)
    cb.ax.tick_params(labelsize=7.5)

    # --- 2. distribuzione della purezza ------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(tab_group["purity"], bins=np.arange(0.2, 1.05, 0.05),
            weights=tab_group["n"] / 1000.0, color="#4c72b0",
            edgecolor="white")
    obs = float(np.average(tab_group["purity"], weights=tab_group["n"]))
    null = float(summary.loc[summary.granularity == "anatomical group",
                             "purity_null"].iloc[0])
    ax.axvline(obs, color="#c44e52", lw=2, label=f"observed {obs:.2f}")
    ax.axvline(null, color="#888888", lw=2, ls="--",
               label=f"random {null:.2f}")
    ax.set_xlabel("share of the group in its modal level", fontsize=9)
    ax.set_ylabel("neurons (thousands)", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.set_title("Groups are not confined to one level,\n"
                 "but are far from randomly spread",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5, frameon=False)

    # --- 3. informazione mutua per granularita' ----------------------------
    ax = fig.add_subplot(gs[1, 1])
    y = np.arange(len(summary))
    ax.barh(y + 0.19, summary["nmi"], height=0.36, color="#4c72b0",
            label="observed")
    ax.barh(y - 0.19, summary["nmi_null"], height=0.36, color="#bbbbbb",
            label="random levels")
    ax.set_yticks(y, summary["granularity"], fontsize=9)
    ax.set_xlabel(r"$I(\mathrm{category};\ \mathrm{level})\ /\ "
                  r"H(\mathrm{level})$", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.set_title("The finer the categories,\n"
                 "the more the level is predictable",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "level_coherence.png")
    fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OUTPUT] {p}")


def main():
    ap = argparse.ArgumentParser(
        description="Coerenza fra categoria anatomica e livello gerarchico.")
    ap.add_argument("--outdir", default=results("level_coherence"))
    ap.add_argument("--figdir", default=None,
                    help="dove scrivere la figura (default: --outdir)")
    ap.add_argument("--n_rep", type=int, default=20,
                    help="ripetizioni del riferimento casuale")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    df = load_neurons()
    print("=" * 74)
    print("  COERENZA FRA CATEGORIA ANATOMICA E LIVELLO")
    print("=" * 74)
    print(f"  {len(df):,} neuroni con gruppo e livello\n")

    summary = summarise(df, a.n_rep, rng)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    tab_group = per_category(df, "group")
    tab_sr = per_category(df, "superregion")

    print("\n  gruppi con livelli occupati consecutivi: "
          f"{tab_group['contiguous'].sum()}/{len(tab_group)} "
          f"({100 * np.average(tab_group['contiguous'].astype(float), weights=tab_group['n']):.1f}% dei neuroni)")
    print("\n  i dieci gruppi piu' popolosi:")
    cols = ["group", "n", "mode_level", "purity", "n_levels_5pct",
            "contiguous"]
    print(tab_group[cols].head(10).to_string(index=False))

    os.makedirs(a.outdir, exist_ok=True)
    summary.to_csv(os.path.join(a.outdir, "granularity_summary.csv"),
                   index=False)
    tab_group.to_csv(os.path.join(a.outdir, "per_group.csv"), index=False)
    tab_sr.to_csv(os.path.join(a.outdir, "per_superregion.csv"), index=False)
    print(f"\n  [OUTPUT] {a.outdir}/")

    figure(df, tab_group, summary, a.figdir or a.outdir)


if __name__ == "__main__":
    main()
