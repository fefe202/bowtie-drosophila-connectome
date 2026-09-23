#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Le figure che servono alle slide e non alla tesi.

Una figura da tesi si guarda da quaranta centimetri, una da slide da cinque
metri. La griglia delle sedici firme, cosi' com'e' nella tesi, mette due
percentuali sotto ciascuna delle sedici celle: su una proiezione non si
leggono, e non servono, perche' i due numeri che contano (38,0% e 82,6%)
stanno scritti sulla slide.

Qui la stessa griglia si ridisegna piu' grande e senza i numerini, con le
etichette di riga e colonna esplicite.

    signature_grid_slide.png    la matrice 4x4 delle firme  (slide 4)

Uso:
    python src/make_slide_figures.py --outdir slides/figures
"""

from make_concept_figures import (signature_shares, dot, arrow, save,
                                  C_FWD, C_BWD, C_LAT, C_MIX,
                                  C_IN, C_WAIST, C_OUT)

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ORDINE = ["FWD", "BWD", "LAT", "mix"]
COLORE = {"FWD": C_FWD, "BWD": C_BWD, "LAT": C_LAT, "mix": C_MIX}


def griglia_firme(outdir):
    """La matrice delle sedici firme, dimensionata per una proiezione."""
    # le quote servono solo per la riga di riepilogo in fondo
    sh = signature_shares()
    bassa = sum(v for k, v in sh["1-2-3"].items() if "LAT" in k.split("->"))
    alta = sum(v for k, v in sh["3-4-5"].items() if "LAT" in k.split("->"))

    fig, axes = plt.subplots(4, 4, figsize=(9.6, 9.0))

    for r, sin in enumerate(ORDINE):
        for c, sout in enumerate(ORDINE):
            ax = axes[r][c]
            ax.set_axis_off()
            ax.set_xlim(-0.40, 2.40)
            ax.set_ylim(-1.30, 1.30)

            if "LAT" in (sin, sout):
                ax.add_patch(FancyBboxPatch(
                    (-0.34, -1.24), 2.68, 2.48,
                    boxstyle="round,pad=0.02",
                    facecolor="#f7f3e8", edgecolor="none", zorder=0))

            def dy(kind, k):
                """La pendenza del ramo dice la direzione."""
                if kind == "FWD":
                    return 0.74
                if kind == "BWD":
                    return -0.74
                if kind == "LAT":
                    return 0.0
                return 0.74 if k == 0 else -0.74      # mix: uno per verso

            for k in range(2):
                y = dy(sin, k)
                if sin != "mix":
                    y += 0.21 if k == 0 else -0.21
                col = COLORE[sin] if sin != "mix" else (
                    C_FWD if k == 0 else C_BWD)
                dot(ax, 0.0, y, C_IN, r=165)
                arrow(ax, (0.0, y), (1.15, 0.0), col, lw=2.6, shrink=10)

            dot(ax, 1.15, 0.0, C_WAIST, r=300)

            for k in range(2):
                y = -dy(sout, k)
                if sout != "mix":
                    y += 0.21 if k == 0 else -0.21
                col = COLORE[sout] if sout != "mix" else (
                    C_FWD if k == 0 else C_BWD)
                dot(ax, 2.3, y, C_OUT, r=165)
                arrow(ax, (1.15, 0.0), (2.3, y), col, lw=2.6, shrink=10)

            ax.set_title("%s->%s" % (sin, sout), fontsize=15,
                         fontweight="bold", family="monospace", pad=6)

    # etichette di riga e di colonna: la matrice va letta, non indovinata
    for c, sout in enumerate(ORDINE):
        axes[0][c].annotate(sout, xy=(0.5, 1.30), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=16,
                            fontweight="bold", color="#B61918")
    for r, sin in enumerate(ORDINE):
        axes[r][0].annotate(sin, xy=(-0.10, 0.5), xycoords="axes fraction",
                            ha="right", va="center", fontsize=16,
                            fontweight="bold", color="#B61918", rotation=90)

    fig.text(0.5, 0.975, "output branch", ha="center", fontsize=13,
             color="#666666")
    fig.text(0.018, 0.5, "input branch", va="center", rotation=90,
             fontsize=13, color="#666666")
    fig.text(0.5, 0.018,
             "shaded: the seven signatures with an entirely lateral branch "
             "—  %.1f%% of the shallow mass,  %.1f%% of the deep"
             % (bassa, alta),
             ha="center", fontsize=13.5, color="#44525e")

    fig.tight_layout(rect=[0.045, 0.05, 1, 0.955])
    save(fig, "signature_grid_slide.png", outdir)


def _ritaglia(sorgente, destinazione, alto=0.0, largo=1.0, outdir="."):
    """Toglie il titolo (o una colonna) a una figura della tesi.

    Sulla slide il titolo lo fa la slide, in italiano: lasciare anche
    quello inglese dentro la figura e' una ripetizione in due lingue.
    """
    from PIL import Image, ImageChops
    im = Image.open(sorgente).convert("RGB")
    w, h = im.size
    im = im.crop((0, int(h * alto), int(w * largo), h))
    bianco = Image.new("RGB", im.size, (255, 255, 255))
    b = ImageChops.difference(im, bianco).getbbox()
    if b:
        im = im.crop((max(0, b[0] - 10), max(0, b[1] - 10),
                      min(im.size[0], b[2] + 10),
                      min(im.size[1], b[3] + 10)))
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, destinazione)
    im.save(p)
    print("    saved: %s  (%dx%d)" % (p, im.size[0], im.size[1]))


def schema_bowtie(outdir, pdf):
    """Lo schema del bow-tie, ritagliato dal PDF della tesi.

    E' disegnato in TikZ dentro il sorgente, quindi non esiste come file:
    l'unico modo di averlo su una slide e' prenderlo dalla pagina stampata.
    """
    import subprocess
    import tempfile
    from PIL import Image, ImageChops

    # 600 dpi e non 200: sulla slide questo schema e' largo quasi cinque
    # pollici, e a 200 dpi ci arriverebbe a centotrenta punti per pollice,
    # cioe' visibilmente sgranato. Le coordinate del ritaglio scalano con la
    # risoluzione, percio' stanno in un fattore.
    DPI = 600
    k = DPI / 200.0

    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "p")
        subprocess.run(["pdftoppm", "-f", "30", "-l", "30", "-r", str(DPI),
                        "-png", pdf, base], check=True)
        reso = [f for f in os.listdir(tmp) if f.endswith(".png")][0]
        im = Image.open(os.path.join(tmp, reso)).convert("RGB")

    # si taglia sotto la riga della condizione di distinzione: e' notazione
    # insiemistica, e su una slide divulgativa non aiuta nessuno
    im = im.crop((int(380 * k), int(330 * k), int(1330 * k), int(678 * k)))
    b = ImageChops.difference(im, Image.new("RGB", im.size,
                                            (255, 255, 255))).getbbox()
    m = int(14 * k)
    im = im.crop((max(0, b[0] - m), max(0, b[1] - m),
                  min(im.size[0], b[2] + m), min(im.size[1], b[3] + m)))
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "bowtie_schema.png")
    im.save(p)
    print("    saved: %s  (%dx%d)" % (p, im.size[0], im.size[1]))


def metagrafo_completo(outdir):
    """Il metagrafo su tutti e cinque i livelli.

    Quello della tesi copre la finestra 1-2-3, perche' li' si contano i
    motif. Sulla slide pero' il testo dice che i livelli sono cinque: una
    figura che ne mostra tre contraddice la riga accanto. Si rigenera con
    la finestra completa e si tiene qui, non in thesis/figures, dove
    sovrascriverebbe la figura della tesi.
    """
    import subprocess
    import shutil
    import sys
    import tempfile

    p = os.path.join(outdir, "metagraph_full.png")
    if os.path.exists(p):
        print("    gia' presente: %s" % p)
        return p

    qui = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable,
                        os.path.join(qui, "make_metagraph_figure.py"),
                        "--window", "1-2-3-4-5", "--outdir", tmp], check=True)
        os.makedirs(outdir, exist_ok=True)
        shutil.copy(os.path.join(tmp, "metagraph.png"), p)
    print("    saved: %s" % p)
    return p


def ritagli(outdir, figdir):
    """Le versioni da slide delle figure della tesi."""
    _ritaglia(metagrafo_completo(outdir),
              "metagraph_slide.png", alto=0.10, largo=0.60, outdir=outdir)
    _ritaglia(os.path.join(figdir, "signature_shift.png"),
              "signature_shift_slide.png", alto=0.055, outdir=outdir)
    _ritaglia(os.path.join(figdir, "olfactory_pathway.png"),
              "olfactory_slide.png", alto=0.052, outdir=outdir)
    _ritaglia(os.path.join(figdir, "micro_macro.png"),
              "micro_macro_slide.png", alto=0.095, outdir=outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="slides/figures")
    ap.add_argument("--figdir", default="thesis/figures",
                    help="da dove prendere le figure della tesi da ritagliare")
    ap.add_argument("--pdf", default="thesis/thesis.pdf",
                    help="il PDF da cui ritagliare lo schema del bow-tie")
    a = ap.parse_args()
    griglia_firme(a.outdir)
    ritagli(a.outdir, a.figdir)
    schema_bowtie(a.outdir, a.pdf)
    print("Fatto.")
