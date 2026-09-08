#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tassonomia dei neuropili e granularita' intermedia.

Il campo `group` di neurons.csv indica in quali neuropili il neurone
arborizza. Non e' una lista arbitraria di 628 etichette: e' un sistema
combinatorio costruito su 44 neuropili di base, ciascuno con una sigla
standard, e ogni gruppo e' o un neuropilo singolo (44 gruppi) o una coppia
scritta col punto (584 gruppi). ME e' un neurone confinato nella medulla,
ME.LO un neurone che arborizza sia nella medulla sia nella lobula.

I 44 neuropili si raggruppano a loro volta nelle super-regioni della
nomenclatura di Ito et al. (2014), che sono tredici. Questo da' una terza
granularita', fra le 9 superclassi funzionali e i 628 gruppi anatomici:
79 categorie e 309 metanodi, contro i 24 delle superclassi e i 1.698 dei
gruppi.

Uso come libreria:

    from neuropil_taxonomy import superregion_of_group, NEUROPILS

Uso da riga di comando, per la tabella dei neuropili della tesi:

    python src/neuropil_taxonomy.py --table
"""

from thesis_paths import data

import argparse
import os

import pandas as pd

# codice -> (nome esteso, super-regione)
NEUROPILS = {
    "ME":      ("Medulla", "OL"),
    "AME":     ("Accessory Medulla", "OL"),
    "LO":      ("Lobula", "OL"),
    "LOP":     ("Lobula Plate", "OL"),
    "LA":      ("Lamina", "OL"),
    "OCG":     ("Ocellar Ganglion", "OCG"),
    "FB":      ("Fan-shaped Body", "CX"),
    "EB":      ("Ellipsoid Body", "CX"),
    "PB":      ("Protocerebral Bridge", "CX"),
    "NO":      ("Noduli", "CX"),
    "MB_CA":   ("Mushroom Body Calyx", "MB"),
    "MB_ML":   ("Mushroom Body Medial Lobe", "MB"),
    "MB_VL":   ("Mushroom Body Vertical Lobe", "MB"),
    "MB_PED":  ("Mushroom Body Pedunculus", "MB"),
    "BU":      ("Bulb", "LX"),
    "LAL":     ("Lateral Accessory Lobe", "LX"),
    "GA":      ("Gall", "LX"),
    "SLP":     ("Superior Lateral Protocerebrum", "SNP"),
    "SIP":     ("Superior Intermediate Protocerebrum", "SNP"),
    "SMP":     ("Superior Medial Protocerebrum", "SNP"),
    "CRE":     ("Crepine", "INP"),
    "SCL":     ("Superior Clamp", "INP"),
    "ICL":     ("Inferior Clamp", "INP"),
    "IB":      ("Inferior Bridge", "INP"),
    "ATL":     ("Antler", "INP"),
    "AL":      ("Antennal Lobe", "AL"),
    "LH":      ("Lateral Horn", "LH"),
    "AOTU":    ("Anterior Optic Tubercle", "VLNP"),
    "AVLP":    ("Anterior Ventrolateral Protocerebrum", "VLNP"),
    "PVLP":    ("Posterior Ventrolateral Protocerebrum", "VLNP"),
    "PLP":     ("Posterior Lateral Protocerebrum", "VLNP"),
    "WED":     ("Wedge", "VLNP"),
    "VES":     ("Vest", "VMNP"),
    "EPA":     ("Epaulette", "VMNP"),
    "GOR":     ("Gorget", "VMNP"),
    "SPS":     ("Superior Posterior Slope", "VMNP"),
    "IPS":     ("Inferior Posterior Slope", "VMNP"),
    "SAD":     ("Saddle", "PENP"),
    "FLA":     ("Flange", "PENP"),
    "CAN":     ("Cantle", "PENP"),
    "PRW":     ("Prow", "PENP"),
    "AMMC":    ("Antennal Mechanosensory and Motor Centre", "PENP"),
    "GNG":     ("Gnathal Ganglia", "GNG"),
    "UNASGD":  ("Unassigned", "UNK"),
}

SUPERREGIONS = {
    "OL":   "Optic Lobe",
    "OCG":  "Ocellar Ganglion",
    "CX":   "Central Complex",
    "MB":   "Mushroom Body",
    "LX":   "Lateral Complex",
    "SNP":  "Superior Neuropils",
    "INP":  "Inferior Neuropils",
    "AL":   "Antennal Lobe",
    "LH":   "Lateral Horn",
    "VLNP": "Ventrolateral Neuropils",
    "VMNP": "Ventromedial Neuropils",
    "PENP": "Periesophageal Neuropils",
    "GNG":  "Gnathal Ganglia",
    "UNK":  "Unassigned",
}

# ordine di presentazione: dalla periferia sensoriale al centro e all'uscita
SUPERREGION_ORDER = ["OL", "OCG", "AL", "LH", "MB", "CX", "LX", "SNP", "INP",
                     "VLNP", "VMNP", "PENP", "GNG", "UNK"]


def tokens(group):
    """I neuropili che compongono un gruppo anatomico."""
    return [t for t in str(group).split(".") if t]


def superregion_of_group(group):
    """
    Super-regione di un gruppo anatomico.

    Se i neuropili del gruppo stanno tutti nella stessa super-regione il
    risultato e' quella; altrimenti e' la coppia, scritta col punto in ordine
    alfabetico, cosi' che ME.LO resti dentro OL mentre LO.PLP diventi
    OL.VLNP.
    """
    srs = sorted({NEUROPILS.get(t, ("", "UNK"))[1] for t in tokens(group)})
    if not srs:
        return "UNK"
    return srs[0] if len(srs) == 1 else ".".join(srs)


def superregion_label(code):
    """Nome esteso di una super-regione, anche composta."""
    return " + ".join(SUPERREGIONS.get(c, c) for c in code.split("."))


def neuropil_counts():
    """Neuroni per neuropilo, ripartendo i gruppi composti fra i due."""
    df = pd.read_csv(data("neurons.csv"), usecols=["root_id", "group"])
    df = df.dropna(subset=["group"])
    df = df[df["group"] != "NO_CONS"]
    out = {}
    for g, c in df["group"].value_counts().items():
        parts = tokens(g)
        for t in parts:
            out[t] = out.get(t, 0.0) + c / len(parts)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Tabella dei neuropili e delle super-regioni.")
    ap.add_argument("--table", action="store_true",
                    help="stampa la tabella dei 44 neuropili")
    ap.add_argument("--latex", default=None,
                    help="scrive la tabella in un file .tex")
    a = ap.parse_args()

    n = neuropil_counts()
    rows = []
    for code in sorted(NEUROPILS, key=lambda c: (
            SUPERREGION_ORDER.index(NEUROPILS[c][1]), -n.get(c, 0))):
        name, sr = NEUROPILS[code]
        rows.append({"code": code, "name": name, "superregion": sr,
                     "superregion_name": SUPERREGIONS[sr],
                     "neurons": round(n.get(code, 0.0))})
    t = pd.DataFrame(rows)

    if a.table or not a.latex:
        print(t.to_string(index=False))
        print(f"\nneuropili: {len(t)}   super-regioni: "
              f"{t['superregion'].nunique()}")

    if a.latex:
        lines = []
        cur = None
        for _, r in t.iterrows():
            if r["superregion"] != cur:
                cur = r["superregion"]
                lines.append("\\midrule")
                lines.append("\\multicolumn{3}{l}{\\textit{%s}} \\\\"
                             % r["superregion_name"])
            lines.append("\\code{%s} & %s & %s \\\\"
                         % (r["code"].replace("_", "\\_"), r["name"],
                            f"{r['neurons']:,}".replace(",", "{,}")))
        os.makedirs(os.path.dirname(os.path.abspath(a.latex)), exist_ok=True)
        with open(a.latex, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  [OUTPUT] {a.latex}")


if __name__ == "__main__":
    main()
