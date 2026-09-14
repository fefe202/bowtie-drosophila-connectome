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

Provenienza delle due parti, che sono diverse e vanno tenute distinte.

  - I 44 neuropili e le loro sigle **vengono dal dataset**. La colonna
    `neuropil` di connections.csv contiene esattamente gli stessi 44 nomi
    nella forma con suffisso emisferico (ME_L, ME_R, ...), e i token della
    colonna `group` di neurons.csv sono esattamente quei 44. Non e' una lista
    inventata, ed e' `--selftest` a ricontrollarlo sui dati.

  - Il raggruppamento in 13 super-regioni **non e' un campo del dataset**:
    nessun file lo contiene. E' la gerarchia della nomenclatura di Ito et al.
    (2014), codificata qui a mano. E' la parte da verificare contro la tabella
    dei neuropili di FlyWire Codex prima di difenderla in sede di discussione.

Il raggruppamento da' una terza granularita', fra le 9 superclassi funzionali
e i 628 gruppi anatomici: 79 categorie e 309 metanodi, contro i 24 delle
superclassi e i 1.698 dei gruppi.

Uso come libreria:

    from neuropil_taxonomy import superregion_of_group, NEUROPILS

Uso da riga di comando:

    python src/neuropil_taxonomy.py --table       # tabella dei 44 neuropili
    python src/neuropil_taxonomy.py --selftest    # verifica contro i dati
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


def selftest():
    """
    Verifica che l'elenco dei neuropili sia quello del dataset.

    Controlla due cose contro i file di input: che i 44 codici di NEUROPILS
    siano esattamente quelli della colonna `neuropil` di connections.csv una
    volta tolto il suffisso emisferico, e che ogni token della colonna `group`
    di neurons.csv sia fra quelli noti. Il raggruppamento in super-regioni non
    e' verificabile qui, perche' non compare in nessun file: quello resta una
    codifica della nomenclatura di Ito et al. (2014).
    """
    import re

    ok = True
    npl = pd.read_csv(data("connections.csv"), usecols=["neuropil"])
    npl = npl["neuropil"].dropna().unique()
    base = {re.sub(r"_(L|R)$", "", x) for x in npl}
    mine = set(NEUROPILS)
    print(f"  neuropili in connections.csv: {len(base)}   "
          f"in NEUROPILS: {len(mine)}")
    for label, diff in (("solo nei dati", base - mine),
                        ("solo nel modulo", mine - base)):
        if diff:
            ok = False
            print(f"  [FALLITO] {label}: {sorted(diff)}")

    g = pd.read_csv(data("neurons.csv"), usecols=["group"])["group"].dropna()
    g = g[g != "NO_CONS"]
    tok = {t for x in g.unique() for t in tokens(x)}
    print(f"  token usati dalla colonna group: {len(tok)}")
    if not tok <= mine:
        ok = False
        print(f"  [FALLITO] token sconosciuti: {sorted(tok - mine)}")

    n_sr = len({sr for _, sr in NEUROPILS.values()} - {"UNK"})
    print(f"  super-regioni codificate: {n_sr}   "
          f"(non verificabili sui dati: nessun file le contiene)")
    print("\n  " + ("OK: l'elenco dei neuropili coincide con il dataset"
                    if ok else "ALCUNI CONTROLLI SONO FALLITI"))
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Tabella dei neuropili e delle super-regioni.")
    ap.add_argument("--table", action="store_true",
                    help="stampa la tabella dei 44 neuropili")
    ap.add_argument("--selftest", action="store_true",
                    help="verifica l'elenco dei neuropili contro i dati")
    ap.add_argument("--latex", default=None,
                    help="scrive la tabella in un file .tex")
    a = ap.parse_args()

    if a.selftest:
        import sys
        sys.exit(0 if selftest() else 1)

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
        # UNASGD non e' una regione: e' l'etichetta dei neuroni non assegnati
        real = t[t["superregion"] != "UNK"]["superregion"].nunique()
        print(f"\nneuropili: {len(t)}   super-regioni: {real}"
              f"   (piu' UNK, per i neuroni non assegnati)")

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
