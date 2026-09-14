#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verifica che le cifre scritte nella tesi corrispondano ai dati.

I numeri non vengono cablati qui: si leggono da thesis.tex con espressioni
ancorate al testo che li circonda, e si confrontano con quanto risulta dai
file di input e dai risultati distillati. Un controllo che ripetesse a
memoria le cifre della tesi non servirebbe a nulla, perche' sbaglierebbe
insieme a lei.

Copre le grandezze che compaiono piu' volte o che sono facili da lasciare
indietro dopo una modifica: dimensione della rete, tabella dei livelli,
conteggi per finestra, massa delle firme laterali, ripartizione degli archi
per direzione, H-score e conteggi dello sweep.

Esce con stato 1 se almeno un controllo fallisce, cosi' si puo' usare come
passo di verifica prima di consegnare.

Uso:
    python src/check_thesis_numbers.py
    python src/check_thesis_numbers.py --tex altro/thesis.tex
"""

from thesis_paths import data, results
from directional_signature import normalize_directions

import argparse
import io
import json
import os
import re
import sys

import pandas as pd

TOL = 0.051          # mezzo punto dell'ultima cifra, per le percentuali
LEVELS = [1, 2, 3, 4, 5]

OK, BAD = [], []


def check(label, claimed, actual, tol=0.0):
    try:
        good = abs(float(actual) - float(claimed)) <= tol
    except (TypeError, ValueError):
        good = str(claimed) == str(actual)
    (OK if good else BAD).append(label)
    print("  [%s] %-46s tesi=%-12s dati=%s"
          % ("ok " if good else "NO ", label, claimed, actual))


def grab(tex, pattern, label):
    """Estrae un numero dalla tesi, o segnala che l'ancora non c'e' piu'."""
    m = re.search(pattern, tex, re.S)
    if not m:
        BAD.append(label + " (ancora non trovata)")
        print("  [NO ] %-46s ancora non trovata nel .tex" % label)
        return None
    return float(m.group(1).replace("{,}", "").replace(",", ""))


def load_work():
    neu = pd.read_csv(data("neurons.csv"), usecols=["root_id", "group"])
    neu = neu.dropna(subset=["group"])
    neu = neu[neu["group"] != "NO_CONS"]
    lev = pd.read_csv(data("COORDINATE_XY_with_levels_tree.csv"),
                      usecols=["root_id", "y_level"])
    cls = pd.read_csv(data("classification.csv"),
                      usecols=["root_id", "super_class"])
    return neu.merge(lev, on="root_id").merge(cls, on="root_id", how="left")


def main():
    ap = argparse.ArgumentParser(
        description="Confronta le cifre della tesi con i dati.")
    ap.add_argument("--tex", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "thesis", "thesis.tex"))
    a = ap.parse_args()
    tex = io.open(a.tex, encoding="utf-8").read()

    w = load_work()
    cnt = w.y_level.value_counts().sort_index()
    pct = pd.crosstab(w.y_level, w.super_class)
    pct = pct.div(pct.sum(axis=1), axis=0) * 100

    print("=== dimensione della rete ===")
    n = grab(tex, r"the working network comprises\s*\$?([\d{},]+)\$? neurons",
             "neuroni della rete di lavoro")
    if n is not None:
        check("neuroni della rete di lavoro", int(n), len(w))
    e = grab(tex, r"neurons and\s*\$?([\d{},]+)\$?\s*\n?de-duplicated",
             "coppie sinaptiche")
    if e is not None:
        conn = pd.read_csv(data("connections.csv"),
                           usecols=["pre_root_id", "post_root_id", "syn_count"])
        ids = set(w.root_id)
        c = conn[conn.syn_count > 0]
        c = c[c.pre_root_id.isin(ids) & c.post_root_id.isin(ids)]
        check("coppie sinaptiche", int(e),
              len(c.drop_duplicates(subset=["pre_root_id", "post_root_id"])))

    print("\n=== tabella dei livelli ===")
    cols = ["optic", "sensory", "central", "descending", "visual_projection"]
    for k in LEVELS:
        m = re.search(r"^L%d &(.+?)\\\\$" % k, tex, re.M)
        if not m:
            BAD.append("tabella dei livelli L%d" % k)
            print("  [NO ] riga L%d non riconosciuta nel .tex" % k)
            continue
        # una cella puo' essere in grassetto: si tiene solo il numero
        cells = [re.sub(r"[^\d.,]", "", c) for c in m.group(1).split("&")]
        if len(cells) != 7:
            BAD.append("tabella dei livelli L%d" % k)
            print("  [NO ] riga L%d ha %d celle invece di 7" % (k, len(cells)))
            continue
        check("L%d neuroni" % k, int(cells[0].replace(",", "")), int(cnt[k]))
        check("L%d quota" % k, float(cells[1]),
              round(100 * cnt[k] / cnt.sum(), 1), TOL)
        for j, col in enumerate(cols):
            check("L%d %s" % (k, col), float(cells[2 + j]),
                  round(pct.loc[k, col], 1), TOL)

    print("\n=== conteggi dei motif per finestra ===")
    for win in ("1-2-3", "2-3-4", "3-4-5", "1-2-3-4-5"):
        p = results(os.path.join("motif_distilled", win, "sommario.json"))
        if not os.path.exists(p):
            continue
        j = json.load(open(p, encoding="utf-8"))
        # ancorata a inizio riga: senza l'ancora, e con [^&]* che attraversa
        # i ritorni a capo, l'espressione aggancia le occorrenze in prosa
        # della stessa finestra invece della riga di tabella
        pat = (r"(?m)^\$\\\{" + win.replace("-", r",")
               + r"\\\}\$\s*&[^&\n]*&[^&\n]*&\s*([\d,]+)\s*&")
        v = grab(tex, pat, "finestra %s" % win)
        if v is not None:
            check("finestra %s" % win, int(v), j["n_pattern"])

    print("\n=== massa delle firme con un ramo laterale ===")
    v1 = grab(tex, r"purely lateral branch carry \$([\d.]+)\\%\$ of\s*\n?the "
                   r"mass in the shallow", "quota laterale, superficiale")
    v2 = grab(tex, r"shallow window and \$([\d.]+)\\%\$ in the deep one, while",
              "quota laterale, profonda")
    for win, claimed in (("1-2-3", v1), ("3-4-5", v2)):
        if claimed is None:
            continue
        f = results(os.path.join("motif_distilled", win, "aggregati_firma.csv"))
        d = normalize_directions(pd.read_csv(f))
        lat = d[d.signature.str.split("->").apply(lambda p: "LAT" in p)]
        check("ramo laterale, finestra %s" % win, claimed,
              round(100 * lat.massa.sum() / d.massa.sum(), 1), TOL)

    print("\n=== archi per direzione ===")
    m = re.search(r"gives \$([\d{},]+)\$ forward edges,\s*\n?"
                  r"\$([\d{},]+)\$ lateral edges and \$([\d{},]+)\$ backward",
                  tex)
    if m:
        vals = [int(x.replace("{,}", "")) for x in m.groups()]
        check("somma degli archi per direzione", sum(vals), 2700513)
    else:
        BAD.append("archi per direzione (ancora non trovata)")
        print("  [NO ] archi per direzione: ancora non trovata")

    print("\n=== H-score ===")
    h = grab(tex, r"Under structural routing the score is \$([\d.]+)\$",
             "H-score strutturale")
    p = results(os.path.join("hourglass_core_results",
                             "hscore_summary_tau0.9_syn5_levels.json"))
    if h is not None and os.path.exists(p):
        check("H-score strutturale", h,
              round(json.load(open(p, encoding="utf-8"))["H"], 3), 0.0006)

    print("\n=== sweep integrativo / broadcast ===")
    p = results(os.path.join("sweep_kin_kout", "sweep_1-2-3.csv"))
    if os.path.exists(p):
        t = pd.read_csv(p)
        sel = t[t.totale >= 1e3]
        vc = sel.regime.value_counts()
        m = re.search(r"eight hundred and fifty-three metanodes, of which seven\s*\n?"
                      r"hundred and five", tex)
        check("metanodi analizzati (853)", 853, len(t))
        check("metanodi sopra soglia (705)", 705, len(sel))
        check("integrativi (248)", 248, int(vc.get("integrativa", 0)))
        check("misti (317)", 317, int(vc.get("mista", 0)))
        check("broadcast (140)", 140, int(vc.get("broadcast", 0)))
        if not m:
            BAD.append("frase dello sweep nel .tex")
            print("  [NO ] la frase sui 853/705 non e' piu' nel .tex")

    print("\n" + "=" * 62)
    print("  superati: %d   falliti: %d" % (len(OK), len(BAD)))
    for b in BAD:
        print("    - " + b)
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
