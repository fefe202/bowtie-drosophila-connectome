#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Riduce il CSV riga-per-riga dei motif agli aggregati e alle selezioni che le
analisi a valle usano davvero.

La ricerca esaustiva produce fino a 513 milioni di righe per finestra, oltre
100 GB per le quattro. Nessuna analisi legge le righe singole: tutte
aggregano oppure prendono un top-N. Il CSV completo e' quindi un intermedio,
non un risultato.

Artefatti prodotti per ciascuna finestra:

    aggregati_firma.csv      pattern e massa per firma direzionale
    aggregati_waist.csv      pattern, massa e massimo per gruppo centrale
    aggregati_catene.csv     massa per terna (livello_in, waist, livello_out)
    top_per_firma.csv        primi N di ciascuna firma, righe complete
    top_per_waist.csv        primi M di ciascun waist, righe complete
    top_per_waist_firma.csv  primi M per coppia (waist, firma)
    top_globale.csv          primi N in assoluto
    sommario.json            totali e parametri del run

Le quattro finestre passano cosi' da oltre 100 GB a 68 MB. La stratificazione
per (waist, firma) serve ai confronti a parita' di waist fra tipi di motif:
senza, le classi rare spariscono dalla selezione.

Per rigenerare le righe grezze: hourglass_areas_fast.py --full_csv.

Uso:
    python src/motif_distill.py --input <csv o csv.gz> --window 3-4-5
"""

from thesis_paths import results

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from directional_signature import add_signature, normalize_directions
from hourglass_areas import timestamp, format_elapsed


def log(msg):
    print(f"  [{timestamp()}] {msg}", flush=True)


class Distiller:
    """
    Accumula aggregati e selezioni top-N senza mai tenere in memoria piu'
    di qualche decina di migliaia di righe.
    """

    def __init__(self, top_per_signature=200, top_per_waist=20,
                 top_global=1000, top_per_waist_sig=10):
        self.n_sig = top_per_signature
        self.n_waist = top_per_waist
        self.n_glob = top_global
        # Stratificazione per (waist, firma), necessaria ai confronti a
        # parita' di waist: con il solo top-per-waist le classi rare come
        # pure_BWD restano rappresentate da pochissimi pattern.
        self.n_ws = top_per_waist_sig
        self.top_ws = {}
        self.agg_sig = defaultdict(lambda: [0, 0.0])      # firma -> [n, massa]
        self.agg_waist = defaultdict(lambda: [0, 0.0, 0.0])
        self.agg_chain = defaultdict(float)
        self.top_sig = {}
        self.top_waist = {}
        self.top_glob = None
        self.n_rows = 0
        self.total_mass = 0.0
        self.k = None

    def add_frame(self, df):
        if self.k is None:
            self.k = (int(df["k_in"].iloc[0]), int(df["k_out"].iloc[0]))
        if "signature" not in df.columns:
            add_signature(df, *self.k)
        else:
            # i CSV prodotti prima della rinomina usano FF/FB/lat
            normalize_directions(df)
        self.n_rows += len(df)
        cnt = df["motif_count"].to_numpy(dtype=float)
        self.total_mass += float(cnt.sum())

        for s, sub in df.groupby("signature", sort=False):
            a = self.agg_sig[s]
            a[0] += len(sub)
            a[1] += float(sub["motif_count"].sum())
            t = sub.nlargest(self.n_sig, "motif_count")
            self.top_sig[s] = t if s not in self.top_sig else \
                pd.concat([self.top_sig[s], t]).nlargest(self.n_sig,
                                                         "motif_count")

        w = (df["bottleneck_area"].astype(str) + "L" +
             df["bottleneck_level"].astype(str))
        for k_, sub in df.groupby(w, sort=False):
            a = self.agg_waist[k_]
            a[0] += len(sub)
            a[1] += float(sub["motif_count"].sum())
            a[2] = max(a[2], float(sub["motif_count"].max()))
            t = sub.nlargest(self.n_waist, "motif_count")
            self.top_waist[k_] = t if k_ not in self.top_waist else \
                pd.concat([self.top_waist[k_], t]).nlargest(self.n_waist,
                                                            "motif_count")

        # stratificazione (waist, firma)
        for (kw, ks), sub in df.groupby([w, "signature"], sort=False):
            key = (kw, ks)
            t = sub.nlargest(self.n_ws, "motif_count")
            self.top_ws[key] = t if key not in self.top_ws else \
                pd.concat([self.top_ws[key], t]).nlargest(self.n_ws,
                                                          "motif_count")

        t = df.nlargest(self.n_glob, "motif_count")
        self.top_glob = t if self.top_glob is None else \
            pd.concat([self.top_glob, t]).nlargest(self.n_glob, "motif_count")

        # catene (livello_in, livello_waist, livello_out)
        k_in, k_out = self.k
        bl = df["bottleneck_level"].to_numpy(dtype=int)
        for i in range(k_in):
            vi = df[f"fan_in_{i}_level"].to_numpy(dtype=int)
            for j in range(k_out):
                vo = df[f"fan_out_{j}_level"].to_numpy(dtype=int)
                key = (vi * 10 + bl) * 10 + vo
                bc = np.bincount(key, weights=cnt)
                for kk in np.flatnonzero(bc):
                    self.agg_chain[(kk // 100, (kk // 10) % 10, kk % 10)] \
                        += bc[kk]

    def finalize(self, outdir, window, params=None):
        os.makedirs(outdir, exist_ok=True)
        pd.DataFrame([{"signature": s, "n_pattern": v[0], "massa": v[1]}
                      for s, v in self.agg_sig.items()]) \
            .sort_values("n_pattern", ascending=False) \
            .to_csv(os.path.join(outdir, "aggregati_firma.csv"), index=False)
        pd.DataFrame([{"waist": k_, "n_pattern": v[0], "massa": v[1],
                       "massa_max": v[2]} for k_, v in self.agg_waist.items()]) \
            .sort_values("massa", ascending=False) \
            .to_csv(os.path.join(outdir, "aggregati_waist.csv"), index=False)
        pd.DataFrame([{"livello_fan_in": a, "livello_waist": b,
                       "livello_fan_out": c, "massa": v}
                      for (a, b, c), v in self.agg_chain.items() if v > 0]) \
            .sort_values("massa", ascending=False) \
            .to_csv(os.path.join(outdir, "aggregati_catene.csv"), index=False)
        pd.concat(self.top_sig.values()).to_csv(
            os.path.join(outdir, "top_per_firma.csv"), index=False)
        pd.concat(self.top_waist.values()).to_csv(
            os.path.join(outdir, "top_per_waist.csv"), index=False)
        pd.concat(self.top_ws.values()).to_csv(
            os.path.join(outdir, "top_per_waist_firma.csv"), index=False)
        self.top_glob.to_csv(os.path.join(outdir, "top_globale.csv"),
                             index=False)
        s = {"finestra": window, "n_pattern": self.n_rows,
             "massa_totale": self.total_mass,
             "k_in": self.k[0], "k_out": self.k[1],
             "n_waist": len(self.agg_waist), "n_firme": len(self.agg_sig),
             "top_per_firma": self.n_sig, "top_per_waist": self.n_waist,
             "top_globale": self.n_glob,
             "top_per_waist_firma": self.n_ws,
             "n_celle_waist_firma": len(self.top_ws)}
        if params:
            s.update(params)
        with open(os.path.join(outdir, "sommario.json"), "w",
                  encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        return s


def distilled_dir(window):
    """Cartella degli artefatti distillati di una finestra."""
    return results(os.path.join("motif_distilled", window))


def load(window, what):
    """
    Carica un artefatto distillato.

    `what` fra: top_per_waist, top_per_firma, top_globale,
                aggregati_firma, aggregati_waist, aggregati_catene

    Le analisi a valle leggono questi artefatti invece di scandire il CSV
    riga-per-riga.
    """
    d = window if os.path.isdir(window) else distilled_dir(window)
    p = os.path.join(d, f"{what}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"artefatto distillato mancante: {p}\n"
            f"Rilanciare la ricerca (hourglass_areas_fast.py) oppure "
            f"distillare un CSV esistente con motif_distill.py")
    return normalize_directions(pd.read_csv(p))


def main():
    ap = argparse.ArgumentParser(
        description="Distilla un CSV di motif negli artefatti effettivamente "
                    "usati dalle analisi.")
    ap.add_argument("--input", required=True, help="CSV o CSV.GZ dei motif")
    ap.add_argument("--window", required=True)
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--top_per_signature", type=int, default=200)
    ap.add_argument("--top_per_waist", type=int, default=20)
    ap.add_argument("--top_global", type=int, default=1000)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    out = a.outdir or results(os.path.join("motif_distilled", a.window))
    t0 = time.time()

    print("=" * 74)
    print(f"  DISTILLAZIONE  finestra {a.window}")
    print("=" * 74)
    src_gb = os.path.getsize(a.input) / 1e9
    log(f"input: {os.path.basename(a.input)}  ({src_gb:.1f} GB)")

    d = Distiller(a.top_per_signature, a.top_per_waist, a.top_global)
    for ch in pd.read_csv(a.input, chunksize=a.chunksize):
        d.add_frame(ch)
        log(f"  {d.n_rows:,} righe")
    s = d.finalize(out, a.window)

    tot = sum(os.path.getsize(os.path.join(out, f))
              for f in os.listdir(out)) / 1e6
    print(f"\n  pattern: {s['n_pattern']:,}   massa: {s['massa_totale']:.4e}")
    print(f"  waist distinti: {s['n_waist']}   firme: {s['n_firme']}")
    print(f"\n  {src_gb:.1f} GB  ->  {tot:.1f} MB   "
          f"(riduzione {src_gb*1000/max(tot,0.001):,.0f}x)")
    print(f"  [OUTPUT] {out}/")
    print(f"  tempo: {format_elapsed(time.time()-t0)}")


if __name__ == "__main__":
    main()
