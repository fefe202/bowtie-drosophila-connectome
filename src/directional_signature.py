#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Firma direzionale dei bow-tie motif.

Ogni ramo del motif si classifica per accordo dei suoi archi:

    signature = fanin_sig -> fanout_sig,  sig in {FWD, BWD, LAT, mix}

dove mix indica che gli archi di quel ramo non concordano. Le firme possibili
sono 16.

Le sigle sono FWD (verso livelli piu' profondi), BWD (verso livelli piu'
superficiali) e LAT (fra neuroni dello stesso livello). Le versioni precedenti
usavano FF/FB/lat: FB collide con la sigla del fan-shaped body, che nel
dataset compare in 19 gruppi anatomici, e la lettura dei risultati ne
risentiva. I CSV gia' prodotti restano leggibili, perche' li converte
`normalize_directions`.

La classificazione grossolana in pure_FWD / pure_BWD / pure_LAT / mixed e'
quasi vuota: con quattro archi l'accordo completo e' improbabile e il 94,4%
dei pattern finisce in mixed. La firma fine scompone quel calderone in 13
classi che si comportano in modo molto diverso.

Dipende solo dai livelli gia' presenti nel CSV, quindi non serve rilanciare
la ricerca per ottenerla.
"""

from thesis_paths import results

import argparse
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd

FWD, BWD, LAT = "FWD", "BWD", "LAT"
DIRS = [FWD, BWD, LAT]

# Notazione precedente, ancora presente nei CSV gia' prodotti.
_LEGACY = {"FF": FWD, "FB": BWD, "lateral": LAT, "lat": LAT}
_LEGACY_RE = re.compile(r"(?<![A-Za-z])(FF|FB|lateral|lat)(?![A-Za-z])")

# Solo queste colonne contengono direzioni: "FB" in una colonna di aree e' il
# fan-shaped body e non va toccato.
SIGNATURE_COLUMNS = ("signature", "fanin_sig", "fanout_sig", "hourglass_type",
                     "direction", "tipo_A", "tipo_B", "firma")


def normalize_directions(obj):
    """
    Converte la notazione FF/FB/lat in FWD/BWD/LAT.

    Accetta Series, Index o DataFrame; sul DataFrame agisce sulle colonne di
    SIGNATURE_COLUMNS e rinomina n_FF/n_FB/n_lat.
    """
    import pandas as pd

    def conv(x):
        return _LEGACY_RE.sub(lambda m: _LEGACY[m.group(1)], str(x))

    if isinstance(obj, pd.DataFrame):
        for c in SIGNATURE_COLUMNS:
            if c in obj.columns:
                obj[c] = obj[c].map(conv)
        obj.rename(columns={"n_FF": "n_FWD", "n_FB": "n_BWD",
                            "n_lat": "n_LAT"}, inplace=True)
        return obj
    if isinstance(obj, pd.Index):
        return obj.map(conv)
    return obj.map(conv)


def log(msg):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def edge_dir(src_level, dst_level):
    """Direzione di un arco, vettorizzata."""
    return np.where(src_level < dst_level, FWD,
                    np.where(src_level > dst_level, BWD, LAT))


def add_signature(df, k_in, k_out):
    """
    Aggiunge le colonne di firma direzionale a un DataFrame di motif.
    Restituisce lo stesso DataFrame modificato in place.
    """
    bl = df["bottleneck_level"].to_numpy()

    # arco di fan-in: A_m -> B   |   arco di fan-out: B -> D_n
    in_dirs = [edge_dir(df[f"fan_in_{i}_level"].to_numpy(), bl)
               for i in range(k_in)]
    out_dirs = [edge_dir(bl, df[f"fan_out_{i}_level"].to_numpy())
                for i in range(k_out)]
    all_dirs = in_dirs + out_dirs

    for d in DIRS:
        df[f"n_{d}"] = sum((arr == d).astype(np.int8) for arr in all_dirs)

    def collapse(arrs):
        """Direzione comune se tutti gli archi concordano, altrimenti 'mix'."""
        same = np.ones(len(arrs[0]), dtype=bool)
        for a in arrs[1:]:
            same &= (a == arrs[0])
        return np.where(same, arrs[0], "mix")

    df["fanin_sig"] = collapse(in_dirs)
    df["fanout_sig"] = collapse(out_dirs)
    df["signature"] = pd.Series(df["fanin_sig"]).str.cat(
        pd.Series(df["fanout_sig"]), sep="->").to_numpy()
    return df


def interpret(sig):
    """Etichetta interpretativa della firma."""
    return {
        "FF->FF": "clessidra puramente feed-forward",
        "FB->FB": "clessidra puramente di feedback",
        "lat->lat": "integrazione laterale, stesso livello",
        "FF->FB": "RI-ENTRANTE: converge in avanti, rimanda indietro",
        "FB->FF": "relay: raccoglie dall'alto, ridistribuisce in avanti",
        "FF->lat": "converge in avanti, distribuisce lateralmente",
        "lat->FF": "raccoglie lateralmente, proietta in avanti",
        "FB->lat": "raccoglie dall'alto, distribuisce lateralmente",
        "lat->FB": "raccoglie lateralmente, rimanda indietro",
    }.get(sig, "")


def main():
    ap = argparse.ArgumentParser(
        description="Firma direzionale fine dei bow-tie motif "
                    "(scompone l'etichetta 'mixed').")
    ap.add_argument("--input", required=True, help="CSV dei motif")
    ap.add_argument("--outdir", default=results("signature_results"))
    ap.add_argument("--top_n", type=int, default=50,
                    help="Quanti pattern di testa salvare per ciascuna firma")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--write_full", action="store_true",
                    help="Riscrive l'intero CSV con le colonne di firma "
                         "(file di parecchi GB; di norma non serve)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    print("=" * 78)
    print("  FIRMA DIREZIONALE FINE DEI BOW-TIE MOTIF")
    print("=" * 78)
    log(f"Input: {args.input}")

    agg_n = {}         # firma -> numero di pattern
    agg_c = {}         # firma -> somma dei motif_count
    cross = {}         # (firma, hourglass_type) -> numero di pattern
    waist = {}         # (firma, waist) -> somma dei motif_count
    tops = {}          # firma -> DataFrame dei top-N
    full_out = None
    rows = 0
    k_in = k_out = None

    for chunk in pd.read_csv(args.input, chunksize=args.chunksize):
        if k_in is None:
            k_in, k_out = int(chunk["k_in"].iloc[0]), int(chunk["k_out"].iloc[0])
            log(f"k_in = {k_in}, k_out = {k_out}  "
                f"({k_in + k_out} archi per motif)")
        rows += len(chunk)
        add_signature(chunk, k_in, k_out)

        g = chunk.groupby("signature")["motif_count"]
        for s, v in g.size().items():
            agg_n[s] = agg_n.get(s, 0) + int(v)
        for s, v in g.sum().items():
            agg_c[s] = agg_c.get(s, 0.0) + float(v)
        for key, v in chunk.groupby(["signature", "hourglass_type"]).size().items():
            cross[key] = cross.get(key, 0) + int(v)

        w = (chunk["bottleneck_area"].astype(str) + "L" +
             chunk["bottleneck_level"].astype(str))
        for key, v in chunk.groupby(["signature", w])["motif_count"].sum().items():
            waist[key] = waist.get(key, 0.0) + float(v)

        for s, sub in chunk.groupby("signature"):
            t = sub.nlargest(args.top_n, "motif_count")
            tops[s] = t if s not in tops else \
                pd.concat([tops[s], t]).nlargest(args.top_n, "motif_count")

        if args.write_full:
            mode, hdr = ("w", True) if full_out is None else ("a", False)
            full_out = os.path.join(args.outdir, "motifs_with_signature.csv")
            chunk.to_csv(full_out, mode=mode, header=hdr, index=False)

        if rows % 5_000_000 == 0:
            log(f"  {rows:,} righe")

    log(f"{rows:,} righe elaborate")

    # ---- tabella riassuntiva
    tot_n = sum(agg_n.values())
    tot_c = sum(agg_c.values())
    summary = pd.DataFrame([
        {"signature": s, "n_pattern": agg_n[s],
         "pct_pattern": 100 * agg_n[s] / tot_n,
         "motif_count": agg_c[s],
         "pct_count": 100 * agg_c[s] / tot_c,
         "interpretazione": interpret(s)}
        for s in sorted(agg_n, key=lambda x: -agg_n[x])
    ])
    summary.to_csv(os.path.join(args.outdir, "signature_summary.csv"), index=False)

    print("\n  [FIRME DIREZIONALI]")
    with pd.option_context("display.width", 200, "display.max_colwidth", 46):
        print(summary.round(3).to_string(index=False))

    # ---- come si scompone 'mixed'
    cx = pd.Series(cross).unstack(fill_value=0)
    cx.index.name = "signature"
    cx.to_csv(os.path.join(args.outdir, "signature_by_type.csv"))
    print("\n  [SCOMPOSIZIONE DELL'ETICHETTA hourglass_type]")
    print(cx.to_string())
    if "mixed" in cx.columns:
        m = cx["mixed"]
        m = m[m > 0].sort_values(ascending=False)
        print("\n  Il calderone 'mixed' si scompone in "
              f"{len(m)} firme distinte:")
        for s, v in m.items():
            print(f"    {s:<12} {v:>12,}  ({100*v/m.sum():5.2f}%)  {interpret(s)}")

    # ---- focus sulle classi ricorrenti
    print("\n" + "=" * 78)
    print("  CLASSI DI INTERESSE PER LE RICORRENZE")
    print("=" * 78)
    ws = pd.Series(waist)
    for s in ("FF->FB", "FB->FF", "FB->FB"):
        if s not in agg_n:
            print(f"\n  {s}: nessun pattern")
            continue
        print(f"\n  [{s}]  {interpret(s)}")
        print(f"    pattern: {agg_n[s]:,} ({100*agg_n[s]/tot_n:.3f}% del totale)"
              f"   massa di conteggio: {agg_c[s]:.4e} "
              f"({100*agg_c[s]/tot_c:.3f}%)")
        sub = ws[s].sort_values(ascending=False).head(8)
        print("    waist dominanti (per massa di conteggio):")
        for w_, v in sub.items():
            print(f"      {w_:<22} {v:.4e}  ({100*v/ws[s].sum():5.2f}%)")
        print("    pattern di testa:")
        for _, r in tops[s].head(5).iterrows():
            print(f"      {r['motif_count']:>18,}  {r['structure_str']}")

    # ---- salvataggio dei top per firma
    d = os.path.join(args.outdir, "top_by_signature")
    os.makedirs(d, exist_ok=True)
    for s, t in tops.items():
        t.sort_values("motif_count", ascending=False).to_csv(
            os.path.join(d, f"{s.replace('->', '_to_')}_top{args.top_n}.csv"),
            index=False)
    print(f"\n  [OUTPUT] {args.outdir}/signature_summary.csv")
    print(f"  [OUTPUT] {args.outdir}/signature_by_type.csv")
    print(f"  [OUTPUT] {d}/  ({len(tops)} file, top-{args.top_n} per firma)")
    if args.write_full:
        print(f"  [OUTPUT] {full_out}")
    print(f"\n  Tempo totale: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
