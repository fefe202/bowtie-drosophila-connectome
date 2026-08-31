#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compressione realizzata del waist.

La metrica nominale, rapporto fra periferia e dimensione del gruppo centrale,
non funziona: un gruppo puo' essere grande mentre solo pochi dei suoi neuroni
partecipano. Qui si usa il participation ratio come numero efficace di
neuroni che portano il flusso,

    n_eff = (sum_i P_i)^2 / sum_i P_i^2

    compressione realizzata = periferia connessa a un waist attivo / n_eff

n_eff vale la dimensione del gruppo quando tutti contribuiscono allo stesso
modo e tende a 1 quando un solo neurone porta tutto. I casi con n_eff vicino
a 1 sono hub, non waist, e si escludono con --min_waist_eff.

Legge di default il distillato della finestra.

Uso:
    python src/compression_realized.py --window 3-4-5
"""

from thesis_paths import data, results

import argparse
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse

from directional_signature import add_signature, interpret
from motif_distill import load as load_distilled


def log(msg):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def select_per_waist(path, top_m, chunksize=1_000_000):
    """Top-M pattern per ogni waist, cosi' ogni gruppo centrale e' rappresentato."""
    log(f"Selezione dei top {top_m} pattern per waist")
    best, rows, k = {}, 0, None
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if k is None:
            k = (int(chunk["k_in"].iloc[0]), int(chunk["k_out"].iloc[0]))
        rows += len(chunk)
        key = (chunk["bottleneck_area"].astype(str) + "|" +
               chunk["bottleneck_level"].astype(str))
        for kk, s in chunk.groupby(key):
            t = s.nlargest(top_m, "motif_count")
            best[kk] = t if kk not in best else \
                pd.concat([best[kk], t]).nlargest(top_m, "motif_count")
    df = pd.concat(best.values()).reset_index(drop=True)
    log(f"  {rows:,} righe scansionate, {len(best):,} waist, "
        f"{len(df):,} pattern selezionati")
    return df, k


def main():
    ap = argparse.ArgumentParser(
        description="Compressione realizzata del waist (versione corretta).")
    ap.add_argument("--window", default="1-2-3",
                    help="Finestra da cui leggere il distillato")
    ap.add_argument("--motifs", default=None,
                    help="CSV riga-per-riga (legacy). Di norma non serve: "
                         "si usa il distillato della finestra.")
    ap.add_argument("--per_waist_top", type=int, default=20)
    ap.add_argument("--min_waist_eff", type=float, default=5.0,
                    help="Numero efficace minimo di neuroni nel waist perche' "
                         "il pattern entri in classifica (default 5)")
    ap.add_argument("--outdir", default=results("compression_results"))
    ap.add_argument("--conn_file", default=data("connections.csv"))
    ap.add_argument("--levels_file", default=data("COORDINATE_XY_with_levels_tree.csv"))
    ap.add_argument("--neurons_file", default=data("neurons.csv"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    print("=" * 78)
    print("  COMPRESSIONE REALIZZATA DEL WAIST")
    print("=" * 78)

    if args.motifs:
        df_hg, (k_in, k_out) = select_per_waist(args.motifs, args.per_waist_top)
    else:
        # il distillato contiene gia' i top-M per waist
        df_hg = load_distilled(args.window, "top_per_waist")
        k_in, k_out = int(df_hg["k_in"].iloc[0]), int(df_hg["k_out"].iloc[0])
        log(f"distillato {args.window}: {len(df_hg):,} pattern, "
            f"{df_hg['bottleneck_area'].nunique()} waist")
    add_signature(df_hg, k_in, k_out)

    pats = []
    for r in df_hg.to_dict("records"):
        pats.append({
            "fan_in": tuple((r[f"fan_in_{i}_area"], int(r[f"fan_in_{i}_level"]))
                            for i in range(k_in)),
            "bn": (r["bottleneck_area"], int(r["bottleneck_level"])),
            "fan_out": tuple((r[f"fan_out_{i}_area"], int(r[f"fan_out_{i}_level"]))
                             for i in range(k_out)),
            "row": r,
        })

    levels = {k[1] for p in pats for k in p["fan_in"] + (p["bn"],) + p["fan_out"]}
    log(f"Livelli coinvolti: {sorted(levels)}. Costruzione del grafo...")

    dn = pd.read_csv(args.neurons_file, usecols=["root_id", "group"]).dropna()
    dn = dn[dn["group"] != "NO_CONS"]
    area = dict(zip(dn["root_id"], dn["group"]))
    dl = pd.read_csv(args.levels_file, usecols=["root_id", "y_level"])
    lev = dict(zip(dl["root_id"], dl["y_level"].astype(int)))
    valid = sorted(n for n in (set(area) & set(lev)) if lev[n] in levels)
    idx = {n: i for i, n in enumerate(valid)}
    groups = defaultdict(list)
    for n in valid:
        groups[(area[n], lev[n])].append(idx[n])

    dc = pd.read_csv(args.conn_file,
                     usecols=["pre_root_id", "post_root_id", "syn_count"])
    e = dc[dc["syn_count"] > 0][["pre_root_id", "post_root_id"]].drop_duplicates()
    e = e[e["pre_root_id"].isin(idx) & e["post_root_id"].isin(idx)]
    A = sparse.csr_matrix((np.ones(len(e)),
                           (e["pre_root_id"].map(idx).astype(int),
                            e["post_root_id"].map(idx).astype(int))),
                          shape=(len(valid), len(valid)))
    log(f"  {len(valid):,} neuroni, {A.nnz:,} archi")

    pairs = set()
    for p in pats:
        for a in p["fan_in"]:
            pairs.add((a, p["bn"]))
        for d in p["fan_out"]:
            pairs.add((p["bn"], d))
    log(f"Estrazione di {len(pairs):,} sottomatrici...")
    sub = {}
    for s, d in pairs:
        if s in groups and d in groups:
            m = A[groups[s], :][:, groups[d]]
            if m.nnz:
                sub[(s, d)] = m.tocsr()

    log("Calcolo delle metriche realizzate...")
    out = []
    for i_p, p in enumerate(pats):
        bn = p["bn"]
        if bn not in groups:
            continue
        nb = len(groups[bn])
        P = np.ones(nb)
        ok = True
        for a in p["fan_in"]:
            if (a, bn) not in sub:
                ok = False
                break
            P = P * np.asarray(sub[(a, bn)].sum(axis=0)).ravel()
        if not ok:
            continue
        for d in p["fan_out"]:
            if (bn, d) not in sub:
                ok = False
                break
            P = P * np.asarray(sub[(bn, d)].sum(axis=1)).ravel()
        if not ok or P.sum() == 0:
            continue

        active = P > 0
        n_act = int(active.sum())
        n_eff = float(P.sum() ** 2 / np.square(P).sum())

        realized = 0
        for a in p["fan_in"]:
            m = sub[(a, bn)][:, active]
            realized += int((np.asarray(m.sum(axis=1)).ravel() > 0).sum())
        for d in p["fan_out"]:
            m = sub[(bn, d)][active, :]
            realized += int((np.asarray(m.sum(axis=0)).ravel() > 0).sum())

        r = p["row"]
        out.append({
            "structure_str": r["structure_str"],
            "signature": r["signature"],
            "hourglass_type": r["hourglass_type"],
            "motif_count": int(r["motif_count"]),
            "count_ricalcolato": int(P.sum()),
            "waist": f"{bn[0]}L{bn[1]}",
            "n_waist_gruppo": nb,
            "n_waist_attivi": n_act,
            "n_waist_efficace": n_eff,
            "n_periferia_realizzata": realized,
            "compressione_realizzata": realized / n_eff if n_eff > 0 else np.nan,
        })
        if (i_p + 1) % 5000 == 0:
            log(f"  {i_p+1:,}/{len(pats):,}")

    res = pd.DataFrame(out)
    rel = (res["count_ricalcolato"] - res["motif_count"]).abs() / \
        res["motif_count"].clip(lower=1)
    print(f"\n  Pattern elaborati: {len(res):,}")
    print(f"  Verifica interna (SigmaPi ricalcolato == motif_count): "
          f"esatti {int((rel < 1e-9).sum()):,}/{len(res):,}")

    res.to_csv(os.path.join(args.outdir, "compression_realized.csv"), index=False)
    elig = res[res["n_waist_efficace"] >= args.min_waist_eff]
    print(f"  Con n_waist_efficace >= {args.min_waist_eff}: {len(elig):,}")

    def show(d, title, col, n=15):
        print(f"\n  [{title}]")
        v = d.nlargest(n, col).copy()
        v["motif_count"] = v["motif_count"].map("{:.3e}".format)
        v["n_waist_efficace"] = v["n_waist_efficace"].map("{:.1f}".format)
        v["compressione_realizzata"] = v["compressione_realizzata"].map("{:.1f}".format)
        with pd.option_context("display.width", 250, "display.max_colwidth", 50):
            print(v[["structure_str", "motif_count", "n_waist_gruppo",
                     "n_waist_efficace", "n_periferia_realizzata",
                     "compressione_realizzata", "signature"]].to_string(index=False))

    show(elig, f"TOP PER COMPRESSIONE REALIZZATA (waist efficace >= "
               f"{args.min_waist_eff})", "compressione_realizzata")

    print("\n  [WAIST DOMINANTI nella classifica per compressione realizzata]")
    for k, v in elig.nlargest(200, "compressione_realizzata")["waist"] \
            .value_counts().head(10).items():
        print(f"    {k:<24} {v}")

    print("\n  [CONFRONTO: compressione nominale vs realizzata su casi noti]")
    for w in ("AL.MB_CAL2", "LO.PVLPL2", "LOL3", "AL.LHL2", "MEL2", "LOPL2"):
        s = res[res["waist"] == w]
        if len(s):
            b = s.nlargest(1, "compressione_realizzata").iloc[0]
            print(f"    {w:<12} gruppo={int(b['n_waist_gruppo']):>5}  "
                  f"attivi={int(b['n_waist_attivi']):>5}  "
                  f"efficaci={b['n_waist_efficace']:>7.1f}  "
                  f"periferia={int(b['n_periferia_realizzata']):>6}  "
                  f"compr.realizzata={b['compressione_realizzata']:>8.1f}")

    print("\n  [MASSIMA COMPRESSIONE REALIZZATA PER FIRMA DIREZIONALE]")
    rr = []
    for s, g in elig.groupby("signature"):
        b = g.nlargest(1, "compressione_realizzata").iloc[0]
        rr.append({"signature": s,
                   "compr_realizzata": b["compressione_realizzata"],
                   "waist": b["waist"], "n_eff": round(b["n_waist_efficace"], 1),
                   "struttura": b["structure_str"],
                   "interpretazione": interpret(s)})
    d = pd.DataFrame(rr).sort_values("compr_realizzata", ascending=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 44):
        print(d.round(1).to_string(index=False))
    d.to_csv(os.path.join(args.outdir, "compression_realized_by_signature.csv"),
             index=False)

    print(f"\n  [OUTPUT] {args.outdir}/compression_realized.csv")
    print(f"  Tempo totale: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
