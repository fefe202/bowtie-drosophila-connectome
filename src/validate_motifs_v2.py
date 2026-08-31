#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validazione statistica dei bow-tie motif contro tre modelli nulli
complementari.

    A  degree-preserving (Milo, Maslov & Sneppen): preserva i gradi,
       distrugge l'anatomia
    B  block-density-preserving: preserva le densita' di area, distrugge i
       gradi
    C  permutazione dell'allineamento: preserva entrambi e randomizza solo
       quale neurone del waist riceve quale grado

Si usano p-value empirici invece dello Z-score, perche' la distribuzione
nulla dei conteggi e' fortemente asimmetrica a destra. L'insieme di pattern
e' pre-registrato (top-20 per ciascuna delle 16 firme) e la correzione per
test multipli e' Benjamini-Hochberg.

Il rapporto reale/random non va usato come dimensione dell'effetto: spazia su
ordini di grandezza fra i nulli ed e' dominato da quanto ciascuno distrugge.

Uso:
    python src/validate_motifs_v2.py --window 1-2-3 --n_random 100 --seed 42
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

from directional_signature import add_signature
from motif_distill import load as load_distilled


def log(msg):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
#  SELEZIONE PRE-REGISTRATA DEI PATTERN
# ============================================================================

def select_patterns(path, per_sig_top, chunksize=1_000_000):
    """Top-K per firma direzionale: tutte le classi rappresentate."""
    log(f"Selezione pre-registrata: top {per_sig_top} per firma direzionale")
    best, rows, k = {}, 0, None
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if k is None:
            k = (int(chunk["k_in"].iloc[0]), int(chunk["k_out"].iloc[0]))
        rows += len(chunk)
        add_signature(chunk, *k)
        for s, sub in chunk.groupby("signature"):
            t = sub.nlargest(per_sig_top, "motif_count")
            best[s] = t if s not in best else \
                pd.concat([best[s], t]).nlargest(per_sig_top, "motif_count")
    df = pd.concat(best.values()).reset_index(drop=True)
    log(f"  {rows:,} righe scansionate -> {len(df)} pattern su "
        f"{len(best)} firme")
    return df, k


# ============================================================================
#  CONTEGGIO SigmaPi SU UN INSIEME DI SOTTOMATRICI
# ============================================================================

def counts_from_submats(patterns, groups, sub):
    """motif_count di ogni pattern, dato un dizionario di sottomatrici."""
    out = np.zeros(len(patterns))
    for i, p in enumerate(patterns):
        bn = p["bn"]
        if bn not in groups:
            continue
        v = np.ones(len(groups[bn]))
        ok = True
        for a in p["fan_in"]:
            m = sub.get((a, bn))
            if m is None:
                ok = False
                break
            v = v * np.asarray(m.sum(axis=0)).ravel()
        if not ok or v.sum() == 0:
            continue
        for d in p["fan_out"]:
            m = sub.get((bn, d))
            if m is None:
                ok = False
                break
            v = v * np.asarray(m.sum(axis=1)).ravel()
            if v.sum() == 0:
                break
        if ok:
            out[i] = v.sum()
    return out


def extract_flow_vectors(patterns, groups, sub):
    """
    Per ogni pattern estrae i vettori di flusso sul waist:
      v_m[i] = numero di neuroni del gruppo di fan-in m che proiettano su i
      w_n[i] = numero di neuroni del gruppo di fan-out n raggiunti da i
    Il motif_count e' sum_i prod_m v_m[i] * prod_n w_n[i].
    """
    out = []
    for p in patterns:
        bn = p["bn"]
        if bn not in groups:
            out.append(None)
            continue
        vs, ok = [], True
        for a in p["fan_in"]:
            m = sub.get((a, bn))
            if m is None:
                ok = False
                break
            vs.append(np.asarray(m.sum(axis=0)).ravel())
        if ok:
            for d in p["fan_out"]:
                m = sub.get((bn, d))
                if m is None:
                    ok = False
                    break
                vs.append(np.asarray(m.sum(axis=1)).ravel())
        out.append(vs if ok else None)
    return out


def alignment_null_counts(flow_vectors, rng):
    """
    NULLO C - permutazione dell'allineamento.

    Preserva ESATTAMENTE, per ogni blocco, il numero di archi e l'intera
    sequenza dei gradi sul waist; randomizza soltanto QUALE neurone del waist
    riceve quale grado, indipendentemente in ciascun blocco.

    E' il nullo che isola l'ipotesi clessidra nella sua forma piu' pura:
    "sono gli STESSI neuroni del waist a ricevere da tutti i gruppi di fan-in
    e a proiettare su tutti quelli di fan-out, piu' di quanto atteso se i
    gradi fossero assegnati indipendentemente?"

    Tutto cio' che non e' allineamento resta invariato, quindi l'arricchimento
    misurato non puo' essere attribuito ne' alla densita' delle aree ne'
    all'eterogeneita' dei gradi: entrambe sono preservate per costruzione.
    """
    out = np.zeros(len(flow_vectors))
    for i, vs in enumerate(flow_vectors):
        if vs is None:
            continue
        acc = None
        for v in vs:
            pv = rng.permutation(v)
            acc = pv if acc is None else acc * pv
        out[i] = acc.sum()
    return out


def extract_submats(A, groups, pairs):
    sub = {}
    for s, d in pairs:
        if s in groups and d in groups:
            m = A[groups[s], :][:, groups[d]]
            if m.nnz:
                sub[(s, d)] = m.tocsr()
    return sub


# ============================================================================
#  NULLO A - degree-preserving edge swap, vettorizzato a batch
# ============================================================================

def swap_randomize(rows, cols, n_nodes, swap_factor, rng, batch=200_000):
    """
    Double-edge swap che preserva in-degree e out-degree.
    Implementazione a batch: propone molti swap insieme, scarta quelli che
    creano self-loop o archi duplicati, e quelli che toccano lo stesso arco
    due volte nello stesso batch.
    """
    rows = rows.copy()
    cols = cols.copy()
    n_edges = len(rows)
    target = int(swap_factor * n_edges)
    done = 0
    keys = np.sort(rows.astype(np.int64) * n_nodes + cols.astype(np.int64))

    while done < target:
        b = min(batch, target - done)
        i1 = rng.integers(0, n_edges, size=b)
        i2 = rng.integers(0, n_edges, size=b)
        a, bb = rows[i1], cols[i1]
        c, d = rows[i2], cols[i2]

        ok = (i1 != i2) & (a != c) & (bb != d) & (a != d) & (c != bb)
        # nessun arco toccato due volte nello stesso batch
        if ok.any():
            idx = np.flatnonzero(ok)
            seen = np.zeros(n_edges, dtype=bool)
            keep = []
            for j in idx:
                if not seen[i1[j]] and not seen[i2[j]]:
                    seen[i1[j]] = seen[i2[j]] = True
                    keep.append(j)
            idx = np.array(keep, dtype=np.int64)
        else:
            done += b
            continue
        if len(idx) == 0:
            done += b
            continue

        k1 = a[idx].astype(np.int64) * n_nodes + d[idx].astype(np.int64)
        k2 = c[idx].astype(np.int64) * n_nodes + bb[idx].astype(np.int64)
        pos1 = np.searchsorted(keys, k1)
        pos2 = np.searchsorted(keys, k2)
        exists1 = (pos1 < len(keys)) & (keys[np.minimum(pos1, len(keys)-1)] == k1)
        exists2 = (pos2 < len(keys)) & (keys[np.minimum(pos2, len(keys)-1)] == k2)
        good = ~exists1 & ~exists2 & (k1 != k2)
        idx = idx[good]
        if len(idx) == 0:
            done += b
            continue

        old1 = rows[i1[idx]].astype(np.int64) * n_nodes + cols[i1[idx]].astype(np.int64)
        old2 = rows[i2[idx]].astype(np.int64) * n_nodes + cols[i2[idx]].astype(np.int64)
        cols[i1[idx]] = d[idx]
        cols[i2[idx]] = bb[idx]
        new1 = k1[good]
        new2 = k2[good]
        keys = np.sort(np.concatenate([
            np.setdiff1d(keys, np.concatenate([old1, old2]), assume_unique=True),
            new1, new2]))
        done += b

    return rows, cols


# ============================================================================
#  NULLO B - ridistribuzione uniforme dentro ciascun blocco
# ============================================================================

def block_randomize(sub_real, groups, rng):
    """
    Per ogni coppia di metanodi si mantiene lo stesso numero di archi,
    ridistribuiti uniformemente fra le |g1| x |g2| coppie possibili.
    Preserva la densita' a livello di aree, randomizza i gradi entro il blocco.
    """
    out = {}
    for (s, d), m in sub_real.items():
        ns, nd = len(groups[s]), len(groups[d])
        total = ns * nd
        k = m.nnz
        if k >= total:
            out[(s, d)] = sparse.csr_matrix(np.ones((ns, nd)))
            continue
        # campionamento senza ripetizioni: proposta con ripetizioni + unique,
        # completata finche' non si raggiunge k
        picked = np.unique(rng.integers(0, total, size=int(k * 1.3) + 8))
        while len(picked) < k:
            picked = np.unique(np.concatenate(
                [picked, rng.integers(0, total, size=(k - len(picked)) * 2 + 8)]))
        picked = picked[:k]
        out[(s, d)] = sparse.csr_matrix(
            (np.ones(k), (picked // nd, picked % nd)), shape=(ns, nd))
    return out


# ============================================================================
#  MAIN
# ============================================================================

def build_graph(patterns, conn_file, levels_file, neurons_file):
    levels = {k[1] for p in patterns
              for k in p["fan_in"] + (p["bn"],) + p["fan_out"]}
    dn = pd.read_csv(neurons_file, usecols=["root_id", "group"]).dropna()
    dn = dn[dn["group"] != "NO_CONS"]
    area = dict(zip(dn["root_id"], dn["group"]))
    dl = pd.read_csv(levels_file, usecols=["root_id", "y_level"])
    lev = dict(zip(dl["root_id"], dl["y_level"].astype(int)))
    valid = sorted(n for n in (set(area) & set(lev)) if lev[n] in levels)
    idx = {n: i for i, n in enumerate(valid)}
    groups = defaultdict(list)
    for n in valid:
        groups[(area[n], lev[n])].append(idx[n])
    dc = pd.read_csv(conn_file, usecols=["pre_root_id", "post_root_id", "syn_count"])
    e = dc[dc["syn_count"] > 0][["pre_root_id", "post_root_id"]].drop_duplicates()
    e = e[e["pre_root_id"].isin(idx) & e["post_root_id"].isin(idx)]
    r = e["pre_root_id"].map(idx).to_numpy(dtype=np.int64)
    c = e["post_root_id"].map(idx).to_numpy(dtype=np.int64)
    A = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(valid), len(valid)))
    return A, groups, r, c, len(valid)


def main():
    ap = argparse.ArgumentParser(description="Validazione statistica rifatta.")
    ap.add_argument("--window", default="1-2-3")
    ap.add_argument("--motifs", default=None,
                    help="CSV riga-per-riga (legacy).")
    ap.add_argument("--per_sig_top", type=int, default=20)
    ap.add_argument("--n_random", type=int, default=100)
    ap.add_argument("--swap_factor", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixing_check", action="store_true",
                    help="Verifica empiricamente a quale fattore di swap i "
                         "conteggi si stabilizzano, poi esce")
    ap.add_argument("--nulls", default="A,B")
    ap.add_argument("--outdir", default=results("validation_v2"))
    ap.add_argument("--conn_file", default=data("connections.csv"))
    ap.add_argument("--levels_file", default=data("COORDINATE_XY_with_levels_tree.csv"))
    ap.add_argument("--neurons_file", default=data("neurons.csv"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    print("=" * 78)
    print("  VALIDAZIONE STATISTICA DEI BOW-TIE MOTIF (v2)")
    print("=" * 78)
    log(f"seed = {args.seed}, N = {args.n_random}, "
        f"swap_factor = {args.swap_factor}")

    if args.motifs:
        df, (k_in, k_out) = select_patterns(args.motifs, args.per_sig_top)
    else:
        d = load_distilled(args.window, "top_per_firma")
        add_signature(d, int(d["k_in"].iloc[0]), int(d["k_out"].iloc[0]))
        df = (d.groupby("signature", group_keys=False)
                .apply(lambda g: g.nlargest(args.per_sig_top, "motif_count"))
                .reset_index(drop=True))
        k_in, k_out = int(df["k_in"].iloc[0]), int(df["k_out"].iloc[0])
        log(f"distillato {args.window}: {len(df)} pattern su "
            f"{df['signature'].nunique()} firme")
    patterns = [{
        "fan_in": tuple((r[f"fan_in_{i}_area"], int(r[f"fan_in_{i}_level"]))
                        for i in range(k_in)),
        "bn": (r["bottleneck_area"], int(r["bottleneck_level"])),
        "fan_out": tuple((r[f"fan_out_{i}_area"], int(r[f"fan_out_{i}_level"]))
                         for i in range(k_out)),
    } for r in df.to_dict("records")]

    log("Costruzione del grafo...")
    A, groups, rows, cols, n_nodes = build_graph(
        patterns, args.conn_file, args.levels_file, args.neurons_file)
    log(f"  {n_nodes:,} neuroni, {len(rows):,} archi")

    pairs = set()
    for p in patterns:
        for a in p["fan_in"]:
            pairs.add((a, p["bn"]))
        for d in p["fan_out"]:
            pairs.add((p["bn"], d))
    sub_real = extract_submats(A, groups, pairs)
    real = counts_from_submats(patterns, groups, sub_real)
    err = np.abs(real - df["motif_count"].to_numpy()) / \
        np.maximum(df["motif_count"].to_numpy(), 1)
    log(f"  verifica: {int((err < 1e-9).sum())}/{len(real)} conteggi reali "
        f"riprodotti esattamente")

    rng = np.random.default_rng(args.seed)

    # ---- verifica del mixing
    if args.mixing_check:
        print("\n  [VERIFICA DEL MIXING]  mediana del rapporto "
              "count_random / count_reale al crescere degli swap")
        prev = None
        for q in (1, 5, 10, 20, 50, 100):
            t = time.time()
            r2, c2 = swap_randomize(rows, cols, n_nodes, q, rng)
            Ar = sparse.csr_matrix((np.ones(len(r2)), (r2, c2)),
                                   shape=(n_nodes, n_nodes))
            cr = counts_from_submats(patterns, groups,
                                     extract_submats(Ar, groups, pairs))
            ratio = np.median(cr / np.maximum(real, 1))
            delta = "" if prev is None else f"   variazione: {ratio-prev:+.3e}"
            print(f"    Q = {q:>4}:  mediana = {ratio:.4e}   "
                  f"({time.time()-t:.0f}s){delta}")
            prev = ratio
        print("    Il mixing e' sufficiente quando la mediana smette di "
              "variare sistematicamente.")
        return

    results = {"real": real}
    for null in [x.strip() for x in args.nulls.split(",")]:
        log(f"\nModello nullo {null}: {args.n_random} randomizzazioni...")
        acc = np.zeros((args.n_random, len(patterns)))
        for i in range(args.n_random):
            t = time.time()
            if null == "A":
                r2, c2 = swap_randomize(rows, cols, n_nodes,
                                        args.swap_factor, rng)
                Ar = sparse.csr_matrix((np.ones(len(r2)), (r2, c2)),
                                       shape=(n_nodes, n_nodes))
                acc[i] = counts_from_submats(
                    patterns, groups, extract_submats(Ar, groups, pairs))
            elif null == "B":
                acc[i] = counts_from_submats(
                    patterns, groups, block_randomize(sub_real, groups, rng))
            elif null == "C":
                if i == 0:
                    flows = extract_flow_vectors(patterns, groups, sub_real)
                acc[i] = alignment_null_counts(flows, rng)
            else:
                raise ValueError(f"nullo sconosciuto: {null}")
            if (i + 1) % 10 == 0 or i == 0:
                log(f"  {null}: {i+1}/{args.n_random}  ({time.time()-t:.1f}s)")
        results[null] = acc

    # ---- p-value empirici, effect size, BH
    out = df[["structure_str", "signature", "hourglass_type",
              "motif_count"]].copy()
    out["count_verificato"] = real.astype(np.int64)
    summary = []
    for null in [x for x in results if x != "real"]:
        acc = results[null]
        ge = (acc >= real[None, :]).sum(axis=0)
        p = (1 + ge) / (args.n_random + 1)
        mean = acc.mean(axis=0)
        ratio = real / np.maximum(mean, 1e-12)
        # Benjamini-Hochberg
        order = np.argsort(p)
        m = len(p)
        q = np.empty(m)
        prev = 1.0
        for rank in range(m - 1, -1, -1):
            j = order[rank]
            prev = min(prev, p[j] * m / (rank + 1))
            q[j] = prev
        out[f"p_{null}"] = p
        out[f"q_{null}"] = q
        out[f"media_random_{null}"] = mean
        out[f"rapporto_{null}"] = ratio
        summary.append({
            "nullo": null,
            "pattern": m,
            "p_minimo_possibile": 1 / (args.n_random + 1),
            "significativi_q<0.05": int((q < 0.05).sum()),
            "rapporto_mediano": float(np.median(ratio)),
            "rapporto_min": float(ratio.min()),
            "rapporto_max": float(ratio.max()),
        })

    out.to_csv(os.path.join(args.outdir, "validation_v2.csv"), index=False)
    ds = pd.DataFrame(summary)

    print("\n" + "=" * 78)
    print("  RISULTATI")
    print("=" * 78)
    print(ds.to_string(index=False))

    for null in [x for x in results if x != "real"]:
        print(f"\n  [NULLO {null}] arricchimento per firma direzionale "
              f"(rapporto reale/random, mediana)")
        g = out.groupby("signature").agg(
            n=("motif_count", "size"),
            rapporto_mediano=(f"rapporto_{null}", "median"),
            q_mediano=(f"q_{null}", "median"),
            significativi=(f"q_{null}", lambda x: int((x < 0.05).sum())))
        print(g.sort_values("rapporto_mediano", ascending=False)
              .round(4).to_string())

    print(f"\n  [OUTPUT] {args.outdir}/validation_v2.csv")
    print(f"  Tempo totale: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
