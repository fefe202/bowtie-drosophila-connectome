#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motore matriciale per la ricerca dei bow-tie motif. Stessi risultati di
hourglass_areas.py, che resta il riferimento di correttezza.

Il ciclo annidato originale e' dominato dall'overhead di chiamata di NumPy,
non dai flop: ~1,7 miliardi di iterazioni sulla finestra 1-2-3. Qui i
conteggi di un bottleneck escono da un unico prodotto matriciale delegato a
BLAS,

    T = P_in @ P_out^T

dove P_in[r] e P_out[j] sono i prodotti elemento per elemento dei flow vector
delle combinazioni. Le metriche di compressione seguono la stessa forma,
quindi il vettore SigmaPi della singola coppia non viene mai materializzato.

Per i waist grandi le matrici non entrano nel budget di memoria e il prodotto
diventa sparso: i flow vector sono quasi tutti zeri, e il prodotto sparso
produce direttamente solo le coppie a conteggio non nullo. Se nemmeno questo
basta, il bottleneck ricade sul ciclo di riferimento: stessi risultati, solo
piu' lento.

Di default scrive solo il distillato (vedi motif_distill.py); --full_csv
aggiunge il CSV riga-per-riga.

Verifica:
    python src/hourglass_areas_fast.py --window 4-5 --verify

Uso:
    python src/hourglass_areas_fast.py --window 1-2-3 --k_in 2 --k_out 2 \
        --max_jump 1 --min_count 10
"""

from thesis_paths import data, results

import argparse
import csv
import gzip
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy import sparse

from motif_distill import Distiller
from hourglass_areas import (DENSE_THRESHOLD, classify_hourglass,
                             format_elapsed, signature_hourglass, timestamp,
                             to_dense_1d)

# Budget di memoria per le matrici di un singolo bottleneck.
MATRIX_BUDGET = 400 * 1024 * 1024

# Righe accumulate prima di passarle al distillatore.
FLUSH_ROWS = 500_000


def log(msg):
    print(f"  [{timestamp()}] {msg}", flush=True)


def build_graph(NL_WINDOW, max_jump, conn_file, levels_file, neurons_file,
                grouping="group", class_file=None):
    """Fasi 1-5: identiche a hourglass_areas.py."""
    df_conn = pd.read_csv(conn_file)
    df_levels = pd.read_csv(levels_file)
    df_neurons = pd.read_csv(neurons_file)

    if grouping == "superclass":
        # Granularita' usata nei lavori di confronto: 9 superclassi funzionali
        # invece dei 628 gruppi anatomici.
        dc = pd.read_csv(class_file, usecols=["root_id", "super_class"])
        dc = dc.dropna(subset=["super_class"])
        area_map = dc.set_index("root_id")["super_class"].to_dict()
    else:
        dn = df_neurons.dropna(subset=["group"])
        dn = dn[dn["group"] != "NO_CONS"]
        area_map = dn.set_index("root_id")["group"].to_dict()
    level_map = df_levels.set_index("root_id")["y_level"].to_dict()
    target = set(NL_WINDOW)
    valid = sorted(n for n in (set(area_map) & set(level_map))
                   if level_map[n] in target)
    idx = {n: i for i, n in enumerate(valid)}

    groups = defaultdict(list)
    for n in valid:
        groups[(area_map[n], level_map[n])].append(idx[n])
    group_keys = sorted(groups.keys())

    e = df_conn[df_conn["syn_count"] > 0][["pre_root_id", "post_root_id"]]
    e = e.drop_duplicates()
    e = e[e["pre_root_id"].isin(idx) & e["post_root_id"].isin(idx)]
    A = sparse.csr_matrix(
        (np.ones(len(e)), (e["pre_root_id"].map(idx).astype(int),
                           e["post_root_id"].map(idx).astype(int))),
        shape=(len(valid), len(valid)))

    submat, incoming, outgoing = {}, defaultdict(list), defaultdict(list)
    for s in group_keys:
        row = A[groups[s], :]
        for d in group_keys:
            if abs(s[1] - d[1]) > max_jump or s == d:
                continue
            m = row[:, groups[d]]
            if m.nnz:
                submat[(s, d)] = m.toarray() if \
                    m.shape[0] * m.shape[1] < DENSE_THRESHOLD else m
                incoming[d].append(s)
                outgoing[s].append(d)
    return groups, group_keys, submat, incoming, outgoing, len(valid), A.nnz


def run(NL_WINDOW, out_folder, k_in, k_out, max_jump, min_count,
        conn_file, levels_file, neurons_file, engine="matrix", limit=0,
        budget=MATRIX_BUDGET, gz=False, full_csv=False,
        grouping="group", class_file=None):
    t0 = time.time()
    os.makedirs(out_folder, exist_ok=True)
    print("=" * 74)
    print(f"  BOW-TIE MOTIF SEARCH - motore: {engine.upper()}")
    print("=" * 74)
    log(f"finestra {NL_WINDOW}, k_in={k_in}, k_out={k_out}, "
        f"max_jump={max_jump}, min_count={min_count}")

    groups, group_keys, submat, incoming, outgoing, n_tot, n_edges = \
        build_graph(NL_WINDOW, max_jump, conn_file, levels_file, neurons_file,
                    grouping, class_file)
    log(f"{n_tot:,} neuroni, {n_edges:,} archi, {len(group_keys)} metanodi, "
        f"{len(submat):,} archi del metagrafo")

    header = ["motif_count", "hourglass_type", "signature", "fanin_sig",
              "fanout_sig", "n_FF", "n_FB", "n_lat", "n_waist",
              "n_waist_active", "n_waist_eff", "n_periphery", "compression",
              "bottleneck_area", "bottleneck_level", "k_in", "k_out",
              "structure_str"]
    for i in range(k_in):
        header += [f"fan_in_{i}_area", f"fan_in_{i}_level"]
    for i in range(k_out):
        header += [f"fan_out_{i}_area", f"fan_out_{i}_level"]

    name = (f"hourglass_areas_nl{'-'.join(map(str, NL_WINDOW))}"
            f"_kin{k_in}_kout{k_out}_jump{max_jump}.csv")
    if gz:
        name += ".gz"
    path = os.path.join(out_folder, name)
    # Di default il CSV riga-per-riga non viene scritto: si accumulano
    # aggregati e selezioni top-N. Con --gzip l'output completo comprime
    # circa 8 volte, e pandas legge .csv.gz senza modifiche a valle.
    fh = wr = None
    if full_csv:
        fh = (gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=1)
              if gz else open(path, "w", newline="", encoding="utf-8"))
        wr = csv.writer(fh)
        wr.writerow(header)
    dist = Distiller()
    buf = []

    n_found = n_saved = n_fallback = n_sparse = 0
    sig_counts, type_counts = defaultdict(int), defaultdict(int)
    search_t0 = time.time()

    n_done = 0
    for bn_i, bn in enumerate(group_keys):
        in_nbrs, out_nbrs = incoming.get(bn, []), outgoing.get(bn, [])
        if len(in_nbrs) < k_in or len(out_nbrs) < k_out:
            continue
        if limit and n_done >= limit:
            break
        n_done += 1
        t_bn = time.time()
        bn_idx_list = groups[bn]
        n_bn = len(bn_idx_list)
        bn_area, bn_level = bn

        fin = {}
        for a in in_nbrs:
            s = submat[(a, bn)]
            fin[a] = to_dense_1d(np.ones(len(groups[a])) @ s)
        fout = {}
        for d in out_nbrs:
            s = submat[(bn, d)]
            v = np.asarray(s.sum(axis=1)).ravel() if sparse.issparse(s) \
                else s.sum(axis=1)
            fout[d] = to_dense_1d(v)

        in_keys, out_keys = list(in_nbrs), list(out_nbrs)
        Vin = np.array([fin[a] for a in in_keys])
        Vout = np.array([fout[d] for d in out_keys])
        support = (Vin > 0).any(axis=0) & (Vout > 0).any(axis=0)
        n_sup = int(support.sum())
        if n_sup == 0:
            continue
        Vin_s, Vout_s = Vin[:, support], Vout[:, support]

        n_ic = comb(len(in_keys), k_in)
        n_oc = comb(len(out_keys), k_out)
        est = (n_ic + n_oc) * n_sup * 8 * 3
        if engine == "loop":
            n_fallback += 1
            res = _loop_engine(bn, in_keys, out_keys, fin, fout, n_bn,
                               k_in, k_out, min_count, groups)
        elif engine == "sparse" or est > budget:
            # waist troppo grande per le matrici dense: si passa allo sparso,
            # che con waist grandi e' anche piu' efficiente perche' i vettori
            # di flusso sono molto sparsi
            n_sparse += 1
            res = _matrix_engine_sparse(bn, in_keys, out_keys, Vin_s, Vout_s,
                                        n_bn, k_in, k_out, min_count, groups)
        else:
            res = _matrix_engine(bn, in_keys, out_keys, Vin_s, Vout_s,
                                 n_bn, k_in, k_out, min_count, groups)

        bn_saved = 0
        for count, fi_combo, fo_combo, n_active, ssq in res:
            n_found += 1
            hg = classify_hourglass(fi_combo, bn, fo_combo)
            n_ff, n_fb, n_lt, fis, fos, sig = signature_hourglass(
                fi_combo, bn, fo_combo)
            type_counts[hg] += 1
            sig_counts[sig] += 1
            n_per = (sum(len(groups[a]) for a in fi_combo) +
                     sum(len(groups[d]) for d in fo_combo))
            n_eff = (count ** 2 / ssq) if ssq > 0 else 0.0
            row = [count, hg, sig, fis, fos, n_ff, n_fb, n_lt, n_bn,
                   n_active, round(n_eff, 3), n_per,
                   round(n_per / n_bn, 3), bn_area, bn_level, k_in, k_out,
                   "[" + ", ".join(f"{a}L{l}" for a, l in fi_combo) + "] -> "
                   + f"{bn_area}L{bn_level} -> ["
                   + ", ".join(f"{a}L{l}" for a, l in fo_combo) + "]"]
            for a, l in fi_combo:
                row += [a, l]
            for a, l in fo_combo:
                row += [a, l]
            if wr is not None:
                wr.writerow(row)
            buf.append(row)
            # Svuotamento a blocchi: alcuni bottleneck emettono decine di
            # milioni di righe, e costruire il DataFrame solo a fine
            # bottleneck esaurisce la memoria.
            if len(buf) >= FLUSH_ROWS:
                dist.add_frame(pd.DataFrame(buf, columns=header))
                buf = []
            n_saved += 1
            bn_saved += 1

        if buf:
            dist.add_frame(pd.DataFrame(buf, columns=header))
            buf = []

        if bn_saved or (bn_i + 1) % 100 == 0 or time.time()-t_bn > 5:
            el = time.time() - search_t0
            pct = (bn_i + 1) / len(group_keys) * 100
            eta = format_elapsed(el / pct * (100 - pct)) if pct > 0 else "?"
            log(f"bottleneck {bn_i+1:>4}/{len(group_keys)} ({pct:5.1f}%) "
                f"{bn_area}L{bn_level} salvati={bn_saved:,} "
                f"totale={n_saved:,} ETA {eta} ({time.time()-t_bn:.1f}s, "
                f"in={len(in_nbrs)} out={len(out_nbrs)} |B|={n_bn})")

    if fh is not None:
        fh.close()
    dist_dir = os.path.join(os.path.dirname(os.path.dirname(out_folder)),
                            "motif_distilled",
                            "-".join(map(str, NL_WINDOW)))
    if dist.k is not None:
        dist.finalize(dist_dir, "-".join(map(str, NL_WINDOW)),
                      {"max_jump": max_jump, "min_count": min_count})
    print()
    print("=" * 74)
    print(f"  motif trovati: {n_found:,}   salvati: {n_saved:,}")
    print(f"  bottleneck su motore sparso: {n_sparse}   su ciclo di "
          f"riferimento: {n_fallback}")
    print("  per firma direzionale:")
    for s in sorted(sig_counts, key=lambda x: -sig_counts[x]):
        print(f"    {s:>12}: {sig_counts[s]:,}")
    print(f"\n  [OUTPUT] {dist_dir}/   (aggregati e selezioni top-N)")
    if full_csv:
        print(f"  [OUTPUT] {path}   (CSV riga-per-riga, chiesto con --full_csv)")
    else:
        print("  Il CSV riga-per-riga NON e' stato scritto: sono decine di GB")
        print("  che nessuna analisi consuma. Usare --full_csv se servono.")
    print(f"  tempo totale: {format_elapsed(time.time()-t0)}")
    return path if full_csv else dist_dir


def _matrix_engine(bn, in_keys, out_keys, Vin, Vout, n_bn, k_in, k_out,
                   min_count, groups):
    """Tutti i conteggi del bottleneck con prodotti matriciali."""
    ic = np.array(list(combinations(range(len(in_keys)), k_in)), dtype=np.int32)
    oc = np.array(list(combinations(range(len(out_keys)), k_out)), dtype=np.int32)

    P_in = Vin[ic[:, 0]].copy()
    for t in range(1, k_in):
        P_in *= Vin[ic[:, t]]
    P_out = Vout[oc[:, 0]].copy()
    for t in range(1, k_out):
        P_out *= Vout[oc[:, t]]

    ki = P_in.sum(axis=1) > 0
    ko = P_out.sum(axis=1) > 0
    P_in, ic = P_in[ki], ic[ki]
    P_out, oc = P_out[ko], oc[ko]
    if not len(ic) or not len(oc):
        return

    out_pos = {k: i for i, k in enumerate(out_keys)}
    shared = {i: out_pos[k] for i, k in enumerate(in_keys) if k in out_pos}
    out_has = {oi: (oc == oi).any(axis=1) for oi in set(shared.values())}

    P_out2 = P_out ** 2
    P_outb = (P_out > 0).astype(np.float64)
    chunk = max(1, int(MATRIX_BUDGET / (3 * len(oc) * 8)))

    for s0 in range(0, len(ic), chunk):
        s1 = min(s0 + chunk, len(ic))
        blk = P_in[s0:s1]
        T = blk @ P_out.T
        T2 = (blk ** 2) @ P_out2.T
        Tb = (blk > 0).astype(np.float64) @ P_outb.T
        for loc in range(s1 - s0):
            r = s0 + loc
            counts = T[loc]
            bad = None
            for t in range(k_in):
                oi = shared.get(int(ic[r, t]))
                if oi is not None:
                    bad = out_has[oi] if bad is None else (bad | out_has[oi])
            if bad is not None:
                counts = np.where(bad, 0.0, counts)
            sel = np.flatnonzero(counts >= min_count)
            if not len(sel):
                continue
            fi = tuple(in_keys[int(ic[r, t])] for t in range(k_in))
            for j in sel:
                yield (int(round(counts[j])), fi,
                       tuple(out_keys[int(oc[j, t])] for t in range(k_out)),
                       int(round(Tb[loc, j])), float(T2[loc, j]))


def _matrix_engine_sparse(bn, in_keys, out_keys, Vin, Vout, n_bn, k_in, k_out,
                          min_count, groups):
    """
    Variante SPARSA, per i bottleneck con waist troppo grande per le matrici
    dense (es. ME_L2, 24.835 neuroni).

    Con waist grandi i vettori di flusso sono molto sparsi: un gruppo di
    fan-in raggiunge solo una frazione dei neuroni del waist, e il prodotto
    di k_in vettori lo e' ancora di piu'. Il prodotto sparso P_in @ P_out^T
    produce direttamente SOLO le coppie a conteggio non nullo, che sono
    esattamente quelle da emettere: non si spreca lavoro sugli zeri, e la
    memoria segue il numero di elementi non nulli invece che |combos| x |B|.

    Le tre quantita' hanno la stessa struttura di sparsita', perche' i valori
    sono conteggi non negativi e quindi un prodotto e' nullo se e solo se lo
    e' uno dei fattori.
    """
    ic = np.array(list(combinations(range(len(in_keys)), k_in)), dtype=np.int32)
    oc = np.array(list(combinations(range(len(out_keys)), k_out)), dtype=np.int32)

    Vin_sp = sparse.csr_matrix(Vin)
    Vout_sp = sparse.csr_matrix(Vout)
    P_in = Vin_sp[ic[:, 0]]
    for t in range(1, k_in):
        P_in = P_in.multiply(Vin_sp[ic[:, t]])
    P_out = Vout_sp[oc[:, 0]]
    for t in range(1, k_out):
        P_out = P_out.multiply(Vout_sp[oc[:, t]])
    P_in = sparse.csr_matrix(P_in)
    P_out = sparse.csr_matrix(P_out)
    P_in.eliminate_zeros()
    P_out.eliminate_zeros()

    ki = np.diff(P_in.indptr) > 0
    ko = np.diff(P_out.indptr) > 0
    P_in, ic = P_in[ki], ic[ki]
    P_out, oc = P_out[ko], oc[ko]
    if not P_in.shape[0] or not P_out.shape[0]:
        return

    out_pos = {k: i for i, k in enumerate(out_keys)}
    shared = {i: out_pos[k] for i, k in enumerate(in_keys) if k in out_pos}
    out_has = {oi: (oc == oi).any(axis=1) for oi in set(shared.values())}

    P_outT = P_out.T.tocsc()
    P_out2T = P_out.multiply(P_out).T.tocsc()
    P_outbT = sparse.csr_matrix(
        (np.ones_like(P_out.data), P_out.indices, P_out.indptr),
        shape=P_out.shape).T.tocsc()

    chunk = max(1, min(2048, P_in.shape[0]))
    for s0 in range(0, P_in.shape[0], chunk):
        s1 = min(s0 + chunk, P_in.shape[0])
        blk = P_in[s0:s1]
        blk2 = blk.multiply(blk)
        blkb = sparse.csr_matrix(
            (np.ones_like(blk.data), blk.indices, blk.indptr), shape=blk.shape)
        T = (blk @ P_outT).tocsr()
        T.sort_indices()
        T2 = (blk2 @ P_out2T).tocsr()
        T2.sort_indices()
        Tb = (blkb @ P_outbT).tocsr()
        Tb.sort_indices()
        if not (T.nnz == T2.nnz == Tb.nnz):
            raise RuntimeError("strutture di sparsita' non allineate")

        for loc in range(s1 - s0):
            a0, a1 = T.indptr[loc], T.indptr[loc + 1]
            if a0 == a1:
                continue
            cols = T.indices[a0:a1]
            vals = T.data[a0:a1]
            r = s0 + loc
            bad = None
            for t in range(k_in):
                oi = shared.get(int(ic[r, t]))
                if oi is not None:
                    m = out_has[oi][cols]
                    bad = m if bad is None else (bad | m)
            keep = vals >= min_count
            if bad is not None:
                keep &= ~bad
            sel = np.flatnonzero(keep)
            if not len(sel):
                continue
            fi = tuple(in_keys[int(ic[r, t])] for t in range(k_in))
            t2 = T2.data[a0:a1]
            tb = Tb.data[a0:a1]
            for j in sel:
                yield (int(round(vals[j])), fi,
                       tuple(out_keys[int(oc[cols[j], t])]
                             for t in range(k_out)),
                       int(round(tb[j])), float(t2[j]))


def _loop_engine(bn, in_keys, out_keys, fin, fout, n_bn, k_in, k_out,
                 min_count, groups):
    """Motore di riferimento, per i bottleneck fuori budget."""
    for fi in combinations(in_keys, k_in):
        prod_in = np.ones(n_bn)
        for a in fi:
            prod_in *= fin[a]
        if prod_in.sum() == 0:
            continue
        fi_set = set(fi)
        valid_out = [d for d in out_keys if d not in fi_set]
        if len(valid_out) < k_out:
            continue
        for fo in combinations(valid_out, k_out):
            pv = prod_in.copy()
            for d in fo:
                pv *= fout[d]
                if pv.sum() == 0:
                    break
            tot = pv.sum()
            if tot < min_count:
                continue
            yield (int(round(tot)), fi, fo,
                   int(np.count_nonzero(pv)), float(pv @ pv))


def main():
    ap = argparse.ArgumentParser(description="Motore matriciale per i bow-tie motif.")
    ap.add_argument("--window", required=True)
    ap.add_argument("--k_in", type=int, default=2)
    ap.add_argument("--k_out", type=int, default=2)
    ap.add_argument("--max_jump", type=int, default=1)
    ap.add_argument("--min_count", type=int, default=10)
    ap.add_argument("--engine", choices=["matrix", "loop", "sparse"],
                    default="matrix")
    ap.add_argument("--grouping", choices=["group", "superclass"],
                    default="group",
                    help="granularita' dei metanodi: gruppo anatomico "
                         "(628 categorie) o superclasse funzionale (9)")
    ap.add_argument("--class_file", default=data("classification.csv"))
    ap.add_argument("--full_csv", action="store_true",
                    help="Scrive anche il CSV riga-per-riga. Di default NON "
                         "viene scritto: sono decine di GB che nessuna analisi "
                         "consuma. Serve solo per query nuove sulle righe grezze.")
    ap.add_argument("--gzip", action="store_true",
                    help="Scrive l output compresso (.csv.gz). Necessario per "
                         "la finestra completa, che in chiaro supera i 200 GB.")
    ap.add_argument("--budget_mb", type=int, default=400,
                    help="Budget di memoria per bottleneck. Abbassarlo forza "
                         "il motore sparso, utile per verificarlo.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Processa solo i primi N bottleneck eleggibili (test)")
    ap.add_argument("--verify", action="store_true",
                    help="Esegue entrambi i motori e confronta i CSV riga per riga")
    ap.add_argument("--outdir", default=results("hourglass_results"))
    ap.add_argument("--conn_file", default=data("connections.csv"))
    ap.add_argument("--levels_file",
                    default=data("COORDINATE_XY_with_levels_tree.csv"))
    ap.add_argument("--neurons_file", default=data("neurons.csv"))
    a = ap.parse_args()
    NL = [int(x) for x in a.window.split("-")]

    if a.verify:
        paths = {}
        for eng in (a.engine, "loop"):
            d = os.path.join(a.outdir, f"_verify_{eng}", a.window)
            paths[eng] = run(NL, d, a.k_in, a.k_out, a.max_jump, a.min_count,
                             a.conn_file, a.levels_file, a.neurons_file, eng,
                             a.limit, a.budget_mb * 1024 * 1024, a.gzip, True)
        key = ["structure_str"]
        d1 = pd.read_csv(paths[a.engine]).sort_values(key).reset_index(drop=True)
        d2 = pd.read_csv(paths["loop"]).sort_values(key).reset_index(drop=True)
        print("\n" + "=" * 74)
        print("  VERIFICA  motore matriciale vs motore di riferimento")
        print("=" * 74)
        print(f"  righe: {len(d1):,} vs {len(d2):,}")
        if len(d1) != len(d2):
            print("  DIVERSO NUMERO DI RIGHE")
            sys.exit(1)
        bad = []
        for c in d1.columns:
            if d1[c].dtype.kind in "fi":
                if not np.allclose(d1[c], d2[c], rtol=1e-9, atol=1e-6):
                    bad.append(c)
            elif not (d1[c].astype(str) == d2[c].astype(str)).all():
                bad.append(c)
        if bad:
            print(f"  COLONNE DIVERSE: {bad}")
            sys.exit(1)
        print(f"  tutte le {len(d1.columns)} colonne coincidono su "
              f"{len(d1):,} righe: I DUE MOTORI SONO EQUIVALENTI")
        return

    out = os.path.join(a.outdir, a.window)
    run(NL, out, a.k_in, a.k_out, a.max_jump, a.min_count,
        a.conn_file, a.levels_file, a.neurons_file, a.engine, a.limit,
        a.budget_mb * 1024 * 1024, a.gzip, a.full_csv,
        a.grouping, a.class_file)


if __name__ == "__main__":
    main()
