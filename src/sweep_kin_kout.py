#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reti integrative, di broadcast e miste: sweep di k_in e k_out.

L'asimmetria fra i due parametri distingue i tre regimi: (k,1) integrativa,
(1,k) broadcast, bilanciata mista.

La somma su tutte le k-combinazioni di un prodotto e' il polinomio simmetrico
elementare e_k, calcolabile in O(p*k) per neurone: nessuna combinazione viene
materializzata e il costo non dipende dal loro numero. Per le forme pure la
disgiunzione richiede un solo termine di inclusione-esclusione, quindi il
conteggio e' esatto.

L'indice grezzo e' confuso con il numero di gruppi disponibili, perche' e_k
su p gruppi cresce come C(p,k). Va usato indice_norm, normalizzato per le
combinazioni disponibili.

Uso:
    python src/sweep_kin_kout.py --selftest
    python src/sweep_kin_kout.py --window 1-2-3 --kmax 5
"""

from thesis_paths import data, results

import argparse
import os
import time
from collections import defaultdict
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse

from hourglass_areas_fast import build_graph
from hourglass_areas import timestamp, format_elapsed, to_dense_1d


def log(msg):
    print(f"  [{timestamp()}] {msg}", flush=True)


def elem_sym(V, kmax):
    """
    Polinomi simmetrici elementari e_0..e_kmax delle righe di V.
    V ha forma (n_gruppi, n_neuroni); restituisce una lista di kmax+1 vettori
    di lunghezza n_neuroni. Costo O(n_gruppi * kmax * n_neuroni).
    """
    n = V.shape[1]
    E = [np.ones(n)] + [np.zeros(n) for _ in range(kmax)]
    for i in range(V.shape[0]):
        vi = V[i]
        for k in range(kmax, 0, -1):
            E[k] = E[k] + vi * E[k - 1]
    return E


def elem_sym_without(E, v_g, kmax):
    """
    e_k dell'insieme PRIVATO dell'elemento g, dalla ricorrenza di cancellazione
        e_k(S\\g) = e_k(S) - v_g * e_{k-1}(S\\g)
    """
    out = [np.ones_like(v_g)]
    for k in range(1, kmax + 1):
        out.append(E[k] - v_g * out[k - 1])
    return out


def counts_pure(Vin, Vout, shared_pairs, k, mode):
    """
    Conteggio ESATTO, con disgiunzione, delle forme pure:
      mode='int'  -> (k_in=k, k_out=1)   integrativa
      mode='bro'  -> (k_in=1, k_out=k)   broadcast

    shared_pairs: lista di (indice_in, indice_out) dei gruppi presenti in
    entrambe le liste.
    """
    if mode == "int":
        A, B, kA, kB = Vin, Vout, k, 1
    else:
        A, B, kA, kB = Vin, Vout, 1, k
    if A.shape[0] < kA or B.shape[0] < kB:
        return 0.0

    EA = elem_sym(A, kA)
    EB = elem_sym(B, kB)
    tot = EA[kA] * EB[kB]

    # correzione |G| = 1: un gruppo che compare in entrambi i rami
    for ia, ib in shared_pairs:
        va, wb = A[ia], B[ib]
        EA_g = elem_sym_without(EA, va, kA)
        EB_g = elem_sym_without(EB, wb, kB)
        tot = tot - va * wb * EA_g[kA - 1] * EB_g[kB - 1]
    return float(np.maximum(tot, 0).sum())


def selftest(n_trials=200, seed=3):
    """Confronta la formula con l'enumerazione esplicita delle combinazioni."""
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(n_trials):
        p = int(rng.integers(2, 7))
        q = int(rng.integers(2, 7))
        n = int(rng.integers(1, 5))
        n_sh = int(rng.integers(0, min(p, q) + 1))
        Vin = rng.integers(0, 4, size=(p, n)).astype(float)
        Vout = rng.integers(0, 4, size=(q, n)).astype(float)
        shared = [(i, i) for i in range(n_sh)]      # i primi n_sh coincidono
        for k in (1, 2, 3):
            for mode in ("int", "bro"):
                kA, kB = (k, 1) if mode == "int" else (1, k)
                if p < kA or q < kB:
                    continue
                got = counts_pure(Vin, Vout, shared, k, mode)
                exp = 0.0
                sh_in = {a for a, _ in shared}
                sh_out = {b for _, b in shared}
                pair = {a: b for a, b in shared}
                for A in combinations(range(p), kA):
                    for D in combinations(range(q), kB):
                        # disgiunzione: nessun gruppo condiviso in entrambi
                        if any(a in sh_in and pair[a] in D for a in A):
                            continue
                        exp += float((np.prod([Vin[a] for a in A], axis=0) *
                                      np.prod([Vout[d] for d in D], axis=0)).sum())
                if abs(got - exp) > 1e-6:
                    bad += 1
    if bad:
        print(f"  [SELF-TEST] FALLITO: {bad} discrepanze")
        return False
    print(f"  [SELF-TEST] {n_trials} casi casuali verificati contro "
          f"enumerazione esplicita delle combinazioni, per k=1,2,3 e per "
          f"entrambe le forme: i conteggi COINCIDONO esattamente.")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Punto 7: reti integrative, di broadcast e miste.")
    ap.add_argument("--window", default="1-2-3")
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--max_jump", type=int, default=1)
    ap.add_argument("--min_total", type=float, default=1e3,
                    help="Soglia sul totale per entrare nella classifica")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--outdir", default=results("sweep_kin_kout"))
    ap.add_argument("--conn_file", default=data("connections.csv"))
    ap.add_argument("--levels_file",
                    default=data("COORDINATE_XY_with_levels_tree.csv"))
    ap.add_argument("--neurons_file", default=data("neurons.csv"))
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    NL = [int(x) for x in a.window.split("-")]

    print("=" * 78)
    print("  PUNTO 7 - RETI INTEGRATIVE, DI BROADCAST, MISTE")
    print("=" * 78)
    print("  (k,1) = integrativa   (1,k) = broadcast   confronto per ogni k")
    log(f"finestra {NL}, k fino a {a.kmax}")

    groups, group_keys, submat, incoming, outgoing, n_tot, n_edges = \
        build_graph(NL, a.max_jump, a.conn_file, a.levels_file, a.neurons_file)
    log(f"{n_tot:,} neuroni, {len(group_keys)} metanodi")

    rows = []
    for bn in group_keys:
        in_nbrs, out_nbrs = incoming.get(bn, []), outgoing.get(bn, [])
        if not in_nbrs or not out_nbrs:
            continue
        n_bn = len(groups[bn])
        Vin = np.array([to_dense_1d(np.ones(len(groups[a_])) @ submat[(a_, bn)])
                        for a_ in in_nbrs])
        Vout = np.array([
            to_dense_1d(np.asarray(submat[(bn, d)].sum(axis=1)).ravel()
                        if sparse.issparse(submat[(bn, d)])
                        else submat[(bn, d)].sum(axis=1))
            for d in out_nbrs])
        pos_out = {k_: i for i, k_ in enumerate(out_nbrs)}
        shared = [(i, pos_out[k_]) for i, k_ in enumerate(in_nbrs)
                  if k_ in pos_out]

        r = {"waist": f"{bn[0]}L{bn[1]}", "area": bn[0], "livello": bn[1],
             "n_waist": n_bn, "n_gruppi_in": len(in_nbrs),
             "n_gruppi_out": len(out_nbrs)}
        for k in range(2, a.kmax + 1):
            r[f"int_k{k}"] = counts_pure(Vin, Vout, shared, k, "int")
            r[f"bro_k{k}"] = counts_pure(Vin, Vout, shared, k, "bro")
        rows.append(r)

    df = pd.DataFrame(rows)
    ks = list(range(2, a.kmax + 1))
    df["tot_int"] = df[[f"int_k{k}" for k in ks]].sum(axis=1)
    df["tot_bro"] = df[[f"bro_k{k}" for k in ks]].sum(axis=1)
    df["totale"] = df["tot_int"] + df["tot_bro"]

    # Indice grezzo, confronto diretto dei conteggi. E' confuso con il
    # numero di gruppi entranti e uscenti, perche' e_k su p gruppi cresce
    # come C(p,k): un metanodo con molti gruppi in ingresso risulta
    # integrativo per pura combinatoria. Usare indice_norm.
    df["indice"] = np.where(
        df["totale"] > 0,
        (df["tot_int"] - df["tot_bro"]) / df["totale"].replace(0, np.nan), 0.0)

    # --- indice NORMALIZZATO: conteggio medio per combinazione DISPONIBILE.
    # Dividendo per il numero di combinazioni possibili si toglie l'effetto
    # della taglia dei vicinati e resta la sola asimmetria di quanto
    # densamente ciascuna forma e' realizzata. E' l'indice da usare.
    from math import comb as _comb
    p = df["n_gruppi_in"].to_numpy()
    q = df["n_gruppi_out"].to_numpy()
    ti = np.zeros(len(df))
    tb = np.zeros(len(df))
    for k in ks:
        cpk = np.array([_comb(int(x), k) if x >= k else 0 for x in p], float)
        cqk = np.array([_comb(int(x), k) if x >= k else 0 for x in q], float)
        den_i = cpk * q
        den_b = p * cqk
        ti += np.divide(df[f"int_k{k}"], den_i, out=np.zeros(len(df)),
                        where=den_i > 0)
        tb += np.divide(df[f"bro_k{k}"], den_b, out=np.zeros(len(df)),
                        where=den_b > 0)
    df["tot_int_norm"], df["tot_bro_norm"] = ti, tb
    tn = ti + tb
    df["indice_norm"] = np.where(tn > 0, (ti - tb) / np.where(tn > 0, tn, 1), 0.0)
    df["regime"] = pd.cut(df["indice_norm"], [-1.01, -0.33, 0.33, 1.01],
                          labels=["broadcast", "mista", "integrativa"])
    df.to_csv(os.path.join(a.outdir, f"sweep_{a.window}.csv"), index=False)

    sel = df[df["totale"] >= a.min_total].copy()
    log(f"{len(df)} metanodi analizzati, {len(sel)} sopra la soglia "
        f"di {a.min_total:.0e}")

    print("\n  [RIPARTIZIONE DEI REGIMI]")
    vc = sel["regime"].value_counts()
    for reg in ["integrativa", "mista", "broadcast"]:
        n = int(vc.get(reg, 0))
        print(f"    {reg:>12}: {n:>4}  ({100*n/max(len(sel),1):5.1f}%)")

    print("\n  [I 12 METANODI PIU' INTEGRATIVI]")
    cols = ["waist", "n_waist", "n_gruppi_in", "n_gruppi_out",
            "tot_int", "tot_bro", "indice", "indice_norm"]
    v = sel.nlargest(12, "indice_norm")[cols].copy()
    for c in ("tot_int", "tot_bro"):
        v[c] = v[c].map("{:.3e}".format)
    print(v.round(3).to_string(index=False))

    print("\n  [I 12 METANODI PIU' DI BROADCAST]")
    v = sel.nsmallest(12, "indice_norm")[cols].copy()
    for c in ("tot_int", "tot_bro"):
        v[c] = v[c].map("{:.3e}".format)
    print(v.round(3).to_string(index=False))

    print("\n  [REGIME PER AREA]  (media pesata sui metanodi dell'area)")
    ag = sel.groupby("area").apply(
        lambda g: pd.Series({
            "n_metanodi": len(g),
            "indice_norm": np.average(g["indice_norm"], weights=g["totale"]),
            "totale": g["totale"].sum(),
        }), include_groups=False).reset_index()
    ag = ag[ag["n_metanodi"] >= 2].sort_values("indice_norm", ascending=False)
    ag["regime"] = pd.cut(ag["indice_norm"], [-1.01, -0.33, 0.33, 1.01],
                          labels=["broadcast", "mista", "integrativa"])
    with pd.option_context("display.width", 200):
        print(ag.head(10).round(3).to_string(index=False))
        print("    ...")
        print(ag.tail(8).round(3).to_string(index=False))
    ag.to_csv(os.path.join(a.outdir, f"regime_per_area_{a.window}.csv"),
              index=False)

    print(f"\n  [OUTPUT] {a.outdir}/")
    print(f"  Tempo totale: {format_elapsed(time.time()-t0)}")


if __name__ == "__main__":
    main()
