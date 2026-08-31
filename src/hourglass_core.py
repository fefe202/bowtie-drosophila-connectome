#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisi hourglass globale del connettoma, secondo Sabrin, Wei, van den Heuvel
& Dovrolis (PLOS Comput Biol 2020).

Si calcola la path centrality di ogni neurone (quanti cammini sorgente-target
lo attraversano), da cui il tau-core: il minimo insieme di neuroni che copre
una frazione tau dei cammini. Il confronto e' con la flat dependency network,
che collega ogni sorgente ai target che raggiunge ignorando la struttura
intermedia. L'H-score e'

    H = 1 - |C(tau)| / |C_f(tau)|

Sorgenti: neuroni sensoriali. Target: motori e discendenti.

Due schemi di routing. Con `levels` un cammino non puo' scendere di livello,
il che rende il routing aciclico ma scarta gli archi laterali, che sono il
39% del connettoma. Con `score` i cammini seguono un punteggio continuo e
recuperano parte di quelli laterali, con un limite esplicito sulla lunghezza.
Il limite e' obbligatorio: senza, il numero di cammini diverge e un solo
neurone arriva a coprire il 97,6% del totale.

Uso:
    python src/hourglass_core.py --tau 0.9 --routing levels
    python src/hourglass_core.py --tau 0.9 --n_random 100 --seed 42
"""

from thesis_paths import data, results

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse


# ============================================================================
#  UTILITY
# ============================================================================

def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"  [{timestamp()}] {msg}", flush=True)


def fmt(x):
    """Formatta numeri potenzialmente enormi (i cammini in un DAG esplodono)."""
    if x == 0:
        return "0"
    if abs(x) >= 1e7:
        return f"{x:.4e}"
    return f"{x:,.0f}"


# ============================================================================
#  1. CARICAMENTO E COSTRUZIONE DEL DAG FEEDFORWARD
# ============================================================================

def load_and_build(conn_file, levels_file, class_file, neurons_file,
                   min_syn, routing):
    """
    Costruisce il DAG feedforward neurone-per-neurone.

    Returns
    -------
    A          : scipy.sparse.csr_matrix  (n x n) binaria, DAG
    meta       : pandas.DataFrame indicizzata per idx con root_id/level/score/
                 super_class/group
    """
    log("Caricamento CSV...")
    t0 = time.time()
    df_conn = pd.read_csv(
        conn_file,
        usecols=["pre_root_id", "post_root_id", "syn_count"],
        dtype={"pre_root_id": np.int64, "post_root_id": np.int64,
               "syn_count": np.int32},
    )
    df_levels = pd.read_csv(levels_file, usecols=["root_id", "score", "y_level"])
    df_class = pd.read_csv(class_file, usecols=["root_id", "super_class"])
    df_neurons = pd.read_csv(neurons_file, usecols=["root_id", "group"])
    log(f"  connections : {len(df_conn):,} righe")
    log(f"  levels      : {len(df_levels):,} righe")
    log(f"  caricati in {time.time()-t0:.1f}s")

    # --- pesi: somma delle sinapsi sulle diverse neuropile della stessa coppia
    log("Aggregazione dei pesi sinaptici per coppia (pre, post)...")
    df_edges = (
        df_conn.groupby(["pre_root_id", "post_root_id"], sort=False)["syn_count"]
        .sum()
        .reset_index()
    )
    n_pairs_all = len(df_edges)
    df_edges = df_edges[df_edges["syn_count"] >= min_syn]
    log(f"  coppie distinte: {n_pairs_all:,}  ->  {len(df_edges):,} "
        f"con syn_count >= {min_syn}")

    # --- universo dei nodi: neuroni con livello assegnato
    df_meta = df_levels.dropna(subset=["y_level"]).copy()
    df_meta["y_level"] = df_meta["y_level"].astype(int)
    df_meta = df_meta.merge(df_class, on="root_id", how="left")
    df_meta = df_meta.merge(df_neurons, on="root_id", how="left")
    df_meta = df_meta.sort_values("root_id").reset_index(drop=True)

    neuron_to_idx = pd.Series(df_meta.index.values, index=df_meta["root_id"].values)
    n = len(df_meta)
    log(f"Neuroni con livello assegnato: {n:,}")

    # --- mappatura archi
    pre_idx = df_edges["pre_root_id"].map(neuron_to_idx)
    post_idx = df_edges["post_root_id"].map(neuron_to_idx)
    keep = pre_idx.notna() & post_idx.notna()
    pre_idx = pre_idx[keep].to_numpy(dtype=np.int64)
    post_idx = post_idx[keep].to_numpy(dtype=np.int64)
    log(f"Archi con entrambi gli estremi nel grafo: {len(pre_idx):,}")

    # --- filtro feedforward => DAG
    level = df_meta["y_level"].to_numpy()
    score = df_meta["score"].to_numpy()
    if routing == "levels":
        ff = level[pre_idx] < level[post_idx]
    elif routing == "score":
        ff = score[pre_idx] > score[post_idx]
    else:
        raise ValueError(f"routing sconosciuto: {routing}")

    n_ff = int(ff.sum())
    n_lat = int((level[pre_idx] == level[post_idx]).sum())
    n_fb = int((level[pre_idx] > level[post_idx]).sum())
    log(f"Classificazione archi (per livello):  FF={n_ff if routing=='levels' else len(pre_idx)-n_lat-n_fb:,}"
        f"  lateral={n_lat:,}  FB={n_fb:,}")
    log(f"Archi mantenuti nel DAG (routing={routing}): {n_ff:,}")

    A = sparse.csr_matrix(
        (np.ones(n_ff, dtype=np.float64), (pre_idx[ff], post_idx[ff])),
        shape=(n, n),
    )
    A.sum_duplicates()
    return A, df_meta


def build_layers(A, level, routing):
    """
    Ordinamento topologico a blocchi.
    routing='levels': i livelli sono gia' un layering topologico valido.
    routing='score' : layering di Kahn (longest-path depth).
    """
    n = A.shape[0]
    if routing == "levels":
        layers = [np.flatnonzero(level == l) for l in sorted(np.unique(level))]
        return [idx for idx in layers if len(idx) > 0]

    # Kahn layering
    indeg = np.asarray(A.sum(axis=0)).ravel()
    remaining = np.ones(n, dtype=bool)
    layers = []
    while remaining.any():
        frontier = remaining & (indeg <= 1e-9)
        if not frontier.any():
            raise RuntimeError("Ciclo rilevato: il grafo non e' un DAG.")
        idx = np.flatnonzero(frontier)
        layers.append(idx)
        remaining[idx] = False
        indeg -= np.asarray(A[idx, :].sum(axis=0)).ravel()
    return layers


# ============================================================================
#  2. CONTEGGIO ESATTO DEI CAMMINI (flow vector su DAG)
# ============================================================================

class DAGPathCounter:
    """
    Conta ESATTAMENTE i cammini S->T di lunghezza <= max_hops in un DAG,
    tramite propagazione a blocchi topologici (prodotti matrice-vettore sparsi).

    Il vincolo sulla lunghezza dei cammini e' richiesto dal paper (p.6,
    principio 4: la bassa affidabilita' di scarica dei neuroni rende
    implausibili cammini lunghi; gli schemi P_4 e P_5 limitano a 4-5 hop).
    Senza vincolo il conteggio e' dominato dai cammini lunghissimi e degenera.

    Si tiene traccia della lunghezza esatta:
      a[v, h] = # cammini di ESATTAMENTE h hop da una sorgente a v  (h=0 sse v in S)
      b[v, h] = # cammini di ESATTAMENTE h hop da v a un target     (h=0 sse v in T)
    e la path centrality (paper p.8) e':
      P(v) = sum_{h1 + h2 <= max_hops} a[v,h1] * b[v,h2]

    Poiche' il grafo e' un DAG, ogni "walk" e' un cammino semplice: il
    conteggio e' esatto, non approssimato.
    Costo per propagazione: O(max_hops * |E|).
    """

    def __init__(self, A, layers, max_hops):
        self.n = A.shape[0]
        self.layers = layers
        self.H = max_hops
        A_csr = A.tocsr()
        A_csc = A.tocsc()
        # blocco degli archi ENTRANTI nei nodi del layer k  -> shape (|L_k|, n)
        self.in_blocks = [A_csc[:, idx].T.tocsr() for idx in layers]
        # blocco degli archi USCENTI dai nodi del layer k   -> shape (|L_k|, n)
        self.out_blocks = [A_csr[idx, :] for idx in layers]

    def forward(self, src_mask, removed=None):
        """a[v, h] = # cammini di h hop da una sorgente a v."""
        a = np.zeros((self.n, self.H + 1), dtype=np.float64)
        for k, idx in enumerate(self.layers):
            inc = self.in_blocks[k] @ a
            val = np.zeros((len(idx), self.H + 1), dtype=np.float64)
            val[:, 1:] = inc[:, :-1]          # un hop in piu'
            val[:, 0] = src_mask[idx]
            if removed is not None:
                val[removed[idx], :] = 0.0
            a[idx] = val
        return a

    def backward(self, tgt_mask, removed=None):
        """b[v, h] = # cammini di h hop da v a un target."""
        b = np.zeros((self.n, self.H + 1), dtype=np.float64)
        for k in range(len(self.layers) - 1, -1, -1):
            idx = self.layers[k]
            out = self.out_blocks[k] @ b
            val = np.zeros((len(idx), self.H + 1), dtype=np.float64)
            val[:, 1:] = out[:, :-1]
            val[:, 0] = tgt_mask[idx]
            if removed is not None:
                val[removed[idx], :] = 0.0
            b[idx] = val
        return b

    def centrality(self, a, b):
        """P(v) = sum_{h1+h2 <= H} a[v,h1] * b[v,h2]."""
        b_cum = np.cumsum(b, axis=1)          # b_cum[v,m] = # cammini v->T di <= m hop
        P = np.zeros(self.n, dtype=np.float64)
        for h1 in range(self.H + 1):
            P += a[:, h1] * b_cum[:, self.H - h1]
        return P

    @staticmethod
    def n_paths_to(a, tgt_idx):
        """Numero totale di cammini S->T (di lunghezza <= H) che finiscono in T."""
        return float(a[tgt_idx, :].sum())


# ============================================================================
#  3. tau-CORE (greedy del paper, p.9)
# ============================================================================

def tau_core(dp, src_mask, tgt_mask, tgt_idx, tau, max_core, verbose=True):
    """
    Minimo insieme di nodi che copre una frazione tau dei cammini S->T.
    Ad ogni passo: nodo con path centrality massima, rimozione dei suoi
    cammini, ricalcolo.
    """
    removed = np.zeros(dp.n, dtype=bool)
    a = dp.forward(src_mask)
    b = dp.backward(tgt_mask)
    total_paths = dp.n_paths_to(a, tgt_idx)
    if total_paths <= 0:
        raise RuntimeError("Nessun cammino S->T: controlla sorgenti/target.")

    core, records, coverage = [], [], 0.0
    while coverage < tau and len(core) < max_core:
        P = dp.centrality(a, b)
        P[removed] = -1.0
        v = int(np.argmax(P))
        if P[v] <= 0:
            break
        pv = float(P[v])
        removed[v] = True
        core.append(v)

        a = dp.forward(src_mask, removed)
        b = dp.backward(tgt_mask, removed)
        remaining = dp.n_paths_to(a, tgt_idx)
        new_cov = 1.0 - remaining / total_paths
        records.append({
            "rank": len(core),
            "node_idx": v,
            "path_centrality": pv,
            "marginal_coverage": new_cov - coverage,
            "cumulative_coverage": new_cov,
        })
        coverage = new_cov
        if verbose and (len(core) <= 25 or len(core) % 25 == 0):
            log(f"    core #{len(core):>4}  P={fmt(pv)}  coverage={coverage:.4f}")

    return core, records, total_paths, coverage


# ============================================================================
#  4. CORE DELLA FLAT DEPENDENCY NETWORK (paper p.9)
# ============================================================================

def flat_core(dp, src_mask, tgt_mask, tau, max_core, verbose=True):
    """
    G_f: solo sorgenti e target, arco s->t di peso W(s,t) = # cammini s->t.
    Il core e' un set-cover greedy sui cammini; ogni nodo scelto (sorgente o
    target) copre tutti i cammini a lui incidenti.

    Guadagno marginale, calcolato con 2 sole propagazioni per iterazione:
      gain(s) = # cammini da s ai target ANCORA scoperti  = b_{T_res}(s)
      gain(t) = # cammini dalle sorgenti ANCORA scoperte a t = a_{S_res}(t)
    """
    res_src = src_mask.copy()   # sorgenti non ancora scelte
    res_tgt = tgt_mask.copy()   # target non ancora scelti

    a = dp.forward(res_src)
    b = dp.backward(res_tgt)
    total_paths = float((a.sum(axis=1) * tgt_mask).sum())

    core, coverage, curve = [], 0.0, []
    while coverage < tau and len(core) < max_core:
        gain_s = b.sum(axis=1) * res_src
        gain_t = a.sum(axis=1) * res_tgt
        i_s = int(np.argmax(gain_s))
        i_t = int(np.argmax(gain_t))
        if gain_s[i_s] >= gain_t[i_t]:
            v, kind, g = i_s, "source", float(gain_s[i_s])
            res_src[v] = 0.0
        else:
            v, kind, g = i_t, "target", float(gain_t[i_t])
            res_tgt[v] = 0.0
        if g <= 0:
            break
        core.append((v, kind))

        a = dp.forward(res_src)
        b = dp.backward(res_tgt)
        remaining = float((a.sum(axis=1) * res_tgt).sum())
        coverage = 1.0 - remaining / total_paths
        curve.append(coverage)
        if verbose and (len(core) <= 10 or len(core) % 100 == 0):
            log(f"    flat-core #{len(core):>5} ({kind})  coverage={coverage:.4f}")

    return core, coverage, curve


# ============================================================================
#  5. MODELLO NULLO (paper p.10-11, Fig. 6)
# ============================================================================

def randomize_preserving_indegree_and_order(A, level, rng):
    """
    Per ogni nodo v: si preserva l'in-degree e si ricampionano i predecessori
    tra i nodi a livello strettamente inferiore (analogo dell'insieme degli
    antenati A(v) del paper). L'out-degree cambia, come nel paper.
    """
    n = A.shape[0]
    A_csc = A.tocsc()
    indeg = np.diff(A_csc.indptr)

    levels_sorted = np.sort(np.unique(level))
    pools = {}
    for l in levels_sorted:
        pools[l] = np.flatnonzero(level < l)

    rows, cols = [], []
    for v in range(n):
        d = indeg[v]
        if d == 0:
            continue
        pool = pools[level[v]]
        if len(pool) == 0:
            continue
        if d >= len(pool):
            chosen = pool
        elif d * 8 < len(pool):
            # campionamento veloce con rifiuto dei duplicati
            cand = rng.integers(0, len(pool), size=d * 2)
            chosen = pool[np.unique(cand)[:d]]
            if len(chosen) < d:
                chosen = pool[rng.choice(len(pool), size=d, replace=False)]
        else:
            chosen = pool[rng.choice(len(pool), size=d, replace=False)]
        rows.append(chosen)
        cols.append(np.full(len(chosen), v, dtype=np.int64))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n, n)
    )


# ============================================================================
#  6. PIPELINE
# ============================================================================

def run_hourglass(A, level, routing, src_mask, tgt_mask, tgt_idx, tau,
                  max_core, max_flat_core, max_hops, verbose=True):
    layers = build_layers(A, level, routing)
    dp = DAGPathCounter(A, layers, max_hops)
    core, records, total_paths, cov = tau_core(
        dp, src_mask, tgt_mask, tgt_idx, tau, max_core, verbose)
    fcore, fcov, fcurve = flat_core(dp, src_mask, tgt_mask, tau,
                                    max_flat_core, verbose)
    C = len(core)
    Cf = len(fcore)
    H = 1.0 - C / Cf if Cf > 0 else float("nan")
    return {
        "core": core, "records": records, "total_paths": total_paths,
        "coverage": cov, "flat_core": fcore, "flat_coverage": fcov,
        "core_curve": [r["cumulative_coverage"] for r in records],
        "flat_curve": fcurve,
        "C": C, "Cf": Cf, "H": H, "dp": dp,
    }


def core_size_at(curve, tau):
    """
    Dimensione del core necessaria a coprire una frazione tau dei cammini,
    letta dalla curva di copertura cumulativa di UNA sola esecuzione del
    greedy. Restituisce None se il greedy si e' fermato prima di tau.
    """
    for i, c in enumerate(curve):
        if c >= tau:
            return i + 1
    return None


def hscore_curve(res, taus):
    """H(tau) = 1 - C(tau)/C_f(tau) per una griglia di tau (paper Fig. 7B)."""
    out = []
    for t in taus:
        C = core_size_at(res["core_curve"], t)
        Cf = core_size_at(res["flat_curve"], t)
        H = 1.0 - C / Cf if (C is not None and Cf) else None
        out.append({"tau": t, "C": C, "Cf": Cf, "H": H})
    return out


# ============================================================================
#  6-bis. SELF-TEST: conteggio esatto vs enumerazione esaustiva
# ============================================================================

def _brute_force_paths(A, S, T, max_hops):
    """
    Enumera esplicitamente tutti i cammini semplici S->T di <= max_hops archi.
    Riferimento di verita' per validare il conteggio algebrico.
    Restituisce (numero_totale, dict nodo -> numero di cammini che lo contengono).
    """
    n = A.shape[0]
    succ = [A.indices[A.indptr[u]:A.indptr[u + 1]].tolist() for u in range(n)]
    total = 0
    through = defaultdict(int)

    def dfs(path, visited):
        nonlocal total
        u = path[-1]
        if u in T and len(path) > 1:
            total += 1
            for w in path:
                through[w] += 1
        if len(path) - 1 >= max_hops:
            return
        for w in succ[u]:
            if w not in visited:
                dfs(path + [w], visited | {w})

    for s in S:
        dfs([s], {s})
    return total, through


def selftest(n_trials=150, seed=0):
    """
    Verifica su DAG casuali che:
      1. il numero totale di cammini S->T calcolato algebricamente coincida
         con l'enumerazione esaustiva;
      2. la path centrality P(v) coincida, per ogni nodo, col numero di
         cammini che lo attraversano;
      3. dopo la rimozione di un nodo, il conteggio residuo sia corretto.
    """
    rng = np.random.default_rng(seed)
    n_ok = 0
    for t in range(n_trials):
        n = int(rng.integers(5, 13))
        perm = rng.permutation(n)
        pos = np.empty(n, dtype=int)
        pos[perm] = np.arange(n)

        rows, cols = [], []
        for u in range(n):
            for v in range(n):
                if pos[u] < pos[v] and rng.random() < 0.35:
                    rows.append(u)
                    cols.append(v)
        if not rows:
            continue
        A = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()

        perm_nodes = rng.permutation(n)
        n_s = int(rng.integers(1, max(2, n // 3) + 1))
        n_t = int(rng.integers(1, max(2, n // 3) + 1))
        S = set(perm_nodes[:n_s].tolist())
        T = set(perm_nodes[n_s:n_s + n_t].tolist())
        if not S or not T:
            continue

        max_hops = int(rng.integers(2, 6))
        src_mask = np.zeros(n); src_mask[list(S)] = 1.0
        tgt_mask = np.zeros(n); tgt_mask[list(T)] = 1.0
        tgt_idx = np.array(sorted(T))

        layers = build_layers(A, np.zeros(n, dtype=int), "score")
        dp = DAGPathCounter(A, layers, max_hops)
        a = dp.forward(src_mask)
        b = dp.backward(tgt_mask)
        algebraic_total = dp.n_paths_to(a, tgt_idx)
        algebraic_P = dp.centrality(a, b)

        bf_total, bf_through = _brute_force_paths(A, S, T, max_hops)

        assert abs(algebraic_total - bf_total) < 1e-6, (
            f"trial {t}: totale {algebraic_total} != atteso {bf_total}")
        for v in range(n):
            assert abs(algebraic_P[v] - bf_through.get(v, 0)) < 1e-6, (
                f"trial {t}: P({v}) = {algebraic_P[v]} != {bf_through.get(v, 0)}")

        # 3. conteggio residuo dopo la rimozione di un nodo
        if n > 2:
            v_rm = int(rng.integers(0, n))
            removed = np.zeros(n, dtype=bool)
            removed[v_rm] = True
            a_r = dp.forward(src_mask, removed)
            algebraic_rem = dp.n_paths_to(a_r, tgt_idx)
            A_r = A.copy().tolil()
            A_r[v_rm, :] = 0
            A_r[:, v_rm] = 0
            bf_rem, _ = _brute_force_paths(
                A_r.tocsr(), S - {v_rm}, T - {v_rm}, max_hops)
            assert abs(algebraic_rem - bf_rem) < 1e-6, (
                f"trial {t}: residuo {algebraic_rem} != atteso {bf_rem}")
        n_ok += 1

    print(f"  [SELF-TEST] {n_ok} DAG casuali verificati contro enumerazione "
          f"esaustiva: conteggio totale, path centrality per nodo e conteggio "
          f"residuo dopo rimozione COINCIDONO esattamente.")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Analisi hourglass globale (tau-core / H-score) "
                    "secondo Sabrin & Dovrolis 2020.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--tau", type=float, default=0.9,
                    help="Soglia di copertura dei cammini (default: 0.9, come nel paper)")
    ap.add_argument("--min_syn", type=int, default=5,
                    help="Soglia minima di sinapsi per coppia (default: 5, standard FlyWire)")
    ap.add_argument("--routing", choices=["levels", "score"], default="levels",
                    help="Schema di routing feedforward (default: levels)")
    ap.add_argument("--max_hops", type=int, default=5,
                    help="Lunghezza massima dei cammini S->T in hop "
                         "(default: 5, analogo dello schema P_5 del paper). "
                         "Con --routing levels il massimo strutturale e' 4.")
    ap.add_argument("--sources", default="sensory",
                    help="super_class delle sorgenti, separate da virgola (default: sensory)")
    ap.add_argument("--targets", default="motor,descending",
                    help="super_class dei target (default: motor,descending)")
    ap.add_argument("--max_core", type=int, default=300)
    ap.add_argument("--max_flat_core", type=int, default=4000)
    ap.add_argument("--n_random", type=int, default=0,
                    help="Numero di reti randomizzate per il p-value (default: 0 = salta)")
    ap.add_argument("--tau_null", type=float, default=None,
                    help="tau usato per il confronto col modello nullo. Sulle reti "
                         "randomizzate il core esplode e il greedy diventa lentissimo "
                         "a tau alti: usare 0.5-0.6 (il paper mostra H(tau) da 0.5). "
                         "Deve essere <= --tau. Default: uguale a --tau.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true",
                    help="Verifica il conteggio dei cammini contro enumerazione "
                         "esaustiva su DAG casuali, poi esce.")
    ap.add_argument("--outdir", default=results("hourglass_core_results"))
    ap.add_argument("--conn_file", default=data("connections.csv"))
    ap.add_argument("--levels_file", default=data("COORDINATE_XY_with_levels_tree.csv"))
    ap.add_argument("--class_file", default=data("classification.csv"))
    ap.add_argument("--neurons_file", default=data("neurons.csv"))
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    t_start = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print("  HOURGLASS ANALYSIS - Drosophila connectome")
    print("  Framework: Sabrin, Wei, van den Heuvel & Dovrolis, PLOS CB 2020")
    print("=" * 78)
    log(f"tau={args.tau}  min_syn={args.min_syn}  routing={args.routing}  "
        f"max_hops={args.max_hops}")
    tag = f"tau{args.tau}_syn{args.min_syn}_{args.routing}_h{args.max_hops}"
    print("-" * 78)

    # ---- 1. grafo
    A, meta = load_and_build(args.conn_file, args.levels_file, args.class_file,
                             args.neurons_file, args.min_syn, args.routing)
    level = meta["y_level"].to_numpy()

    # ---- 2. sorgenti e target
    src_classes = [s.strip() for s in args.sources.split(",")]
    tgt_classes = [s.strip() for s in args.targets.split(",")]
    sc = meta["super_class"].fillna("NA").to_numpy()
    is_src = np.isin(sc, src_classes)
    is_tgt = np.isin(sc, tgt_classes)
    overlap = is_src & is_tgt
    if overlap.any():
        log(f"[WARN] {overlap.sum()} neuroni sia sorgente sia target: rimossi dai target")
        is_tgt &= ~overlap

    src_mask = is_src.astype(np.float64)
    tgt_mask = is_tgt.astype(np.float64)
    tgt_idx = np.flatnonzero(is_tgt)
    log(f"Sorgenti S ({'+'.join(src_classes)}): {int(is_src.sum()):,}")
    log(f"Target   T ({'+'.join(tgt_classes)}): {int(is_tgt.sum()):,}")

    # ---- 3. analisi
    print()
    log("Calcolo tau-core e flat-core sulla rete reale...")
    res = run_hourglass(A, level, args.routing, src_mask, tgt_mask, tgt_idx,
                        args.tau, args.max_core, args.max_flat_core,
                        args.max_hops)

    print()
    print("=" * 78)
    print("  RISULTATI")
    print("=" * 78)
    print(f"  Cammini S->T totali (esatti):     {fmt(res['total_paths'])}")
    print(f"  |C({args.tau})|   core della rete reale : {res['C']:,}"
          f"   (copertura {res['coverage']:.4f})")
    print(f"  |Cf({args.tau})|  core della flat network: {res['Cf']:,}"
          f"   (copertura {res['flat_coverage']:.4f})")
    print(f"  H-score = 1 - C/Cf              : {res['H']:.4f}")
    print(f"  [riferimento C. elegans, Tab.2] : H = 0.74 - 0.87")

    # curva H(tau), ricavata gratis dalla stessa esecuzione (paper Fig. 7B)
    taus = [t for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95) if t <= args.tau + 1e-9]
    curve = hscore_curve(res, taus)
    print()
    print("  [CURVA H(tau)]")
    print(f"    {'tau':>6} {'C(tau)':>8} {'Cf(tau)':>9} {'H(tau)':>8}")
    for r in curve:
        Hs = f"{r['H']:.4f}" if r["H"] is not None else "n/d"
        print(f"    {r['tau']:>6.2f} {str(r['C']):>8} {str(r['Cf']):>9} {Hs:>8}")
    # Esportata su file: le figure la leggono da qui invece di avere i
    # valori scritti dentro il codice, che mentirebbero al primo rilancio
    # con parametri diversi.
    pd.DataFrame(curve).to_csv(
        os.path.join(args.outdir, f"hscore_curve_{tag}.csv"), index=False)

    # ---- 4. annotazione dei neuroni del core
    rows = []
    for r in res["records"]:
        i = r["node_idx"]
        rows.append({
            "rank": r["rank"],
            "root_id": int(meta.at[i, "root_id"]),
            "group": meta.at[i, "group"],
            "y_level": int(meta.at[i, "y_level"]),
            "super_class": meta.at[i, "super_class"],
            "path_centrality": r["path_centrality"],
            "marginal_coverage": r["marginal_coverage"],
            "cumulative_coverage": r["cumulative_coverage"],
        })
    df_core = pd.DataFrame(rows)
    core_path = os.path.join(args.outdir, f"core_neurons_{tag}.csv")
    df_core.to_csv(core_path, index=False)

    # aggregazione per (group, level): confronto diretto con i waist dei motif
    df_grp = (
        df_core.groupby(["group", "y_level"], dropna=False)
        .agg(n_core_neurons=("root_id", "size"),
             coverage=("marginal_coverage", "sum"),
             best_rank=("rank", "min"))
        .reset_index()
        .sort_values("coverage", ascending=False)
    )
    grp_path = os.path.join(args.outdir, f"core_by_group_{tag}.csv")
    df_grp.to_csv(grp_path, index=False)

    print()
    print("  [COMPOSIZIONE DEL tau-CORE PER GRUPPO DI CONNETTIVITA']")
    print(df_grp.head(20).to_string(index=False))
    print()
    print("  [PRIMI 20 NEURONI DEL CORE]")
    with pd.option_context("display.width", 200, "display.max_colwidth", 30):
        print(df_core.head(20).to_string(index=False))

    summary = {
        "tau": args.tau, "min_syn": args.min_syn, "routing": args.routing,
        "max_hops": args.max_hops,
        "sources": src_classes, "targets": tgt_classes,
        "n_nodes": int(A.shape[0]), "n_dag_edges": int(A.nnz),
        "n_sources": int(is_src.sum()), "n_targets": int(is_tgt.sum()),
        "total_paths": res["total_paths"],
        "C": res["C"], "Cf": res["Cf"], "H": res["H"],
        "coverage": res["coverage"], "flat_coverage": res["flat_coverage"],
    }

    # ---- 5. modello nullo
    if args.n_random > 0:
        print()
        log(f"Modello nullo: {args.n_random} reti randomizzate "
            f"(in-degree + ordinamento parziale preservati)...")
        tau_null = args.tau_null if args.tau_null else args.tau
        H_real_null = next(r["H"] for r in hscore_curve(res, [tau_null]))
        if H_real_null is None:
            log("[ERROR] tau_null non raggiunto dalla rete reale.")
            sys.exit(1)
        log(f"Confronto a tau_null={tau_null} (H reale = {H_real_null:.4f})")
        rng = np.random.default_rng(args.seed)
        H_rand = []
        for r in range(args.n_random):
            t0 = time.time()
            A_r = randomize_preserving_indegree_and_order(A, level, rng)
            res_r = run_hourglass(A_r, level, args.routing, src_mask, tgt_mask,
                                  tgt_idx, tau_null, args.max_core,
                                  args.max_flat_core, args.max_hops,
                                  verbose=False)
            H_rand.append(res_r["H"])
            log(f"  random {r+1}/{args.n_random}: C={res_r['C']} "
                f"Cf={res_r['Cf']} H={res_r['H']:.4f}  ({time.time()-t0:.1f}s)")
        H_rand = np.array(H_rand)
        p_emp = (np.sum(H_rand >= H_real_null) + 1) / (len(H_rand) + 1)
        print()
        print(f"  tau usato    : {tau_null}")
        print(f"  H reale      : {H_real_null:.4f}")
        print(f"  H random     : {H_rand.mean():.4f} +/- {H_rand.std():.4f}")
        print(f"  p-value emp. : {p_emp:.4g}   (paper C. elegans: p < 1e-3)")
        summary.update({
            "tau_null": tau_null,
            "H_real_at_tau_null": float(H_real_null),
            "H_random_mean": float(H_rand.mean()),
            "H_random_std": float(H_rand.std()),
            "H_random_values": H_rand.tolist(),
            "p_empirical": float(p_emp),
            "n_random": args.n_random,
        })
        pd.DataFrame({"H_random": H_rand}).to_csv(
            os.path.join(args.outdir, f"hscore_random_{tag}.csv"), index=False)

    sum_path = os.path.join(args.outdir, f"hscore_summary_{tag}.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"  [OUTPUT] {core_path}")
    print(f"  [OUTPUT] {grp_path}")
    print(f"  [OUTPUT] {sum_path}")
    print(f"\n  Tempo totale: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
