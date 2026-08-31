#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_hourglass.py
=====================
Validazione statistica dei bow-tie motif (convergenza-divergenza) tramite
confronto con un modello nullo a grado conservato.

Metodo (Milo et al., Science 2002 - framework dei network motif):
  1. Si generano N reti randomizzate (degree-preserving edge swap)
  2. Per ogni rete randomizzata, si ricalcola il motif_count di ogni pattern
  3. Z = (N_reale - mean(N_random)) / std(N_random)

ATTRIBUZIONE: l'edge swap a grado conservato e lo Z-score sono il framework
di Milo et al. (2002), NON quello di Sabrin & Dovrolis (2020). Questi ultimi
non usano Z-score sui motif: randomizzano preservando in-degree e ordinamento
parziale, e testano l'H-score con un p-value empirico su 1000 reti. Quella
procedura e' implementata in hourglass_core.py --n_random.

AVVERTENZE STATISTICHE (da tenere presenti nell'interpretare l'output):
  * Il motif_count e' un prodotto di gradi: la sua distribuzione nulla e'
    fortemente non gaussiana. Uno Z-score dell'ordine di 1e5-1e6 NON e' una
    misura di significativita' interpretabile, e' solo il sintomo di una
    sigma empirica minuscola stimata su poche randomizzazioni. Riportare un
    p-value empirico basato sul rango, oppure il rapporto N_reale/N_random.
  * L'edge swap globale distrugge completamente la struttura per gruppi:
    sotto questo nullo qualunque pattern gruppo-specifico risulta
    iper-arricchito. Per rispondere alla domanda "questo bow-tie e'
    arricchito rispetto all'organizzazione anatomica?" serve un nullo che
    preservi anche il numero di connessioni tra coppie (gruppo, livello).
  * Con ~1e7 pattern testati serve una correzione per test multipli (FDR).
  * n_swaps_factor=5 e' basso: la letteratura usa ~100 * |E| per il mixing.
  * rng non ha seed: i risultati non sono riproducibili.

Uso:
    python validate_hourglass.py --input hourglass_results/1-2-3/hourglass_areas_nl1-2-3_kin2_kout2_jump1_top1000.csv --n_random 30
    python validate_hourglass.py --input top1000.csv --n_random 50 --top_k 100
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
import time
import csv
from datetime import datetime
from scipy import sparse
from collections import defaultdict
from itertools import combinations


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def to_dense_1d(v):
    if sparse.issparse(v):
        return np.asarray(v.todense()).flatten()
    elif isinstance(v, np.matrix):
        return np.asarray(v).flatten()
    return np.asarray(v).flatten()


DENSE_THRESHOLD = 250_000


def edge_swap_randomize(adj_matrix, n_swaps_factor=5):
    """
    Degree-preserving randomization via edge swaps.
    Modifica la matrice di adiacenza in-place mantenendo
    la distribuzione di in-degree e out-degree.
    """
    adj_coo = adj_matrix.tocoo()
    rows = adj_coo.row.tolist()
    cols = adj_coo.col.tolist()
    n_edges = len(rows)
    n_nodes = adj_matrix.shape[0]

    # Crea un set di archi per lookup veloce
    edge_set = set(zip(rows, cols))

    n_swaps = int(n_swaps_factor * n_edges)
    rng = np.random.default_rng()

    # Pre-generazione indici random in vettori per evitare l'overhead di chiamate inside loop
    idx1_arr = rng.integers(0, n_edges, size=n_swaps).tolist()
    idx2_arr = rng.integers(0, n_edges, size=n_swaps).tolist()

    for idx_i in range(n_swaps):
        idx1 = idx1_arr[idx_i]
        idx2 = idx2_arr[idx_i]
        if idx1 == idx2:
            continue
        a = rows[idx1]
        b = cols[idx1]
        c = rows[idx2]
        d = cols[idx2]

        # Controlla vincoli
        if a == d or c == b:
            continue
        if a == c or b == d:
            continue
        if (a, d) in edge_set or (c, b) in edge_set:
            continue

        # Esegui lo swap
        edge_set.discard((a, b))
        edge_set.discard((c, d))
        edge_set.add((a, d))
        edge_set.add((c, b))

        rows[idx1] = a
        cols[idx1] = d
        rows[idx2] = c
        cols[idx2] = b

    # Ricostruisci la matrice sparsa
    data = np.ones(n_edges)
    new_adj = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    return new_adj


def compute_hourglass_counts_for_patterns(adj_matrix, groups, patterns, max_jump=1):
    """
    Dato un insieme di pattern hourglass (lista di tuple di chiavi),
    calcola il motif_count per ciascuno sulla matrice di adiacenza data.
    Pre-computa solo le sottomatrici necessarie per questi specifici pattern.
    """
    # 1. Trova le coppie (sorgente, destinazione) necessarie per i pattern
    required_pairs = set()
    for pat in patterns:
        bn_key = pat['bottleneck_key']
        for fi_key in pat['fan_in_keys']:
            required_pairs.add((fi_key, bn_key))
        for fo_key in pat['fan_out_keys']:
            required_pairs.add((bn_key, fo_key))

    # 2. Pre-computa solo le sottomatrici richieste
    submat_map = {}
    for src_key, dst_key in required_pairs:
        if src_key not in groups or dst_key not in groups:
            continue
        src_indices = groups[src_key]
        dst_indices = groups[dst_key]
        if len(src_indices) == 0 or len(dst_indices) == 0:
            continue

        row_submat = adj_matrix[src_indices, :]
        submat = row_submat[:, dst_indices]

        if submat.nnz > 0:
            rows, cols = submat.shape
            if (rows * cols) < DENSE_THRESHOLD:
                submat_map[(src_key, dst_key)] = submat.toarray()
            else:
                submat_map[(src_key, dst_key)] = submat

    # 3. Calcola motif_count per ogni pattern
    counts = []
    for pat in patterns:
        bn_key = pat['bottleneck_key']
        fan_in_keys = pat['fan_in_keys']
        fan_out_keys = pat['fan_out_keys']

        if bn_key not in groups or len(groups[bn_key]) == 0:
            counts.append(0)
            continue

        n_bn = len(groups[bn_key])

        # Vettori fan-in
        prod_in = np.ones(n_bn, dtype=np.float64)
        valid = True
        for a_key in fan_in_keys:
            if (a_key, bn_key) not in submat_map:
                valid = False
                break
            sub = submat_map[(a_key, bn_key)]
            n_a = len(groups[a_key])
            v = np.ones(n_a, dtype=np.float64) @ sub
            prod_in *= to_dense_1d(v)

        if not valid or np.sum(prod_in) == 0:
            counts.append(0)
            continue

        # Vettori fan-out
        product_vector = prod_in.copy()
        for d_key in fan_out_keys:
            if (bn_key, d_key) not in submat_map:
                valid = False
                break
            sub = submat_map[(bn_key, d_key)]
            if sparse.issparse(sub):
                v = np.asarray(sub.sum(axis=1)).flatten()
            else:
                v = sub.sum(axis=1)
            product_vector *= to_dense_1d(v)
            if np.sum(product_vector) == 0:
                valid = False
                break

        if not valid:
            counts.append(0)
        else:
            counts.append(int(np.sum(product_vector)))

    return counts


def validate_hourglass(input_csv, n_random=30, top_k=None, max_jump=1,
                       conn_file="connections.csv",
                       levels_file="COORDINATE_XY_with_levels_tree.csv",
                       neurons_file="neurons.csv",
                       output_csv=None):
    """
    Validazione Z-score degli hourglass.
    """
    start_time = time.time()

    # Load hourglass patterns
    print("=" * 65)
    print("  HOURGLASS VALIDATION (Z-Score)")
    print("=" * 65)
    print(f"  [{timestamp()}] Input:    {input_csv}")
    print(f"  [{timestamp()}] N random: {n_random}")
    print(f"  [{timestamp()}] Top K:    {top_k if top_k else 'ALL'}")
    print("-" * 65)

    df_hg = pd.read_csv(input_csv)
    df_hg = df_hg.sort_values(by='motif_count', ascending=False)
    if top_k:
        df_hg = df_hg.head(top_k)
    n_patterns = len(df_hg)
    print(f"  [{timestamp()}] Patterns to validate: {n_patterns}")

    # Reconstruct patterns
    patterns = []
    for _, row in df_hg.iterrows():
        k_in = int(row['k_in'])
        k_out = int(row['k_out'])
        fan_in_keys = []
        for i in range(k_in):
            fan_in_keys.append((row[f'fan_in_{i}_area'], int(row[f'fan_in_{i}_level'])))
        fan_out_keys = []
        for i in range(k_out):
            fan_out_keys.append((row[f'fan_out_{i}_area'], int(row[f'fan_out_{i}_level'])))
        bn_key = (row['bottleneck_area'], int(row['bottleneck_level']))
        patterns.append({
            'fan_in_keys': tuple(fan_in_keys),
            'bottleneck_key': bn_key,
            'fan_out_keys': tuple(fan_out_keys),
        })

    # Collect all relevant group keys
    all_keys = set()
    for p in patterns:
        all_keys.add(p['bottleneck_key'])
        for k in p['fan_in_keys']:
            all_keys.add(k)
        for k in p['fan_out_keys']:
            all_keys.add(k)

    # Determine NL_WINDOW from patterns
    all_levels = set()
    for k in all_keys:
        all_levels.add(k[1])
    NL_WINDOW = sorted(all_levels)
    TARGET_LEVELS_SET = set(NL_WINDOW)

    # Load data
    print(f"\n  [{timestamp()}] Loading CSV files...")
    t0 = time.time()
    df_conn = pd.read_csv(conn_file)
    df_levels = pd.read_csv(levels_file)
    df_neurons = pd.read_csv(neurons_file)
    print(f"  [{timestamp()}] Loaded in {time.time()-t0:.1f}s")

    # Build neuron mappings
    print(f"  [{timestamp()}] Building neuron mappings...")
    df_neurons_valid = df_neurons.dropna(subset=["group"])
    df_neurons_valid = df_neurons_valid[df_neurons_valid["group"] != "NO_CONS"]
    area_map = df_neurons_valid.set_index("root_id")["group"].to_dict()
    level_map = df_levels.set_index("root_id")["y_level"].to_dict()

    common_ids = set(area_map.keys()) & set(level_map.keys())
    valid_neurons = sorted(
        n for n in common_ids if level_map[n] in TARGET_LEVELS_SET
    )
    neuron_to_idx = {nid: i for i, nid in enumerate(valid_neurons)}
    n_total = len(valid_neurons)
    print(f"  [{timestamp()}] Valid neurons: {n_total:,}")

    # Build groups
    groups = defaultdict(list)
    for n in valid_neurons:
        groups[(area_map[n], level_map[n])].append(neuron_to_idx[n])
    group_keys = sorted(groups.keys())

    # Build adjacency matrix
    print(f"  [{timestamp()}] Building adjacency matrix...")
    t0 = time.time()
    df_edges = df_conn[df_conn["syn_count"] > 0][["pre_root_id", "post_root_id"]]
    df_edges = df_edges.drop_duplicates()
    mask = (
        df_edges["pre_root_id"].isin(neuron_to_idx)
        & df_edges["post_root_id"].isin(neuron_to_idx)
    )
    df_edges = df_edges[mask].copy()
    s_pre = df_edges["pre_root_id"].map(neuron_to_idx).astype(int)
    s_post = df_edges["post_root_id"].map(neuron_to_idx).astype(int)
    data = np.ones(len(s_pre))
    adj_matrix = sparse.csr_matrix(
        (data, (s_pre, s_post)), shape=(n_total, n_total)
    )
    print(f"  [{timestamp()}] Adjacency: {adj_matrix.nnz:,} edges ({time.time()-t0:.1f}s)")

    # Real counts (should match the input CSV)
    print(f"\n  [{timestamp()}] Computing real counts...")
    real_counts = compute_hourglass_counts_for_patterns(
        adj_matrix, groups, patterns, max_jump
    )

    # Random counts
    print(f"  [{timestamp()}] Starting randomization ({n_random} iterations)...")
    random_counts = np.zeros((n_random, n_patterns), dtype=np.float64)

    for r in range(n_random):
        t0 = time.time()
        rand_adj = edge_swap_randomize(adj_matrix, n_swaps_factor=5)
        counts = compute_hourglass_counts_for_patterns(
            rand_adj, groups, patterns, max_jump
        )
        random_counts[r, :] = counts
        elapsed_r = time.time() - t0
        total_elapsed = time.time() - start_time
        eta = total_elapsed / (r + 1) * (n_random - r - 1)
        print(f"    [{timestamp()}] Random {r+1}/{n_random} "
              f"({elapsed_r:.1f}s) | ETA: {eta/60:.1f}min")

    # Compute Z-scores
    print(f"\n  [{timestamp()}] Computing Z-scores...")
    mean_random = random_counts.mean(axis=0)
    std_random = random_counts.std(axis=0)

    z_scores = np.zeros(n_patterns)
    for i in range(n_patterns):
        if std_random[i] > 0:
            z_scores[i] = (real_counts[i] - mean_random[i]) / std_random[i]
        else:
            z_scores[i] = 0.0 if real_counts[i] == mean_random[i] else float('inf')

    # Add Z-score columns to dataframe
    df_hg = df_hg.copy()
    df_hg['real_count_verified'] = real_counts
    df_hg['mean_random'] = mean_random
    df_hg['std_random'] = std_random
    df_hg['z_score'] = z_scores

    # Save
    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_validated{ext}"
    df_hg.to_csv(output_csv, index=False)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 65}")
    print(f"  RESULTS  [{timestamp()}]  (total: {elapsed/60:.1f} min)")
    print(f"{'=' * 65}")
    print(f"  Patterns validated: {n_patterns}")
    print(f"  Z > 2 (significant): {(z_scores > 2).sum()}")
    print(f"  Z > 3 (highly significant): {(z_scores > 3).sum()}")
    print(f"  Z < -2 (under-represented): {(z_scores < -2).sum()}")
    print(f"  Output: {output_csv}")

    # Top 10
    print(f"\n  [TOP 10 BY Z-SCORE]")
    df_sorted = df_hg.sort_values('z_score', ascending=False)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 200)
    print(
        df_sorted[['structure_str', 'motif_count', 'z_score', 'hourglass_type']]
        .head(10)
        .to_string(index=False)
    )

    return df_hg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate hourglass patterns with Z-score (null model)."
    )
    parser.add_argument("--input", required=True,
                        help="Input CSV with hourglass patterns")
    parser.add_argument("--n_random", type=int, default=30,
                        help="Number of random networks (default: 30)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Validate only top-K patterns (default: all)")
    parser.add_argument("--max_jump", type=int, default=1)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()
    validate_hourglass(args.input, args.n_random, args.top_k, args.max_jump,
                       output_csv=args.output)
