#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ricerca di bow-tie motif fra gruppi di connettivita': k_in gruppi
convergono su un gruppo centrale (waist) che diverge verso k_out gruppi.
Conteggio con flow vector e somma di prodotti (SigmaPi), senza enumerare le
istanze.

Questo e' il motore di riferimento: leggibile e lento. Il motore di
produzione e' hourglass_areas_fast.py, che si verifica contro questo.

I motif contati qui sono locali, nel senso di Milo et al. (2002). L'hourglass
effect di Sabrin & Dovrolis (2020), che e' una proprieta' globale della rete,
lo misura hourglass_core.py. I nomi dei file e delle colonne conservano
"hourglass" per continuita' con i risultati gia' prodotti.

La propagazione del flow vector (next_counts = counts @ submat) riprende
l'implementazione di D. Curcio, level-based-drosophila-motifs,
computation/compute.py. L'estensione a SigmaPi per i pattern con diramazioni
e' di questo lavoro.

Uso:
    python src/hourglass_areas.py --window 1-2-3 --k_in 2 --k_out 2 \
        --max_jump 1 --min_count 10
"""

import pandas as pd
import numpy as np
from thesis_paths import data, results

import argparse
import csv
import os
import time
import sys
from datetime import datetime, timedelta
from scipy import sparse
from collections import defaultdict
from itertools import combinations
from math import comb

# parametri globali
# Soglia per convertire sottomatrici sparse in dense (per velocita')
# 250,000 float64 ~ 2 MB di RAM
DENSE_THRESHOLD = 250_000


# utilita'

def timestamp():
    """Restituisce timestamp corrente per i log."""
    return datetime.now().strftime("%H:%M:%S")


def to_dense_1d(v):
    """
    Converte un vettore (scipy sparse matrix, np.matrix, o np.ndarray)
    in un array numpy 1D denso.
    """
    if sparse.issparse(v):
        return np.asarray(v.todense()).flatten()
    elif isinstance(v, np.matrix):
        return np.asarray(v).flatten()
    return np.asarray(v).flatten()


def classify_edge(src_level, dst_level):
    """Classifica un arco in base alla direzione nei livelli gerarchici."""
    if src_level < dst_level:
        return "FWD"
    elif src_level > dst_level:
        return "BWD"
    return "LAT"


def classify_hourglass(fan_in_keys, bottleneck_key, fan_out_keys):
    """
    Classifica un hourglass completo.
    Returns: 'pure_FWD', 'pure_BWD', 'mixed', o 'pure_LAT'
    """
    edge_types = set()
    bn_level = bottleneck_key[1]

    for (_, level) in fan_in_keys:
        edge_types.add(classify_edge(level, bn_level))
    for (_, level) in fan_out_keys:
        edge_types.add(classify_edge(bn_level, level))

    if edge_types == {"FWD"}:
        return "pure_FWD"
    if edge_types == {"BWD"}:
        return "pure_BWD"
    if edge_types == {"LAT"}:
        return "pure_LAT"
    return "mixed"


def signature_hourglass(fan_in_keys, bottleneck_key, fan_out_keys):
    """
    Firma direzionale FINE, che scompone l'etichetta unica `hourglass_type`.

    Con k_in + k_out archi, `pure_FWD` richiede che TUTTI siano in avanti:
    la dominanza di `mixed` (94% dei pattern) e' quasi tautologica e non dice
    nulla sulle ricorrenze, che sono l'interesse principale del docente.
    Qui il ramo entrante e quello uscente vengono descritti separatamente:

        fanin_sig  = direzione comune degli archi di fan-in  (FWD/BWD/LAT/mix)
        fanout_sig = direzione comune degli archi di fan-out (FWD/BWD/LAT/mix)
        signature  = "fanin_sig->fanout_sig"

    La classe FWD->BWD e' la *clessidra ri-entrante*: l'informazione converge in
    avanti sul waist, che poi la rimanda a livelli inferiori. E' il substrato
    strutturale della ricorrenza.

    Returns: (n_FWD, n_BWD, n_LAT, fanin_sig, fanout_sig, signature)
    """
    bn_level = bottleneck_key[1]
    in_dirs = [classify_edge(lv, bn_level) for (_, lv) in fan_in_keys]
    out_dirs = [classify_edge(bn_level, lv) for (_, lv) in fan_out_keys]
    all_dirs = in_dirs + out_dirs

    def collapse(dirs):
        return dirs[0] if len(set(dirs)) == 1 else "mix"

    fanin_sig = collapse(in_dirs)
    fanout_sig = collapse(out_dirs)
    return (
        all_dirs.count("FWD"), all_dirs.count("BWD"), all_dirs.count("LAT"),
        fanin_sig, fanout_sig, f"{fanin_sig}->{fanout_sig}",
    )


def format_elapsed(seconds):
    """Formatta secondi in stringa leggibile."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h{m}m{s}s"


# funzione principale

def compute_hourglass_motifs(NL_WINDOW, out_folder, k_in=2, k_out=2, max_jump=1,
                             min_count=1, top_n=None,
                             conn_file=data("connections.csv"),
                             levels_file=data("COORDINATE_XY_with_levels_tree.csv"),
                             neurons_file=data("neurons.csv")):
    """
    Cerca tutte le strutture hourglass (bow-tie) a livello di aree cerebrali.

    Parametri
    ---------
    NL_WINDOW : list[int]
        Livelli gerarchici da considerare.
    k_in, k_out : int
        Numero di rami fan-in e fan-out (parametrico).
    max_jump : int
        Distanza massima tra livelli connessi.
    min_count : int
        Soglia minima di occorrenze per salvare un hourglass.
        Valori con motif_count < min_count vengono scartati.
    top_n : int or None
        Se specificato, salva solo i top_n hourglass per motif_count.
    """
    TARGET_LEVELS_SET = set(NL_WINDOW)
    os.makedirs(out_folder, exist_ok=True)
    start_time = time.time()

    print("=" * 70)
    print("  BOW-TIE MOTIF SEARCH (convergenza-divergenza) - Flow Vector + SigmaPi")
    print("  Analisi MICRO-strutturale. Per l'hourglass globale: hourglass_core.py")
    print("=" * 70)
    print(f"  [{timestamp()}] Started")
    print(f"  Window      = {NL_WINDOW}")
    print(f"  k_in        = {k_in}")
    print(f"  k_out       = {k_out}")
    print(f"  max_jump    = {max_jump}")
    print(f"  min_count   = {min_count}")
    print(f"  top_n       = {top_n if top_n else 'ALL'}")
    print("-" * 70)

    # == 1. LOAD DATA ==
    print(f"\n[{timestamp()}] [1/6] Loading CSV files...")
    t0 = time.time()
    try:
        df_conn = pd.read_csv(conn_file)
        print(f"  connections.csv     : {len(df_conn):>10,} rows  ({time.time()-t0:.1f}s)")
        t1 = time.time()
        df_levels = pd.read_csv(levels_file)
        print(f"  levels.csv          : {len(df_levels):>10,} rows  ({time.time()-t1:.1f}s)")
        t1 = time.time()
        df_neurons = pd.read_csv(neurons_file)
        print(f"  neurons.csv         : {len(df_neurons):>10,} rows  ({time.time()-t1:.1f}s)")
    except Exception as e:
        print(f"  [ERROR] Failed to load data: {e}")
        sys.exit(1)
    print(f"  Total load time: {time.time()-t0:.1f}s")

    # == 2. NEURON MAPPINGS ==
    print(f"\n[{timestamp()}] [2/6] Building neuron -> (area, level) mappings...")
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

    # Count per area e per level
    area_counts = defaultdict(int)
    level_counts = defaultdict(int)
    for n in valid_neurons:
        area_counts[area_map[n]] += 1
        level_counts[level_map[n]] += 1

    print(f"  Valid neurons in window: {n_total:,}")
    print(f"  Distinct areas: {len(area_counts)}")
    print(f"  Neurons per level:")
    for lev in sorted(level_counts.keys()):
        print(f"    L{lev}: {level_counts[lev]:,}")

    # == 3. BUILD METAGRAPH GROUPS ==
    print(f"\n[{timestamp()}] [3/6] Grouping neurons by (area, level)...")
    groups = defaultdict(list)
    for n in valid_neurons:
        groups[(area_map[n], level_map[n])].append(neuron_to_idx[n])

    group_keys = sorted(groups.keys())
    print(f"  Metagraph nodes: {len(group_keys)}")

    # Statistiche sui gruppi
    group_sizes = [len(groups[k]) for k in group_keys]
    print(f"  Group size: min={min(group_sizes)}, "
          f"median={int(np.median(group_sizes))}, "
          f"max={max(group_sizes)}, "
          f"mean={np.mean(group_sizes):.1f}")

    # == 4. BUILD ADJACENCY MATRIX ==
    print(f"\n[{timestamp()}] [4/6] Building sparse adjacency matrix...")
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
    print(f"  Directed edges: {adj_matrix.nnz:,}  ({time.time()-t0:.1f}s)")

    # == 5. PRE-COMPUTE SUBMATRICES ==
    print(f"\n[{timestamp()}] [5/6] Pre-computing submatrices...")
    t0 = time.time()
    submat_map = {}
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    n_dense = 0

    for src_key in group_keys:
        src_lev = src_key[1]
        src_indices = groups[src_key]
        row_submat = adj_matrix[src_indices, :]

        for dst_key in group_keys:
            dst_lev = dst_key[1]
            if abs(src_lev - dst_lev) > max_jump:
                continue
            if src_key == dst_key:
                continue

            dst_indices = groups[dst_key]
            submat = row_submat[:, dst_indices]

            if submat.nnz > 0:
                rows, cols = submat.shape
                if (rows * cols) < DENSE_THRESHOLD:
                    final_mat = submat.toarray()
                    n_dense += 1
                else:
                    final_mat = submat

                submat_map[(src_key, dst_key)] = final_mat
                incoming[dst_key].append(src_key)
                outgoing[src_key].append(dst_key)

    print(f"  Metagraph edges: {len(submat_map):,} ({n_dense} densified)")
    print(f"  Pre-computation time: {time.time()-t0:.1f}s")

    # Stats on connectivity
    in_degrees = [len(incoming.get(k, [])) for k in group_keys]
    out_degrees = [len(outgoing.get(k, [])) for k in group_keys]
    eligible = sum(1 for k in group_keys
                   if len(incoming.get(k, [])) >= k_in
                   and len(outgoing.get(k, [])) >= k_out)
    print(f"  Eligible bottlenecks (in>={k_in}, out>={k_out}): {eligible}/{len(group_keys)}")

    # == 6. HOURGLASS SEARCH WITH SigmaPi ==
    print(f"\n[{timestamp()}] [6/6] Searching hourglass patterns...")
    print(f"  Method: SigmaPi (sum-of-products) over bottleneck neurons")
    print(f"  min_count filter: discarding patterns with count < {min_count}")
    print()

    # Prepare streaming CSV output to avoid MemoryError
    out_name = (
        f"hourglass_areas_nl{'-'.join(map(str, NL_WINDOW))}"
        f"_kin{k_in}_kout{k_out}_jump{max_jump}.csv"
    )
    out_path = os.path.join(out_folder, out_name)

    # Build CSV header
    header = [
        "motif_count", "hourglass_type",
        "signature", "fanin_sig", "fanout_sig", "n_FF", "n_FB", "n_lat",
        "n_waist", "n_waist_active", "n_waist_eff", "n_periphery", "compression",
        "bottleneck_area", "bottleneck_level",
        "k_in", "k_out", "structure_str",
    ]
    for i in range(k_in):
        header.extend([f"fan_in_{i}_area", f"fan_in_{i}_level"])
    for i in range(k_out):
        header.extend([f"fan_out_{i}_area", f"fan_out_{i}_level"])

    csv_file = open(out_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=header)
    csv_writer.writeheader()

    total_found = 0
    total_saved = 0
    total_pruned_in = 0     # pruned by fan-in zero product
    total_pruned_out = 0    # pruned by fan-out dead flow
    total_below_min = 0     # pruned by min_count
    total_excluded_overlap = 0  # scartati: gruppo comune a fan-in e fan-out
    search_start = time.time()

    # Type counters
    type_counts = defaultdict(int)
    sig_counts = defaultdict(int)

    for bn_idx, bottleneck_key in enumerate(group_keys):
        in_nbrs = incoming.get(bottleneck_key, [])
        out_nbrs = outgoing.get(bottleneck_key, [])

        if len(in_nbrs) < k_in or len(out_nbrs) < k_out:
            continue

        n_bn = len(groups[bottleneck_key])
        bn_area, bn_level = bottleneck_key

        # Number of combos for this bottleneck (limite superiore: il vincolo
        # di disgiunzione fan-in/fan-out ne scarta una parte, vedi sotto)
        n_combos_in = comb(len(in_nbrs), k_in)
        n_combos_out = comb(len(out_nbrs), k_out)
        n_combos_total = n_combos_in * n_combos_out

        # Pre-compute fan-in flow vectors
        fan_in_vec = {}
        for a_key in in_nbrs:
            sub = submat_map[(a_key, bottleneck_key)]
            n_a = len(groups[a_key])
            v = np.ones(n_a, dtype=np.float64) @ sub
            fan_in_vec[a_key] = to_dense_1d(v)

        # Pre-compute fan-out flow vectors
        fan_out_vec = {}
        for d_key in out_nbrs:
            sub = submat_map[(bottleneck_key, d_key)]
            if sparse.issparse(sub):
                v = np.asarray(sub.sum(axis=1)).flatten()
            else:
                v = sub.sum(axis=1)
            fan_out_vec[d_key] = to_dense_1d(v)

        bn_found = 0
        bn_saved = 0

        for fan_in_combo in combinations(in_nbrs, k_in):
            prod_in = np.ones(n_bn, dtype=np.float64)
            for a_key in fan_in_combo:
                prod_in *= fan_in_vec[a_key]

            # Vincolo di disgiunzione fanIN / FAN-OUT ---
            # Un gruppo che compare sia in fan-in sia in fan-out produce
            # istanze degeneri in cui lo stesso neurone e' contemporaneamente
            # sorgente e destinazione (coppia reciproca a -> b -> a): non e'
            # un motif a k_in + 1 + k_out nodi distinti.
            # I gruppi di fan-in sono gia' distinti tra loro (combinations),
            # idem quelli di fan-out, e il bottleneck e' escluso da entrambi
            # (submat_map salta src_key == dst_key). Resta da imporre solo
            # l'intersezione vuota tra i due insiemi.
            fi_set = set(fan_in_combo)
            valid_out = [d for d in out_nbrs if d not in fi_set]
            n_valid_out = comb(len(valid_out), k_out) if len(valid_out) >= k_out else 0
            total_excluded_overlap += n_combos_out - n_valid_out

            if np.sum(prod_in) == 0:
                total_pruned_in += n_valid_out
                continue

            if n_valid_out == 0:
                continue

            for fan_out_combo in combinations(valid_out, k_out):
                product_vector = prod_in.copy()

                dead = False
                for d_key in fan_out_combo:
                    product_vector *= fan_out_vec[d_key]
                    if np.sum(product_vector) == 0:
                        dead = True
                        break

                if dead:
                    total_pruned_out += 1
                    continue

                total = int(np.sum(product_vector))
                total_found += 1
                bn_found += 1

                if total < min_count:
                    total_below_min += 1
                    continue

                # Classify
                hg_type = classify_hourglass(
                    fan_in_combo, bottleneck_key, fan_out_combo
                )
                type_counts[hg_type] += 1
                n_ff, n_fb, n_lat, fi_sig, fo_sig, sig = signature_hourglass(
                    fan_in_combo, bottleneck_key, fan_out_combo
                )
                sig_counts[sig] += 1

                # metriche di compressione del waist
                # participation ratio: numero efficace di neuroni del waist
                # che portano il flusso. Se un solo neurone fa tutto vale ~1.
                # Il vettore SigmaPi e' gia' calcolato, quindi il costo e'
                # nullo.
                # `compression` e' la versione nominale, basata sulle
                # dimensioni dei gruppi: da sola e' fuorviante e va letta
                # insieme a n_waist_eff, o sostituita dalla compressione
                # realizzata di compression_realized.py.
                n_periphery = (sum(len(groups[a]) for a in fan_in_combo) +
                               sum(len(groups[d]) for d in fan_out_combo))
                n_active = int(np.count_nonzero(product_vector))
                ssq = float(product_vector @ product_vector)
                n_eff = (float(total) ** 2 / ssq) if ssq > 0 else 0.0

                # Build row
                fi_str = ", ".join(f"{a}L{l}" for a, l in fan_in_combo)
                fo_str = ", ".join(f"{a}L{l}" for a, l in fan_out_combo)
                bn_str = f"{bn_area}L{bn_level}"

                row = {
                    "motif_count": total,
                    "hourglass_type": hg_type,
                    "signature": sig,
                    "fanin_sig": fi_sig,
                    "fanout_sig": fo_sig,
                    "n_FF": n_ff, "n_FB": n_fb, "n_lat": n_lat,
                    "n_waist": n_bn,
                    "n_waist_active": n_active,
                    "n_waist_eff": round(n_eff, 3),
                    "n_periphery": n_periphery,
                    "compression": round(n_periphery / n_bn, 3),
                    "bottleneck_area": bn_area,
                    "bottleneck_level": bn_level,
                    "k_in": k_in,
                    "k_out": k_out,
                    "structure_str": f"[{fi_str}] -> {bn_str} -> [{fo_str}]",
                }
                for i, (a, l) in enumerate(fan_in_combo):
                    row[f"fan_in_{i}_area"] = a
                    row[f"fan_in_{i}_level"] = l
                for i, (a, l) in enumerate(fan_out_combo):
                    row[f"fan_out_{i}_area"] = a
                    row[f"fan_out_{i}_level"] = l

                csv_writer.writerow(row)
                total_saved += 1
                bn_saved += 1

        # Progress output per bottleneck
        elapsed = time.time() - search_start
        pct = (bn_idx + 1) / len(group_keys) * 100
        if elapsed > 0 and pct > 0:
            eta = elapsed / pct * (100 - pct)
            eta_str = format_elapsed(eta)
        else:
            eta_str = "?"

        if bn_found > 0 or (bn_idx + 1) % 50 == 0 or bn_idx == 0:
            print(
                f"  [{timestamp()}] Bottleneck {bn_idx+1:>4}/{len(group_keys)} "
                f"({pct:5.1f}%) | "
                f"{bn_area}L{bn_level} "
                f"(in={len(in_nbrs)},out={len(out_nbrs)},combos={n_combos_total:,}) | "
                f"found={bn_found:,} saved={bn_saved:,} | "
                f"Total saved: {total_saved:,} | "
                f"ETA: {eta_str}"
            )

    csv_file.close()

    # == RESULTS ==
    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print(f"  RESULTS  [{timestamp()}]  (total time: {format_elapsed(elapsed)})")
    print("=" * 70)
    print(f"  Motif found (raw):                {total_found:>12,}")
    print(f"  Excluded (fan-in/fan-out overlap):{total_excluded_overlap:>12,}")
    print(f"  Pruned (fan-in zero):             {total_pruned_in:>12,}")
    print(f"  Pruned (fan-out dead flow):       {total_pruned_out:>12,}")
    print(f"  Pruned (count < {min_count}):{'':>13}{total_below_min:>12,}")
    print(f"  Saved to CSV:                     {total_saved:>12,}")
    print()

    if sig_counts:
        print("  [SUMMARY BY DIRECTIONAL SIGNATURE]")
        for s in sorted(sig_counts, key=lambda x: -sig_counts[x]):
            print(f"    {s:>12}: {sig_counts[s]:,}")
        print()

    if type_counts:
        print("  [SUMMARY BY TYPE]")
        for t in sorted(type_counts.keys()):
            print(f"    {t:>15}: {type_counts[t]:,}")
        print()

    print(f"  [OUTPUT] {out_path}")

    # If top_n requested, re-read and filter
    if top_n and total_saved > top_n:
        print(f"\n  [{timestamp()}] Applying --top_n={top_n}: "
              f"keeping only top {top_n} by motif_count...")
        df = pd.read_csv(out_path)
        df = df.sort_values(by="motif_count", ascending=False).head(top_n)
        df.to_csv(out_path, index=False)
        print(f"  CSV trimmed from {total_saved:,} to {len(df):,} rows.")
        total_saved = len(df)

    # Show top results
    if total_saved > 0:
        df = pd.read_csv(out_path, nrows=15)
        print(f"\n  [TOP {min(15, total_saved)} HOURGLASS MOTIFS]")
        pd.set_option("display.max_colwidth", 120)
        pd.set_option("display.width", 200)
        print(
            df[["structure_str", "motif_count", "hourglass_type"]]
            .to_string(index=False)
        )
    else:
        print("  [WARN] No hourglass motifs found in the specified window.")

    print(f"\n  [{timestamp()}] Done.")
    return out_path


# interfaccia a riga di comando

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Ricerca di bow-tie motif (convergenza-divergenza) a livello di "
            "gruppi di connettivita', con classificazione FF/FB.\n\n"
            "Usa il metodo Flow Vector + SigmaPi per contare i motif nel "
            "connettoma della Drosophila.\n\n"
            "I 'gruppi' sono definiti dalla colonna 'group' di neurons.csv: "
            "sono neuropile singole (es. 'AL', 'ME') o coppie di neuropile "
            "collegate da neuroni di proiezione (es. 'AL.MB_CA'), NON aree "
            "anatomiche in senso stretto.\n\n"
            "NOTA: questa e' l'analisi micro-strutturale. L'hourglass effect "
            "globale (tau-core, H-score, Sabrin & Dovrolis 2020) e' in "
            "hourglass_core.py\n\n"
            "IMPORTANTE: usare --min_count per filtrare i pattern a basso "
            "conteggio. Senza, i risultati sono nell'ordine dei milioni."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--window", required=True,
        help="Level window, e.g.: 1-2-3",
    )
    parser.add_argument(
        "--k_in", type=int, default=2,
        help="Number of fan-in branches (default: 2)",
    )
    parser.add_argument(
        "--k_out", type=int, default=2,
        help="Number of fan-out branches (default: 2)",
    )
    parser.add_argument(
        "--max_jump", type=int, default=1,
        help="Maximum level distance between connected groups (default: 1)",
    )
    parser.add_argument(
        "--min_count", type=int, default=10,
        help="Minimum motif_count to save a pattern (default: 10). "
             "Patterns below this threshold are discarded.",
    )
    parser.add_argument(
        "--top_n", type=int, default=None,
        help="Keep only the top N hourglass by motif_count (default: keep all)",
    )
    parser.add_argument(
        "--outdir", default=results("hourglass_results"),
        help="Output directory (default: hourglass_results)",
    )

    args = parser.parse_args()

    try:
        NL_WINDOW = [int(x) for x in args.window.split("-")]
    except ValueError:
        print("Error: --window format must be integers separated by hyphens")
        sys.exit(1)

    if len(NL_WINDOW) < 2:
        print("Error: --window must include at least 2 levels")
        sys.exit(1)

    out_folder = os.path.join(args.outdir, args.window)

    compute_hourglass_motifs(
        NL_WINDOW,
        out_folder,
        k_in=args.k_in,
        k_out=args.k_out,
        max_jump=args.max_jump,
        min_count=args.min_count,
        top_n=args.top_n,
    )
