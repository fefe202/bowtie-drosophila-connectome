#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-processing del CSV dei motif: lettura a blocchi ed estrazione delle
righe di testa.

Superato da motif_distill.py, che produce le stesse selezioni durante la
ricerca invece che dopo.

Uso:
    python src/filter_top_hourglass.py --input <csv> --top_n 1000
"""

import pandas as pd
import argparse
import os
import sys
import time
import heapq
from datetime import datetime


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def filter_top_hourglass(input_csv, top_n=1000, min_count=None, output_csv=None):
    """
    Filtra il CSV degli hourglass mantenendo solo i top_n per motif_count.

    Usa un approccio heap-based: scorre il CSV in chunks e mantiene in memoria
    solo i top_n record con motif_count piu' alto, evitando di caricare
    tutto il file in RAM.

    Parametri
    ---------
    input_csv : str
        Percorso al CSV degli hourglass (puo' essere molto grande).
    top_n : int
        Numero di hourglass da mantenere.
    min_count : int or None
        Soglia minima aggiuntiva su motif_count.
    output_csv : str or None
        Percorso di output. Se None, generato automaticamente.
    """
    start_time = time.time()

    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_top{top_n}{ext}"

    print("=" * 65)
    print("  FILTER TOP HOURGLASS")
    print("=" * 65)
    print(f"  [{timestamp()}] Input:     {input_csv}")
    print(f"  [{timestamp()}] Output:    {output_csv}")
    print(f"  [{timestamp()}] Top N:     {top_n}")
    print(f"  [{timestamp()}] Min count: {min_count if min_count else 'None'}")
    print("-" * 65)

    # Fase 1: scorrere in chunks e mantenere un heap dei top-N
    print(f"\n  [{timestamp()}] Phase 1: Scanning CSV in chunks...")
    chunk_size = 100_000
    total_rows = 0
    filtered_rows = 0

    # Heap min: manteniamo i top_n piu' grandi
    # Ogni elemento e' (motif_count, row_index, row_dict)
    top_heap = []

    for chunk_idx, chunk in enumerate(pd.read_csv(input_csv, chunksize=chunk_size)):
        total_rows += len(chunk)

        if min_count is not None:
            chunk = chunk[chunk['motif_count'] >= min_count]

        filtered_rows += len(chunk)

        for _, row in chunk.iterrows():
            count = row['motif_count']
            row_tuple = (count, total_rows + _, row.to_dict())

            if len(top_heap) < top_n:
                heapq.heappush(top_heap, row_tuple)
            elif count > top_heap[0][0]:
                heapq.heapreplace(top_heap, row_tuple)

        if (chunk_idx + 1) % 10 == 0:
            print(f"    [{timestamp()}] Processed {total_rows:,} rows, "
                  f"heap size: {len(top_heap)}, "
                  f"min in heap: {top_heap[0][0]:,}")

    print(f"\n  [{timestamp()}] Phase 1 complete:")
    print(f"    Total rows scanned: {total_rows:,}")
    print(f"    Rows after min_count filter: {filtered_rows:,}")
    print(f"    Top-{top_n} collected")

    if len(top_heap) == 0:
        print("  [WARN] No hourglass found matching criteria!")
        return None

    # Fase 2: ordinare e salvare
    print(f"\n  [{timestamp()}] Phase 2: Sorting and saving...")
    top_rows = sorted(top_heap, key=lambda x: -x[0])
    df_top = pd.DataFrame([r[2] for r in top_rows])
    df_top.to_csv(output_csv, index=False)

    elapsed = time.time() - start_time
    print(f"\n  [{timestamp()}] Done in {elapsed:.1f}s")
    print(f"  [{timestamp()}] Saved {len(df_top)} rows to: {output_csv}")

    # Statistiche rapide
    print(f"\n  [STATS]")
    print(f"    motif_count: min={df_top['motif_count'].min():,}, "
          f"max={df_top['motif_count'].max():,}, "
          f"median={int(df_top['motif_count'].median()):,}")
    if 'hourglass_type' in df_top.columns:
        print(f"    By type:")
        for t, c in df_top['hourglass_type'].value_counts().items():
            print(f"      {t}: {c}")

    return df_top


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter hourglass CSV to keep only top-N by motif_count."
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--top_n", type=int, default=1000,
                        help="Number of top hourglass to keep (default: 1000)")
    parser.add_argument("--min_count", type=int, default=None,
                        help="Additional minimum motif_count filter")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: auto-generated)")

    args = parser.parse_args()
    filter_top_hourglass(args.input, args.top_n, args.min_count, args.output)
