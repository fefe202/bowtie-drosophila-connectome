#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure dei bow-tie motif.

In entrambe le modalita' l'asse verticale e' il livello gerarchico, cosi' la
direzione di un arco si legge dalla sua inclinazione. La modalita' single
disegna un motif per figura; la modalita' catalog produce una griglia
raggruppata per firma direzionale, spezzata in piu' immagini perche' in un
file unico le sedici firme danno una striscia illeggibile.

Uso:
    python src/visualize_hourglass.py --input <csv> --top_n 10
    python src/visualize_hourglass.py --input <csv> --mode catalog \
        --per_type 2 --types_per_figure 4
"""

from thesis_paths import results
from directional_signature import normalize_directions

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import argparse
import os
import sys
import hashlib
from collections import defaultdict
from directional_signature import add_signature


# --- MACRO-AREA EXTRACTION ---
def get_macro_area(group_name):
    """
    Estrae la macro-area dalle prime 2-3 lettere del nome del gruppo.
    """
    if not isinstance(group_name, str):
        return "UNK"
    
    # Sostituisci i punti con underscore per dividere in parti coerenti
    parts = group_name.replace('.', '_').split('_')
    prefix = ""
    for p in parts:
        if p:
            prefix = p
            break
    if not prefix:
        prefix = group_name

    # Prendi le prime 3 lettere, converti in maiuscolo e tieni solo caratteri alfanumerici
    prefix = prefix[:3].upper()
    prefix = "".join(c for c in prefix if c.isalnum())
    return prefix if prefix else "UNK"


# palette dei colori
# Palette per macro-aree (colori vivaci e distinti)
MACRO_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    '#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939',
    '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39',
]


def get_color_for_macro(macro_area, macro_to_color):
    """Assegna un colore stabile a una macro-area."""
    if macro_area not in macro_to_color:
        idx = len(macro_to_color) % len(MACRO_COLORS)
        macro_to_color[macro_area] = MACRO_COLORS[idx]
    return macro_to_color[macro_area]


# classificazione degli archi
def classify_edge_color(src_level, dst_level):
    """Colore dell'arco secondo la direzione rispetto ai livelli."""
    if dst_level > src_level:
        return '#666666'  # FWD: grigio scuro
    elif dst_level < src_level:
        return '#d62728'  # BWD: rosso
    else:
        return '#1f77b4'  # LAT: blu


# disegno del singolo motif
def visualize_single_hourglass(row, rank, macro_to_color, output_dir,
                               show_z_score=True):
    """
    Disegna un singolo hourglass nello stile di Kevin:
    layout gerarchico per livelli, nodi colorati per macro-area.
    """
    k_in = int(row['k_in'])
    k_out = int(row['k_out'])
    count = int(row['motif_count'])
    hg_type = row['hourglass_type']

    # Extract nodes
    bn_area = row['bottleneck_area']
    bn_level = int(row['bottleneck_level'])
    bn_node = f"{bn_area}\nL{bn_level}"

    fan_in_nodes = []
    for i in range(k_in):
        area = row[f'fan_in_{i}_area']
        level = int(row[f'fan_in_{i}_level'])
        fan_in_nodes.append((f"{area}\nL{level}", area, level))

    fan_out_nodes = []
    for i in range(k_out):
        area = row[f'fan_out_{i}_area']
        level = int(row[f'fan_out_{i}_level'])
        fan_out_nodes.append((f"{area}\nL{level}", area, level))

    # Build graph
    G = nx.DiGraph()
    G.add_node(bn_node, area=bn_area, level=bn_level,
               macro=get_macro_area(bn_area))

    for (node_id, area, level) in fan_in_nodes:
        G.add_node(node_id, area=area, level=level,
                   macro=get_macro_area(area))
        G.add_edge(node_id, bn_node)

    for (node_id, area, level) in fan_out_nodes:
        G.add_node(node_id, area=area, level=level,
                   macro=get_macro_area(area))
        G.add_edge(bn_node, node_id)

    # Determine all levels present
    all_levels_in_graph = set()
    for n in G.nodes():
        all_levels_in_graph.add(G.nodes[n]['level'])
    min_level = min(all_levels_in_graph)
    max_level = max(all_levels_in_graph)
    ALL_LEVELS = range(min_level, max_level + 1)

    # Layout: group nodes by level
    nodes_by_level = defaultdict(list)
    for n in G.nodes():
        lev = G.nodes[n]['level']
        nodes_by_level[lev].append(n)

    pos = {}
    spacing_x = 4.0
    spacing_y = 4.5
    for level in ALL_LEVELS:
        nodes = nodes_by_level.get(level, [])
        if not nodes:
            continue
        n_nodes = len(nodes)
        start_x = -((n_nodes - 1) * spacing_x) / 2
        for i, node_id in enumerate(nodes):
            pos[node_id] = (start_x + i * spacing_x, -level * spacing_y)

    # Drawing
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw nodes
    node_colors = []
    for n in G.nodes():
        macro = G.nodes[n]['macro']
        node_colors.append(get_color_for_macro(macro, macro_to_color))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=2500, alpha=0.9,
                           edgecolors='black', linewidths=0.5)

    # Node labels (area name, truncated)
    labels = {}
    for n in G.nodes():
        area = G.nodes[n]['area']
        # Trunca il nome se troppo lungo
        if len(area) > 10:
            labels[n] = area[:10]
        else:
            labels[n] = area
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels,
                            font_size=8, font_weight='normal')

    # Draw edges with colors
    node_radius = np.sqrt(2500) / 100.0
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)

        if dist > 0:
            start_x = x1 + (dx/dist) * node_radius
            start_y = y1 + (dy/dist) * node_radius
            end_x = x2 - (dx/dist) * node_radius
            end_y = y2 - (dy/dist) * node_radius

            src_lev = G.nodes[u]['level']
            tgt_lev = G.nodes[v]['level']
            color = classify_edge_color(src_lev, tgt_lev)

            arrow = FancyArrowPatch(
                (start_x, start_y), (end_x, end_y),
                arrowstyle='->', mutation_scale=20,
                linewidth=2.5, color=color,
                connectionstyle='arc3,rad=0.18',
                zorder=5
            )
            ax.add_patch(arrow)

    # Level labels and grid
    for level in ALL_LEVELS:
        y = -level * spacing_y
        if pos:
            all_x = [p[0] for p in pos.values()]
            label_x = min(all_x) - 2.5
        else:
            label_x = -5.0
        ax.text(label_x, y, f"L{level}",
                fontsize=14, fontweight='bold', va='center', color='#444')

    # Bande di livello: rendono leggibile la direzione degli archi senza
    # dover confrontare le etichette dei due nodi.
    for level in ALL_LEVELS:
        y = -level * spacing_y
        ax.axhspan(y - spacing_y / 2, y + spacing_y / 2, zorder=0,
                   facecolor='#f2f2f2' if level % 2 else '#fafafa',
                   edgecolor='none')
        ax.axhline(y - spacing_y / 2, color='#e2e2e2', linewidth=0.8,
                   zorder=0)
    ax.axhline(-max(ALL_LEVELS) * spacing_y - spacing_y / 2,
               color='#e2e2e2', linewidth=0.8, zorder=0)

    # Nel titolo non si riporta lo Z-score, che non e' una misura di
    # significativita' interpretabile per questi conteggi. Al suo posto la
    # firma direzionale e, se disponibili, le metriche di compressione.
    parts = [f"Rank #{rank}", f"count: {count:,}"]
    if 'signature' in row.index:
        parts.append(f"signature: {row['signature']}")
    else:
        parts.append(f"tipo: {hg_type}")
    if 'n_waist_eff' in row.index and pd.notna(row['n_waist_eff']):
        parts.append(f"effective waist: {float(row['n_waist_eff']):.1f}")
    if 'compressione_realizzata' in row.index and pd.notna(row['compressione_realizzata']):
        parts.append(f"compression: {float(row['compressione_realizzata']):.0f}")
    plt.title("  |  ".join(parts), fontsize=11, fontweight='bold', pad=20)

    # Legend for macros
    macros_in_plot = set(G.nodes[n]['macro'] for n in G.nodes())
    legend_patches = [
        mpatches.Patch(color=get_color_for_macro(m, macro_to_color), label=m)
        for m in sorted(macros_in_plot)
    ]
    # Edge legend
    legend_patches.append(mpatches.Patch(color='#666666',
                                        label='forward (deeper)'))
    legend_patches.append(mpatches.Patch(color='#d62728', label='backward'))
    legend_patches.append(mpatches.Patch(color='#1f77b4', label='lateral'))
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8,
              framealpha=0.9)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    sig = str(row["signature"]).replace("->", "_to_") if "signature" in row.index else hg_type
    safe_name = f"Rank{rank}_{sig}_Count{count}"
    out_path = os.path.join(output_dir, f"{safe_name}.png")
    plt.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)

    return out_path


# catalogo per firma
def level_bands(ax, levels, x0, x1, label=True):
    """
    Fondo a bande orizzontali, una per livello gerarchico.

    Con la profondita' sull'asse verticale la direzione di un arco si legge
    dalla sua inclinazione e non serve piu' confrontare le etichette dei due
    nodi: un arco che scende va in avanti, uno che sale torna indietro, uno
    orizzontale resta nello stesso livello.
    """
    for lv in levels:
        y = -lv
        ax.axhspan(y - 0.5, y + 0.5, xmin=0, xmax=1, zorder=0,
                   facecolor="#f2f2f2" if lv % 2 else "#fafafa",
                   edgecolor="none")
        ax.axhline(y - 0.5, color="#e2e2e2", lw=0.6, zorder=0)
        if label:
            ax.text(x0 - 0.30, y, f"L{lv}", fontsize=7.5,
                    color="#777777", ha="right", va="center", zorder=2)
    ax.axhline(-max(levels) - 0.5, color="#e2e2e2", lw=0.6, zorder=0)


def spread_x(base, levels_of_nodes, width=0.62):
    """
    Ascisse dei nodi di un ramo, a partire dalla colonna `base`.

    I nodi dello stesso livello finirebbero sovrapposti, perche' l'ordinata
    la decide il livello: quelli in collisione si separano orizzontalmente
    dentro la propria colonna.
    """
    from collections import defaultdict
    by_level = defaultdict(list)
    for i, lv in enumerate(levels_of_nodes):
        by_level[lv].append(i)
    xs = [base] * len(levels_of_nodes)
    for lv, idxs in by_level.items():
        if len(idxs) == 1:
            continue
        step = width / (len(idxs) - 1)
        for j, i in enumerate(idxs):
            xs[i] = base - width / 2 + j * step
    return xs


def draw_motif_panel(ax, row, macro_to_color, levels, show_level_labels=True):
    """Un singolo motif: ruolo sull'asse x, livello gerarchico sull'asse y."""
    k_in, k_out = int(row["k_in"]), int(row["k_out"])
    bl = int(row["bottleneck_level"])

    in_levels = [int(row[f"fan_in_{i}_level"]) for i in range(k_in)]
    out_levels = [int(row[f"fan_out_{i}_level"]) for i in range(k_out)]
    in_areas = [str(row[f"fan_in_{i}_area"]) for i in range(k_in)]
    out_areas = [str(row[f"fan_out_{i}_area"]) for i in range(k_out)]

    X_IN, X_WA, X_OUT = 0.0, 1.15, 2.30
    level_bands(ax, levels, X_IN, X_OUT, label=show_level_labels)

    xs_in = spread_x(X_IN, in_levels)
    xs_out = spread_x(X_OUT, out_levels)
    p_wa = (X_WA, -bl)
    p_in = [(x, -lv) for x, lv in zip(xs_in, in_levels)]
    p_out = [(x, -lv) for x, lv in zip(xs_out, out_levels)]

    for p, lv in zip(p_in, in_levels):
        ax.annotate("", xy=p_wa, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, shrinkA=8,
                                    shrinkB=9,
                                    color=classify_edge_color(lv, bl)))
    for p, lv in zip(p_out, out_levels):
        ax.annotate("", xy=p, xytext=p_wa,
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, shrinkA=9,
                                    shrinkB=8,
                                    color=classify_edge_color(bl, lv)))

    def node(p, area, big=False):
        # L'etichetta sta sotto il nodo: i nomi dei gruppi arrivano a undici
        # caratteri e dentro il cerchio verrebbero tagliati.
        ax.scatter([p[0]], [p[1]], s=340 if big else 230,
                   c=[get_color_for_macro(get_macro_area(area),
                                          macro_to_color)],
                   zorder=3, edgecolors="white", linewidths=1.4)
        ax.annotate(area, p, textcoords="offset points", xytext=(0, -11),
                    ha="center", va="top", fontsize=6.0, zorder=4,
                    color="#333333",
                    fontweight="bold" if big else "normal")

    for p, a in zip(p_in, in_areas):
        node(p, a)
    node(p_wa, str(row["bottleneck_area"]), big=True)
    for p, a in zip(p_out, out_areas):
        node(p, a)

    ax.set_xlim(X_IN - 0.72, X_OUT + 0.62)
    ax.set_ylim(-max(levels) - 0.62, -min(levels) + 0.62)
    ax.set_axis_off()


def visualize_catalog(df, macro_to_color, output_dir, max_per_type=2,
                      types_per_figure=4):
    """
    Catalogo dei motif raggruppati per FIRMA DIREZIONALE.

    Le colonne danno il ruolo (ingressi a sinistra, waist al centro, uscite a
    destra) e le righe il livello gerarchico, cosi' la struttura del motif si
    legge dalla forma e non dalle etichette. Il catalogo si spezza in piu'
    immagini da `types_per_figure` firme ciascuna: in un file unico le 16
    firme danno una striscia illeggibile.

    Restituisce la lista dei file prodotti.
    """
    col = 'signature' if 'signature' in df.columns else 'hourglass_type'
    types = sorted(df[col].unique())

    # bande comuni a tutti i pannelli: i motif restano confrontabili fra loro
    lv_cols = ([c for c in df.columns if c.endswith("_level")])
    levels = sorted(set(int(v) for c in lv_cols for v in df[c].dropna()))

    os.makedirs(output_dir, exist_ok=True)
    out_paths = []
    chunks = [types[i:i + types_per_figure]
              for i in range(0, len(types), types_per_figure)]

    for part, chunk in enumerate(chunks, 1):
        n_rows = len(chunk)
        fig, axes = plt.subplots(n_rows, max_per_type,
                                 figsize=(4.9 * max_per_type, 2.9 * n_rows),
                                 squeeze=False)

        for row_idx, hg_type in enumerate(chunk):
            subset = df[df[col] == hg_type].head(max_per_type)

            for col_idx in range(max_per_type):
                ax = axes[row_idx, col_idx]
                if col_idx >= len(subset):
                    ax.set_visible(False)
                    continue
                row = subset.iloc[col_idx]
                draw_motif_panel(ax, row, macro_to_color, levels,
                                 show_level_labels=(col_idx == 0))
                ax.set_title(f"#{col_idx + 1}   {int(row['motif_count']):,}",
                             fontsize=8.5, fontweight="bold")

            axes[row_idx, 0].text(-0.16, 0.5, hg_type,
                                  transform=axes[row_idx, 0].transAxes,
                                  fontsize=10.5, fontweight='bold',
                                  va='center', ha='right', family='monospace')

        handles = [mpatches.Patch(color='#666666', label='forward (deeper)'),
                   mpatches.Patch(color='#d62728', label='backward'),
                   mpatches.Patch(color='#1f77b4', label='lateral')]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5,
                   frameon=False, bbox_to_anchor=(0.5, -0.005))
        fig.suptitle(f"Representative motifs by directional signature "
                     f"({part}/{len(chunks)}). "
                     f"Depth increases downwards.",
                     fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0.03, 0.03, 1, 0.96])

        out_path = os.path.join(output_dir, f"hourglass_catalog_{part}.png")
        fig.savefig(out_path, dpi=200, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        print(f"  [{timestamp()}] Saved catalog part {part}: {out_path}")
        out_paths.append(out_path)

    return out_paths


def timestamp():
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize hourglass patterns (Kevin-style or Tricoli-style)."
    )
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--mode", choices=['single', 'catalog', 'both'],
                        default='both',
                        help="Visualization mode (default: both)")
    parser.add_argument("--per_type", type=int, default=2,
                        help="esempi per firma nel catalogo")
    parser.add_argument("--types_per_figure", type=int, default=4,
                        help="firme per immagine: il catalogo viene spezzato")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of top hourglass to visualize (single mode)")
    parser.add_argument("--outdir", default=results("figures"),
                        help="Output directory")

    args = parser.parse_args()

    df = normalize_directions(pd.read_csv(args.input))
    if 'signature' not in df.columns and 'bottleneck_level' in df.columns:
        add_signature(df, int(df['k_in'].iloc[0]), int(df['k_out'].iloc[0]))
    df = df.sort_values(by='motif_count', ascending=False)

    macro_to_color = {}

    print("=" * 65)
    print("  HOURGLASS VISUALIZATION")
    print("=" * 65)
    print(f"  [{timestamp()}] Input: {args.input}")
    print(f"  [{timestamp()}] Mode:  {args.mode}")
    print(f"  [{timestamp()}] Total patterns: {len(df)}")
    print("-" * 65)

    if args.mode in ('single', 'both'):
        print(f"\n  [{timestamp()}] Generating single hourglass plots...")
        df_top = df.head(args.top_n)
        for rank, (_, row) in enumerate(df_top.iterrows(), 1):
            out_path = visualize_single_hourglass(
                row, rank, macro_to_color, args.outdir
            )
            print(f"    [{timestamp()}] Rank {rank}: {out_path}")

    if args.mode in ('catalog', 'both'):
        print(f"\n  [{timestamp()}] Generating catalog...")
        visualize_catalog(df, macro_to_color, args.outdir,
                          args.per_type, args.types_per_figure)

    print(f"\n  [{timestamp()}] Done!")


if __name__ == "__main__":
    main()
