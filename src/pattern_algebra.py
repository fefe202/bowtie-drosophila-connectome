#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classificazione dei pattern di connettivita' per costo di conteggio.

Contare le occorrenze significa contare gli omomorfismi dal pattern al grafo,
e il costo dipende solo dalla forma del pattern:

    cammino   prodotto di matrici, O(n) memoria, l'unione si preserva
    albero    matvec lungo gli archi piu' prodotto di Hadamard nelle
              diramazioni, O(n) memoria, l'unione non si preserva
    ciclico   non fattorizzabile su vettori, serve una matrice, O(n^2)

Il confine e' la presenza di diramazioni. Il modulo enumera i pattern
connessi non isomorfi fino a 4 nodi, ne deriva la formula fattorizzata e la
verifica contro enumerazione esaustiva degli omomorfismi.

Gia' a 4 nodi l'87% dei pattern e' ciclico, quindi fuori dalla portata del
metodo flow-vector.

Uso:
    python src/pattern_algebra.py --max_nodes 4 --n_verify 25
"""

from thesis_paths import results

import argparse
import itertools
import os
from collections import defaultdict

import numpy as np
import pandas as pd


# ============================================================================
#  ENUMERAZIONE DEI PATTERN
# ============================================================================

def canonical(edges, k):
    """Forma canonica di un pattern: minimo su tutte le permutazioni dei nodi."""
    best = None
    for perm in itertools.permutations(range(k)):
        e = tuple(sorted((perm[u], perm[v]) for u, v in edges))
        if best is None or e < best:
            best = e
    return best


def weakly_connected(edges, k):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    seen = {0}
    stack = [0]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == k


def enumerate_patterns(k):
    """Tutti i pattern diretti connessi su k nodi, senza cappi, non isomorfi."""
    pairs = [(u, v) for u in range(k) for v in range(k) if u != v]
    seen = set()
    out = []
    for mask in range(1, 1 << len(pairs)):
        edges = [pairs[i] for i in range(len(pairs)) if mask >> i & 1]
        if len(edges) < k - 1:
            continue
        if not weakly_connected(edges, k):
            continue
        c = canonical(edges, k)
        if c in seen:
            continue
        seen.add(c)
        out.append(list(c))
    return out


def undirected_skeleton(edges):
    return {frozenset(e) for e in edges}


def is_tree(edges, k):
    """Il grafo non orientato sottostante e' un albero?"""
    sk = undirected_skeleton(edges)
    return len(sk) == k - 1


def is_path(edges, k):
    """Il grafo non orientato sottostante e' un cammino?"""
    if not is_tree(edges, k):
        return False
    deg = defaultdict(int)
    for e in undirected_skeleton(edges):
        for x in e:
            deg[x] += 1
    return sorted(deg.values()) == [1, 1] + [2] * (k - 2)


def treewidth_upper(edges, k):
    """
    Treewidth per eliminazione greedy (min-degree). Esatta per i casi
    piccoli qui considerati; serve solo a quantificare il salto di costo.
    """
    adj = {i: set() for i in range(k)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    tw = 0
    nodes = set(range(k))
    while nodes:
        x = min(nodes, key=lambda n: len(adj[n] & nodes))
        nb = adj[x] & nodes
        tw = max(tw, len(nb))
        for a in nb:
            for b in nb:
                if a != b:
                    adj[a].add(b)
        nodes.discard(x)
    return tw


# ============================================================================
#  CONTEGGIO
# ============================================================================

def brute_force_hom(edges, k, A):
    """Numero di omomorfismi del pattern in A, per enumerazione esaustiva."""
    n = A.shape[0]
    tot = 0
    for assign in itertools.product(range(n), repeat=k):
        if all(A[assign[u], assign[v]] for u, v in edges):
            tot += 1
    return tot


def tree_message_passing(edges, k, A):
    """
    Conteggio fattorizzato su un albero: message passing dalle foglie alla
    radice. Lungo ogni arco un prodotto matrice-vettore; in ogni nodo il
    prodotto ELEMENTWISE dei messaggi entranti (e' il passo SigmaPi).

    Restituisce (conteggio, n_prodotti_matvec, n_prodotti_hadamard).
    """
    n = A.shape[0]
    directed = set(edges)

    def constraint(parent, child):
        """
        Matrice del vincolo sull'arco dello SCHELETRO {parent, child},
        indicizzata (parent, child).

        Se il pattern ha entrambe le direzioni fra i due nodi si tratta di una
        connessione RECIPROCA: il vincolo e' "esiste l'arco in entrambi i
        versi", cioe' la matrice A (*) A^T (prodotto elementwise). Le
        reciprocita' NON aumentano quindi la complessita': restano un singolo
        vincolo su un arco dello scheletro, e l'albero resta un albero.
        """
        fw = (parent, child) in directed
        bw = (child, parent) in directed
        if fw and bw:
            return A * A.T
        return A if fw else A.T

    nb = defaultdict(set)
    for u, v in edges:
        nb[u].add(v)
        nb[v].add(u)

    n_mv = n_had = 0
    root = 0
    visited = {root}
    order = []

    def dfs(x):
        for y in sorted(nb[x]):
            if y not in visited:
                visited.add(y)
                dfs(y)
                order.append((y, x))      # messaggio da y verso x
    dfs(root)

    msg = {x: np.ones(n) for x in range(k)}
    got_child = defaultdict(int)
    for child, parent in order:
        m = constraint(parent, child) @ msg[child]
        n_mv += 1
        # il primo messaggio non e' un Hadamard (il padre parte da tutti uno):
        # l'Hadamard vero, cioe' il passo SigmaPi, avviene dal secondo figlio
        # in poi, ed e' esattamente dove l'unione non si preserva
        if got_child[parent]:
            n_had += 1
        got_child[parent] += 1
        msg[parent] = msg[parent] * m
    return float(msg[root].sum()), n_mv, n_had


def cyclic_count_matrix(edges, k, A):
    """
    Per i pattern ciclici non esiste una fattorizzazione su vettori.
    Qui si conta comunque, in modo esaustivo, solo per verificare che la
    classificazione sia corretta: il punto e' proprio che servirebbe un
    oggetto di dimensione superiore.
    """
    return brute_force_hom(edges, k, A)


def shape_name(edges, k):
    if is_path(edges, k):
        return "cammino"
    if is_tree(edges, k):
        deg = defaultdict(int)
        for e in undirected_skeleton(edges):
            for x in e:
                deg[x] += 1
        if max(deg.values()) == k - 1:
            return "stella"
        return "albero"
    return "ciclico"


# ============================================================================
#  MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Classificazione algebrica dei pattern: dove si preserva "
                    "l'unione e dove serve SigmaPi (punti 6, 9, 10).")
    ap.add_argument("--max_nodes", type=int, default=4,
                    help="Enumerazione esaustiva fino a questo numero di nodi")
    ap.add_argument("--n_verify", type=int, default=25,
                    help="Grafi casuali su cui verificare ogni formula")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=results("pattern_algebra"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 78)
    print("  ALGEBRA DEI PATTERN: dove si preserva l'unione")
    print("=" * 78)
    print("  cammino  -> puro prodotto di matrici, nessun Hadamard")
    print("             (l'unione si preserva: basta il flow vector)")
    print("  albero   -> matvec + Hadamard nei nodi di diramazione = SigmaPi")
    print("             (l'unione NON si preserva nelle diramazioni:")
    print("              serve imporre che sia lo STESSO neurone)")
    print("  ciclico  -> non fattorizzabile su vettori: serve una matrice")
    print()

    rows = []
    for k in range(3, args.max_nodes + 1):
        pats = enumerate_patterns(k)
        print(f"  [{k} nodi] {len(pats)} pattern connessi non isomorfi")
        for edges in pats:
            shape = shape_name(edges, k)
            tw = treewidth_upper(edges, k)
            ok = True
            n_mv = n_had = 0
            for _ in range(args.n_verify):
                n = int(rng.integers(4, 7))
                A = (rng.random((n, n)) < 0.4).astype(float)
                np.fill_diagonal(A, 0)
                exp = brute_force_hom(edges, k, A)
                if shape != "ciclico":
                    got, n_mv, n_had = tree_message_passing(edges, k, A)
                    if abs(got - exp) > 1e-6:
                        ok = False
                        break
            rows.append({
                "n_nodi": k,
                "archi": len(edges),
                "forma": shape,
                "treewidth": tw,
                "unione_preservata": shape == "cammino",
                "fattorizzabile_su_vettori": shape != "ciclico",
                "n_matvec": n_mv if shape != "ciclico" else None,
                "n_hadamard": n_had if shape != "ciclico" else None,
                "formula_verificata": ok if shape != "ciclico" else None,
                "pattern": " ".join(f"{u}->{v}" for u, v in edges),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "pattern_catalogue.csv"), index=False)

    print("\n  [RIEPILOGO]")
    summ = df.groupby(["n_nodi", "forma"]).agg(
        n_pattern=("archi", "size"),
        treewidth_max=("treewidth", "max"),
        formule_verificate=("formula_verificata",
                            lambda x: int(x.dropna().sum())),
    ).reset_index()
    print(summ.to_string(index=False))

    tree_ok = df[df["forma"] != "ciclico"]
    n_bad = int((~tree_ok["formula_verificata"].fillna(False)).sum())
    print(f"\n  Formule su albero verificate contro enumerazione esaustiva: "
          f"{len(tree_ok) - n_bad}/{len(tree_ok)}"
          f"{'  TUTTE CORRETTE' if n_bad == 0 else f'  {n_bad} ERRATE'}")

    print("\n  [DOVE SI PRENDE IL SALTO DI COSTO]")
    for k in sorted(df["n_nodi"].unique()):
        d = df[df["n_nodi"] == k]
        n_path = int((d["forma"] == "cammino").sum())
        n_tree = int((d["forma"].isin(["albero", "stella"])).sum())
        n_cyc = int((d["forma"] == "ciclico").sum())
        print(f"    {k} nodi: {n_path} cammini (solo matvec), "
              f"{n_tree} alberi (SigmaPi), "
              f"{n_cyc} ciclici (servono matrici) "
              f"= {100*n_cyc/len(d):.0f}% non fattorizzabili su vettori")

    # ---- il bow-tie della tesi
    print("\n  [DOVE STA IL BOW-TIE DELLA TESI]")
    bt = [(0, 2), (1, 2), (2, 3), (2, 4)]     # a1,a2 -> b -> d1,d2
    print(f"    pattern: {' '.join(f'{u}->{v}' for u, v in bt)}")
    print(f"    forma: {shape_name(bt, 5)}   treewidth: {treewidth_upper(bt, 5)}")
    A = (rng.random((6, 6)) < 0.4).astype(float)
    np.fill_diagonal(A, 0)
    got, n_mv, n_had = tree_message_passing(bt, 5, A)
    exp = brute_force_hom(bt, 5, A)
    print(f"    formula fattorizzata = {got:.0f}, enumerazione = {exp:.0f}  "
          f"{'COINCIDONO' if abs(got-exp) < 1e-6 else 'DISCREPANZA'}")
    print(f"    costo: {n_mv} prodotti matrice-vettore + {n_had} Hadamard")
    print("    E' un ALBERO: il metodo della tesi lo conta esattamente,")
    print("    con memoria O(n). E' il caso (2).")

    # ---- il caso minimo non fattorizzabile
    print("\n  [IL CASO MINIMO CHE IL METODO NON SA FARE SU VETTORI]")
    dia = [(0, 1), (0, 2), (1, 3), (2, 3)]    # diamante
    print(f"    diamante: {' '.join(f'{u}->{v}' for u, v in dia)}")
    print(f"    forma: {shape_name(dia, 4)}   treewidth: {treewidth_upper(dia, 4)}")
    A = (rng.random((6, 6)) < 0.4).astype(float)
    np.fill_diagonal(A, 0)
    N = A @ A
    print(f"    conteggio corretto = sum_(a,d) N[a,d]^2 con N = M M  ->  "
          f"{float((N * N).sum()):.0f}")
    print(f"    enumerazione esaustiva                              ->  "
          f"{brute_force_hom(dia, 4, A):.0f}")
    print("    Serve materializzare la MATRICE N, non un vettore: e' il")
    print("    salto di costo del caso (3), ed e' il limite del metodo.")

    print(f"\n  [OUTPUT] {args.outdir}/pattern_catalogue.csv")


if __name__ == "__main__":
    main()
