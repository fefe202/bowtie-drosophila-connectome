#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bow-tie motif fra neuroni della stessa area, classificati per direzione, e
confronto dei profili di frequenza fra aree.

Un'area e' un neuropilo, e un neurone appartiene a ogni neuropilo citato nel
suo gruppo: le aree si sovrappongono, il che e' anatomicamente corretto.

Il conteggio usa una formula chiusa con inclusione-esclusione per imporre che
i cinque neuroni siano distinti, dato che dentro una singola area le
connessioni reciproche sono frequenti. Enumerare sarebbe impraticabile: in ME
servirebbero circa 10^10 combinazioni, mentre la formula copre l'intero
connettoma in 4,5 secondi.

--selftest confronta la formula con l'enumerazione esaustiva su grafi
casuali.

Uso:
    python src/motifs_within_area.py --selftest
    python src/motifs_within_area.py --min_neurons 200 --n_random 200 --seed 42
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
from scipy.stats import spearmanr

SIGS = ["FWD", "LAT", "BWD", "mix"]
SIGNATURES = [f"{a}->{b}" for a in SIGS for b in SIGS]


def log(msg):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def c2(x):
    """Coefficiente binomiale (x scelto 2), elementwise, mai negativo."""
    x = np.maximum(x, 0)
    return x * (x - 1) / 2.0


def pair_counts(n_F, n_L, n_B):
    """Numero di coppie per firma, a partire dai conteggi delle tre classi."""
    return {
        "FWD": c2(n_F),
        "LAT": c2(n_L),
        "BWD": c2(n_B),
        "mix": n_F * n_L + n_F * n_B + n_L * n_B,
    }


def pairs_containing(sig, cls, n_F, n_L, n_B):
    """
    Quante coppie di firma `sig` contengono un dato neurone di classe `cls`.
    Serve alla correzione di inclusione-esclusione.
    """
    same = {"F": n_F, "L": n_L, "B": n_B}[cls]
    others = {"F": n_L + n_B, "L": n_F + n_B, "B": n_F + n_L}[cls]
    pure = {"F": "FWD", "L": "LAT", "B": "BWD"}[cls]
    if sig == pure:
        return np.maximum(same - 1, 0)
    if sig == "mix":
        return others
    return np.zeros_like(same, dtype=np.float64)


def counts_by_signature(p, q, r):
    """
    Conteggio esatto dei bow-tie per firma, vettorizzato su tutti i waist.

    p = (p_F, p_L, p_B)   classi degli in-vicini
    q = (q_F, q_L, q_B)   classi degli out-vicini
    r = (r_lo, r_eq, r_hi) vicini reciproci per posizione di livello
    """
    T_A = pair_counts(*p)
    T_D = pair_counts(*q)
    r_lo, r_eq, r_hi = r

    # Un reciproco sotto il waist e' in-classe F e out-classe B, e viceversa.
    recip = [(r_lo, "F", "B"), (r_eq, "L", "L"), (r_hi, "B", "F")]

    # Coppie di reciproci: contributo al termine A == D
    deq = {
        ("FWD", "BWD"): c2(r_lo),
        ("LAT", "LAT"): c2(r_eq),
        ("BWD", "FWD"): c2(r_hi),
        ("mix", "mix"): r_lo * r_eq + r_lo * r_hi + r_eq * r_hi,
    }

    out = {}
    for sA in SIGS:
        for sD in SIGS:
            total = T_A[sA] * T_D[sD]
            shared = np.zeros_like(total)
            for r_c, ci, co in recip:
                shared = shared + r_c * \
                    pairs_containing(sA, ci, *p) * pairs_containing(sD, co, *q)
            total = total - shared + deq.get((sA, sD), 0.0)
            out[f"{sA}->{sD}"] = np.maximum(total, 0.0)
    return out


def class_counts(A_sub, levels, n_lv):
    """
    Per ogni neurone dell'area restituisce i conteggi di in-vicini,
    out-vicini e vicini reciproci suddivisi per livello.
    Costo: 3 * n_lv prodotti matrice-vettore sparsi.
    """
    n = A_sub.shape[0]
    R = A_sub.multiply(A_sub.T)          # archi reciproci
    uniq = np.arange(1, n_lv + 1)
    M_in = np.zeros((n, n_lv))
    M_out = np.zeros((n, n_lv))
    M_rec = np.zeros((n, n_lv))
    for j, lv in enumerate(uniq):
        ind = (levels == lv).astype(np.float64)
        M_in[:, j] = ind @ A_sub          # quanti in-vicini a livello lv
        M_out[:, j] = A_sub @ ind         # quanti out-vicini a livello lv
        M_rec[:, j] = ind @ R
    # maschere "livello del vicino minore / uguale / maggiore di quello del waist"
    lv_idx = levels.astype(int) - 1
    grid = np.arange(n_lv)[None, :]
    lower = grid < lv_idx[:, None]
    equal = grid == lv_idx[:, None]
    higher = grid > lv_idx[:, None]

    p = (( M_in * lower).sum(1), (M_in * equal).sum(1), (M_in * higher).sum(1))
    # per gli archi USCENTI la direzione e' invertita: FF = verso l'alto
    q = ((M_out * higher).sum(1), (M_out * equal).sum(1), (M_out * lower).sum(1))
    r = ((M_rec * lower).sum(1), (M_rec * equal).sum(1), (M_rec * higher).sum(1))
    return p, q, r


def null_profiles(A_sub, lv_sub, n_lv, n_random, rng):
    """
    MODELLO NULLO - permutazione delle etichette di livello dentro l'area.

    Si tiene il GRAFO FISSO (tutti i gradi, la reciprocita', la densita', la
    topologia) e il multinsieme dei livelli dell'area, e si randomizza solo
    QUALE neurone porta quale livello.

    Domanda: l'assegnazione dei livelli e' allineata alla struttura di
    connettivita', oppure il profilo osservato si otterrebbe da una qualunque
    assegnazione degli stessi livelli allo stesso grafo?

    Due proprieta' rendono questo nullo adatto al caso:

    1. Il NUMERO TOTALE di bow-tie non dipende dai livelli: questi
       determinano solo la classificazione in firme, non quante quaterne
       esistono. Il nullo preserva quindi il totale per costruzione e testa
       esattamente la COMPOSIZIONE del profilo, che e' l'oggetto in esame.

    2. Controlla automaticamente l'artefatto P16: se un'area ha quasi tutti i
       neuroni allo stesso livello, permutare le etichette non cambia nulla,
       il profilo nullo coincide con quello osservato e l'arricchimento
       risulta ~1. Il metodo dichiara da solo di non avere informazione,
       invece di produrre una dominanza laterale spuria.

    Restituisce una matrice (n_random x 16) di profili (frazioni).
    """
    out = np.zeros((n_random, len(SIGNATURES)))
    for i in range(n_random):
        perm = rng.permutation(lv_sub)
        counts = counts_by_signature(*class_counts(A_sub, perm, n_lv))
        tot = sum(v.sum() for v in counts.values())
        if tot > 0:
            out[i] = [counts[s].sum() / tot for s in SIGNATURES]
    return out


def benjamini_hochberg(p):
    """q-value BH, procedura step-up."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        j = order[rank]
        prev = min(prev, p[j] * m / (rank + 1))
        q[j] = prev
    return q


def selftest(n_trials=300, seed=7):
    """
    Verifica la formula chiusa contro ENUMERAZIONE ESAUSTIVA su grafi casuali.
    L'inclusione-esclusione per il vincolo di distinzione dei cinque neuroni
    e' intricata: va verificata, non assunta.
    """
    import itertools
    from collections import Counter

    def brute(A, lv):
        n = A.shape[0]
        Ad = A.toarray()
        def d(x, y):
            return ("FWD" if lv[x] < lv[y]
                    else ("BWD" if lv[x] > lv[y] else "LAT"))
        def sig(ds):
            return ds[0] if ds[0] == ds[1] else "mix"
        out = Counter()
        for b in range(n):
            ins = [u for u in range(n) if Ad[u, b]]
            outs = [v for v in range(n) if Ad[b, v]]
            for a1, a2 in itertools.combinations(ins, 2):
                for d1, d2 in itertools.combinations(outs, 2):
                    if len({a1, a2, d1, d2, b}) != 5:
                        continue
                    out[f"{sig([d(a1, b), d(a2, b)])}->"
                        f"{sig([d(b, d1), d(b, d2)])}"] += 1
        return out

    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(n_trials):
        n = int(rng.integers(5, 11))
        nl = 4
        lv = rng.integers(1, nl + 1, size=n)
        M = (rng.random((n, n)) < 0.35).astype(float)
        np.fill_diagonal(M, 0)
        A = sparse.csr_matrix(M)
        if A.nnz == 0:
            continue
        got = counts_by_signature(*class_counts(A, lv, nl))
        exp = brute(A, lv)
        for k in set(got) | set(exp):
            g = float(got[k].sum()) if k in got else 0.0
            if abs(g - float(exp.get(k, 0))) > 1e-6:
                bad += 1
    if bad:
        print(f"  [SELF-TEST] FALLITO: {bad} discrepanze")
        return False
    print(f"  [SELF-TEST] {n_trials} grafi casuali verificati contro "
          f"enumerazione esaustiva: i conteggi per tutte e 16 le firme, con il "
          f"vincolo dei cinque neuroni distinti, COINCIDONO esattamente.")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Approccio B: bow-tie motif DENTRO ogni singola area.")
    ap.add_argument("--selftest", action="store_true",
                    help="Verifica la formula chiusa contro enumerazione "
                         "esaustiva su grafi casuali, poi esce.")
    ap.add_argument("--min_neurons", type=int, default=200,
                    help="Numero minimo di neuroni perche' un'area sia analizzata")
    ap.add_argument("--n_random", type=int, default=0,
                    help="Randomizzazioni del modello nullo per il profilo "
                         "(permutazione dei livelli). 0 = salta.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_areas", type=int, default=20,
                    help="Quante aree mostrare a schermo")
    ap.add_argument("--outdir", default=results("within_area_results"))
    ap.add_argument("--conn_file", default=data("connections.csv"))
    ap.add_argument("--levels_file",
                    default=data("COORDINATE_XY_with_levels_tree.csv"))
    ap.add_argument("--neurons_file", default=data("neurons.csv"))
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    print("=" * 78)
    print("  APPROCCIO B - BOW-TIE MOTIF DENTRO LA SINGOLA AREA")
    print("=" * 78)

    log("Caricamento...")
    dn = pd.read_csv(args.neurons_file, usecols=["root_id", "group"]).dropna()
    dn = dn[dn["group"] != "NO_CONS"]
    dl = pd.read_csv(args.levels_file, usecols=["root_id", "y_level"])
    lev = dict(zip(dl["root_id"], dl["y_level"].astype(int)))

    # un neurone appartiene a ogni neuropilo citato nel suo gruppo
    area_members = defaultdict(list)
    for rid, grp in zip(dn["root_id"].to_numpy(), dn["group"].to_numpy()):
        if rid not in lev:
            continue
        for part in str(grp).split("."):
            if part:
                area_members[part].append(rid)
    log(f"  {len(area_members):,} neuropili distinti")

    valid = sorted(set(lev) & set(dn["root_id"]))
    idx = {n: i for i, n in enumerate(valid)}
    levels_all = np.array([lev[n] for n in valid])
    n_lv = int(levels_all.max())

    dc = pd.read_csv(args.conn_file,
                     usecols=["pre_root_id", "post_root_id", "syn_count"])
    e = dc[dc["syn_count"] > 0][["pre_root_id", "post_root_id"]].drop_duplicates()
    e = e[e["pre_root_id"].isin(idx) & e["post_root_id"].isin(idx)]
    A = sparse.csr_matrix(
        (np.ones(len(e)), (e["pre_root_id"].map(idx).astype(int),
                           e["post_root_id"].map(idx).astype(int))),
        shape=(len(valid), len(valid))).tocsr()
    log(f"  {len(valid):,} neuroni, {A.nnz:,} archi, {n_lv} livelli")

    areas = {a: m for a, m in area_members.items()
             if len(m) >= args.min_neurons}
    log(f"Aree con almeno {args.min_neurons} neuroni: {len(areas)}")

    rows, null_rows = [], []
    rng = np.random.default_rng(args.seed)
    for k, (area, members) in enumerate(sorted(areas.items())):
        ids = np.array(sorted(idx[m] for m in members))
        A_sub = A[ids, :][:, ids]
        if A_sub.nnz == 0:
            continue
        lv_sub = levels_all[ids]
        p, q, r = class_counts(A_sub, lv_sub, n_lv)
        counts = counts_by_signature(p, q, r)
        tot = sum(v.sum() for v in counts.values())
        lvc = np.bincount(lv_sub, minlength=n_lv + 1)[1:]
        row = {"area": area, "n_neuroni": len(ids), "n_archi": int(A_sub.nnz),
               "quota_livello_modale": float(lvc.max() / lvc.sum()),
               "motif_totali": tot,
               "motif_per_neurone": tot / len(ids) if len(ids) else 0.0}
        for s, v in counts.items():
            row[s] = float(v.sum())
        if args.n_random > 0:
            obs = np.array([counts[s].sum() / tot for s in SIGNATURES]) \
                if tot > 0 else np.zeros(len(SIGNATURES))
            nul = null_profiles(A_sub, lv_sub, n_lv, args.n_random, rng)
            mean = nul.mean(axis=0)
            for j, s in enumerate(SIGNATURES):
                # p a una coda per sovra-rappresentazione della firma
                ge = int((nul[:, j] >= obs[j]).sum())
                null_rows.append({
                    "area": area, "firma": s,
                    "quota_osservata": obs[j],
                    "quota_nulla_media": mean[j],
                    "quota_nulla_sd": nul[:, j].std(),
                    "arricchimento": obs[j] / mean[j] if mean[j] > 0 else np.nan,
                    "p": (1 + ge) / (args.n_random + 1),
                })

        rows.append(row)
        if (k + 1) % 20 == 0:
            log(f"  {k+1}/{len(areas)} aree")

    df = pd.DataFrame(rows).sort_values("motif_totali", ascending=False)
    df = df[df["motif_totali"] > 0].reset_index(drop=True)
    log(f"Aree con almeno un motif: {len(df)}")

    # ---- profili e arricchimenti
    prof = df[SIGNATURES].div(df["motif_totali"], axis=0)
    prof.insert(0, "area", df["area"])
    globale = df[SIGNATURES].sum() / df["motif_totali"].sum()
    enr = prof[SIGNATURES].div(globale.replace(0, np.nan), axis=1)
    enr.insert(0, "area", df["area"])

    df.to_csv(os.path.join(args.outdir, "within_area_counts.csv"), index=False)
    prof.to_csv(os.path.join(args.outdir, "within_area_profiles.csv"), index=False)
    enr.to_csv(os.path.join(args.outdir, "within_area_enrichment.csv"), index=False)

    print("\n  [AREE PER NUMERO DI MOTIF]")
    show = df.head(args.top_areas)[
        ["area", "n_neuroni", "n_archi", "motif_totali", "motif_per_neurone"]].copy()
    show["motif_totali"] = show["motif_totali"].map("{:.3e}".format)
    show["motif_per_neurone"] = show["motif_per_neurone"].map("{:.3e}".format)
    print(show.to_string(index=False))

    print("\n  [PROFILO: % dei motif dell'area per firma direzionale]")
    keep = [s for s in SIGNATURES if globale.get(s, 0) > 0.005]
    pv = prof.head(args.top_areas).set_index("area")[keep] * 100
    print(pv.round(1).to_string())
    print(f"\n    riga di riferimento (tutte le aree insieme):")
    print("    " + "  ".join(f"{s}={globale[s]*100:.1f}%" for s in keep))

    # Controllo: il profilo e' solo un riflesso della distribuzione
    # dei livelli dentro l'area? Se un'area ha quasi tutti i neuroni allo
    # stesso livello, la dominanza laterale e' FORZATA per costruzione:
    # ogni arco interno e' per forza laterale. In quel caso il profilo non
    # e' una scoperta sulla funzione dell'area, ma un riflesso di come i
    # livelli sono stati assegnati.
    rho, pv_ = spearmanr(df["quota_livello_modale"], prof["lat->lat"])
    print("\n  [CONTROLLO: il profilo e' solo un riflesso dei livelli?]")
    print(f"    Spearman(concentrazione sul livello modale, quota lat->lat) "
          f"= {rho:+.3f}   p = {pv_:.3g}")
    if pv_ < 0.05:
        print("    CORRELAZIONE SIGNIFICATIVA: le aree con livelli concentrati")
        print("    hanno profilo laterale per costruzione. Il loro profilo NON")
        print("    va presentato come caratterizzazione funzionale.")
    info = df[["area", "quota_livello_modale"]].copy()
    info["lat_lat"] = prof["lat->lat"].to_numpy()
    strong = info[(info["lat_lat"] > 0.25) & (info["quota_livello_modale"] < 0.75)]
    forced = info[info["quota_livello_modale"] >= 0.75].nlargest(5, "lat_lat")
    print("\n    aree con profilo laterale marcato NONOSTANTE livelli "
          "distribuiti (le informative):")
    print("      " + (", ".join(f"{r.area} (conc.={r.quota_livello_modale:.2f}, "
                                f"lat->lat={r.lat_lat:.0%})"
                                for r in strong.itertuples())
                      if len(strong) else "nessuna"))
    print("    aree il cui profilo laterale e' spiegato dalla concentrazione:")
    print("      " + (", ".join(f"{r.area} (conc.={r.quota_livello_modale:.2f})"
                                for r in forced.itertuples())
                      if len(forced) else "nessuna"))

    print("\n  [FIRMA PIU' CARATTERISTICA DI CIASCUNA AREA]")
    print("    (arricchimento = frazione nell'area / frazione globale;")
    print("     si considerano solo firme con almeno l'1% dei motif dell'area)")
    car = []
    for _, row in enr.iterrows():
        a = row["area"]
        pr = prof.loc[prof["area"] == a, SIGNATURES].iloc[0]
        cand = [s for s in SIGNATURES if pr[s] >= 0.01 and np.isfinite(row[s])]
        if not cand:
            continue
        best = max(cand, key=lambda s: row[s])
        car.append({"area": a, "firma": best,
                    "arricchimento": row[best],
                    "quota_nell_area": pr[best],
                    "n_neuroni": int(df.loc[df["area"] == a, "n_neuroni"].iloc[0])})
    dc_ = pd.DataFrame(car).sort_values("arricchimento", ascending=False)
    with pd.option_context("display.width", 200):
        print(dc_.head(args.top_areas).round(3).to_string(index=False))
    dc_.to_csv(os.path.join(args.outdir, "within_area_characteristic.csv"),
               index=False)

    # ---- modello nullo: permutazione dei livelli
    if null_rows:
        nd = pd.DataFrame(null_rows)
        nd["q"] = benjamini_hochberg(nd["p"].to_numpy())
        nd.to_csv(os.path.join(args.outdir, "within_area_null.csv"), index=False)

        print("\n" + "=" * 78)
        print("  MODELLO NULLO - permutazione delle etichette di livello")
        print("=" * 78)
        print(f"  {args.n_random} permutazioni per area, seed {args.seed}")
        print(f"  Grafo tenuto FISSO: gradi, reciprocita', densita', topologia.")
        print(f"  Randomizzato solo QUALE neurone porta quale livello.")
        print(f"  Il totale dei bow-tie e' invariante per costruzione: si testa")
        print(f"  esclusivamente la composizione del profilo.")
        print(f"\n  Test: {len(nd)}  |  p minimo possibile: "
              f"{1/(args.n_random+1):.4f}  |  significativi q<0.05: "
              f"{int((nd['q'] < 0.05).sum())}")

        sig = nd[(nd["q"] < 0.05) & (nd["quota_osservata"] >= 0.05)]
        sig = sig.sort_values("arricchimento", ascending=False)
        print("\n  [FIRME SIGNIFICATIVAMENTE SOVRA-RAPPRESENTATE]")
        print("    (q<0.05 e almeno il 5% dei motif dell'area)")
        if len(sig):
            v = sig.head(args.top_areas).copy()
            for c in ("quota_osservata", "quota_nulla_media", "arricchimento"):
                v[c] = v[c].map("{:.3f}".format)
            print(v[["area", "firma", "quota_osservata", "quota_nulla_media",
                     "arricchimento", "q"]].to_string(index=False))
        else:
            print("    nessuna")

        print("\n  [AREE SENZA INFORMAZIONE: profilo indistinguibile dal nullo]")
        print("    (arricchimento ~1 su tutte le firme: i livelli non sono")
        print("     allineati alla topologia, oppure sono tutti uguali)")
        flat = nd.groupby("area")["arricchimento"].apply(
            lambda x: float(np.nanmax(np.abs(np.log(x.replace(0, np.nan))))))
        piatte = flat[flat < 0.35].index.tolist()
        print("    " + (", ".join(piatte) if piatte else "nessuna"))

    print(f"\n  [OUTPUT] {args.outdir}/")
    print(f"  Tempo totale: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
