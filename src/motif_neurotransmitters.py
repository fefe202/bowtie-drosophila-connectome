#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Composizione eccitatoria/inibitoria dei motif e dei gruppi anatomici, dai
neurotrasmettitori predetti in neurons.csv.

Due modalita'. Con --mode global si calcola la composizione E/I per gruppo e
per direzione dell'arco. Con --mode motifs si analizzano i pattern del
distillato, decomponendo il conteggio di ciascuno sulle classi di
neurotrasmettitore del waist.

Il confronto fra tipi di motif si fa a parita' di waist: gruppi anatomici
diversi hanno composizione E/I molto diversa, quindi un confronto grezzo
misura la regione e non il tipo di circuito. La selezione stratificata per
(waist, firma) prodotta dal distillatore serve esattamente a questo.

Uso:
    python src/motif_neurotransmitters.py --mode global
    python src/motif_neurotransmitters.py --mode motifs --window 3-4-5
"""

from thesis_paths import data, results

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from scipy import sparse

from motif_distill import load as load_distilled
from directional_signature import normalize_directions

NT_CLASS_DEFAULT = {
    "ACH": "E",
    "GABA": "I", "GLUT": "I",
    "DA": "M", "SER": "M", "OCT": "M",
}
CLASSES = ["E", "I", "M", "U"]


def log(msg):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_nt_map(neurons_file, min_nt_score, glut_excitatory):
    """root_id -> classe E/I/M/U, dal nt_type consensuale del neurone."""
    mapping = dict(NT_CLASS_DEFAULT)
    if glut_excitatory:
        mapping["GLUT"] = "E"
    df = pd.read_csv(neurons_file,
                     usecols=["root_id", "nt_type", "nt_type_score"])
    nt = df["nt_type"].fillna("")
    score = df["nt_type_score"].fillna(-1.0)
    cls = nt.map(mapping).fillna("U")
    cls = cls.where(score >= min_nt_score, "U")
    return dict(zip(df["root_id"].to_numpy(), cls.to_numpy()))


# ============================================================================
#  MODE GLOBAL
# ============================================================================

def mode_global(args):
    log("Caricamento dati...")
    nt_map = load_nt_map(args.neurons_file, args.min_nt_score,
                         args.glut_excitatory)
    df_lev = pd.read_csv(args.levels_file, usecols=["root_id", "y_level"])
    level = dict(zip(df_lev["root_id"].to_numpy(),
                     df_lev["y_level"].to_numpy().astype(int)))

    log("Aggregazione delle sinapsi per coppia (pre, post)...")
    df = pd.read_csv(args.conn_file,
                     usecols=["pre_root_id", "post_root_id", "syn_count"])
    df = df.groupby(["pre_root_id", "post_root_id"], sort=False)["syn_count"] \
           .sum().reset_index()
    log(f"  {len(df):,} coppie distinte")

    pre = df["pre_root_id"].to_numpy()
    syn = df["syn_count"].to_numpy()
    lv_pre = np.array([level.get(p, -999) for p in pre])
    lv_post = np.array([level.get(p, -999) for p in df["post_root_id"].to_numpy()])
    keep = (lv_pre != -999) & (lv_post != -999)
    syn = syn[keep]
    lv_pre, lv_post = lv_pre[keep], lv_post[keep]
    ntcls = np.array([nt_map.get(p, "U") for p in pre[keep]])
    log(f"  {keep.sum():,} coppie con livello su entrambi gli estremi")

    direction = np.where(lv_pre < lv_post, "FWD",
                         np.where(lv_pre > lv_post, "BWD", "LAT"))
    res = pd.DataFrame({"direction": direction, "nt": ntcls, "syn": syn})

    print("\n" + "=" * 78)
    print("  PUNTO 11 - NEUROTRASMETTITORI PER DIREZIONE DELL'ARCO")
    print("=" * 78)
    print(f"  GLUT = {'ECCITATORIO' if args.glut_excitatory else 'INIBITORIO'}"
          f"   |   soglia nt_type_score = {args.min_nt_score}")

    for label, weights in (("ARCHI", None), ("SINAPSI", "syn")):
        if weights is None:
            tab = pd.crosstab(res["direction"], res["nt"])
        else:
            tab = pd.crosstab(res["direction"], res["nt"],
                              values=res[weights], aggfunc="sum").fillna(0)
        tab = tab.reindex(index=["FWD", "LAT", "BWD"],
                          columns=[c for c in CLASSES if c in tab.columns],
                          fill_value=0)
        pct = tab.div(tab.sum(axis=1), axis=0) * 100
        print(f"\n  [conteggio per {label}]")
        print(tab.astype(np.int64).to_string())
        print(f"\n  [composizione % per {label}]")
        print(pct.round(2).to_string())
        if "E" in tab.columns and "I" in tab.columns:
            print(f"\n  [rapporto E/I per {label}]")
            for d in tab.index:
                den = tab.loc[d, "I"]
                r = tab.loc[d, "E"] / den if den else float("nan")
                print(f"    {d:<8} E/I = {r:.3f}")

    sub = res[res["nt"].isin(["E", "I"])]
    tab_ei = pd.crosstab(sub["direction"], sub["nt"])
    tab_ei = tab_ei.reindex(index=["FWD", "LAT", "BWD"], fill_value=0)
    try:
        from scipy.stats import chi2_contingency, fisher_exact
        chi2, p, dof, _ = chi2_contingency(tab_ei.values)
        print(f"\n  [test] chi-quadro direzione x (E,I): "
              f"chi2 = {chi2:,.1f}, dof = {dof}, p = {p:.3g}")
        odds, pf = fisher_exact(tab_ei.loc[["FWD", "BWD"], ["E", "I"]].values)
        print(f"  [test] Fisher FWD vs BWD: odds ratio = {odds:.3f}, "
              f"p = {pf:.3g}")
        print("         odds ratio > 1  =>  gli archi in avanti sono piu "
              "eccitatori di quelli all'indietro")
    except Exception as e:
        print(f"  [WARN] test non calcolato: {e}")

    out = os.path.join(args.outdir,
                       f"nt_by_direction_score{args.min_nt_score}.csv")
    pd.crosstab(res["direction"], res["nt"]).to_csv(out)
    print(f"\n  [OUTPUT] {out}")

    # Controllo stratificato per regione
    # Il confronto FF vs FB sull'intero connettoma e' confuso con la regione:
    # aree diverse hanno sia composizioni E/I sia bilanci FF/FB diversi.
    # Qui ogni gruppo presinaptico fa da controllo di se stesso: si confronta
    # la frazione di archi eccitatori fra i SUOI archi FF e i SUOI archi FB.
    df_g = pd.read_csv(args.neurons_file, usecols=["root_id", "group"])
    gmap = dict(zip(df_g["root_id"].to_numpy(), df_g["group"].to_numpy()))
    res2 = res.copy()
    res2["grp"] = [gmap.get(p, None) for p in pre[keep]]
    res2 = res2[res2["nt"].isin(["E", "I"]) & res2["grp"].notna()]
    res2 = res2[res2["grp"] != "NO_CONS"]

    rows = []
    for g, sub_g in res2.groupby("grp"):
        ff = sub_g[sub_g["direction"] == "FWD"]
        fb = sub_g[sub_g["direction"] == "BWD"]
        if len(ff) >= args.strat_min and len(fb) >= args.strat_min:
            rows.append({
                "group": g, "n_FWD": len(ff), "n_BWD": len(fb),
                "fracE_FWD": (ff["nt"] == "E").mean(),
                "fracE_BWD": (fb["nt"] == "E").mean(),
            })
    st = pd.DataFrame(rows)
    print("\n" + "-" * 78)
    print("  CONTROLLO STRATIFICATO PER GRUPPO PRESINAPTICO")
    print("-" * 78)
    if len(st) < 5:
        print(f"  Troppi pochi gruppi con >= {args.strat_min} archi FF e FB "
              f"({len(st)}): controllo non eseguito.")
    else:
        st["delta"] = st["fracE_FWD"] - st["fracE_BWD"]
        n_pos = int((st["delta"] > 0).sum())
        print(f"  Gruppi con >= {args.strat_min} archi FWD e "
              f">= {args.strat_min} archi BWD: {len(st)}")
        print(f"  Frazione eccitatoria media, archi FWD: "
              f"{st['fracE_FWD'].mean():.4f}")
        print(f"  Frazione eccitatoria media, archi BWD: "
              f"{st['fracE_BWD'].mean():.4f}")
        print(f"  Delta (FWD - BWD): mediana = {st['delta'].median():+.4f}, "
              f"media = {st['delta'].mean():+.4f}")
        print(f"  Gruppi in cui gli archi FWD sono piu' eccitatori dei BWD: "
              f"{n_pos}/{len(st)} ({100*n_pos/len(st):.1f}%)")
        try:
            from scipy.stats import wilcoxon
            stat, pw = wilcoxon(st["fracE_FWD"], st["fracE_BWD"])
            print(f"  [test] Wilcoxon appaiato: statistica = {stat:,.0f}, "
                  f"p = {pw:.3g}")
            print("         Se il gradiente FWD/BWD sopravvive a questo test,")
            print("         NON e' un artefatto della composizione regionale.")
        except Exception as e:
            print(f"  [WARN] Wilcoxon non calcolato: {e}")
        out3 = os.path.join(args.outdir,
                            f"nt_stratified_score{args.min_nt_score}.csv")
        st.sort_values("delta").to_csv(out3, index=False)
        print(f"  [OUTPUT] {out3}")

    # Decomposizione di kitagawa
    # Il test appaiato esclude i gruppi specializzati in una sola direzione.
    # Qui si usano TUTTI i gruppi e si scompone esattamente il divario
    # P(E|FWD) - P(E|BWD) in due contributi:
    #   COMPOSIZIONE  : quali gruppi emettono archi FWD vs BWD
    #   EFFETTO INTERNO: a parita' di gruppo, differenza fra FWD e BWD
    # Identita' algebrica esatta (media dei pesi / media delle proporzioni).
    agg = res2.groupby(["grp", "direction"])["nt"].agg(
        n="size", pE=lambda s: (s == "E").mean()).reset_index()
    piv_n = agg.pivot(index="grp", columns="direction", values="n").fillna(0.0)
    piv_p = agg.pivot(index="grp", columns="direction", values="pE")
    for c in ("FWD", "BWD"):
        if c not in piv_n.columns:
            piv_n[c] = 0.0
            piv_p[c] = np.nan
    # dove una direzione manca, si usa la proporzione dell'altra: il termine
    # interno di quel gruppo e' nullo e tutto ricade sulla composizione
    piv_p["FWD"] = piv_p["FWD"].fillna(piv_p["BWD"])
    piv_p["BWD"] = piv_p["BWD"].fillna(piv_p["FWD"])
    ok_rows = piv_p[["FWD", "BWD"]].notna().all(axis=1)
    piv_n, piv_p = piv_n[ok_rows], piv_p[ok_rows]

    wFF = piv_n["FWD"] / piv_n["FWD"].sum()
    wFB = piv_n["BWD"] / piv_n["BWD"].sum()
    pFF, pFB = piv_p["FWD"], piv_p["BWD"]
    P_FF = float((wFF * pFF).sum())
    P_FB = float((wFB * pFB).sum())
    comp = float(((wFF - wFB) * (pFF + pFB) / 2).sum())
    within = float(((wFF + wFB) / 2 * (pFF - pFB)).sum())

    print("\n" + "-" * 78)
    print("  DECOMPOSIZIONE DEL DIVARIO FWD vs BWD (Kitagawa)")
    print("-" * 78)
    print(f"  Gruppi usati: {len(piv_n)}")
    print(f"  P(E | FF) = {P_FF:.4f}")
    print(f"  P(E | FB) = {P_FB:.4f}")
    print(f"  Divario   = {P_FF - P_FB:+.4f}")
    print(f"    di cui COMPOSIZIONE (quali gruppi emettono FF vs FB): "
          f"{comp:+.4f}  ({100*comp/(P_FF-P_FB):.1f}%)")
    print(f"    di cui EFFETTO INTERNO (a parita' di gruppo):          "
          f"{within:+.4f}  ({100*within/(P_FF-P_FB):.1f}%)")
    print(f"    residuo di verifica: {abs((comp+within)-(P_FF-P_FB)):.2e}")

    # composizione E/I per gruppo di connettivita'
    df_grp = pd.read_csv(args.neurons_file, usecols=["root_id", "group"])
    df_grp["nt"] = df_grp["root_id"].map(nt_map).fillna("U")
    df_grp["y_level"] = df_grp["root_id"].map(level)
    df_grp = df_grp.dropna(subset=["group", "y_level"])
    df_grp = df_grp[df_grp["group"] != "NO_CONS"]
    comp = pd.crosstab([df_grp["group"], df_grp["y_level"].astype(int)],
                       df_grp["nt"])
    comp["n"] = comp.sum(axis=1)
    for c in CLASSES:
        if c in comp.columns:
            comp[f"frac_{c}"] = comp[c] / comp["n"]
    out2 = os.path.join(args.outdir,
                        f"nt_by_group_score{args.min_nt_score}.csv")
    comp.to_csv(out2)
    print(f"  [OUTPUT] {out2}")

    big = comp[comp["n"] >= 100]
    if "frac_I" in big.columns:
        cols = ["n", "frac_E", "frac_I"]
        print("\n  [I 12 GRUPPI PIU' INIBITORI (>=100 neuroni)]")
        print(big.nlargest(12, "frac_I")[cols].round(3).to_string())
        print("\n  [I 12 GRUPPI PIU' ECCITATORI (>=100 neuroni)]")
        print(big.nlargest(12, "frac_E")[cols].round(3).to_string())


# ============================================================================
#  MODE MOTIFS
# ============================================================================

def select_patterns(path, per_type_top, chunksize=1_000_000):
    """Top-N pattern per ciascun hourglass_type, letti in chunk."""
    log(f"Selezione dei top {per_type_top} pattern per tipo da {path}")
    best = {}
    rows = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        rows += len(chunk)
        normalize_directions(chunk)
        for t, s in chunk.groupby("hourglass_type"):
            top = s.nlargest(per_type_top, "motif_count")
            best[t] = top if t not in best else \
                pd.concat([best[t], top]).nlargest(per_type_top, "motif_count")
    log(f"  {rows:,} righe scansionate")
    df = pd.concat(best.values()).reset_index(drop=True)
    log("  selezionati " + str(len(df)) + " pattern: " +
        ", ".join(f"{t}={len(g)}" for t, g in df.groupby("hourglass_type")))
    return df


def select_patterns_per_waist(path, top_m, chunksize=1_000_000):
    """
    Top-M pattern per OGNI combinazione (waist, hourglass_type).

    Serve al confronto A PARITA' DI WAIST: selezionando i top-N globali per
    tipo si finisce per confrontare regioni cerebrali diverse (i top mixed
    stanno quasi tutti in LO/MB_ML, i top pure_FWD in AL.MB_CA), non tipi di
    motif. Qui ogni gruppo centrale fa da controllo di se stesso.
    """
    log(f"Selezione dei top {top_m} pattern per (waist, tipo) da {path}")
    best = {}
    rows = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        rows += len(chunk)
        key = (chunk["bottleneck_area"].astype(str) + "|" +
               chunk["bottleneck_level"].astype(str) + "|" +
               chunk["hourglass_type"].astype(str))
        for k, s in chunk.groupby(key):
            top = s.nlargest(top_m, "motif_count")
            best[k] = top if k not in best else \
                pd.concat([best[k], top]).nlargest(top_m, "motif_count")
    log(f"  {rows:,} righe scansionate, {len(best):,} celle (waist x tipo)")
    df = pd.concat(best.values()).reset_index(drop=True)
    log("  selezionati " + str(len(df)) + " pattern: " +
        ", ".join(f"{t}={len(g)}" for t, g in df.groupby("hourglass_type")))
    return df


def within_waist_analysis(res, outdir, min_nt_score):
    """
    Confronto fra tipi di motif A PARITA' DI WAIST.

    Per ogni gruppo centrale si calcola la frazione eccitatoria del waist
    separatamente per ciascun tipo di motif, poi si confrontano i tipi a
    coppie usando SOLO i waist in cui entrambi i tipi sono presenti
    (test appaiato di Wilcoxon). Cosi' la differenza regionale e' eliminata
    per costruzione.
    """
    from itertools import combinations as _comb
    res = res.copy()
    res["waist"] = res["bottleneck_area"].astype(str) + "L" + \
        res["bottleneck_level"].astype(str)

    def wavg_E(x):
        w = x["motif_count"].to_numpy(dtype=float)
        return float(np.average(x["waist_E"].to_numpy(), weights=w)) \
            if w.sum() > 0 else np.nan

    piv = (res.groupby(["waist", "hourglass_type"])[["waist_E", "motif_count"]]
           .apply(wavg_E).unstack())

    print("\n" + "=" * 78)
    print("  CONFRONTO A PARITA' DI WAIST  (frazione eccitatoria del waist)")
    print("=" * 78)
    print(f"  Waist distinti nella selezione: {len(piv)}")
    print("  Waist per tipo di motif: " +
          ", ".join(f"{c}={int(piv[c].notna().sum())}" for c in piv.columns))

    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    rows = []
    print(f"\n  {'confronto':<28} {'n waist':>8} {'media A':>9} {'media B':>9} "
          f"{'delta med.':>11} {'A>B':>9} {'p':>10}")
    print("  " + "-" * 78)
    for t1, t2 in _comb(sorted(piv.columns), 2):
        both = piv[[t1, t2]].dropna()
        if len(both) < 5:
            continue
        d = both[t1] - both[t2]
        p = np.nan
        if wilcoxon is not None and (d != 0).any():
            try:
                p = wilcoxon(both[t1], both[t2]).pvalue
            except Exception:
                pass
        frac = (d > 0).mean()
        rows.append({"tipo_A": t1, "tipo_B": t2, "n_waist": len(both),
                     "media_A": both[t1].mean(), "media_B": both[t2].mean(),
                     "delta_mediano": d.median(), "quota_A_maggiore": frac,
                     "p_wilcoxon": p})
        print(f"  {t1+' vs '+t2:<28} {len(both):>8} {both[t1].mean():>9.4f} "
              f"{both[t2].mean():>9.4f} {d.median():>+11.4f} "
              f"{frac:>8.1%} {p:>10.3g}")

    out = os.path.join(outdir, f"waist_paired_score{min_nt_score}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    piv.to_csv(os.path.join(outdir, f"waist_fracE_by_type_score{min_nt_score}.csv"))
    print(f"\n  [OUTPUT] {out}")

    # scomposizione del divario pure_FWD vs pure_BWD
    if "pure_FWD" in piv.columns and "pure_BWD" in piv.columns:
        tot = res.groupby("hourglass_type").apply(
            lambda g: np.average(g["waist_E"], weights=g["motif_count"]))
        both = piv[["pure_FWD", "pure_BWD"]].dropna()
        print("\n  [SCOMPOSIZIONE del divario pure_FWD - pure_BWD]")
        print(f"    divario grezzo (tutti i waist)        : "
              f"{tot.get('pure_FWD', np.nan) - tot.get('pure_BWD', np.nan):+.4f}")
        print(f"    divario a parita' di waist ({len(both)} waist): "
              f"{(both['pure_FWD'] - both['pure_BWD']).mean():+.4f}")
        print("    Se il secondo e' molto minore del primo, il divario grezzo")
        print("    era un effetto di composizione regionale, non di tipo di motif.")


def mode_motifs(args):
    t0 = time.time()
    nt_map = load_nt_map(args.neurons_file, args.min_nt_score,
                         args.glut_excitatory)
    if args.motifs:
        df_hg = (select_patterns_per_waist(args.motifs, args.per_waist_top)
                 if args.per_waist_top > 0
                 else select_patterns(args.motifs, args.per_type_top))
    else:
        # stratificato per (waist, firma): il confronto a parita' di waist
        # richiede che ogni tipo sia rappresentato in ciascun waist
        df_hg = load_distilled(args.window, "top_per_waist_firma")
        log(f"distillato {args.window}: {len(df_hg):,} pattern")

    k_in = int(df_hg["k_in"].iloc[0])
    k_out = int(df_hg["k_out"].iloc[0])
    patterns = []
    for _, r in df_hg.iterrows():
        patterns.append({
            "fan_in": tuple((r[f"fan_in_{i}_area"], int(r[f"fan_in_{i}_level"]))
                            for i in range(k_in)),
            "bn": (r["bottleneck_area"], int(r["bottleneck_level"])),
            "fan_out": tuple((r[f"fan_out_{i}_area"], int(r[f"fan_out_{i}_level"]))
                             for i in range(k_out)),
        })

    levels_needed = set()
    for p in patterns:
        for k in p["fan_in"] + (p["bn"],) + p["fan_out"]:
            levels_needed.add(k[1])
    log(f"Livelli coinvolti: {sorted(levels_needed)}")

    log("Costruzione del grafo...")
    df_neurons = pd.read_csv(args.neurons_file, usecols=["root_id", "group"])
    df_neurons = df_neurons.dropna(subset=["group"])
    df_neurons = df_neurons[df_neurons["group"] != "NO_CONS"]
    area_map = dict(zip(df_neurons["root_id"], df_neurons["group"]))
    df_lev = pd.read_csv(args.levels_file, usecols=["root_id", "y_level"])
    level_map = dict(zip(df_lev["root_id"], df_lev["y_level"].astype(int)))

    valid = sorted(n for n in (set(area_map) & set(level_map))
                   if level_map[n] in levels_needed)
    idx = {n: i for i, n in enumerate(valid)}
    n_total = len(valid)
    groups = defaultdict(list)
    for n in valid:
        groups[(area_map[n], level_map[n])].append(idx[n])
    nt_arr = np.array([nt_map.get(n, "U") for n in valid])
    log(f"  {n_total:,} neuroni")

    df_conn = pd.read_csv(args.conn_file,
                          usecols=["pre_root_id", "post_root_id", "syn_count"])
    e = df_conn[df_conn["syn_count"] > 0][["pre_root_id", "post_root_id"]] \
        .drop_duplicates()
    e = e[e["pre_root_id"].isin(idx) & e["post_root_id"].isin(idx)]
    A = sparse.csr_matrix(
        (np.ones(len(e)),
         (e["pre_root_id"].map(idx).astype(int),
          e["post_root_id"].map(idx).astype(int))), shape=(n_total, n_total))
    log(f"  {A.nnz:,} archi")

    pairs = set()
    for p in patterns:
        for a in p["fan_in"]:
            pairs.add((a, p["bn"]))
        for d in p["fan_out"]:
            pairs.add((p["bn"], d))
    log(f"Estrazione di {len(pairs):,} sottomatrici...")
    sub = {}
    for s, d in pairs:
        if s in groups and d in groups:
            m = A[groups[s], :][:, groups[d]]
            if m.nnz:
                sub[(s, d)] = m
    log(f"  {len(sub):,} non vuote")

    log("Decomposizione SigmaPi per neurotrasmettitore...")
    out_rows = []
    orig_rows = list(df_hg.to_dict("records"))
    for pi, (p, orig) in enumerate(zip(patterns, orig_rows)):
        bn = p["bn"]
        if bn not in groups:
            continue
        bn_idx = np.array(groups[bn])
        bn_nt = nt_arr[bn_idx]

        # flusso in uscita: il NT e' quello del neurone del waist (Dale)
        prod_out = np.ones(len(bn_idx))
        ok = True
        for d in p["fan_out"]:
            if (bn, d) not in sub:
                ok = False
                break
            prod_out = prod_out * np.asarray(sub[(bn, d)].sum(axis=1)).ravel()
        if not ok or prod_out.sum() == 0:
            continue

        # flusso in ingresso, separato per NT del neurone presinaptico
        vin = []
        for a in p["fan_in"]:
            if (a, bn) not in sub:
                ok = False
                break
            m = sub[(a, bn)]
            a_nt = nt_arr[np.array(groups[a])]
            per_cls = {}
            for c in CLASSES:
                mask = (a_nt == c).astype(np.float64)
                if mask.sum():
                    per_cls[c] = np.asarray(mask @ m).ravel()
            vin.append(per_cls)
        if not ok:
            continue

        total = 0.0
        by_waist = defaultdict(float)
        by_fanin = defaultdict(float)
        for combo in product(CLASSES, repeat=k_in):
            if any(c not in vin[m] for m, c in enumerate(combo)):
                continue
            v = prod_out.copy()
            for m, c in enumerate(combo):
                v = v * vin[m][c]
            if v.sum() == 0:
                continue
            s = set(combo)
            fk = "tutti_E" if s == {"E"} else \
                 "tutti_I" if s == {"I"} else \
                 "con_U" if "U" in s else "misto"
            for c in CLASSES:
                val = float(v[bn_nt == c].sum())
                if val:
                    by_waist[c] += val
                    by_fanin[fk] += val
                    total += val
        if total == 0:
            continue

        row = {
            "structure_str": orig["structure_str"],
            "hourglass_type": orig["hourglass_type"],
            "motif_count": int(orig["motif_count"]),
            "count_ricalcolato": int(round(total)),
            "bottleneck_area": bn[0], "bottleneck_level": bn[1],
        }
        for c in CLASSES:
            row[f"waist_{c}"] = by_waist.get(c, 0.0) / total
        for k in ("tutti_E", "tutti_I", "misto", "con_U"):
            row[f"fanin_{k}"] = by_fanin.get(k, 0.0) / total
        out_rows.append(row)

        if (pi + 1) % 2000 == 0:
            log(f"  {pi+1:,}/{len(patterns):,}")

    res = pd.DataFrame(out_rows)
    out = os.path.join(args.outdir,
                       f"motif_nt_decomposition_score{args.min_nt_score}.csv")
    res.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("  PUNTO 12 - CIRCUITI ECCITATORI / INIBITORI NEI BOW-TIE MOTIF")
    print("=" * 78)
    rel = (res["count_ricalcolato"] - res["motif_count"]).abs() / \
        res["motif_count"].clip(lower=1)
    print(f"  Pattern analizzati: {len(res):,}")
    print(f"  Verifica interna (somma sulle classi == motif_count): "
          f"errore relativo max = {rel.max():.2e}, "
          f"esatti = {(rel < 1e-9).sum():,}/{len(res):,}")

    def wavg(g, cols):
        return pd.Series({c: np.average(g[c], weights=g["motif_count"])
                          for c in cols})

    wcols = [f"waist_{c}" for c in CLASSES]
    fcols = [f"fanin_{k}" for k in ("tutti_E", "tutti_I", "misto", "con_U")]

    print("\n  [NT DEL WAIST, pesato sul conteggio, per tipo di motif]")
    a1 = res.groupby("hourglass_type").apply(lambda g: wavg(g, wcols))
    a1.insert(0, "n_pattern", res.groupby("hourglass_type").size())
    print(a1.round(4).to_string())

    print("\n  [COMPOSIZIONE DEL FAN-IN, pesata sul conteggio, per tipo]")
    print(res.groupby("hourglass_type").apply(lambda g: wavg(g, fcols))
          .round(4).to_string())

    print(f"\n  [OUTPUT] {out}")
    within_waist_analysis(res, args.outdir, args.min_nt_score)

    print(f"  Tempo totale: {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(
        description="Neurotrasmettitori e circuiti E/I sui bow-tie motif "
                    "(punti 11-12 di prof_tip.md)")
    ap.add_argument("--mode", choices=["global", "motifs"], required=True)
    ap.add_argument("--window", default="1-2-3",
                    help="Finestra da cui leggere il distillato")
    ap.add_argument("--motifs", default=None,
                    help="CSV riga-per-riga (legacy).")
    ap.add_argument("--per_type_top", type=int, default=2000,
                    help="Quanti pattern per hourglass_type analizzare")
    ap.add_argument("--per_waist_top", type=int, default=0,
                    help="Se >0, seleziona i top-M pattern per OGNI (waist, "
                         "tipo) invece dei top-N globali per tipo: e la "
                         "selezione corretta per il confronto a parita di waist")
    ap.add_argument("--min_nt_score", type=float, default=0.5,
                    help="Soglia su nt_type_score; sotto, il neurone e' 'U'")
    ap.add_argument("--glut-excitatory", action="store_true",
                    help="Tratta GLUT come eccitatorio (convenzione dei "
                         "vertebrati) invece che inibitorio")
    ap.add_argument("--strat_min", type=int, default=50,
                    help="Archi FF e FB minimi perche un gruppo entri nel "
                         "controllo stratificato")
    ap.add_argument("--outdir", default=results("nt_results"))
    ap.add_argument("--conn_file", default=data("connections.csv"))
    ap.add_argument("--levels_file", default=data("COORDINATE_XY_with_levels_tree.csv"))
    ap.add_argument("--neurons_file", default=data("neurons.csv"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.mode == "global":
        mode_global(args)
    else:
        mode_motifs(args)


if __name__ == "__main__":
    main()
