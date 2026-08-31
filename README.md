# bowtie-drosophila-connectome

Structural analysis of the adult *Drosophila melanogaster* connectome, asking whether the network is organised as a **bow-tie**: many inputs converging onto a narrow set of intermediate elements that then diverge towards many outputs.

The question is addressed at two independent scales. Locally, we exhaustively count bow-tie motifs in which `k_in` groups of neurons converge onto one *waist* group that projects onto `k_out` further groups. Globally, we compute the τ-core of the network and its hourglass score. Where the two agree the result is robust; where they disagree, the disagreement is itself informative.

Master thesis, University of Calabria. The thesis source is in [`thesis/thesis.tex`](thesis/thesis.tex).

---

## Results at a glance

| | |
| :--- | :--- |
| Network | 134,181 neurons, 2,700,513 synaptic pairs (FlyWire) |
| Bow-tie motifs counted | up to 512,969,374 on the full hierarchy |
| Hourglass score | **H = 0.498** under structural routing |
| Search speed-up | **219×** on the worst bottleneck, after a sparse reformulation |
| Output footprint | 100 GB reduced to 68 MB with no loss to any analysis |

Exhaustive counts for all four level windows:

| window | neurons | metanodes | motifs | runtime |
| :--- | ---: | ---: | ---: | ---: |
| 1-2-3 | 104,915 | 879 | 31,184,800 | 14m39s |
| 2-3-4 | 108,438 | 1,155 | 119,026,375 | 57m43s |
| 3-4-5 | 73,531 | 1,220 | 433,546,629 | 3h05m |
| 1-2-3-4-5 | 134,181 | — | 512,969,374 | 3h35m |

Motif counts grow with hierarchical depth while neuron counts do not: window 3-4-5 contains fewer neurons than 1-2-3 and yields fourteen times as many motifs. The growth is driven by connection density, which rises from 1.28 to 18.00 edges per 10,000 possible pairs between the shallowest and deepest levels.

---

## Requirements

Python 3.11 or later, with

```
numpy  pandas  scipy  matplotlib  networkx
```

Install them with

```bash
pip install -r requirements.txt
```

Compiling the thesis additionally requires a LaTeX distribution with pdfLaTeX. The source is a single self-contained file with an inlined bibliography.

## Data

The input data are not distributed with this repository. Download the following tables from the [FlyWire Codex](https://codex.flywire.ai) portal and place them in `data/`:

| file | content | size |
| :--- | :--- | ---: |
| `connections.csv` | synaptic pairs with contact counts | 198 MB |
| `neurons.csv` | identifiers, anatomical group, predicted neurotransmitter | 8.6 MB |
| `classification.csv` | functional super-class labels | 9.9 MB |
| `COORDINATE_XY_with_levels_tree.csv` | hierarchical level assignment | 8.3 MB |

`src/thesis_paths.py` resolves paths relative to the project root rather than the working directory, so scripts run from anywhere. To read the data from elsewhere, set `THESIS_DATA`:

```bash
THESIS_DATA=/other/path python src/hourglass_areas_fast.py --window 1-2-3
```

## Quick start

The distilled results for all four windows are committed, so the analyses run without repeating the search:

```bash
python src/compression_realized.py --window 3-4-5
```

```bash
python src/level_transitions.py --window 3-4-5
```

To repeat the search itself, which takes from fifteen minutes to three and a half hours depending on the window:

```bash
python src/hourglass_areas_fast.py --window 3-4-5 \
    --k_in 2 --k_out 2 --max_jump 1 --min_count 10
```

This writes only the distilled output. Add `--full_csv` to also write the row-level CSV, which reaches tens of gigabytes and is needed only for queries the distilled artefacts do not cover.

## Pipeline

```bash
# global hourglass analysis
python src/hourglass_core.py --tau 0.9 --routing levels
python src/hourglass_core.py --tau 0.9 --n_random 100 --tau_null 0.5 --seed 42

# motif search and downstream analyses
python src/hourglass_areas_fast.py --window 3-4-5 --k_in 2 --k_out 2 \
    --max_jump 1 --min_count 10
python src/compression_realized.py    --window 3-4-5
python src/motif_neurotransmitters.py --mode motifs --window 3-4-5
python src/validate_motifs_v2.py      --window 3-4-5 --n_random 100 --seed 42
python src/compare_micro_macro.py     --window 3-4-5 --macro <core_by_group.csv>
python src/level_transitions.py       --window 3-4-5

# motifs within a single area
python src/motifs_within_area.py --min_neurons 200 --n_random 200 --seed 42

# pattern algebra and the integrative/broadcast sweep
python src/pattern_algebra.py --max_nodes 4 --n_verify 25
python src/sweep_kin_kout.py --window 1-2-3 --kmax 5

# figures
python src/visualize_hourglass.py --input results/motif_distilled/3-4-5/top_globale.csv
python src/make_summary_figures.py --lang en --outdir thesis/figures
python src/make_thesis_figures.py --outdir thesis/figures
```

## Method

Counting occurrences of a pattern means counting homomorphisms from the pattern into the graph, and both the cost and the way counts compose depend only on the shape of the pattern:

| shape | how it is counted | memory | counts add over unions |
| :--- | :--- | :--- | :--- |
| path | matrix product | O(n) | yes |
| tree | matvec along edges, Hadamard product at branchings | O(n) | no |
| cyclic | not factorisable over vectors | O(n²) | no |

A bow-tie is a tree, so its count is a sum of products over the waist,

```
count = Σ_i∈B  ( Π_m v_Am[i] ) ( Π_n w_Dn[i] )
```

where `v_Am[i]` is the number of neurons of group `A_m` projecting onto waist neuron `i`. The products force all branches onto the same neuron; the sum aggregates over the waist. Nothing is enumerated and nothing is sampled.

The boundary between the tractable and the intractable is therefore the presence of branching. An exhaustive catalogue up to four nodes shows that 87% of connected patterns are already cyclic, and so outside what this method covers.

`src/hourglass_areas.py` implements the formulation directly and serves as the correctness reference. `src/hourglass_areas_fast.py` is the production engine: it computes all combinations of a waist as a single matrix product delegated to BLAS, and switches to a sparse product where the dense form exceeds the memory budget.

## Verification

Every count is checked rather than assumed.

```bash
python src/hourglass_areas_fast.py --window 4-5 --verify
python src/motifs_within_area.py --selftest
python src/sweep_kin_kout.py --selftest
python src/hourglass_core.py --selftest
```

The `--verify` run compares the fast engine against the reference implementation row by row, across all 1,043,872 rows and all 26 columns, including waists deliberately forced onto the sparse path. The `--selftest` runs compare closed-form counts against exhaustive enumeration on random graphs: 300 graphs for the within-area formula, with all sixteen signatures matching exactly, and 31 of 31 derived formulae for the pattern algebra.

Each analysis also recomputes occurrence counts directly from the graph and compares them against the stored values, so a corrupted intermediate is detected at the point of use.

## Repository layout

```
├── src/                  19 scripts, 6,722 lines
│   └── deprecated/       superseded scripts, kept for traceability
├── thesis/
│   ├── thesis.tex        single-file LaTeX source, 60 pages
│   └── figures/          the 19 figures of the thesis
├── results/
│   ├── motif_distilled/  committed: reproduces every table of the thesis
│   └── figures/          committed: summary figures
├── data/                 FlyWire tables, not committed
├── docs/                 working notes, not committed
└── logs/                 run logs, not committed
```

### Why the row-level output is not kept

Exhaustive counting produces one row per pattern, up to 513 million for the full hierarchy and over 100 GB across the four windows. No downstream analysis reads those rows: every one either aggregates them or takes a top-N subset. The row-level CSV is therefore an intermediate, not a result.

`src/motif_distill.py` retains the aggregates and the top-N selections with complete rows, which comes to 68 MB for all four windows and reproduces every table in the thesis. Since the search now takes from fifteen minutes to three and a half hours, regenerating the raw rows costs less than storing them.

## License

The code, the build files and the distilled results are released under the MIT License; see [LICENSE](LICENSE).

Two things are excluded. The thesis text and its figures, under `thesis/` and `results/figures/`, are the author's academic work: all rights reserved, quotable with attribution but not redistributable. The connectome data expected under `data/` are not distributed here and remain subject to the terms of the [FlyWire Codex](https://codex.flywire.ai) portal.

## References

- Milo, Shen-Orr, Itzkovitz, Kashtan, Chklovskii & Alon (2002). Network
motifs: simple building blocks of complex networks. *Science* 298, 824–827.
- Sabrin, Wei, van den Heuvel & Dovrolis (2020). The hourglass organization of
the *Caenorhabditis elegans* connectome. *PLOS Computational Biology* 16, e1007526.
- Lin et al. (2024). Network statistics of the whole-brain connectome of
*Drosophila*. *Nature*.
- Benjamini & Hochberg (1995). Controlling the false discovery rate. *JRSS B*
57, 289–300.
