#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
thesis_paths.py
===============
Percorsi del progetto, risolti rispetto alla posizione di QUESTO file e non
alla directory di lavoro. Serve a poter lanciare gli script da qualunque
cartella senza che i default dei file di input si rompano.

    tesi/
    |-- src/      <- questo file
    |-- data/     <- CSV di input
    |-- results/  <- output
    `-- logs/

Le variabili d'ambiente THESIS_DATA e THESIS_RESULTS permettono di puntare
altrove, per esempio se i dati stanno su un disco esterno:

    THESIS_DATA=/mnt/dati python src/hourglass_core.py --tau 0.9
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("THESIS_DATA", os.path.join(ROOT, "data"))
RESULTS_DIR = os.environ.get("THESIS_RESULTS", os.path.join(ROOT, "results"))
LOGS_DIR = os.path.join(ROOT, "logs")


def data(name):
    """Percorso di un file di input."""
    return os.path.join(DATA_DIR, name)


def results(name):
    """Percorso di una cartella o file di output."""
    return os.path.join(RESULTS_DIR, name)
