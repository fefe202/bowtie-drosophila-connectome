#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Costruisce la presentazione della discussione.

Cinque minuti, otto slide di contenuto piu' il titolo, che non si conta
perche' la legge il presidente di commissione.

Le slide sono in inglese, come la tesi, e percio' i numeri usano il punto
decimale: 0.498, non 0,498. Il parlato nelle note e' in inglese anche lui;
la versione italiana resta in `slides/parlato_it.md`.

Chi ascolta non sa nulla dell'argomento, ma non per questo si tolgono i
termini tecnici: si spiegano dove compaiono. «Motif» resta «motif», e
accanto c'e' scritto che cos'e'. Le parole che reggono il discorso sono in
rosso, cosi' si vede dove guardare senza leggere tutto.

Ogni slide deve rispondere a due domande: che cosa sto guardando, e perche'.

I colori vengono dal poster della triennale e dal logo dell'ateneo: rosso
#B61918, grigio #9E9E9D, bianco e quasi-nero.

Uso:
    python slides/build_slides.py
"""

import os
import re

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

QUI = os.path.dirname(os.path.abspath(__file__))
RADICE = os.path.dirname(QUI)
FIG_SLIDE = os.path.join(QUI, "figures")
FIG_TESI = os.path.join(RADICE, "thesis", "figures")

# ---------------------------------------------------------------- palette
ROSSO = RGBColor(0xB6, 0x19, 0x18)      # il rosso del logo dell'ateneo
GRIGIO = RGBColor(0x9E, 0x9E, 0x9D)     # il grigio del logo
TESTO = RGBColor(0x1A, 0x1A, 0x1A)
TENUE = RGBColor(0x55, 0x5D, 0x66)
PANNELLO = RGBColor(0xF4, 0xF3, 0xF1)
FILETTO = RGBColor(0xE2, 0xE0, 0xDE)

FONT = "Calibri"

# --------------------------------------------------------------- geometria
W, H = 13.333, 7.5
MARG = 0.72
TITOLO_Y = 0.42
RIGA_Y = 1.32                 # il filetto sotto il titolo
CORPO_Y = 1.58
CORPO_H = H - CORPO_Y - 0.62

# *cosi'* va in rosso grassetto, k§in§ mette "in" a pedice
PEZZI = re.compile(r"(\*[^*]+\*|§[^§]+§)")


def testo(slide, x, y, w, h, righe, size=18, colore=TESTO, bold=False,
          align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, interlinea=1.18,
          spazio=10):
    """Una casella di testo. `righe` e' una lista di (testo, opzioni)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    for i, riga in enumerate(righe):
        if isinstance(riga, str):
            riga = (riga, {})
        t, opt = riga
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opt.get("align", align)
        p.line_spacing = opt.get("interlinea", interlinea)
        p.space_after = Pt(opt.get("spazio", spazio))
        if opt.get("bullet"):
            t = "•   " + t
        for pezzo in PEZZI.split(t):
            if not pezzo:
                continue
            pedice = pezzo.startswith("§")
            forte = pezzo.startswith("*")
            if pedice or forte:
                pezzo = pezzo[1:-1]
            r = p.add_run()
            r.text = pezzo
            f = r.font
            f.name = FONT
            f.size = Pt(opt.get("size", size))
            f.italic = opt.get("italic", False)
            # la parola forte prende rosso e grassetto, il resto resta com'e'
            f.bold = True if forte else opt.get("bold", bold)
            f.color.rgb = ROSSO if forte else opt.get("colore", colore)
            if pedice:
                r.font._rPr.set("baseline", "-25000")
    return tb


def rettangolo(slide, x, y, w, h, riempi, bordo=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                               Inches(w), Inches(h))
    s.shadow.inherit = False
    if riempi is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = riempi
    if bordo is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = bordo
        s.line.width = Pt(1)
    s.text_frame.text = ""
    return s


def _converti(p):
    """Se il .png non c'e' ma c'e' lo stesso nome in un altro formato, lo
    converte. Serve perche' i browser oggi salvano in .webp, che PowerPoint
    non apre: cosi' basta lasciare il file scaricato nella cartella.
    """
    from PIL import Image
    radice = os.path.splitext(p)[0]
    for est in (".webp", ".jpg", ".jpeg", ".avif"):
        alt = radice + est
        if not os.path.exists(alt):
            continue
        # si riconverte anche se il .png c'e' gia', purche' l'originale sia
        # piu' recente: altrimenti un segnaposto vecchio bloccherebbe per
        # sempre l'immagine buona appena scaricata
        if (os.path.exists(p)
                and os.path.getmtime(alt) <= os.path.getmtime(p)):
            return False
        Image.open(alt).convert("RGB").save(p)
        print("    convertita: %s -> %s"
              % (os.path.basename(alt), os.path.basename(p)))
        return True
    return False


def figura(slide, nome, cartella, x, y, w=None, h=None):
    """Inserisce l'immagine rispettandone le proporzioni.

    Se il file non c'e' disegna un riquadro al suo posto, cosi' il mazzo si
    costruisce lo stesso e si vede dove va messa.
    """
    from PIL import Image
    p = os.path.join(cartella, nome)
    _converti(p)
    if not os.path.exists(p):
        w = w or 5.0
        h = h or 3.5
        rettangolo(slide, x, y, w, h, PANNELLO, FILETTO)
        testo(slide, x, y + h / 2 - 0.45, w, 0.9,
              [("manca %s" % nome,
                {"size": 15, "bold": True, "colore": ROSSO,
                 "align": PP_ALIGN.CENTER, "spazio": 3}),
               ("va messa in slides/figures/",
                {"size": 13, "colore": TENUE, "align": PP_ALIGN.CENTER})])
        print("    [manca] %s" % p)
        return None, w, h
    iw, ih = Image.open(p).size
    rapp = iw / float(ih)
    if w is None:
        w = h * rapp
    elif h is None:
        h = w / rapp
    else:                       # entrambi dati: si sta dentro al riquadro
        if w / h > rapp:
            w = h * rapp
        else:
            h = w / rapp
    return slide.shapes.add_picture(p, Inches(x), Inches(y),
                                    Inches(w), Inches(h)), w, h


def nuova(prs, titolo=None, numero=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    if titolo is not None:
        testo(slide, MARG, TITOLO_Y, W - 2 * MARG, 0.8,
              [(titolo, {"size": 30, "bold": True, "colore": ROSSO})])
        rettangolo(slide, MARG, RIGA_Y, 2.4, 0.045, ROSSO)
        rettangolo(slide, MARG + 2.4, RIGA_Y, W - 2 * MARG - 2.4, 0.045,
                   FILETTO)
    if numero is not None:
        testo(slide, W - MARG - 1.2, H - 0.62, 1.2, 0.35,
              [(str(numero), {"size": 12, "colore": GRIGIO,
                              "align": PP_ALIGN.RIGHT})])
    return slide


# ----------------------------------------------------------- il parlato
# Indicizzato sulla posizione della slide. Fra parentesi quadre il tempo.
PARLATO = {
    0: "[read out by the chair: do not read it again]",

    1: "[40 s]  This is a connectome: the complete map of every neuron in a "
       "brain and every connection between them, traced one synapse at a "
       "time. For the adult fruit fly it was finished in twenty twenty-four, "
       "and it is the first one ever completed for an adult brain of this "
       "size. Everything I will show is computed on that map and on nothing "
       "else: no recordings, no experiments, only the wiring. "
       "And the question I ask of it is whether the brain funnels its "
       "traffic through a few central elements, the way an hourglass does.",

    2: "[35 s]  Two things before the results. First, every neuron carries a "
       "level, from one near the senses to five near the muscles, so the "
       "brain has a front-to-back order. Second, a single neuron tells you "
       "nothing and a hundred and thirty-four thousand are too many, so I "
       "group them: one brain region at one level makes one node. Sixteen "
       "hundred nodes remain, and this is the map. Circle size is how many "
       "neurons; line thickness is how strong the connection. A motif is a "
       "small shape of connections that keeps recurring on this map.\n\n"
       "[se chiedono perche' la tesi dice 879] Quello e' il conto sulla "
       "sola finestra 1-2-3, dove si contano i motif. Qui la mappa e' su "
       "tutti e cinque i livelli, e i metanodi sono 1.698.",

    3: "[40 s]  Now, an hourglass is a claim about shape, and shape can be "
       "measured in two different ways. Locally I take one motif at a time "
       "and ask how often several regions converge on a single middle one, "
       "which then fans out again: that tells me which structures are "
       "bottlenecks. Globally I ask the same of the whole network, and that "
       "tells me whether there is a bottleneck at all: a hundred and fifty-four "
       "neurons cover ninety per cent of the routes, which scores zero point "
       "five, against zero point seven-four to eight-seven for the worm C. "
       "elegans. Neither view answers the question alone.",

    4: "[30 s]  Back to the local view. Each motif has two branches, the one "
       "entering the middle and the one leaving it. Labelling all four "
       "connections at once is useless, because it puts ninety-four per cent "
       "of the motifs into a single bucket. So I label the two branches "
       "separately: each can run deeper, run back towards the senses, stay "
       "level, or be mixed. Four by four gives sixteen signatures.",

    5: "[35 s]  And this is the main local result. I compare the shallow part "
       "of the brain with the deep part. Near the senses the flow runs "
       "forward, as you would expect. Deep down it flips: signatures with a "
       "sideways branch go from thirty-eight per cent to eighty-three per "
       "cent of the traffic. The deep brain is not pushing the signal "
       "onwards. It is stirring it in place.",

    6: "[25 s]  Do the two views agree? Out of six hundred and sixty "
       "regions, both put the same four near the top, and all four sit on "
       "the olfactory pathway. Overall the agreement is weak, so I am not "
       "claiming the two measures are interchangeable. But where they do "
       "agree, they agree on something that matters.",

    7: "[45 s]  Which is the last result. So far every motif had the same "
       "number of branches in and out. Letting those vary gives an index "
       "that is negative when a circuit fans out and positive when it "
       "funnels in, computed from the wiring only. I applied it to the "
       "olfactory pathway because it is the one circuit whose answer was "
       "already known from physiology, so it is a test I could fail. I "
       "picked the five rows by anatomy, before looking at any numbers. The "
       "index comes out negative where the circuit expands and at the "
       "maximum where it compresses: the textbook description, recovered "
       "from the wiring alone.\n\n"
       "[if asked about artificial neural networks] The shape is that of a "
       "high-dimensional expansion followed by a readout, but that is a "
       "reading: the thesis does not make the comparison.\n"
       "[if asked how general it is] It is validated on one circuit only, "
       "the one where the answer was already known. The other seven hundred "
       "nodes carry index values too, but nothing here establishes that "
       "those are equally meaningful.",

    8: "[30 s]  To sum up. The hourglass is there, but how narrow it looks "
       "depends on how you trace the routes. Deep in the brain the signal "
       "stops advancing and is stirred instead. And an index built from "
       "wiring alone recovers a circuit already known from physiology. I "
       "also tested whether a connection's direction predicts its "
       "chemistry: it does not hold up, and I report it as a negative "
       "result. The method does not depend on the species. Thank you.",
}


# =========================================================================
def costruisci(out=None):
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)

    # ------------------------------------------------ 0. titolo
    s = nuova(prs)
    rettangolo(s, 0, 0, W, 0.28, ROSSO)
    figura(s, "logo.png", FIG_TESI, W - MARG - 1.05, 0.72, h=1.05)
    testo(s, MARG, 2.05, W - 2 * MARG - 1.4, 2.0,
          [("Bow-Tie Architecture in the",
            {"size": 30, "colore": TESTO, "spazio": 2}),
           ("Drosophila melanogaster Connectome",
            {"size": 30, "colore": TESTO, "italic": True, "spazio": 2}),
           ("A Two-Scale Structural Analysis",
            {"size": 30, "colore": ROSSO, "bold": True, "spazio": 0})])
    rettangolo(s, MARG, 4.42, 3.1, 0.045, ROSSO)
    testo(s, MARG, 4.78, 6.0, 1.6,
          [("Candidate", {"size": 13, "colore": GRIGIO, "spazio": 1}),
           ("Federico Di Franco — student ID 263678",
            {"size": 17, "colore": TESTO, "spazio": 12}),
           ("Supervisor", {"size": 13, "colore": GRIGIO, "spazio": 1}),
           ("Prof. Giorgio Terracina", {"size": 17, "colore": TESTO})])
    testo(s, W - MARG - 4.0, H - 1.0, 4.0, 0.4,
          [("Academic Year 2025/2026",
            {"size": 13, "colore": GRIGIO, "align": PP_ALIGN.RIGHT})])

    # ------------------------------------------------ 1. il connettoma
    s = nuova(prs, "A connectome, and the question I ask of it", 1)
    # si dimensiona in larghezza e non in altezza: l'immagine del cervello
    # e' molto piu' larga che alta, e data in altezza mangerebbe la colonna
    # del testo
    _, fw, fh = figura(s, "connectome.png", FIG_SLIDE,
                       MARG, CORPO_Y, w=6.75, h=CORPO_H - 0.6)
    # centrata in verticale nella fascia del corpo, altrimenti resta un
    # buco bianco sotto
    s.shapes[-1].top = Emu(int(Inches(CORPO_Y + (CORPO_H - 0.5 - fh) / 2)))
    xt = MARG + 6.95
    testo(s, xt, CORPO_Y - 0.05, W - MARG - xt, 5.2,
          [("A *connectome* is the complete map of a brain: every neuron, "
            "and every connection between them, traced one synapse at a "
            "time.", {"size": 19, "spazio": 16}),
           ("For the adult fruit fly it was finished in *2024* — the "
            "first ever for an adult brain of this size.",
            {"size": 19, "spazio": 18}),
           ("134,181 neurons     2,700,513 connections",
            {"size": 21, "bold": True, "colore": ROSSO, "spazio": 18}),
           ("Everything here is computed on that map and *nothing else*: "
            "no experiments, only the wiring.",
            {"size": 18, "spazio": 20}),
           ("The question: does the brain funnel its traffic through "
            "*a few central elements*, the way an *hourglass* does?",
            {"size": 19})])
    testo(s, MARG, H - 0.62, 6.75, 0.35,
          [("rendering: the 50 largest neurons — FlyWire",
            {"size": 11, "colore": GRIGIO, "italic": True})])

    # ------------------------------------------------ 2. la mappa
    s = nuova(prs, "The map I work on", 2)
    # immagine a sinistra e testo a destra, come nella slide del connettoma:
    # affiancati si leggono meglio che impilati
    _, fw, fh = figura(s, "metagraph_slide.png", FIG_SLIDE,
                       MARG, CORPO_Y, w=7.1, h=CORPO_H - 0.2)
    s.shapes[-1].top = Emu(int(Inches(CORPO_Y + (CORPO_H - 0.2 - fh) / 2)))
    xt = MARG + 7.35
    testo(s, xt, CORPO_Y + 0.35, W - MARG - xt, 4.6,
          [("Every neuron carries a *level*, from 1 near the senses to 5 "
            "near the muscles.", {"size": 19, "spazio": 18}),
           ("One brain region at one level makes one node: *1,698 nodes* "
            "instead of 134,181 neurons.", {"size": 19, "spazio": 18}),
           ("A *motif* is a small shape of connections that keeps recurring "
            "between those nodes.", {"size": 19, "spazio": 22}),
           ("Circle size is how many neurons; line thickness is how strong "
            "the connection.", {"size": 16, "colore": TENUE})])

    # ------------------------------------------------ 3. le due scale
    s = nuova(prs, "Two ways to measure the same shape", 3)
    testo(s, MARG, CORPO_Y - 0.06, W - 2 * MARG, 0.5,
          [("“Hourglass” is a claim about *shape*, and shape can "
            "be measured from two distances. *Neither one answers the "
            "question alone.*", {"size": 19})])

    meta = (W - 2 * MARG - 0.75) / 2.0
    ytop = CORPO_Y + 0.72
    for i, (eti, dom) in enumerate([
            ("Close up — one motif at a time",
             "How often do several regions converge on a single middle "
             "region, which then fans out again?"),
            ("Far away — the whole network at once",
             "How few neurons does it take to cover every route from the "
             "senses to the muscles?")]):
        x = MARG + i * (meta + 0.75)
        rettangolo(s, x, ytop, meta, 0.052, ROSSO)
        testo(s, x, ytop + 0.18, meta, 0.42,
              [(eti, {"size": 19, "bold": True, "colore": ROSSO})])
        testo(s, x, ytop + 0.72, meta, 1.0,
              [(dom, {"size": 17, "colore": TENUE})])

    # a sinistra il disegno, a destra i numeri
    _, fw, fh = figura(s, "bowtie_schema.png", FIG_SLIDE,
                       MARG + 0.35, ytop + 1.80, w=meta - 0.7)
    testo(s, MARG, ytop + 1.90 + fh, meta, 0.9,
          [("Tells me *which* structures are bottlenecks.",
            {"size": 18, "align": PP_ALIGN.CENTER})])

    xr = MARG + meta + 0.75
    testo(s, xr, ytop + 1.95, meta, 2.6,
          [("154 neurons", {"size": 32, "bold": True, "colore": ROSSO,
                            "spazio": 2}),
           ("cover 90% of the routes — a score of *H = 0.498*, against "
            "0.74–0.87 for the worm *C. elegans*",
            {"size": 17, "colore": TENUE, "spazio": 20}),
           ("Tells me *whether* the brain has a bottleneck at all.",
            {"size": 18})])

    # ------------------------------------------------ 4. le sedici firme
    s = nuova(prs, "Sixteen directional signatures", 4)
    _, fw, fh = figura(s, "signature_grid_slide.png", FIG_SLIDE,
                       MARG, CORPO_Y - 0.12, h=CORPO_H + 0.10)
    xt = MARG + fw + 0.65
    testo(s, xt, CORPO_Y + 0.3, W - MARG - xt, 4.6,
          [("Labelling all four connections at once is useless: it puts "
            "*94% of motifs* into one bucket.",
            {"size": 18, "spazio": 18}),
           ("So each motif gets *two labels*, one per branch — the one "
            "going in, the one coming out.", {"size": 18, "spazio": 18}),
           ("Each branch runs *deeper*, runs *back*, stays *level*, or is "
            "*mixed*. Four by four: *sixteen signatures*.",
            {"size": 18, "spazio": 18}),
           ("Depth increases downwards, so the signature can be read from "
            "the slope of the arrows alone.",
            {"size": 16, "colore": TENUE})])

    # ------------------------------------------------ 5. il profondo e' laterale
    s = nuova(prs, "The deep brain stirs instead of advancing", 5)
    _, fw, fh = figura(s, "signature_shift_slide.png", FIG_SLIDE,
                       MARG, CORPO_Y - 0.05, h=CORPO_H)
    xt = MARG + fw + 0.6
    testo(s, xt, CORPO_Y + 0.35, W - MARG - xt, 4.6,
          [("Near the senses the flow runs *forward*. Deep down it turns "
            "*sideways*.", {"size": 19, "spazio": 20}),
           ("38.0%  →  82.6%", {"size": 30, "bold": True,
                                     "colore": ROSSO, "spazio": 2}),
           ("of the motif traffic, in signatures with an entirely sideways "
            "branch", {"size": 16, "colore": TENUE, "spazio": 22}),
           ("The deep brain does not push the signal onwards: it *stirs it "
            "in place*.", {"size": 18})])

    # ------------------------------------------------ 6. le scale concordano?
    s = nuova(prs, "Do the two views agree?", 6)
    testo(s, MARG, CORPO_Y, W - 2 * MARG, 0.6,
          [("Out of *660 regions*, both views put the *same four* near the "
            "top — and all four sit on the *olfactory pathway*.",
            {"size": 21})])

    rx, ry, rw, rh = MARG + 0.3, CORPO_Y + 1.0, 8.4, 0.62
    intest = [("Brain region", 3.4), ("rank, close up", 2.5),
              ("rank, far away", 2.5)]
    x = rx
    for t, lw in intest:
        testo(s, x, ry, lw, 0.45,
              [(t, {"size": 15, "colore": GRIGIO,
                    "align": PP_ALIGN.LEFT if x == rx else PP_ALIGN.CENTER})])
        x += lw
    rettangolo(s, rx, ry + 0.48, rw, 0.022, ROSSO)

    righe = [("antennal lobe → mushroom body", "8", "12"),
             ("antennal lobe → lateral horn", "10", "2"),
             ("antennal lobe", "20", "4"),
             ("antennal lobe → posterior lateral", "50", "8")]
    for i, (nome, a, b) in enumerate(righe):
        y = ry + 0.62 + i * rh
        if i % 2 == 0:
            rettangolo(s, rx - 0.12, y - 0.04, rw + 0.24, rh - 0.06,
                       PANNELLO)
        testo(s, rx, y, 3.4, 0.5, [(nome, {"size": 17})])
        testo(s, rx + 3.4, y, 2.5, 0.5,
              [(a, {"size": 19, "bold": True, "colore": ROSSO,
                    "align": PP_ALIGN.CENTER})])
        testo(s, rx + 5.9, y, 2.5, 0.5,
              [(b, {"size": 19, "bold": True, "colore": ROSSO,
                    "align": PP_ALIGN.CENTER})])

    testo(s, MARG, H - 1.35, W - 2 * MARG, 0.9,
          [("Across all regions the agreement is *weak* "
            "(ρ = +0.121), so the two measures are not "
            "interchangeable. But where they agree, they agree on the "
            "circuit that comes next.",
            {"size": 17, "colore": TENUE})])

    # ------------------------------------------------ 7. la via olfattiva
    s = nuova(prs, "Wiring alone recovers the olfactory pathway", 7)
    _, fw, fh = figura(s, "olfactory_slide.png", FIG_SLIDE,
                       MARG, CORPO_Y + 0.22, h=4.35)
    xt = MARG + fw + 0.55
    testo(s, xt, CORPO_Y + 0.02, W - MARG - xt, 4.8,
          [("*Why this circuit:* it is the one whose answer was already "
            "known from physiology — so it is a test I could fail.",
            {"size": 18, "spazio": 22}),
           ("An index, *negative* when a circuit fans out and *positive* "
            "when it funnels in, computed from the *wiring only*.",
            {"size": 18, "spazio": 22}),
           ("The five rows were chosen by anatomy, *before* looking at any "
            "numbers.", {"size": 18, "spazio": 26}),
           ("–0.895  →  +0.061  →  +1.000",
            {"size": 23, "bold": True, "colore": ROSSO, "spazio": 4}),
           ("fans out, holds, funnels in — in that order",
            {"size": 16, "colore": TENUE})])
    testo(s, MARG, H - 0.98, W - 2 * MARG - 1.0, 0.6,
          [("It is the same shape as a high-dimensional expansion followed "
            "by a readout — but the comparison with artificial neural "
            "networks is a reading, not a result of the thesis.",
            {"size": 14.5, "italic": True, "colore": TENUE})])

    # ------------------------------------------------ 8. conclusioni
    s = nuova(prs, "Conclusions and future work", 8)
    # font e interlinea piu' generosi: la slide finiva a due terzi
    # dell'altezza e lo spazio in fondo non serviva a nulla
    testo(s, MARG, CORPO_Y, 6.6, 5.0,
          [("What was found",
            {"size": 23, "bold": True, "colore": ROSSO, "spazio": 18}),
           ("The hourglass is there, but how narrow it looks *depends on "
            "how you trace the routes*: 0.247 to 0.498.",
            {"size": 19, "bullet": True, "spazio": 20}),
           ("Deep in the brain the signal *stops advancing* and is stirred "
            "instead: 38.0% → 82.6%.",
            {"size": 19, "bullet": True, "spazio": 20}),
           ("An index built from *wiring alone* recovers a circuit already "
            "known from physiology.",
            {"size": 19, "bullet": True, "spazio": 20}),
           ("Whether a connection's direction predicts its chemistry was "
            "also tested: it *does not hold up*, and is reported as a "
            "negative result.",
            {"size": 19, "bullet": True, "colore": TENUE})])
    rettangolo(s, MARG + 7.0, CORPO_Y, 0.02, 4.9, FILETTO)
    testo(s, MARG + 7.45, CORPO_Y, W - MARG - (MARG + 7.45), 5.0,
          [("Future work", {"size": 23, "bold": True, "colore": ROSSO,
                            "spazio": 18}),
           ("Repeat the analyses on the *deep* part of the brain, not only "
            "the shallow one.", {"size": 19, "bullet": True, "spazio": 20}),
           ("Let the motif shape vary *across the whole network*, not only "
            "on the olfactory pathway.",
            {"size": 19, "bullet": True, "spazio": 20}),
           ("Apply the same counting to *other connectomes*: the method "
            "does not depend on the species.",
            {"size": 19, "bullet": True})])

    # --------------------------------------------- il parlato, nelle note
    for i, n in PARLATO.items():
        prs.slides[i].notes_slide.notes_text_frame.text = n

    out = out or os.path.join(QUI, "presentazione.pptx")
    try:
        prs.save(out)
    except PermissionError:
        raise SystemExit("Il file e' aperto in PowerPoint e non si puo' "
                         "sovrascrivere:\n  %s\nChiudilo e rilancia, "
                         "oppure passa --out con un altro percorso." % out)
    print("salvata: %s" % out)
    print("slide: %d (la prima e' il titolo e non si conta)"
          % len(prs.slides._sldIdLst))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None,
                    help="dove salvare (default: slides/presentazione.pptx)")
    costruisci(ap.parse_args().out)
