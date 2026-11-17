import re

import numpy as np

COMPARTMENTS = ["nucleus", "cytosol", "ER_membrane", "neurite"]
OUTPUT = "prob"
MAX_LEN = 8000
CROP = "3p"
BATCH_SIZE = 256
IS_DEMO = True

_MOTIFS = {
    "PUM": re.compile(r"TGTA[ACT]ATA"),
    "IGF2BP_GGAC": re.compile(r"[AG]GGAC"),
    "A2RE": re.compile(r"GCCAAGGAGCC"),
    "QKI": re.compile(r"ACTAA[CT]"),
    "ARE": re.compile(r"ATTTA"),
}
_W = np.array([
    [-0.30,  0.10,  0.00,  0.60],
    [-0.10,  0.20,  0.10,  0.50],
    [-0.20,  0.00,  0.00,  1.50],
    [ 0.40, -0.20,  0.00,  0.30],
    [ 0.10,  0.30, -0.10,  0.00],
])


def load():
    return None


def predict(model, seqs):
    out = np.zeros((len(seqs), len(COMPARTMENTS)))
    for i, s in enumerate(seqs):
        L = max(len(s), 1) / 1000.0
        counts = np.array([len(m.findall(s)) for m in _MOTIFS.values()]) / L
        gc = (s.count("G") + s.count("C")) / max(len(s), 1)
        logits = counts @ _W + np.array([2.0 * (gc - 0.45), 0.0, 0.5 * (gc - 0.45), -0.5])
        out[i] = 1.0 / (1.0 + np.exp(-logits))
    return out
