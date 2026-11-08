"""Loaders for the generated datasets.

import sys; sys.path.append("data")
import loader
cfg   = loader.load_config("ivancevic_soliton")
train = loader.load_train("ivancevic_soliton", seed=0)
test  = loader.load_test("ivancevic_soliton")
Y     = loader.exact("ivancevic_soliton", X)
"""
import json
from pathlib import Path

import numpy as np

try:  # works both as a package (from data import loader) and as a plain module
    from .generate_data import exact  # noqa: F401
except ImportError:
    from generate_data import exact  # noqa: F401

HERE = Path(__file__).resolve().parent


def load_config(problem):
    return json.loads((HERE / problem / "config.json").read_text())


def _load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def load_train(problem, seed):
    return _load(HERE / problem / f"train_seed{seed}.npz")


def load_test(problem):
    return _load(HERE / problem / "test.npz")
