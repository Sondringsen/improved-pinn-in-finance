#!/usr/bin/env python3
"""Synthetic data for the three problems of Bai, Chaolu & Bilige (2022)

python data/generate_data.py          # write <problem>/{config.json, test.npz, train_seed*.npz}
python data/generate_data.py --check  # verify that each exact solution solves its PDE
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
SEEDS = range(5)

# Setups from Sect. 3 of the paper.  N_u = n_ic + 2 n_bc supervised points, n_f collocation points.
PROBLEMS = {
    "black_scholes": dict(  # Sect. 3.3: European call, N_V = 200; N_f = 2000 as for the paper's put (call: 10 000)
        params=dict(K=10.0, r=0.1, sigma=0.05, option="call"),
        x=[0.0, 20.0], t=[0.0, 1.0], n_ic=100, n_bc=50, n_f=2000, nx_test=201, nt_test=101),
    "ivancevic_soliton": dict(  # Sect. 3.2: psi = 2 sech(2S + 8t) exp(i(1 - 2S)), N_psi = 100, N_f = 2000
        params=dict(sigma=2.0, beta=2.0, a=2.0, k=-2.0, phi0=1.0),
        x=[-5.0, 5.0], t=[-0.5, 0.5], n_ic=50, n_bc=25, n_f=2000, nx_test=512, nt_test=501),
    "ivancevic_rogue": dict(  # Sect. 3.1: psi = [1 - 4(1 + 2it)/(1 + 4S^2 + 4t^2)] exp(it), N_psi = 100, N_f = 2000
        params=dict(sigma=1.0, beta=1.0, alpha=2**0.5, k=0.0),
        x=[-2.0, 2.0], t=[-1.5, 1.5], n_ic=50, n_bc=25, n_f=2000, nx_test=512, nt_test=501),
}


# ---- exact solutions --------------------------------------------------------------------------
def black_scholes(S, tau, K, r, sigma, option):
    """European option price; tau = time to maturity, so tau = 0 gives the payoff."""
    sgn = 1 if option == "call" else -1
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        price = sgn * (S * ndtr(sgn * d1) - K * np.exp(-r * tau) * ndtr(sgn * d2))
    return np.where(tau > 0, price, np.maximum(sgn * (S - K), 0.0))


def ivancevic_soliton(S, t, sigma, beta, a, k, phi0):
    """Bright soliton of i psi_t + sigma/2 psi_SS + beta |psi|^2 psi = 0."""
    phase = np.exp(1j * (k * S + 0.5 * sigma * (a**2 - k**2) * t + phi0))
    return a * np.sqrt(sigma / beta) / np.cosh(a * (S - sigma * k * t)) * phase


def ivancevic_rogue(S, t, sigma, beta, alpha, k):
    """First-order financial rogue wave of the same equation (eq. 10 of the paper)."""
    u = S - sigma * k * t
    phase = np.exp(1j * (k * S + 0.5 * sigma * (alpha**2 - k**2) * t))
    frac = 4 * (1 + 1j * sigma * alpha**2 * t) / (1 + 2 * alpha**2 * u**2 + sigma**2 * alpha**4 * t**2)
    return alpha * np.sqrt(sigma / (2 * beta)) * (1 - frac) * phase


SOLUTIONS = {"black_scholes": black_scholes, "ivancevic_soliton": ivancevic_soliton, "ivancevic_rogue": ivancevic_rogue}


def exact(problem, X):
    """Exact solution at X (N, 2) with columns (x, t) -> (N, n_out) real array: (V,) or (Re psi, Im psi)."""
    X = np.asarray(X, dtype=float)
    u = SOLUTIONS[problem](X[:, 0], X[:, 1], **PROBLEMS[problem]["params"])
    return np.column_stack([u.real, u.imag]) if np.iscomplexobj(u) else u[:, None]


# ---- sampling ---------------------------------------------------------------------------------
def lhs(n, lo, hi, rng):
    """n Latin-hypercube points in the box [lo, hi]."""
    return qmc.scale(qmc.LatinHypercube(d=len(lo), seed=rng).random(n), lo, hi)


def make_train(problem, seed):
    cfg, rng = PROBLEMS[problem], np.random.default_rng(seed)
    (x0, x1), (t0, t1) = cfg["x"], cfg["t"]
    X_f = lhs(cfg["n_f"], [x0, t0], [x1, t1], rng)
    x_ic = lhs(cfg["n_ic"], [x0], [x1], rng)[:, 0]
    t_bc = lhs(cfg["n_bc"], [t0], [t1], rng)[:, 0]
    X_u = np.vstack([np.column_stack([x_ic, np.full_like(x_ic, t0)]),   # initial condition t = t0
                     np.column_stack([np.full_like(t_bc, x0), t_bc]),   # left boundary x = x0
                     np.column_stack([np.full_like(t_bc, x1), t_bc])])  # right boundary x = x1
    return dict(X_f=X_f, X_u=X_u, Y_u=exact(problem, X_u), lb=np.array([x0, t0]), ub=np.array([x1, t1]))


def make_test(problem):
    cfg = PROBLEMS[problem]
    x, t = np.linspace(*cfg["x"], cfg["nx_test"]), np.linspace(*cfg["t"], cfg["nt_test"])
    XX, TT = np.meshgrid(x, t)  # (nt, nx): u_star.reshape(nt, nx, -1) recovers the grid
    X_star = np.column_stack([XX.ravel(), TT.ravel()])
    return dict(x=x, t=t, X_star=X_star, u_star=exact(problem, X_star))


# ---- check: finite-difference residual of the exact solution ----------------------------------
def fd_residual(problem, nx, nt):
    """max |PDE residual| / max |u| with central differences on an (nt, nx) grid."""
    cfg, p = PROBLEMS[problem], PROBLEMS[problem]["params"]
    (x0, x1), (t0, t1) = cfg["x"], cfg["t"]
    if problem == "black_scholes":
        t0 += 0.05 * (t1 - t0)  # stay away from the payoff kink at tau = 0
    x, t = np.linspace(x0, x1, nx), np.linspace(t0, t1, nt)
    dx, dt = x[1] - x[0], t[1] - t[0]
    XX, TT = np.meshgrid(x, t)
    U = exact(problem, np.column_stack([XX.ravel(), TT.ravel()])).reshape(nt, nx, -1)
    f = U[..., 0] if U.shape[-1] == 1 else U[..., 0] + 1j * U[..., 1]
    f_t = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * dt)
    f_x = (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * dx)
    f_xx = (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dx**2
    fi, S = f[1:-1, 1:-1], XX[1:-1, 1:-1]
    if problem == "black_scholes":
        res = f_t - 0.5 * p["sigma"] ** 2 * S**2 * f_xx - p["r"] * S * f_x + p["r"] * fi
    else:
        res = 1j * f_t + 0.5 * p["sigma"] * f_xx + p["beta"] * np.abs(fi) ** 2 * fi
    return np.abs(res).max() / np.abs(f).max()


def check():
    failed = False
    for name in PROBLEMS:
        r1, r2 = fd_residual(name, 801, 801), fd_residual(name, 1601, 1601)
        ok = r2 < 1e-2 and r1 / r2 > 3  # second-order scheme: residual should drop ~4x when halving h
        failed |= not ok
        print(f"[{'OK ' if ok else 'BAD'}] {name:18s} residual/|u|: {r1:.1e} -> {r2:.1e} when halving h")
    sys.exit(failed)


def main():
    if "--check" in sys.argv:
        check()
    for name in PROBLEMS:
        d = HERE / name
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text(json.dumps(PROBLEMS[name], indent=2) + "\n")
        np.savez_compressed(d / "test.npz", **make_test(name))
        for s in SEEDS:
            np.savez_compressed(d / f"train_seed{s}.npz", **make_train(name, s))
        print(f"wrote {d}")


if __name__ == "__main__":
    main()
