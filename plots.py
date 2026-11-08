"""Figures and the results table from results/*.pt.

python plots.py   # writes figures/*.png and report/table.tex, prints the table
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("data")
import loader
from pinn import Net, predict
from run import METHODS, OUT, SEEDS, WIDTH

matplotlib.use("Agg")
FIG = Path("figures")
LABEL = {"pinn": "PINN", "laaf": "IPINN w/o slope recovery", "ipinn": "IPINN"}
STYLE = {"pinn": dict(color="#4C78A8", ls="--"), "laaf": dict(color="#54A24B", ls="-."), "ipinn": dict(color="#E45756", ls=":")}
TITLE = {"black_scholes": "Black-Scholes call", "ivancevic_soliton": "Ivancevic soliton", "ivancevic_rogue": "Ivancevic rogue wave"}
YLABEL = {"black_scholes": "V", "ivancevic_soliton": r"$|\psi|$", "ivancevic_rogue": r"$|\psi|$"}


def load(problem, method, seed):
    return torch.load(OUT / f"{problem}_{method}_seed{seed}.pt")


def rebuild(r, cfg, n_out):
    net = Net([2] + [WIDTH[r["problem"]]] * 8 + [n_out], [cfg["x"][0], cfg["t"][0]], [cfg["x"][1], cfg["t"][1]],
              adaptive=r["method"] != "pinn")
    net.load_state_dict(r["state_dict"])
    return net


def magnitude(U):
    return U[:, 0] if U.shape[1] == 1 else np.hypot(U[:, 0], U[:, 1])


def sci(mean, std):
    e = int(np.floor(np.log10(mean)))
    return rf"$({mean / 10**e:.2f} \pm {std / 10**e:.2f}) \times 10^{{{e}}}$"


def table(problems):
    """Mean +- std over seeds of the two errors; prints markdown, writes the LaTeX tabular to report/table.tex."""
    print("| problem | method | rel. L2 | max abs | min/run |\n|---|---|---|---|---|")
    tex = [r"\begin{tabular}{llcc}", r"\toprule", r"problem & method & relative $L^2$ error & max.\ abs.\ error \\", r"\midrule"]
    for p in problems:
        for m in METHODS:
            rs = [load(p, m, s) for s in SEEDS]
            rel, mx = (np.array([r[k] for r in rs]) for k in ("rel_l2", "max_abs"))
            minutes = np.mean([r["minutes"] for r in rs])
            print(f"| {TITLE[p]} | {LABEL[m]} | {rel.mean():.2e} ± {rel.std():.1e} | {mx.mean():.2e} ± {mx.std():.1e} | {minutes:.1f} |")
            name = TITLE[p] if m == METHODS[0] else ""
            tex.append(f"{name} & {LABEL[m]} & {sci(rel.mean(), rel.std())} & {sci(mx.mean(), mx.std())} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    Path("report/table.tex").write_text("\n".join(tex) + "\n")


def solution_figure(problem, seed=0):
    """Top: heat maps of the exact solution and the PINN / IPINN-w/o-slope-recovery predictions. Bottom: slices at three times."""
    cfg, te = loader.load_config(problem), loader.load_test(problem)
    x, t = te["x"], te["t"]
    nt, nx = len(t), len(x)
    exact = magnitude(te["u_star"]).reshape(nt, nx)
    pred = {m: magnitude(predict(rebuild(load(problem, m, seed), cfg, te["u_star"].shape[1]), te["X_star"])).reshape(nt, nx)
            for m in METHODS}
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))
    for a, (title, Z) in zip(ax[0], [("exact", exact), (LABEL["pinn"], pred["pinn"]), (LABEL["laaf"], pred["laaf"])]):
        im = a.imshow(Z, origin="lower", extent=[x[0], x[-1], t[0], t[-1]], aspect="auto", cmap="viridis",
                      vmin=exact.min(), vmax=exact.max())
        a.set(title=title, xlabel="S", ylabel="t")
        fig.colorbar(im, ax=a)
    for a, it in zip(ax[1], [nt // 4, nt // 2, 3 * nt // 4]):
        a.plot(x, exact[it], color="black", lw=1.5, label="exact")
        for m in METHODS:
            a.plot(x, pred[m][it], lw=1.5, label=LABEL[m], **STYLE[m])
        a.set(title=f"t = {t[it]:.2f}", xlabel="S", ylabel=YLABEL[problem])
    ax[1, 0].legend(frameon=False)
    fig.suptitle(f"{TITLE[problem]} (seed {seed})")
    fig.tight_layout()
    fig.savefig(FIG / f"{problem}.png", dpi=150)


def loss_figure(problems, seed=0):
    fig, axes = plt.subplots(1, len(problems), figsize=(4 * len(problems), 3.2))
    for a, p in zip(axes, problems):
        for m in METHODS:
            r = load(p, m, seed)
            a.semilogy(r["loss"].numpy(), lw=0.8, color=STYLE[m]["color"], label=LABEL[m])
            a.axvline(r["adam_steps"], color="gray", lw=0.8, ls=":")
        a.set(title=TITLE[p], xlabel="loss evaluations (Adam | L-BFGS)")
    axes[0].set_ylabel("MSE loss")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "loss.png", dpi=150)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    problems = [p for p in WIDTH if all((OUT / f"{p}_{m}_seed{s}.pt").exists() for m in METHODS for s in SEEDS)]
    table(problems)
    loss_figure(problems)
    for p in problems:
        solution_figure(p)
