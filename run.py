"""Train every method on every problem for 5 seeds; one results/<problem>_<method>_seed<k>.pt per run.

pinn: fixed tanh.  laaf: trainable slope per neuron.  ipinn: laaf + slope-recovery loss (the paper's method).

python run.py                 # everything (existing result files are skipped)
python run.py black_scholes   # one problem
SEEDS=1 python run.py         # seed 0 only, e.g. for a quick check
"""
import os
import sys
import time
from pathlib import Path

import torch

sys.path.append("data")
import loader
from pinn import Net, errors, predict, train

WIDTH = {"black_scholes": 50, "ivancevic_soliton": 40, "ivancevic_rogue": 40}  # 8 hidden layers each, as in the paper
METHODS = ("pinn", "laaf", "ipinn")
ADAM_STEPS, LBFGS_STEPS = 2000, 5000  # paper: 10 000 + 15 000; L-BFGS usually stops early on tolerance
SEEDS = range(int(os.environ.get("SEEDS", 5)))
OUT = Path("results")


def run(problem, method, seed):
    cfg, tr, te = loader.load_config(problem), loader.load_train(problem, seed), loader.load_test(problem)
    torch.manual_seed(seed)
    net = Net([2] + [WIDTH[problem]] * 8 + [tr["Y_u"].shape[1]], tr["lb"], tr["ub"], adaptive=method != "pinn")
    t0 = time.time()
    loss = train(net, problem, cfg["params"], tr["X_u"], tr["Y_u"], tr["X_f"], ADAM_STEPS, LBFGS_STEPS,
                 slope_recovery=method == "ipinn")
    rel_l2, max_abs = errors(predict(net, te["X_star"]), te["u_star"])
    return dict(problem=problem, method=method, seed=seed, rel_l2=float(rel_l2), max_abs=float(max_abs),
                minutes=(time.time() - t0) / 60, adam_steps=ADAM_STEPS, loss=torch.tensor(loss),
                state_dict=net.cpu().state_dict())


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for seed in SEEDS:
        for method in METHODS:
            for problem in sys.argv[1:] or WIDTH:
                out = OUT / f"{problem}_{method}_seed{seed}.pt"
                if out.exists():
                    continue
                r = run(problem, method, seed)
                torch.save(r, out)
                print(f"{problem:18s} {method:5s} seed {seed}: rel L2 {r['rel_l2']:.2e}  max abs {r['max_abs']:.2e}  "
                      f"{r['minutes']:.1f} min", flush=True)
