# Improved PINNs for option pricing (CEE292 project 1)

Re-implementation of Bai, Chaolu & Bilige, *The application of improved
physics-informed neural network (IPINN) method in finance*, Nonlinear Dyn. 107
(2022): a standard PINN and the IPINN (neuron-wise adaptive activation slopes
with a slope-recovery loss term) on the Ivancevic option-pricing equation
(rogue wave and soliton) and the Black-Scholes equation, compared over 5 seeds.

```
pip install -r requirements.txt
python data/generate_data.py      # training/test data (not committed)
python run.py                     # all problems x {PINN, IPINN} x 5 seeds -> results/
python plots.py                   # figures -> figures/
```

## Layout

```
data/        generate_data.py (synthetic data), loader.py
pinn.py      network (PINN and IPINN), PDE residuals, loss, training
run.py       every problem x method x seed, writes results/
plots.py     figures from results/, writes figures/
results/     per-run errors and loss curves (json)
figures/     plots used in the report
report/      report.tex and pdf
```

The paper itself is not included for copyright reasons.
