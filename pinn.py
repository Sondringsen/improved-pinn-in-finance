"""PINN and IPINN for the Black-Scholes and Ivancevic equations.

IPINN = PINN with a trainable slope per hidden neuron, tanh(n * a * z), and the slope-recovery
loss term 1 / mean_l exp(mean_i a_li)  (Jagtap et al. 2020; Bai, Chaolu & Bilige 2022).
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, layers, lb, ub, adaptive=False, n=10.0, a0=0.1):
        super().__init__()
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32))
        self.linears = nn.ModuleList(nn.Linear(i, o) for i, o in zip(layers[:-1], layers[1:]))
        for lin in self.linears:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
        # IPINN: one slope per hidden neuron; n * a0 = 1 so training starts from a plain tanh net
        self.n = n
        self.a = nn.ParameterList(nn.Parameter(torch.full((w,), a0)) for w in layers[1:-1]) if adaptive else None

    def forward(self, X):
        h = 2 * (X - self.lb) / (self.ub - self.lb) - 1
        for i, lin in enumerate(self.linears[:-1]):
            h = lin(h)
            if self.a is not None:
                h = self.n * self.a[i] * h
            h = torch.tanh(h)
        return self.linears[-1](h)

    def slope_recovery(self):
        return 1 / torch.stack([torch.exp(a.mean()) for a in self.a]).mean()


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def residual(problem, net, X, p):
    """PDE residual at collocation points X (N, 2) with columns (S, t); p = problem parameters."""
    X = X.clone().requires_grad_(True)
    S, U = X[:, :1], net(X)
    if problem == "black_scholes":  # V_tau - sigma^2 S^2 V_SS / 2 - r S V_S + r V = 0
        V = U
        V_S, V_t = grad(V, X).split(1, dim=1)
        V_SS = grad(V_S, X)[:, :1]
        return V_t - 0.5 * p["sigma"] ** 2 * S**2 * V_SS - p["r"] * S * V_S + p["r"] * V
    u, v = U.split(1, dim=1)  # psi = u + i v, paper eq. (14)
    u_S, u_t = grad(u, X).split(1, dim=1)
    v_S, v_t = grad(v, X).split(1, dim=1)
    u_SS, v_SS = grad(u_S, X)[:, :1], grad(v_S, X)[:, :1]
    mod2 = u**2 + v**2
    f_u = u_t + 0.5 * p["sigma"] * v_SS + p["beta"] * mod2 * v
    f_v = v_t - 0.5 * p["sigma"] * u_SS - p["beta"] * mod2 * u
    return torch.cat([f_u, f_v], dim=1)


def train(net, problem, p, X_u, Y_u, X_f, adam_steps, lbfgs_steps, slope_recovery=False, lr=1e-3):
    """Adam then L-BFGS on MSE_u + MSE_f (+ slope recovery). Returns the MSE part at every evaluation."""
    net.to(DEVICE)
    X_u, Y_u, X_f = (torch.tensor(a, dtype=torch.float32, device=DEVICE) for a in (X_u, Y_u, X_f))

    history = []

    def loss_fn():
        mse = F.mse_loss(net(X_u), Y_u) + residual(problem, net, X_f, p).pow(2).mean()
        history.append(mse.item())  # the paper reports the MSE part, without the slope-recovery term
        return mse + net.slope_recovery() if slope_recovery else mse

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(adam_steps):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()

    opt = torch.optim.LBFGS(net.parameters(), max_iter=lbfgs_steps, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss.detach()  # L-BFGS only reads the value; detaching silences a torch warning

    opt.step(closure)
    return np.array(history)


def predict(net, X):
    device = next(net.parameters()).device
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()


def errors(u_pred, u_star):
    """Relative L2 error and max abs error."""
    err = u_pred - u_star
    return np.linalg.norm(err) / np.linalg.norm(u_star), np.abs(err).max()
