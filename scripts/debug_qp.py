"""Debug QP optimization in factor_residual."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.optimize import minimize

# Simulate the QP
n_assets = 5

# Simulated inputs (from diagnostic output)
signals = np.array([-23.2754, 455.6082, 166.4287, 166.4287, 166.4287])  # Rough values
vol = np.array([0.15, 0.15, 0.15, 0.15, 0.15])  # 15% annualized vol
mu = signals / vol  # This gives values like -155, 3037, etc.

# Simulated betas (from diagnostic)
# For 5 assets with 4 factors, betas are near 1 for factor assets
betas = np.array([
    [1.0, 0.0, 1.0, 0.0],   # DBC = INFL
    [0.0, 0.0, 0.0, 0.5],   # GLD = part of SAFE
    [0.0, 0.0, 0.0, 0.0],   # IEF = not a factor (low beta)
    [1.0, 0.0, 0.0, 0.0],   # SPY = MKT
    [0.0, 1.0, 0.0, 0.5],   # TLT = RATES + part of SAFE
])

print(f"mu (scaled signals): {mu}")
print(f"betas shape: {betas.shape}")

# Constraints
# 1. sum(w) = 0
# 2. betas.T @ w = 0

# Try QP
reg_lambda = 0.1

def objective(w):
    return -float(np.dot(mu, w)) + reg_lambda * float(np.dot(w, w))

def gradient(w):
    return -mu + 2 * reg_lambda * w

constraints = [
    {"type": "eq", "fun": lambda w: float(np.sum(w))},
]
for j in range(betas.shape[1]):
    factor_beta = betas[:, j]
    constraints.append({
        "type": "eq",
        "fun": lambda w, b=factor_beta: float(np.dot(w, b)),
    })

bounds = [(-2.0, 2.0) for _ in range(n_assets)]

w0 = mu / max(np.std(mu), 1e-10)
w0 = np.clip(w0, -2.0, 2.0)

print(f"\nInitial guess w0: {w0}")
print(f"Initial objective: {objective(w0)}")

result = minimize(
    objective,
    w0,
    method="SLSQP",
    jac=gradient,
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 1000, "ftol": 1e-10, "disp": True},
)

print(f"\nQP Result:")
print(f"  Success: {result.success}")
print(f"  Message: {result.message}")
print(f"  Weights: {result.x}")
print(f"  Objective: {result.fun}")

# Check constraints
print(f"\nConstraint checks:")
print(f"  sum(w) = {np.sum(result.x)}")
for j in range(betas.shape[1]):
    print(f"  beta[{j}] @ w = {betas[:, j] @ result.x}")

# The issue: With betas having ~1 for factor assets, and we need betas.T @ w = 0,
# this means the weights for factor assets must cancel out.
# But the signals are huge for some assets and near-zero for others.
# The optimizer might be returning all zeros because it can't satisfy constraints and maximize signal.

print("\n\nAlternative: What if we relax factor neutrality?")
print("Let me check what happens with just market neutrality...")

constraints_simple = [
    {"type": "eq", "fun": lambda w: float(np.sum(w))},
]

result2 = minimize(
    objective,
    w0,
    method="SLSQP",
    jac=gradient,
    bounds=bounds,
    constraints=constraints_simple,
    options={"maxiter": 1000, "ftol": 1e-10, "disp": True},
)

print(f"\nMarket-neutral only:")
print(f"  Weights: {result2.x}")
print(f"  Factor exposures: {betas.T @ result2.x}")
