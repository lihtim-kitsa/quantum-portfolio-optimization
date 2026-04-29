#!pip install -q qiskit[visualization]
#!pip install -q qiskit_aer
#!pip install -q qiskit_ibm_runtime
#!pip install -q matplotlib
#!pip install -q pylatexenc
#!pip install -q qiskit-algorithms
#!pip install qiskit qiskit-finance yfinance matplotlib 

import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.minimum_eigensolvers import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_finance.data_providers import YahooDataProvider

# 1. Configuration & Data Fetching
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
# Use 'Close' and set group_by='column' to ensure a cleaner structure
raw_data = yf.download(tickers, start="2023-01-01", end="2024-01-01")

# Extract only the 'Close' prices
data = raw_data['Close'] 

# Drop any missing values to prevent math errors in covariance
data = data.dropna()

# Calculate expected returns (mu) and covariance matrix (sigma)
returns = data.pct_change().dropna()
mu = returns.mean().to_numpy()
sigma = returns.cov().to_numpy()

num_assets = len(tickers)
budget = num_assets // 2  # Max assets to select
risk_factor = 0.5         # Risk aversion (lambda)

# 2. Formulate the Portfolio Optimization Problem
# This creates a Quadratic Program representing: 
# min [lambda * x^T * sigma * x - mu^T * x]
portfolio = PortfolioOptimization(
    expected_returns=mu, 
    covariances=sigma, 
    risk_factor=risk_factor, 
    budget=budget
)
qp = portfolio.to_quadratic_program()

# 3. Setup the Quantum Solver (VQE)
# RealAmplitudes is a heuristic ansatz suitable for NISQ devices
ansatz = RealAmplitudes(num_assets, reps=2)
optimizer = COBYLA(maxiter=100)
sampler = Sampler()

vqe_solver = SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=optimizer)
quantum_optimizer = MinimumEigenOptimizer(vqe_solver)

# 4. Solve the problem
result = quantum_optimizer.solve(qp)

# 5. Results Visualization
print(f"Optimal Selection: {result.x}")
print(f"Selected Tickers: {[tickers[i] for i, val in enumerate(result.x) if val > 0]}")

selection = result.x
labels = tickers
colors = ['#66b3ff' if val > 0 else '#ff9999' for val in selection]

plt.figure(figsize=(8, 5))
plt.bar(labels, selection, color=colors)
plt.title("Quantum Portfolio Selection (Binary)")
plt.ylabel("Selected (1) vs Not Selected (0)")
plt.show()