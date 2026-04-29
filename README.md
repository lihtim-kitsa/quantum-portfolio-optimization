# Quantum Portfolio Optimization via VQE

This project implements a quantum-enhanced version of the **Markowitz Mean-Variance Optimization** model. It translates a classical financial selection problem into a quantum Hamiltonian, which is then solved using the Variational Quantum Eigensolver (VQE).

## 🚀 How it Works

1.  **Classical Pre-processing**: We fetch historical stock data and compute the daily returns' mean (reward) and covariance (risk).
2.  **QUBO Mapping**: The portfolio problem is formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem:
    $$\min_{x \in \{0,1\}^n} q \sum_{i,j} \sigma_{ij}x_i x_j - \sum_{i} \mu_i x_i$$
    Subject to: $\sum x_i = B$ (Budget constraint).
3.  **Ising Hamiltonian**: Qiskit maps this QUBO into an Ising Hamiltonian where binary variables $\{0, 1\}$ become spin operators $\{I, Z\}$.
4.  **Quantum Variation**: The VQE uses a parameterized trial state (Ansatz) and a classical optimizer (COBYLA) to find the ground state of the Hamiltonian, which corresponds to the optimal portfolio.

## 🛠 Tech Stack
* **Python 3.9+**
* **Qiskit**: Quantum circuit construction and optimization.
* **Qiskit Finance**: Specialized tools for financial problem translation.
* **YFinance**: Market data API.
* **Matplotlib**: Visualization of the optimized weights.

## 📈 Key Learnings
* **Mapping Constraints**: Learned how to use penalty methods to enforce "Budget" constraints within a quantum circuit.
* **Ansatz Selection**: Using `RealAmplitudes` to explore the Hilbert space efficiently with minimal gate depth.
* **Hybrid Workflow**: Understanding that the "heavy lifting" of the optimization happens in a feedback loop between a classical CPU and a Quantum Processor (or simulator).
