# MSAI 451 – Programming Assignment 2: Monte Carlo Portfolio Optimization

## Overview: Mapping the Diversified Investment Opportunity Set

This project applies the **Monte Carlo simulation** technique to visualize and analyze the complete feasible investment set for a diversified, four-asset portfolio. The primary goal is to quantitatively and visually assess how two distinct investment constraints—**Long-Only** vs. **Shorts Permitted (Long + Short)**—impact the shape and location of the portfolio **Efficient Frontier** in the risk–return space.

The simulation generates thousands of random portfolio weight combinations to map the entire feasible space, providing an intuitive basis for understanding Mean–Variance Optimization principles.

### Scenario and Assets

The portfolio includes assets chosen for their diversification properties across distinct asset classes:

| Ticker | Asset Name | Asset Class | Expected Role |
| :--- | :--- | :--- | :--- |
| **NVDA** | NVIDIA Corporation | Technology Equity | Growth / High Volatility |
| **XOM** | Exxon Mobil Corporation | Energy Equity | Value / Cyclical |
| **TLT** | iShares 20+ Year Treasury Bond ETF | Bonds | Low-Correlation Anchor |
| **GLD** | SPDR Gold Shares | Gold / Commodity | Inflation Hedge |

### Key Technologies

| Category | Tools |
|----------|-------|
| Modeling | Python, NumPy (for linear algebra and vector operations) |
| Simulation | SciPy (for Dirichlet distribution sampling) |
| Visualization | Matplotlib / Seaborn |

---

## For Users: Getting Started

This section guides you through executing the simulation to generate the results and the key visualization of the feasible investment set.

### Prerequisites

You need a Python environment installed. The required packages are standard for quantitative finance:

| Package | Purpose |
|---------|---------|
| numpy | Numerical operations, matrix algebra (covariance calculation) |
| scipy | Statistical functions, specifically the Dirichlet distribution for constrained sampling |
| matplotlib | Generating the scatter plot of the risk-return space |
| pandas | Data structuring and preparation (optional, depending on final script structure) |

To install the dependencies:

`pip install numpy scipy matplotlib pandas`

### Running the Code

1. **Clone the Repository:**

    ```bash
    git clone <your_repo_url>.git
    cd folder_name
    ```

2. **Execute the Main Script:**
    The script contains the hardcoded parameters (returns, volatilities, correlations) from the assignment prompt.

    ```bash
    python3 assign2_code.py
    ```

**Expected Output:** The script will generate a console output detailing the number of simulations run and save one primary visualization:
- `Figure_1_Feasible_Set.png` (A plot showing the Long-Only and Long + Short feasible sets in the risk-return plane).

---

##  For Developers: Building and Testing

This section provides details for those interested in the underlying methodology and code structure.

### Repository Structure

| File/Folder | Description |
|-------------|-------------|
| `assign2_code.py` | Main execution script: Defines parameters, runs both Monte Carlo simulations, calculates $\mu_p$ and $\sigma_p$, and generates the final plot. |
| `report.pdf` | The detailed written analysis, including parameter assumptions, interpretation of results, and discussion. |

### Simulation Methodology Overview

The core simulation requires two distinct methods for sampling the weights ($\mathbf{w}$):

1.  **Long-Only Sampling:** Achieved efficiently by drawing samples from a **Dirichlet distribution**, which naturally enforces both $w_i \geq 0$ and $\sum w_i = 1$.
2.  **Shorts Permitted Sampling:** Achieved by drawing random samples (e.g., from a standard Gaussian or Uniform distribution) and then normalizing them such that $\sum w_i = 1$. A manual filter is applied post-sampling to manage excessive leverage.

### Key Calculations

The script implements the fundamental formulas from Mean–Variance theory:

$$\mu_p = \mathbf{w}^\top \boldsymbol{\mu} \quad \text{and} \quad \sigma_p = \sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}$$

where $\boldsymbol{\Sigma}$ is derived from the input volatilities and correlation matrix.

---

## Use of AI Tools

AI assistance (ChatGPT/GPT-4) was primarily used for refining grammar in the report. All core model choices, parameter definitions, mathematical implementation, and result interpretations were executed and validated by the author.
