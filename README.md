# Estimating the Number of Legal Chess Diagrams Using Monte Carlo Simulations

This repository contains the code, data, and results for the paper:

> **Estimating the Number of Legal Chess Diagrams Using Monte Carlo Simulations**  
> Sven Hans, November 2025

📄 **[Read the full paper (PDF)](./paper.pdf)**

---

## Overview

The goal of this project is to provide an estimate for the total number of *legal chess diagrams* (without promotions).  
We approach this by:

1. **Enumerating material classes** (distinct piece combinations for White and Black).
2. **Counting all diagrams per class** using combinatorial formulas.
3. **Estimating legality ratios** per class via **Monte Carlo sampling** with the [`python-chess`](https://python-chess.readthedocs.io/) library.
4. **Aggregating results** across all classes to estimate the global number of legal diagrams and its statistical uncertainty.

Our results suggest an upper bound of  
**≈ 4.1 × 10^41 ± 2.9% legal diagrams**,  
meaning that roughly **1 in 800 random piece placements** is valid.

---
