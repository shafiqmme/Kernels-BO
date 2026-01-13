# Bayesian Optimization over Kernel Space

This repository implements a kernel optimization framework that searches over kernel space.
The framework is designed to compare BO-based kernel search against LLM-guided genetic algorithms (LLM-GA) and other baselines under controlled computational budgets.

## 1. Overview

Gaussian Process performance is highly sensitive to kernel choice. Instead of selecting kernels heuristically or via greedy composition, this work:

- Constructs a symbolic grammar over GP kernels

- Defines a probabilistic distance between kernels by marginalizing over hyperparameters

- Embeds kernels into a Euclidean latent space using MDS

- Performs Bayesian Optimization in kernel space

- This enables principled, sample-efficient kernel discovery.

## 2. Methodology Summary

The pipeline consists of the following steps:

- Kernel Grammar Construction: Generate a discrete kernel space using symbolic grammar rules.

- Expected Distance Computation: For each kernel pair, compute the expected √Jensen–Shannon distance by integrating over kernel hyperparameters using Quasi–Monte Carlo (QMC).

- Metric Embedding: Embed the kernel distance matrix into a low-dimensional Euclidean space using classical MDS.

- Kernel Bayesian Optimization: Perform BO over the embedded kernel space to identify high-performing kernel structures.

- Application-Level Evaluation: Apply optimized kernels to real tasks (e.g., printability modeling).

## 3. Directory Structure

```
/kernels-BO
│
├── kernels/
│   ├── grammar.py          # Kernel grammar and expression generation
│   ├── builders.py         # Construction of GP kernels from grammar expressions
│
├── utils/
│   ├── qmc.py              # quasi monte carlo
│
├── distances/
│   ├── js.py               # Jensen–Shannon divergence utilities
│   ├── expectations.py     # Expected √JS distance via QMC over hyperparameters
│
├── embedding/
│   ├── mds.py              # Classical MDS for squared-distance matrices
│
├── analysis/
│   ├── diagnostics.py      # MDS reconstruction error, parity plots, metrics
│
├── io_utils/
│   ├── export.py           # CSV export of kernel embeddings
│
├── output/
│   ├── kernels_JS.csv      # MDS-embedded kernel descriptors
│   ├── mds_error_vs_k.pdf  # Reconstruction error vs embedding dimension
│   ├── mds_parity.pdf      # Parity plot
│
├── main/
│   ├── timetemp.csv        # an in house data for benchmark
│   ├── airline.csv         # international airline passenger data
│   ├── mauna.csv           # Mauna Loa CO2 conc. data
│   ├── input_space.ipynb
│   ├── main.ipynb
│
├── application/
│   ├── BO_with_Optimized_Kernels.ipynb
│
└── README.md
```

## 4. Requirements

- Python ≥ 3.9

- NumPy

- SciPy

- PyTorch

- GPyTorch

- BoTorch

- scikit-learn

- matplotlib

## 5. Experiments & Benchmarks

Benchmarks include:

- Ackley

- Dropwave

- Schwefel

- Rastrigin

- Levy

- Bukin

- Airline

- Mauna Loa CO₂

- Eggholder

- Thermal History (In house data)

- Printability Data (In house data)

## Cite this work
[![DOI](https://zenodo.org/badge/1128343109.svg)](https://doi.org/10.5281/zenodo.18189227)
