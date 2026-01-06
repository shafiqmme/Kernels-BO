## Directory Structure

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
│   ├── kernels_JS-1D.csv   # MDS-embedded kernel descriptors
│   ├── mds_error_vs_k.pdf  # Reconstruction error vs embedding dimension
│   ├── mds_parity.pdf      # Parity plot
│
├── main/
│   ├── build_kernel_geometry.py  # End-to-end pipeline
│
├── notebooks/
│   ├── kernel_geometry_demo.ipynb
│
└── README.md
```
