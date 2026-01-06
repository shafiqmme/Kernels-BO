Project: The Kernel Manifold: A Geometric Approach to Gaussian Process Model Selection
Directory Structure
-------------------

/kernels-BO
│
├── kernels/
│   ├── grammar.py              # Kernel grammar and expression generation
│   ├── builders.py             # Construction of GP kernels from grammar expressions
│
├── distances/
│   ├── js.py                   # Sqrt Jensen–Shannon divergence utilities
│   ├── expectations.py         # Expected √JS squared distance via QMC over hyperparameters
│
├── embedding/
│   ├── mds.py                  # Classical MDS for squared-distance matrices
│
├── analysis/
│   ├── diagnostics.py          # MDS reconstruction error, parity plots, metrics
│
├── io_utils/
│   ├── export.py               # CSV export of kernel embeddings
│
├── output/
│   ├── kernels_JS.csv          # MDS-embedded kernel descriptors
│   ├── mds_error_vs_k.pdf      # Reconstruction error vs embedding dimension
│   ├── mds_parity.pdf          # Parity plot: original vs reconstructed distances
│
├── main/                       # End-to-end kernel geometry construction
│
├── main/
│   ├── kernel_geometry_demo.ipynb # Interactive exploration and visualization
│
└── README.md
