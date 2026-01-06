#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch

def mds_from_squared_distance(D2: torch.Tensor, k=None, tol=1e-12):
    D2 = 0.5 * (D2 + D2.T)
    D2.fill_diagonal_(0.0)

    N = D2.shape[0]
    J = torch.eye(N, dtype=D2.dtype) - torch.ones((N, N), dtype=D2.dtype) / N
    B = -0.5 * J @ D2 @ J

    evals, evecs = torch.linalg.eigh(B)
    pos = evals > tol

    if not torch.any(pos):
        raise RuntimeError("No positive eigenvalues in MDS.")

    evals_pos = evals[pos]
    evecs_pos = evecs[:, pos]

    if k is not None:
        idx = torch.argsort(evals_pos, descending=True)[:k]
        evals_pos = evals_pos[idx]
        evecs_pos = evecs_pos[:, idx]

    X = evecs_pos * torch.sqrt(evals_pos.clamp(min=0.0))
    return X, evals

