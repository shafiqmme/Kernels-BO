#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    p = p.clamp(min=0)
    q = q.clamp(min=0)

    ps = p / (p.sum() + eps)
    qs = q / (q.sum() + eps)
    m = 0.5 * (ps + qs)

    kl_pm = torch.sum(ps * torch.log((ps + eps) / (m + eps)))
    kl_qm = torch.sum(qs * torch.log((qs + eps) / (m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def sqrt_js_distance_squared_from_K(
    K1: torch.Tensor, K2: torch.Tensor, eps: float = 1e-9
) -> float:
    n = K1.shape[-1]
    I = torch.eye(n, dtype=K1.dtype, device=K1.device)

    K1s = 0.5 * (K1 + K1.T) + eps * I
    K2s = 0.5 * (K2 + K2.T) + eps * I

    evals1, _ = torch.linalg.eigh(K1s)
    evals2, _ = torch.linalg.eigh(K2s)

    evals1 = torch.clamp(evals1, min=0.0)
    evals2 = torch.clamp(evals2, min=0.0)

    return js_divergence(evals1, evals2)

