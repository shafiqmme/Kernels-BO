#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.quasirandom import SobolEngine

def sample_hyperparams_qmc(n_samples, bounds):
    sobol = SobolEngine(bounds.size(0), scramble=True)
    samples = sobol.draw(n_samples).double()
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * samples

