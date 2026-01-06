#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
from utils.qmc import sample_hyperparams_qmc
from kernels.builders import set_hyperparams
from distances.js import sqrt_js_distance_squared_from_K


def expected_sqrtjs_distance_squared(
    kernel1_class,
    kernel2_class,
    train_X,
    n_samples=256,
    bounds=torch.tensor([[0.01, 2.5], [0.01, 2.5]]),
):
    thetas = sample_hyperparams_qmc(n_samples, bounds)
    vals = []

    for theta in thetas:
        l, s = theta[0].item(), theta[1].item()
        k1 = kernel1_class()
        k2 = kernel2_class()

        set_hyperparams(k1, l, s)
        set_hyperparams(k2, l, s)

        K1 = k1(train_X).evaluate()
        K2 = k2(train_X).evaluate()

        vals.append(sqrt_js_distance_squared_from_K(K1, K2))

    return float(np.mean(vals))

