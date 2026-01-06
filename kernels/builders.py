#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gpytorch.kernels import ScaleKernel

def build_module(expr):
    if not expr.children:
        base = expr.module
        if isinstance(base, ScaleKernel):
            return ScaleKernel(base.base_kernel.__class__())
        return base.__class__()

    A, B = map(build_module, expr.children)
    return A + B if expr.name == "+" else A * B


def set_hyperparams(kernel, l, s):
    if isinstance(kernel, ScaleKernel):
        kernel.outputscale = s
        set_hyperparams(kernel.base_kernel, l, s)
    elif hasattr(kernel, "kernels"):
        for sub_k in kernel.kernels:
            set_hyperparams(sub_k, l, s)
    elif hasattr(kernel, "lengthscale"):
        kernel.lengthscale = l

