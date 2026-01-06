#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel, RQKernel

# Base kernels
base_kernels = {
    "SE":  ScaleKernel(RBFKernel()),
    "PER": ScaleKernel(PeriodicKernel()),
    "RQ":  ScaleKernel(RQKernel()),
}

class KernelExpr:
    def __init__(self, name, module=None, children=()):
        self.name = name
        self.module = module
        self.children = tuple(children)

    def __str__(self):
        if not self.children:
            return self.name
        return f"({self.children[0]} {self.name} {self.children[1]})"

    def __hash__(self):
        return hash((self.name, self.children))

    def __eq__(self, other):
        return (
            isinstance(other, KernelExpr)
            and self.name == other.name
            and self.children == other.children
        )


def grammar_neighbors(expr):
    nbrs = set()
    for bname, bmod in base_kernels.items():
        base = KernelExpr(bname, module=bmod)
        nbrs |= {
            KernelExpr("+", children=(expr, base)),
            KernelExpr("+", children=(base, expr)),
            KernelExpr("*", children=(expr, base)),
            KernelExpr("*", children=(base, expr)),
        }

    if expr.children:
        for i, child in enumerate(expr.children):
            if not child.children:
                for bname, bmod in base_kernels.items():
                    if bname != child.name:
                        new_ch = list(expr.children)
                        new_ch[i] = KernelExpr(bname, module=bmod)
                        nbrs.add(KernelExpr(expr.name, children=tuple(new_ch)))
    return nbrs


def generate_kernel_set(max_depth=1):
    level = {KernelExpr(n, module=m) for n, m in base_kernels.items()}
    all_expr = set(level)

    for _ in range(max_depth):
        nxt = set()
        for e in level:
            nxt |= grammar_neighbors(e)
        nxt -= all_expr
        all_expr |= nxt
        level = nxt

    return sorted(all_expr, key=str)

