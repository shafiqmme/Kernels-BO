#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

def export_kernel_embedding(X, kernel_exprs, path):
    X_np = X.detach().cpu().numpy()
    cols = [f"C{i+1}" for i in range(X_np.shape[1])]

    df = pd.DataFrame(X_np, columns=cols)
    df.insert(0, "kernel", [str(k) for k in kernel_exprs])
    df.to_csv(path, index=False)

