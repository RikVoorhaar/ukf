# %%
"""It seems that the UT has an expression for computing P that can be optimized
significantly over a triple for loop."""
import numpy as np

def ut_P(sigmas, x, Wc):
    y = sigmas - x[np.newaxis, :]
    P = np.dot(y.T, np.dot(np.diag(Wc), y))
    return P

n = 100
sigmas = np.random.normal(size=(2*n+1,n))
x = np.random.normal(size=(n,))
Wc = np.random.normal(size=(2*n+1,))

P_correct = ut_P(sigmas, x, Wc)

%timeit ut_P(sigmas,  x, Wc)
#%%
