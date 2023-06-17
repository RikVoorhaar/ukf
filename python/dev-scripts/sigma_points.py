# %%
"""To start with we want to figure out how the sigma points are implemented. 
Let's start with the following:
- Use filter py's sigma points on a PSD
- Reimplement it in numpy
- Reimplement it in rust
- Expose the implementation to python
- Test, in Python, that the implementations are equivalent
- Benchmark

I think the sigma points themselves really just need a PSD. Then we apply a non-linear
transform to those points. 
"""
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
from abc import ABC, abstractmethod
import numpy as np


# def test_merwe_sigma_points():
#     n = 2
#     sigma_point_gen = MerweScaledSigmaPoints(n, alpha=1e-3, beta=2, kappa=0)

#     np.random.seed(179)
#     x = np.arange(n)  # np.random.normal(size=n)
#     P = np.diag(np.arange(1, n + 1))
#     sigma = sigma_point_gen.sigma_points(x, P)

#     expected_sigma = np.array(
#         [[0.0, 1.0], [0.00141421, 1.0], [0.0, 1.002], [-0.00141421, 1.0], [0.0, 0.998]]
#     )
#     assert np.linalg.norm(sigma - expected_sigma) < 1e-6

#     expected_Wc = np.array(
#         [
#             -999995.99997224,
#             249999.99999281,
#             249999.99999281,
#             249999.99999281,
#             249999.99999281,
#         ]
#     )
#     assert np.linalg.norm(sigma_point_gen.Wc - expected_Wc) < 1e-6

#     expected_Wm = np.array(
#         [
#             -999998.99997124,
#             249999.99999281,
#             249999.99999281,
#             249999.99999281,
#             249999.99999281,
#         ]
#     )
#     assert np.linalg.norm(sigma_point_gen.Wm - expected_Wm) < 1e-6

n = 2
sigma_point_gen = MerweScaledSigmaPoints(n, alpha=1, beta=2, kappa=0)

np.random.seed()
x = np.arange(n)  # np.random.normal(size=n)
P = np.diag(np.arange(1, n + 1))
sigma = sigma_point_gen.sigma_points(x, P)

expected_sigma = np.array(
    [[0.0, 1.0], [0.00141421, 1.0], [0.0, 1.002], [-0.00141421, 1.0], [0.0, 0.998]]
)
print(f"sigma = \n{sigma}")
print(f"\nWc = \n{sigma_point_gen.Wc}")
print(f"\nWm = \n{sigma_point_gen.Wm}")