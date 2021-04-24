from GenerateMatrix import GenerateMatrix
from Jacobi import Jacobi
from GaussSeidel import GaussSeidel

import numpy as np

system = GenerateMatrix(lower_threshold = 0.6, upper_threshold = 0.91)
A, b = system.generate_system(1000)

print("Spectral Radius: ", system.s_radius)
print("Number of times matrix generated: ", system.counter)

jacobi_obj = Jacobi(A, b, 1000)
jacobi_obj.solver()

gauss_obj = GaussSeidel(A, b, 1000)
gauss_obj.solver()

print(str(jacobi_obj))
print(str(gauss_obj))

print(np.allclose(jacobi_obj.sol, gauss_obj.sol, atol = 1e-6))