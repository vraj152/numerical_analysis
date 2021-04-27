from GenerateMatrix import GenerateMatrix
from Jacobi import Jacobi
from GaussSeidel import GaussSeidel
from SOR import SOR
import numpy as np

system = GenerateMatrix(method = "Random-DD", lower_threshold = 0.5, upper_threshold = 0.7)
A, b = system.generate_system(1000)

print("Spectral Radius: ", system.s_radius)
print("Number of times matrix generated: ", system.counter)

jacobi_obj = Jacobi(A, b, system.method, system.jacobi_rad, 1000)
jacobi_obj.solver()
print(str(jacobi_obj))

gauss_obj = GaussSeidel(A, b, system.method, system.gs_rad, 1000)
gauss_obj.solver()
print(str(gauss_obj))

sor_obj = SOR(A, b, system.method, omega = 1, debug = False)
sor_obj.solver()
print(str(sor_obj))

print("isSimilar? (Jacobi and Gauss): ", np.allclose(jacobi_obj.sol, gauss_obj.sol, atol = 1e-6))
print("isSimilar? (Jacobi and SoR)", np.allclose(jacobi_obj.sol, sor_obj.sol, atol = 1e-6))