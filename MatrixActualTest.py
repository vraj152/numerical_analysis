from scipy.io import mmread
from scipy.sparse import coo_matrix
import numpy as np
from GaussSeidel import GaussSeidel
from Jacobi import Jacobi
from GenerateMatrix import GenerateMatrix
from SOR import SOR

a = mmread("TestingMatrix/bcsstm20.mtx")
A = coo_matrix(a, dtype = np.float64).toarray()

n = A.shape[0]
b = np.random.rand(n, 1)

"""
system = GenerateMatrix(upper_threshold = 0.95)
A, b = system.generate_system(1000)
"""

system = GenerateMatrix()
system.foreign_system(A, b)

jacobi_obj = Jacobi(A, b, system.jacobi_rad, it = 3000)
jacobi_obj.solver()
print(str(jacobi_obj))

gauss_obj = GaussSeidel(A, b, system.gs_rad, it = 1000)
gauss_obj.solver()
print(str(gauss_obj))

print("Gauss and Jacobi's sol: ",np.allclose(jacobi_obj.sol, gauss_obj.sol, atol = 1e-8))

sor_obj = SOR(A, b, omega = 0.8)
sor_obj.solver()
print(str(sor_obj))

print("Gauss and SOR's sol: ",np.allclose(jacobi_obj.sol, sor_obj.sol, atol = 1e-8))