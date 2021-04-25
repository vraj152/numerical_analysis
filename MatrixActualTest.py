from scipy.io import mmread
from scipy.sparse import coo_matrix
import numpy as np
from GaussSeidel import GaussSeidel
from Jacobi import Jacobi
from GenerateMatrix import GenerateMatrix
from SOR import SOR

a = mmread(r'C:\Users\Dell\Downloads\gr_30_30.mtx')
A = coo_matrix(a, dtype = np.float64).toarray()

n = A.shape[0]
b = np.random.rand(n, 1)

system = GenerateMatrix()
system.foreign_system(A, b)

jacobi_obj = Jacobi(A, b, system.jacobi_rad, it = 3500)
jacobi_obj.solver()
print(str(jacobi_obj))

gauss_obj = GaussSeidel(A, b, system.gs_rad, it = 1000)
gauss_obj.solver()
print(str(gauss_obj))

sor_obj = SOR(A, b, omega = 1.9)
sor_obj.solver()
print(str(sor_obj))

print(np.allclose(jacobi_obj.sol, gauss_obj.sol, atol = 1e-8))
print(np.allclose(jacobi_obj.sol, sor_obj.sol, atol = 1e-8))

"""
Dataset Used: gr_30_30.mtx

+-----------------+--------+
|     Method      | Jacobi |
+=================+========+
|   Iterations    |  2690  |
+-----------------+--------+
| Spectral Radius | 0.992  |
+-----------------+--------+
| Error - L1 Norm | 0.000  |
+-----------------+--------+
|   Time taken    | 46.959 |
+-----------------+--------+
+-----------------+--------------+
|     Method      | Gauss Seidel |
+=================+==============+
|   Iterations    |     927      |
+-----------------+--------------+
| Spectral Radius |    0.985     |
+-----------------+--------------+
| Error - L1 Norm |    0.000     |
+-----------------+--------------+
|   Time taken    |    13.859    |
+-----------------+--------------+
+-----------------+----------------------------+
|     Method      | Successive Over-Relaxation |
+=================+============================+
|   Iterations    |            177             |
+-----------------+----------------------------+
|      Omega      |           1.900            |
+-----------------+----------------------------+
| Error - L1 Norm |           0.000            |
+-----------------+----------------------------+
|   Time taken    |          663.937           |
+-----------------+----------------------------+


"""