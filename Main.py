from GenerateMatrix import GenerateMatrix
from Jacobi import Jacobi
from GaussSeidel import GaussSeidel
from SOR import SOR

system = GenerateMatrix(lower_threshold = 0.6, upper_threshold = 0.8)
A, b = system.generate_system(1000)

print("Spectral Radius: ", system.s_radius)
print("Number of times matrix generated: ", system.counter)

jacobi_obj = Jacobi(A, b, system.jacobi_rad, 1000)
jacobi_obj.solver()
print(str(jacobi_obj))

gauss_obj = GaussSeidel(A, b, system.gs_rad, 1000)
gauss_obj.solver()
print(str(gauss_obj))

sor_obj = SOR(A, b, omega = 1)
sor_obj.solver()
print(str(sor_obj))

#print(np.allclose(jacobi_obj.sol, gauss_obj.sol, atol = 1e-6))
#print(np.allclose(jacobi_obj.sol, sor_obj.sol, atol = 1e-6))