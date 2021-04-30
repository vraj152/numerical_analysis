from GenerateMatrix import GenerateMatrix
from Jacobi import Jacobi
from GaussSeidel import GaussSeidel
from SOR import SOR
import numpy as np


system = GenerateMatrix(method = "Z-DD", lower_threshold = 0.5, upper_threshold = 0.9)
A, b = system.generate_system(500)

"""
if(self.method == "Random-DD"):
elif(self.method == "Lower-DD"):
elif(self.method == "Upper-DD"):
elif(self.method == "TriDiagonal-DD"):
elif(self.method == "Symmetric-DD"):
elif(self.method == "Z-DD"):
elif(self.method == "Q-DD"):
"""

print("Spectral Radius: ", system.s_radius)
print("Number of times matrix generated: " + str(system.gmc + system.spc))

jacobi_obj = Jacobi(A, b, system.method, system.jacobi_rad, 3000)
jacobi_obj.solver()
print(str(jacobi_obj))

gauss_obj = GaussSeidel(A, b, system.method, system.gs_rad, 3000)
gauss_obj.solver()
print(str(gauss_obj))

fault_tolerance_SOR = 1e-4

sor_obj = SOR(A, b, system.method, omega = 1, debug = True, fault_tolerance = fault_tolerance_SOR)
sor_obj.solver()
print(str(sor_obj))

print("isSimilar? (Jacobi and Gauss): ", np.allclose(jacobi_obj.sol, gauss_obj.sol, atol = 1e-6))
print("isSimilar? (Jacobi and SoR)", np.allclose(jacobi_obj.sol, sor_obj.sol, atol = fault_tolerance_SOR))