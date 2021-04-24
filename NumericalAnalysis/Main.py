import Jacobi as jb
import GenerateMatrix as gm

system = gm.GenerateMatrix()
A, b = system.generate_system(3)
print(system.calculate_spactral_radius())

jacobi_obj = jb.Jacobi(A, b, 1000)
jacobi_obj.solver()

print(str(jacobi_obj))