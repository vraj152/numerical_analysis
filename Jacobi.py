import numpy as np
import texttable
import time

class Jacobi:
    
# =============================================================================
#     Constructor:
#         A (numpy 2D array) = [NxN] Matrix
#         b (numpy 1D array) = [Nx1] Matrix
#         method (str) = Type of matrix that has been given to the system
#         isDiagonalDominant (bool) = Whether provided matrix is diagonal dominant
#         radius (float) = Spectral Radius of the provided matrix
#         it (int) = Maximum number of iterations
# =============================================================================
    
    def __init__(self, A, b, method, isDiagonalDominant, radius = "NA", it = 1000):
        self.A = A
        self.b = b
        self.method = method
        self.isDiagonalDominant = isDiagonalDominant
        
        self.iterations = it
        self.sol = None
        self.count = 0
        self.time_taken = 0
        self.radius = radius
    
# =============================================================================
#     This function solves the linear system provided.
# =============================================================================
    
    def solver(self):
        start = time.time()
        
        self.sol = np.zeros_like(self.b)
        
        for it in range(self.iterations):
            self.count += 1
            temp = np.zeros_like(self.sol)
                
            for i in range(self. A.shape[0]):
                s1 = np.dot(self.A[i, :i], self.sol[:i])
                s2 = np.dot(self.A[i, i + 1:], self.sol[i + 1:])
                temp[i] = (self.b[i] - s1 - s2) / self.A[i, i]
                
                if temp[i] == temp[i-1]:
                  break
        
            if np.allclose(self.sol, temp, atol=1e-10, rtol=0.):
                break
    
            self.sol = temp
            
        self.time_taken = time.time() - start
            
# =============================================================================
#     This function will calculate error -> L1 Norm.
# =============================================================================
    
    def calculate_error_L1(self):
        b_dash = np.dot(self.A, self.sol)
        return np.sum(np.abs(b_dash - self.b))
        
# =============================================================================
#     Overridden toString() method. 
#     When object of this class is printed, it will print details of the object 
#     in tabular form.
# =============================================================================
    
    def __str__(self):
        table = texttable.Texttable()
        
        table.set_cols_align(["c", "c"])
        table.set_cols_valign(["m", "m"])
        
        values = [["Method", "Jacobi"],
                  ["Matrix Type", self.method + " Matrix"],
                  ["isDiagonalDominant", str(self.isDiagonalDominant)],
                  ["Matrix Size", self.A.shape],
                  ["Iterations", self.count],
                  ["Spectral Radius", self.radius],
                  ["Error - L1 Norm", self.calculate_error_L1()],
                  ["Time taken", self.time_taken]]
        
        table.add_rows(values)
        return table.draw()