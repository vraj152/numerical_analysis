import numpy as np
import texttable
import time

class Jacobi:
    def __init__(self, A, b, method, radius = "NA", it = 1000):
        self.A = A
        self.b = b
        self.iterations = it
        self.sol = None
        self.count = 0
        self.time_taken = 0
        self.radius = radius
        self.method = method
    
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
            
    def calculate_error_L1(self):
        b_dash = np.dot(self.A, self.sol)
        return np.sum(np.abs(b_dash - self.b))
        
    def __str__(self):
        table = texttable.Texttable()
        
        table.set_cols_align(["c", "c"])
        table.set_cols_valign(["m", "m"])
        
        values = [["Method", "Jacobi"],
                  ["Matrix Type", self.method],
                  ["Matrix Size", self.A.shape],
                  ["Iterations", self.count],
                  ["Spectral Radius", self.radius],
                  ["Error - L1 Norm", self.calculate_error_L1()],
                  ["Time taken", self.time_taken]]
        
        table.add_rows(values)
        return table.draw()