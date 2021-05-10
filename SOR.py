import numpy as np
import texttable
import time

class MyException(Exception):
    pass
    
class SOR:
    def __init__(self, A, b, method, isDiagonalDominant, initial_guess = 0, omega = 0.5, it = 1000, debug = False, fault_tolerance = 1e-4):
        self.A = A
        self.b = b
        self.method = method
        self.isDiagonalDominant = isDiagonalDominant
        
        self.iterations = it
        self.sol = None
        self.count = 0
        self.omega = omega
        self.initial_guess = initial_guess
        self.time_taken = 0
        self.debug = debug
        self.fault_tolerance = fault_tolerance
        
    def solver(self):
        start = time.time()
        residual_convergence = self.fault_tolerance
        
        if(self.initial_guess == 0):
            self.initial_guess = np.zeros_like(self.b)
        
        try:
            if(self.b.shape != self.initial_guess.shape):
                raise MyException("b and Initital Guess do not have same shape!")
        except Exception as e:
            print("Exception:" + str(e))
            return
        
        self.sol = self.initial_guess[:]
        
        curr_residual = np.linalg.norm(np.matmul(self.A, self.sol) - self.b)
        
        while(curr_residual > residual_convergence):
            self.count += 1
            
            for i in range(self.A.shape[0]):
                total = 0
                
                for j in range(self.A.shape[1]):
                    if(j != i):
                        total += self.A[i, j] * self.sol[j]
                
                self.sol[i] = (1 - self.omega) * self.sol[i] + (self.omega / self.A[i, i]) * (self.b[i] - total)
            
            curr_residual = np.linalg.norm(np.matmul(self.A, self.sol) - self.b)
            if(self.debug):
                print("Iteration: {} and Residual: {}".format(self.count, curr_residual))
            
        self.time_taken = time.time() - start
    
    def calculate_error_L1(self):
        b_dash = np.dot(self.A, self.sol)
        return np.sum(np.abs(b_dash - self.b))
        
    def __str__(self):
        table = texttable.Texttable()
        
        table.set_cols_align(["c", "c"])
        table.set_cols_valign(["m", "m"])
        
        values = [["Method", "Successive Over-Relaxation"],
                  ["Matrix Type", self.method + " Matrix"],
                  ["isDiagonalDominant", str(self.isDiagonalDominant)],
                  ["Matrix Size", self.A.shape],
                  ["Iterations", self.count],
                  ["Omega", self.omega],
                  ["Fault Tolerance", self.fault_tolerance],
                  ["Error - L1 Norm", self.calculate_error_L1()],
                  ["Time taken", self.time_taken]]
        
        table.add_rows(values)
        return table.draw()