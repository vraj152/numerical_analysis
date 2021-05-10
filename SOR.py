import numpy as np
import texttable
import time

class MyException(Exception):
    pass
    
class SOR:
# =============================================================================
#     Constructor:
#         A (numpy 2D array) = [NxN] Matrix
#         b (numpy 1D array) = [Nx1] Matrix
#         method (str) = Type of matrix that has been given to the system
#         isDiagonalDominant (bool) = Whether provided matrix is diagonal dominant
#         initial_guess (numpy 1D array) = Initial guess for SoR method.
#         omega (float) = Relaxation Factor
#         it (int) = Maximum number of iterations
#         debug (bool) = Setting to view the logs generated during the convergence
#         fault_tolerance (float) = Residual allowed.    
# =============================================================================
    
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

# =============================================================================
#     This function solves the linear system provided.
# =============================================================================
    
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