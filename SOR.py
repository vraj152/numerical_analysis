import numpy as np

class MyException(Exception):
    pass
    
class SOR:
    def __init__(self, A, b, initial_guess = 0, omega = 0.5, it = 1000):
        self.A = A
        self.b = b
        self.iterations = it
        self.sol = None
        self.count = 0
        self.omega = omega
        self.initial_guess = initial_guess
        
    def solver(self):
        residual_convergence = 1e-8
        
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
        
    def __str__(self):
        return "SOR Iterations: " + str(self.count)