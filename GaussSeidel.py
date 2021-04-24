import numpy as np

class GaussSeidel:
    def __init__(self, A, b, it = 100):
        self.A = A
        self.b = b
        self.iteration = it
        self.sol = None
        self.count = 0
    
    def solver(self):
        self.sol = np.zeros_like(self.b)
        
        for it in range(self.iteration):
            self.count += 1
            temp = np.zeros_like(self.sol)
            
            for i in range(self.A.shape[0]):
                s1 = np.dot(self.A[i, :i], temp[:i])
                s2 = np.dot(self.A[i, i + 1 :], self.sol[i + 1 :])
                temp[i] = (self.b[i] - s1 - s2) / self.A[i, i]
                
            if np.allclose(self.sol, temp, rtol=1e-8):
                break
            
            self.sol = temp

    def calculate_error(self):
        return np.dot(self.A, self.sol) - self.b
    
    def __str__(self):
        return "Gauss Seidel Iteration: " + str(self.count)
    
    
    
    

            