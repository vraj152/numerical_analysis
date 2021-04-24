import numpy as np

class GenerateMatrix:
    def __init__(self):
        self.counter = 0
        self.A = None
        self.b = None
    
    def generate_A(self):
        mat = np.random.rand(self.n, self.n)
        
        for r in range(self.n):
            mat[r, r] = sum(abs(mat[r, :]) - abs(mat[r, r])) + np.random.randint(self.n, 2 * self.n)
        
        return mat
        
    def generate_matrix(self):
        self.A = self.generate_A()
        self.b = np.random.rand(self.n, 1)
        
        eigen_values, _ = np.linalg.eig(self.A)
        
        if(np.any(np.isclose(eigen_values,np.zeros(self.n),atol=1e-10))):
            print("Generated matrix is not convertible!")
            self.counter += 1
            self.generate_matrix()
    
    def calculate_spactral_radius(self):
        T = np.copy(self.A)
        
        for r in range(self.n):
            T[r, :] = -T[r, :] / T[r, r]
            T[r, r] = 0
            
        return max(abs(np.linalg.eigvals(T)))
    
    def will_converge(self):
        if(self.calculate_spactral_radius() <= 0.2):
            return True
        
        return False
    
    def generate_system(self, n):
        self.n = n
        
        self.generate_matrix()
        
        while(self.counter < 100):
            if(self.will_converge()):
                break
            else:
                print("Generated matrix will diverge!")
                self.counter += 1
                
        if(self.counter != 0):
            print("System generated %s times!" % (self.counter))
        
        return self.A, self.b