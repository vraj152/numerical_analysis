import numpy as np

class GenerateMatrix:
    def __init__(self, lower_threshold = 0, upper_threshold = 1):
        self.counter = 0
        self.A = None
        self.b = None
        self.lower = lower_threshold
        self.upper = upper_threshold
        self.s_radius = 0
    
    def generate_A(self):
        mat = np.random.rand(self.n, self.n)
        
        for r in range(self.n):
            all_sum = sum(abs(mat[r, :])) - abs(mat[r, r])
            mat[r, r] = all_sum * (1 / self.upper)
        
        return mat
        
    def generate_matrix(self):
        self.A = self.generate_A()
        self.b = np.random.rand(self.n, 1)
        
        eigen_values = np.linalg.eigvals(self.A)
        
        if(np.any(np.isclose(eigen_values,np.zeros(self.n),atol=1e-10))):
            print("Generated matrix is not convertible!")
            self.counter += 1
            self.generate_matrix()
    
    def calculate_spactral_radius(self):
        n = self.n
        A = self.A
        
        D = np.zeros((n,n))
        D[range(n),range(n)] = A[range(n),range(n)]
        
        L = np.zeros((n,n))
        for r in range(n):
            L[r,:r] = -A[r,:r]
            
        U = np.zeros((n,n))
        for r in range(n):
            U[r,r+1:] = -A[r,r+1:]
            
        T_jac = np.matmul(np.linalg.inv(D),(L+U))
        ev_jac = max(abs(np.linalg.eigvals(T_jac)))
        
        T_gs = np.matmul(np.linalg.inv(D-L),U)
        ev_gs = max(abs(np.linalg.eigvals(T_gs)))
        
        return max(ev_jac, ev_gs)
    
        """
        By returning maximum among these two:
            we are making sure we have the larger spectral radius so that
            we can have more iterations.
            
            -> we can also set the solver method. Based on which
            spectral radius will be calculated.
        
        if self.method == 'jacobi':
            return ev_jac
        elif self.method == 'gauss_seidel':
            return ev_gs
        
        """
    
    def will_converge(self):
        self.s_radius = self.calculate_spactral_radius()
        
        if(self.lower <= self.s_radius <= self.upper) or np.isclose(self.s_radius,self.upper,atol=1e-4) or np.isclose(self.s_radius,self.lower,atol=1e-4):
            return True
        
        return False
    
    def generate_system(self, n):
        self.n = n
        
        self.generate_matrix()
        
        while(self.counter < 100):
            if(self.will_converge()):
                break
            else:
                print("Generated matrix has spectral radius {0} which is out of range!".format(self.s_radius))
                self.counter += 1
                
        if(self.counter != 0):
            print("System generated %s times!" % (self.counter))
        
        return self.A, self.b