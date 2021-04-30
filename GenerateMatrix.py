import numpy as np

class GenerateMatrix:
    def __init__(self, method = "Random-DD", lower_threshold = 0, upper_threshold = 1):
        self.A = None
        self.b = None
        self.lower = lower_threshold
        self.upper = upper_threshold
        self.s_radius = 0
        self.method = method
        self.gmc = 0
        self.spc = 0
    
    def make_diagonal_dominant(self, mat):
        for r in range(self.n):
            if(np.count_nonzero(mat[r]) > 1):
                all_sum = sum(abs(mat[r, :])) - abs(mat[r, r])
                mat[r, r] = all_sum * (1 / self.upper)
                
        return mat
    
    def random_dd(self, mat):
        return self.make_diagonal_dominant(mat)
    
    def lower_dd(self, mat):
        for r in range(self.n):
            mat[r, r+1:] = 0
        
        return self.make_diagonal_dominant(mat)
    
    def upper_dd(self, mat):
        for r in range(self.n):
            mat[r, :r] = 0
        
        return self.make_diagonal_dominant(mat)
    
    def tri_dd(self, mat):
        for r in range(self.n):
            x = max(0, r-1)
            
            mat[r, r+2:] = 0
            mat[r, :x] = 0
            
        return self.make_diagonal_dominant(mat)
    
    def symmetrix_matrix(self, mat):
        for r in range(self.n):
            mat[r+1:, r] = mat[r, r+1:]
        
        return self.make_diagonal_dominant(mat)
        
    def z_matrix(self, mat, mode = ""):
        mat = mat * -1
        mat[range(self.n), range(self.n)] = -mat[range(self.n), range(self.n)]
        
        if(mode == "q"):
            return mat
            
        return self.make_diagonal_dominant(mat)
    
    def q_matrix(self, mat):
        mat = self.z_matrix(mat, "q")
        mat[range(self.n), range(self.n)] = 0
        
        for c in range(self.n):
            mat[c, c] = -sum(mat[:,c])

        return mat
        
    def generate_A(self):
        mat = np.random.rand(self.n, self.n)
        
        if(self.method == "Random-DD"):
            return self.random_dd(mat)
        elif(self.method == "Lower-DD"):
            return self.lower_dd(mat)
        elif(self.method == "Upper-DD"):
            return self.upper_dd(mat)
        elif(self.method == "TriDiagonal-DD"):
            return self.tri_dd(mat)
        elif(self.method == "Symmetric-DD"):
            return self.symmetrix_matrix(mat)
        elif(self.method == "Z-DD"):
            return self.z_matrix(mat)
        elif(self.method == "Q-DD"):
            return self.q_matrix(mat)
        
    def check_invertibility(self):
        eigen_values = np.linalg.eigvals(self.A)
        
        if(np.any(np.isclose(eigen_values,np.zeros(self.n),atol=1e-10))):
            return False
        
        return True
    
    def generate_matrix(self):
        self.A = self.generate_A()
        #print(self.A)
        
        self.b = np.random.rand(self.n, 1)
        
        if(not self.check_invertibility() and self.gmc < 100):
            print("Generated matrix is Singular!")
            self.gmc += 1
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
        
        self.jacobi_rad = ev_jac
        self.gs_rad = ev_gs
        
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
        
        if(self.method != "Random-DD" or self.lower <= self.s_radius <= self.upper) or np.isclose(self.s_radius, self.upper, atol=1e-6) or np.isclose(self.s_radius, self.lower, atol=1e-6):
            return True
        
        return False
    
    def generate_system(self, n):
        self.n = n
        
        self.generate_matrix()
        
        while(self.spc < 100):
            if(self.will_converge()):
                print("System generated successfully!")
                break
            else:
                print("Generated matrix has spectral radius {0} which is out of range!".format(self.s_radius))
                self.generate_system(n)
                self.spc += 1
                
        if(self.spc + self.gmc != 0):
            print("System generated %s times!" % (self.spc + self.gmc))
        
        return self.A, self.b
    
    def foreign_system(self, A, b):
        self.A = A
        self.b = b
        
        self.n = self.A.shape[0]
        
        if(not self.check_invertibility()):
            print("Exception: Provided matrix is Singular!" )
            return
            
        if(not self.will_converge()):
            print("Exception: Provided matrix will not converge!")
            print("It has Spectral radius: ", max(self.jacobi_rad,self.gs_rad))
            return
        
        print("Foreign System Initialized Successfully!")