import numpy as np

class GenerateMatrix:
    
# =============================================================================
#     Constructor Arguments:
#         * method (str) = Type of matrix to be generated.
#         * lower_threshold (float) = lower limit for spectral radius
#         * upper_threshold (float) = upper limit for spectra radius
#         (If radius will not not be between this range then system will generate another matrix)
#         * diagonal_dom (bool) = whether diagonal dominant matrix should be generated. 
# =============================================================================
        
    def __init__(self, method = "Random-DD", lower_threshold = 0, upper_threshold = 1, diagonal_dom = True):
        self.A = None
        self.b = None
        self.lower = lower_threshold
        self.upper = upper_threshold
        self.diagonal_dom = diagonal_dom
        
        self.s_radius = 0
        self.method = method
        self.gmc = 0
        self.spc = 0
    
# =============================================================================
#     Accepts:
#         mat = (NxN) matrix
#     Returns:
#         mat = (NxN) matrix
#         
#     It will make matrix diagonal dominant if enabled.
# =============================================================================
    
    def make_diagonal_dominant(self, mat):
        if(not self.diagonal_dom):
            return mat
        
        for r in range(self.n):
            if(np.count_nonzero(mat[r]) > 1):
                all_sum = sum(abs(mat[r, :])) - abs(mat[r, r])
                mat[r, r] = all_sum * (1 / self.upper)
                
        return mat
# =============================================================================
#     
#     Helper functions for generating A matrix.
#     All following functions will take mat -> (NxN) matrix as an argument
#     and will return (NxN) matrix.
#     
#     1) random_dd = Will not modify matrix as A is already randomly generated.
#     2) lower_dd = Will make A lower triangular.
#     3) upper_dd = Will make A upper triangular.
#     4) tri_dd = Will make A tridiagonal.
#     5) symmetrix_matrix = Will make A  symmetric.
#     6) z_matrix = Will make A Z-Matrix.
#     7) q_matrix = Will make A Q-Matrix.
#     
# =============================================================================
    
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
    
# =============================================================================
#     This function generates matrix A randomly of dimension [self.n x self.n]
#     And will invoke method acccordingly based on the argument provided.
#     
# =============================================================================
    
    def generate_A(self):
        mat = np.random.rand(self.n, self.n)
        
        if(self.method == "Random"):
            return self.random_dd(mat)
        elif(self.method == "Lower"):
            return self.lower_dd(mat)
        elif(self.method == "Upper"):
            return self.upper_dd(mat)
        elif(self.method == "TriDiagonal"):
            return self.tri_dd(mat)
        elif(self.method == "Symmetric"):
            return self.symmetrix_matrix(mat)
        elif(self.method == "Z"):
            return self.z_matrix(mat)
        elif(self.method == "Q"):
            return self.q_matrix(mat)
        
# =============================================================================
#     This matrix checks whether the matrix is invertible or not.
#     Returns:
#         bool = False if singular, True otherwise.
#         
# =============================================================================
    
    def check_invertibility(self):
        eigen_values = np.linalg.eigvals(self.A)
        
        if(np.any(np.isclose(eigen_values,np.zeros(self.n),atol=1e-10))):
            return False
        
        return True
    
# =============================================================================
#     This function generates A and b matrix and uses generate_A() as helper
#     function. It will check invertibility.
#     
#     If singular matrix is generated then it will recursively generates new matrix.
#     It will stop if invertible matrix is generated or it has generated matrix 100
#     times. Whichever is earlier.
#     
# =============================================================================
    
    def generate_matrix(self):
        self.A = self.generate_A()
        #print(self.A)
        
        self.b = np.random.rand(self.n, 1)
        
        if(not self.check_invertibility() and self.gmc < 100):
            print("Generated matrix is Singular!")
            self.gmc += 1
            self.generate_matrix()
    
# =============================================================================
#     This function calculates the spectral radius of the matrix.
#     By returning maximum among these two (Jacobi & Gauss Seidel):
#         we are making sure we have the larger spectral radius so that
#         we can have more iterations.
#         
#         -> we can also set the solver method. Based on which
#         spectral radius will be calculated.
#         
#     Returns:
#         int - Spectral radius of the matrix A. 
#         
# =============================================================================
    
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
        if self.method == 'jacobi':
            return ev_jac
        elif self.method == 'gauss_seidel':
            return ev_gs
        
        """
    
# =============================================================================
#     This function will check whether matrix A will converge or not.
#     Check is performed based on spectral radius.
#     
#     If system's spectral radius is between lower bound and upper bound then
#     system will converge. 
#     
#     Returns:
#         bool -> True is returned if system will converge; False otherwise.
#     
# =============================================================================
    
    def will_converge(self):
        self.s_radius = self.calculate_spactral_radius()
        
        if(self.method != "Random-DD" or self.lower <= self.s_radius <= self.upper) or np.isclose(self.s_radius, self.upper, atol=1e-6) or np.isclose(self.s_radius, self.lower, atol=1e-6):
            return True
        
        return False
    
# =============================================================================
#     This function will be called.
#     Accepts:
#         n = (int) -> Dimension of the matrix to be generated.
#     Returns:
#         A, b -> Will return system generated.
#      
#     If generated system will not converge then it will recursively generates new matrix.
#     It will stop if matrix will converge or it has generated matrix 100 times. 
#     Whichever is earlier.    
# 
# =============================================================================
    
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
    
# =============================================================================
#     Accepts:
#         A, b matrix
#     Returns:
#         None
#     
#     Function loads foreign systems to the native system. 
#     While loading the data -> It will check the spectral radius and convergence of the system.
#     
#     If both the check passes -> we have successfully loaded the system.
#     Otherwise -> It will generate exception.
# =============================================================================
    
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