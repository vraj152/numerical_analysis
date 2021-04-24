import numpy as np

class SOR:
    def __init__(self, A, b, it = 1000):
        self.A = A
        self.b = b
        self.iterations = it
        self.sol = None
        self.count = 0
    
    def solver(self):
        pass
    