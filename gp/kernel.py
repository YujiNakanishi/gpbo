import numpy as np

class RBFkernel:
    """
    RBFkernel:k(xi, xj) = exp{-sum((xi-xj)^2)/2/l^2}
    """

    def __init__(self, l = 1.):
        self.l = l


    def __call__(self, xi, xj):
        """
        calculate kernel value
        
        input:
            xi, xj -> <np:float:(D, )>
                D -> <int> dimension of x
        output:
            <float> k(xi, xj)
        """
        val = np.sum((xi-xj)**2)
        return np.exp(-val/2./(self.l**2))
    

    def grad(self, xi, xj, t):
        """
        calculate gradient of kernel function
        input:
            xi, xj -> <np:float:(D, )>
                D -> <int> dimension of x
            t -> <int> 0 or 1
        output:
            (grad_xi)k(xi, xj) if t == 0
            (grad_xj)k(xi, xj) if t == 1
        """
        return -self(xi, xj)*(xi - xj)/(self.l**2) if t == 0 else self(xi, xj)*(xi - xj)/(self.l**2)
    
    
    def val_grad(self, xi, xj, t):
        """
        same with
            return self(xi, xj), self.grad(xi, xj, t)
        """
        val = self(xi, xj)
        grad = -val*(xi - xj)/(self.l**2) if t == 0 else val*(xi - xj)/(self.l**2)
        return val, grad