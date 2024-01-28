import numpy as np
from gpbo.gp.kernel import *

def getKernelMatrix(kernel, Xn, Xm = None):
    """
    calculate kernel matrix
    input:
        kernel -> <kernel class>
        Xn -> <np:float:(N, D)> input1
            N -> <int> num of data
            D -> <int> dimension
        Xm -> <np:float:(M, D)> input2
            M -> <int> num of data
    output:
        if Xm is None
            K -> <np:float:(N, N)> K[i, j] = kernel(Xn[i], Xn[j])
        else
            K -> <np:float:(N, M)> K[i, j] = kernel(Xn[i], Xm[j])
    """
    N = len(Xn)
    if Xm is None:
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                K[i, j] = K[j, i] = kernel(Xn[i], Xn[j])

    else:
        M = len(Xm)
        K = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                K[i, j] = kernel(Xn[i], Xm[j])

    return K


class element:
    """
    Element class for gaussian process.
    Define some methods which are useful for any gaussian process.
    """
    def __init__(self, kernel, Xn, yn, sigma):
        self.kernel = kernel #<kernel class>
        self.Xn = Xn #<np:float:(N, D)> input data
        self.sigma = sigma #<float> standard dev of observation
        self.y_mean = np.mean(yn) #<float>
        self.yn = yn - self.y_mean #<np:float:(N, )> output data
    
    def inverse_wsigma(self, K):
        return np.linalg.inv(K + (self.sigma**2)*np.eye(len(K)))


class GP(element):
    """
    class of pure gaussian process
    """
    def __init__(self, kernel, Xn, yn, sigma):
        super().__init__(kernel, Xn, yn, sigma)
        self.Knn = getKernelMatrix(self.kernel, self.Xn) #<np:float:(N, N)> kernel matrix
        self.Knn_inv_wsigma = self.inverse_wsigma(self.Knn) #(Knn + sigma^2I)
    

    def __call__(self, Xm):
        Knm = getKernelMatrix(self.kernel, self.Xn, Xm)
        Kmm = getKernelMatrix(self.kernel, Xm)

        mu = (Knm.T)@(self.Knn_inv_wsigma@self.yn) + self.y_mean
        Sigma = Kmm - (Knm.T)@(self.Knn_inv_wsigma@Knm)

        return mu, Sigma
    

    def addData(self, Xm, ym):
		Knm = getKernelMatrix(self.kernel, self.Xn, Xm)
		Kmm = getKernelMatrix(self.kernel, Xm)

		K1 = np.concatenate((self.Knn, Knm), axis = 1)
		K2 = np.concatenate((Knm.T, Kmm), axis = 1)

		self.Knn = np.concatenate((K1, K2), axis = 0)
		self.Knn_inv_wsigma = self.inverse_wsigma(self.Knn)
		
		self.Xn = np.concatenate((self.Xn, Xm), axis = 0)
		yn = np.concatenate((self.yn + self.y_mean, ym))
		self.y_mean = np.mean(yn)
		self.yn = yn - self.y_mean


class sor(element):
    """
    class of SOR gaussian process
    """
    def __init__(self, kernel, Xn, yn, sigma, Us):
        super().__init__(kernel, Xn, yn, sigma)
		self.Us = Us #<np:float:(S, D)> latent inputs
		self.Ksn = getKernelMatrix(kernel, Us, Xn)
		self.Kss = getKernelMatrix(kernel, Us)
		self.S = np.linalg.inv(self.Ksn@(self.Ksn.T)/(self.sigma**2) + self.Kss)
    
    def __call__(self, Xm):
        Kms = getKernelMatrix(self.kernel, Xm, self.Us)
        mu = Kms@self.S@self.Ksn@self.yn/(self.sigma**2) + self.y_mean
        Sigma = Kms@self.S@(Kms.T)

        return mu, Sigma


class dtc(sor):
    """
    class of DTC gaussian process
    """
    def __init__(self, kernel, Xn, yn, sigma, Us):
        super().__init__(kernel, Xn, yn, sigma, Us)
        self.Kss_inv = np.linalg.inv(self.Kss)

    def __call__(self, Xm):
        Kmm = getKernelMatrix(self.kernel, Xm)
        Kms = getKernelMatrix(self.kernel, Xm, self.Us)
        Qmm = Kms@self.Kss_inv@(Kms.T)

        mu = Kms@self.S@self.Ksn@self.yn/(self.sigma**2) + self.y_mean
        Sigma = Kms@self.S@(Kms.T) + Kmm - Qmm

        return mu, Sigma