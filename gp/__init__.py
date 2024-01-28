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