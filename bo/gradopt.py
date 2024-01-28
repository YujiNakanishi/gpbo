import numpy as np
import copy

def sgd(gp_model, x, grad_f, lr, max_itr):
    x_next = copy.copy(x)
    for i in range(max_itr):
        ac_grad = grad_f(gp_model, x)
        x_next += lr*ac_grad
    return x_next