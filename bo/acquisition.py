import numpy as np
from scipy.stats import norm

def prob_imprv(gp_model, Xm):
    y_min = np.min(gp_model.yn)
    mu, Sigma = gp_model(Xm)
    sigma = np.sqrt(np.diag(Sigma))

    return norm.cdf((y_min - mu)/sigma)

def prob_imprv_grad(gp_model, x):
    y_min = np.min(gp_model.yn)
    mu, sigma, grad_mu, grad_sigma = gp_model.pred_grad(x)
    return norm.pdf((y_min - mu)/sigma)*(mu*grad_sigma - sigma*grad_mu)/(sigma**2)


def expected_imprv(gp_model, Xm):
    y_min = np.min(gp_model.yn)
    mu, Sigma = gp_model(Xm)
    sigma = np.sqrt(np.diag(Sigma))

    return (y_min - mu)*norm.cdf((y_min - mu)/sigma) + sigma*norm.pdf((y_min - mu)/sigma)

def expected_imprv_grad(gp_model, x):
    y_min = np.min(gp_model.yn)
    mu, sigma, grad_mu, grad_sigma = gp_model.pred_grad(x)
    return grad_sigma*norm.pdf((y_min-mu)/sigma) - grad_mu*norm.cdf((y_min-mu)/sigma)


def low_conf(gp_model, x):
    n = float(len(gp_model.yn))
    beta = np.sqrt(np.log(n)/n)

    mu, Sigma = gp_model(Xm)
    sigma = np.sqrt(np.diag(Sigma))

    return -mu + beta*sigma

def low_conf_grad(gp_model, x):
    n = float(len(gp_model.yn))
    beta = np.sqrt(np.log(n)/n)

    mu, sigma, grad_mu, grad_sigma = gp_model.pred_grad(x)

    return -grad_mu + beta*grad_sigma