import torch
from torch import nn

def get_loss(loss):
    if loss == "MSE":
        return nn.MSELoss()
    elif loss == "PNLL":
        return nn.PoissonNLLLoss(log_input=False)
    elif loss == "CPNLL":
        return censored_poisson_negative_log_likelihood
    else:
        raise NotImplementedError(f"Loss function \"{loss}\" not implemented")

def poisson_negative_log_likelihood(y_predict, y):
    """ https://en.wikipedia.org/wiki/Poisson_regression#Maximum_likelihood-based_parameter_estimation """
    pois = torch.distributions.poisson.Poisson(y_predict)
    return -torch.sum(pois.log_prob(y))

''' NOT IN USE CURRENTLY 
def poisson_negative_log_likelihood(y_predict, y, C):
    return nn.PoissonNLLLoss(log_input=False)

def mean_squared_error(y_predict, y, C):
    return nn.MSELoss()
'''

def censored_poisson_negative_log_likelihood(y_predict, y, C):
    """ 
    y_predict: lambda for Poisson
    y: observed data
    C: censoring threshold
    https://findit.dtu.dk/en/catalog/53282c10c18e77205dd0f8ae """
    pois = torch.distributions.poisson.Poisson(y_predict)

    # Pytorch doesn't have the cdf function for the poisson distribution so we use the regularized gamma function.
    # Poisson CDF: Q(floor[k+1], lambda). We are prediction P(X <= C-1) = Q(floor[C+1-1], lambda) = Q(floor[C+1-1], lambda)
    # https://en.wikipedia.org/wiki/Poisson_distribution 
    poiss_cdf = 1 - torch.special.gammaincc(torch.floor(C), y_predict)
    poiss_cdf += 1e-8
    d_t = (C > y).int()

    return -torch.sum((d_t * pois.log_prob(y)) + ((1-d_t) * (torch.log(poiss_cdf))))