import torch
from scipy.stats import poisson
from torch import nn

def get_loss(loss):
    if loss == "mse":
        return nn.MSELoss()
    elif loss == "PNLL":
        return nn.PoissonNLLLoss(log_input=False)
    elif loss == "CPNLL":
        return censored_poisson_negative_log_likelihood
    else:
        raise NotImplementedError("Loss function not implemented")

def poisson_negative_log_likelihood(y_predict, y):
    """ https://en.wikipedia.org/wiki/Poisson_regression#Maximum_likelihood-based_parameter_estimation """
    return -torch.sum(y*y_predict - torch.exp(y_predict))

def poisson_cdf(k, lamb):
    """Assume k is a tensor where every element is the same!!!!"""
    _pois = torch.distributions.Poisson(lamb)
    cdf = torch.zeros(k.shape)
    k_int = torch.max(k).int()
    for i in range(k_int+1):
        cdf += _pois.log_prob(torch.tensor(i)).exp()
    return cdf

def censored_poisson_negative_log_likelihood(y_predict, y, C):
    """ 
    y_predict: lambda for Poisson
    y: observed data
    C: censoring threshold
    https://findit.dtu.dk/en/catalog/53282c10c18e77205dd0f8ae """
    pois = torch.distributions.poisson.Poisson(y_predict)

    # Pytorch doesn't have the cdf function for the poisson distribution
    poiss_cdf = 1 - poisson_cdf(k=C-1, lamb=y_predict) # Is C-1 correct?

    d_t = (C > y).int()

    return -torch.sum((d_t * pois.log_prob(y)) + ((1-d_t) * (torch.log(poiss_cdf)))) #Do we sum on the correct axis here?
