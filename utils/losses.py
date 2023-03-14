import torch
from scipy.stats import poisson
from torch import nn

def get_loss(loss):
    if loss == "mse":
        return nn.MSELoss()
    elif loss == "PNLL":
        return nn.PoissonNLLLoss(log_input=False)
    else:
        raise NotImplementedError("Loss function not implemented")
    #elif loss == "CPNLL":
    #    return censored_poisson_negative_log_likelihood

def poisson_negative_log_likelihood(y_predict, y):
    """ https://en.wikipedia.org/wiki/Poisson_regression#Maximum_likelihood-based_parameter_estimation """
    return -torch.sum(y*y_predict - torch.exp(y_predict))

def censored_poisson_negative_log_likelihood(y_predict, y, C):
    """ 
    y_predict: lambda for Poisson
    y: observed data
    C: censoring threshold
    https://findit.dtu.dk/en/catalog/53282c10c18e77205dd0f8ae """

    pois = torch.distributions.poisson.Poisson(y_predict)

    # Pytorch doesn't have the cdf function for the poisson distribution
    poiss_cdf = torch.tensor(poisson.cdf(k=C, mu=y_predict.detach().numpy()))
    

    d_t = (y<C).int()
    return 1 - torch.sum(pois.log_prob(y)) - torch.sum(d_t * pois.log_prob(y) + (1 - d_t) * torch.log(poiss_cdf)) #Do we sum on the correct axis here?



'''
def tobit_loss(y_predict, y, C):
    """ 
    y_predict: lambda for Poisson
    y: observed data
    C: censoring threshold
    https://findit.dtu.dk/en/catalog/53282c10c18e77205dd0f8ae """

    pois = torch.distributions.poisson.Poisson(y_predict)

    # Pytorch doesn't have the cdf function for the poisson distribution
    poiss_cdf = torch.tensor(poisson.cdf(k=C, mu=y_predict))

    d_t = (y<C).int()
    return 1 - torch.sum(pois.log_prob(y)) - torch.sum(d_t * pois.log_prob(y) + (1 - d_t) * torch.log(poiss_cdf)) # Do we sum on the correct axis here?
'''
