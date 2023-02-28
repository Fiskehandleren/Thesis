import torch
from scipy.stats import poisson

def poisson_negative_log_likelihood(y_predict, y):
    """ https://en.wikipedia.org/wiki/Poisson_regression#Maximum_likelihood-based_parameter_estimation """
    return -torch.sum(y*y_predict - torch.exp(y_predict))


def censored_poisson_negative_log_likelihood(y_predict, y, threshold):
    pois = torch.distributions.poisson.Poisson(y_predict)
    """ https://findit.dtu.dk/en/catalog/53282c10c18e77205dd0f8ae """
    stupid_cdf = torch.tensor(poisson.cdf(y_predict, y))
    return -torch.sum((y<threshold).int() * pois.log_prob(y) + (1 - (y<threshold).int() * torch.log(stupid_cdf)))
