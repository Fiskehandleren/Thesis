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

def poisson_cdf_non_identical(k, lamb):
    """ If k is not a tensor of idential shape to lamb, this will still work.
        k: tensor of shape (n) !! ALWAYS INTEGER !!
    """
    _pois = torch.distributions.Poisson(lamb) # Initialize Poisson distribution with rates
    cdf = torch.zeros(k.shape) # placeholder
    n = len(k) # number of samples
    k_int = torch.ceil(k).int()
    max_k_int = torch.max(k_int)
    # Matrix of 0-max_k_int repeated n times across columns, each column being reserved for one value of lambda. 
    # so each columns is a range of 0-max_k_int.
    k_mtrx = torch.repeat_interleave(torch.arange(0, max_k_int+1), n).view(max_k_int+1, n)
    # Calculate the pdf of 0, 1, .., max_k_int for each lambda.
    pdf_mtrx = _pois.log_prob(k_mtrx).exp()
    print("torch.max(k_int)", torch.max(k_int))
    for i in range(len(k)):
        # Sum the pdf of 0, 1, .., k_int[i] for each lambda to get the cdf for each k
        cdf[i] = torch.sum(pdf_mtrx[:k_int[i]+1,i])
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