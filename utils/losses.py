import torch
from torch import nn
from torch import sqrt
import torch.nn.functional as F

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

    return -torch.mean((d_t * pois.log_prob(y)) + ((1-d_t) * (torch.log(poiss_cdf))))


def get_loss_metrics(self, batch, y_hat, stage):
    """
    Returns a dictionary of loss metrics for the given batch and predictions.
    args:
        batch: a batch of data
        y_hat: the predictions for the batch
        stage: the stage of the model (train, val, test)
    
    returns:
        loss_metrics: a dictionary of loss metrics
        y: the true, censored values for the batch
        y_hat: the predictions for the batch
        y_true: the true values for the batch
    """
    _, y, tau, y_true = batch

    if self.censored:
        loss = self.loss_fn(y_hat, y, tau)
        loss_uncen = nn.PoissonNLLLoss(log_input=False)
        loss_true = loss_uncen(y_hat, y_true)

        mse = F.mse_loss(y_hat, y_true)
        mae = F.l1_loss(y_hat, y_true) 
    else:
        loss = self.loss_fn(y_hat, y)
        loss_true = loss
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)  

    return {
        f"{stage}_loss": loss,
        f"{stage}_loss_true": loss_true.item(),
        f"{stage}_mae": mae.item(),
        f"{stage}_rmse": sqrt(mse).item(),
        f"{stage}_mse": mse.item(),
    }, y, y_true, y_hat
