import torch
from torch import nn
from torch import sqrt
import torch.nn.functional as F
from typing import Dict, Tuple


def get_loss(loss):
    if loss == "MSE":
        return nn.MSELoss()
    elif loss == "PNLL":
        return poisson_negative_log_likelihood
    elif loss == "CPNLL":
        return censored_poisson_negative_log_likelihood
    else:
        raise NotImplementedError(f'Loss function "{loss}" not implemented')


def censored_poisson_negative_log_likelihood(y_predict, y, C) -> torch.Tensor:
    """
    y_predict: lambda for Poisson
    y: observed data
    C: censoring threshold
    https://findit.dtu.dk/en/catalog/53282c10c18e77205dd0f8ae"""
    pois = torch.distributions.poisson.Poisson(y_predict)

    # Pytorch doesn't have the cdf function for the poisson distribution so we use the regularized gamma function.
    # Poisson CDF: Q(floor[k+1], lambda). We are prediction P(X <= C-1) = Q(floor[C+1-1], lambda) = Q(floor[C+1-1], lambda)
    # https://en.wikipedia.org/wiki/Poisson_distribution
    poiss_cdf = 1 - torch.special.gammaincc(torch.floor(C), y_predict)
    poiss_cdf += 1e-8  # to avoid log(0)
    d_t = (C > y).int()

    return -torch.mean((d_t * pois.log_prob(y)) + ((1 - d_t) * (torch.log(poiss_cdf))))


def poisson_negative_log_likelihood(y_predict, y) -> torch.Tensor:
    pois = torch.distributions.poisson.Poisson(y_predict)
    return -torch.mean(pois.log_prob(y))


def calculate_losses(y_hat, y, tau, y_true, censored, loss_fn):
    """
    Calculate loss metrics based on whether the data is censored or not.

    args:
        y_hat: the predictions for the batch
        y: the true, censored values for the batch
        tau: censoring indicator
        y_true: the true values for the batch
        censored: boolean flag indicating whether the data is censored or not
        loss_fn: loss function used to calculate the main loss

    returns:
        A tuple consisting of the main loss, the true loss,
        Mean Absolute Error (MAE), Mean Squared Error (MSE),
        and Root Mean Squared Error (RMSE).
    """
    if censored:
        loss = loss_fn(y_hat, y, tau)
        loss_uncen = nn.PoissonNLLLoss(log_input=False)
        loss_true = loss_uncen(y_hat, y_true)

        mse = F.mse_loss(y_hat, y_true)
        mae = F.l1_loss(y_hat, y_true)
    else:
        # This is the loss the unaware model is optimizing after (censored target)
        loss = loss_fn(y_hat, y)
        # We evaluate the model on losses between the true target (latent) and the predictions
        loss_uncen = nn.PoissonNLLLoss(log_input=False)
        loss_true = loss_uncen(y_hat, y_true)
        mse = F.mse_loss(y_hat, y_true)
        mae = F.l1_loss(y_hat, y_true)

    rmse = torch.sqrt(mse)

    return loss, loss_true, mae, mse, rmse


def get_loss_metrics(
    self, batch, y_hat, stage
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
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

    loss, loss_true, mae, mse, rmse = calculate_losses(
        y_hat, y, tau, y_true, self.censored, self.loss_fn
    )

    return (
        {
            f"{stage}_loss": loss,
            f"{stage}_loss_true": loss_true.item(),
            f"{stage}_mae": mae.item(),
            f"{stage}_rmse": rmse.item(),
            f"{stage}_mse": mse.item(),
        },
        y,
        y_true,
        y_hat,
    )
