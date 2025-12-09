import torch
import torch.nn.functional as F
from torch import tensor, abs as torch_abs, logical_not, log, clamp

def masked_adt_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = torch.abs(target-input) * mask
    return loss.sum() / mask.sum()

 

def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def quantile_loss(
    pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mask=mask.float()
    # q = tensor([0.1, 0.25, 0.75, 0.9], device = pred.device)
    q = tensor([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95], device = pred.device)
    bias = pred - truth[:, :, None]
    
    I_over = bias.detach() > 0.
    q_weight = I_over * (1 - q) + logical_not(I_over) * q
    
    q_loss = torch_abs(bias) * q_weight
    q_loss = q_loss.sum(axis = 2) * mask
    return q_loss.sum()/ mask.sum()

        


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()
