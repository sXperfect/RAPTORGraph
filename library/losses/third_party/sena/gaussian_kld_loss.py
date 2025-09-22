import torch

class SENA_GaussianVAEKLDivergenceLoss:
    """
    A class representing the Gaussian KL divergence loss for a Variational Autoencoder (VAE).
    Based on SENA & Discrepancy VAE paper

    Notes
    ---
    This class computes the KL divergence loss between the approximate posterior distribution
    and the prior distribution (standard normal distribution).

    Parameters
    ----------
    reduction : str
        Specifies the reduction to apply to the loss.

    Returns
    -------
    GaussianKLDivergenceVAELossV1
        An instance of GaussianKLDivergenceVAELossV1.

    Examples
    --------

    """
    def __init__(self, 
        reduction: str = 'mean'
    ):

        self.reduction = reduction
    
    # @validate_call
    def __call__(self, 
        mu: torch.Tensor, 
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the KL divergence loss.

        Notes
        -----
        This function computes the element-wise KL divergence and then applies the reduction.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the approximate posterior distribution.
        log_var : torch.Tensor
            Logarithm of the variance of the approximate posterior distribution.

        Returns
        -------
        torch.Tensor
            The KL divergence loss, either summed or averaged based on the reduction parameter.

        Examples
        --------

        """
        
        KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        KLD_loss = torch.mean(KLD_element).mul_(-0.5) / mu.shape[0]

        return KLD_loss