import torch

from .third_party import SENA_GaussianVAEKLDivergenceLoss

class GaussianKLDivergenceLoss:
    """
    A class representing the Gaussian KL divergence loss with dynamic batch handling.

    Notes
    ---
    This class computes the KL divergence between two Gaussian distributions.
    The class takes care of numerical stability by adding a small value to variances.

    Parameters
    ----------
    epsilon : float
        Small value to add to variances for numerical stability.
    reduction : str
        Specifies the reduction to apply to the loss.

    Returns
    -------
    GaussianKLDivergenceLossV1
        An instance of GaussianKLDivergenceLossV1.

    Examples
    --------

    """

    def __init__(self,
        epsilon: float = 1e-8,
        reduction: str = 'mean'
    ):

        self.epsilon = epsilon
        self.reduction = reduction

    def __call__(self,
        mu1: torch.Tensor,
        var1: torch.Tensor,
        mu2: torch.Tensor,
        var2: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the KL divergence between two Gaussian distributions.

        Notes
        -----
        This function splits the computation into several steps for better readability.
        It first ensures numerical stability by adding epsilon to variances, then computes
        the log of variances, and finally applies the KL divergence formula.

        Parameters
        ----------
        mu1 : torch.Tensor
            Mean of the first distribution.
        var1 : torch.Tensor
            Variance of the first distribution.
        mu2 : torch.Tensor
            Mean of the second distribution.
        var2 : torch.Tensor
            Variance of the second distribution.

        Examples
        --------

        """

        # Ensure numerical stability
        var1 = var1 + self.epsilon
        var2 = var2 + self.epsilon

        # Check for positive variances
        if (var1 <= 0).any() or (var2 <= 0).any():
            raise ValueError("Variances must be positive.")

        # Compute log of variances
        logvar1 = torch.log(var1)
        logvar2 = torch.log(var2)

        # KL divergence formula
        kl_element = (
            (var1 + (mu1 - mu2).pow(2)) / var2
            - 1
            + (logvar2 - logvar1)
        )

        # Apply reduction
        if self.reduction == 'sum':
            kl_loss = torch.sum(kl_element)
        elif self.reduction == 'mean':
            kl_loss = torch.mean(kl_element)
        else:
            raise ValueError("Invalid reduction parameter. Accepted values are 'sum' and 'mean'.")

        return kl_loss

class GaussianVAEKLDivergenceVAELoss:
    """
    A class representing the Gaussian KL divergence loss for a Variational Autoencoder (VAE).

    Notes
    -----
    This class computes the KL divergence loss between the approximate posterior distribution
    and the prior distribution (standard normal distribution).

    Parameters
    ----------
    reduction : str
        Specifies the reduction to apply to the loss.
    input_type : str
        Specifies whether the input is log variance ('logvar') or variance ('var').

    Returns
    -------
    GaussianKLDivergenceVAELossV2
        An instance of GaussianKLDivergenceVAELossV2.

    Examples
    --------

    """
    def __init__(self,
        reduction: str = 'sum',
        input_type: str = 'var'
    ):
        self.reduction = reduction
        self.input_type = input_type
        if self.input_type not in ['logvar', 'var']:
            raise ValueError("Invalid input_type parameter. Accepted values are 'logvar' and 'var'.")

    def __call__(self,
        mu: torch.Tensor,
        log_var_or_var: torch.Tensor
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
        log_var_or_var : torch.Tensor
            Logarithm of the variance (if input_type='logvar') or variance (if input_type='var')
            of the approximate posterior distribution.

        Returns
        -------
        torch.Tensor
            The KL divergence loss, either summed or averaged based on the reduction parameter.

        Examples
        --------

        """

        if self.input_type == 'var':
            var = log_var_or_var
            log_var = torch.log(log_var_or_var + 1e-8)  # Adding small epsilon to avoid log(0)
        else:
            var = log_var_or_var.exp()
            log_var = log_var_or_var

        #? Compute the element-wise KL divergence
        kl_elements = torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        # kl_elements = torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

        if self.reduction == 'sum':
            kl_loss = torch.sum(kl_elements)
        elif self.reduction == 'mean':
            kl_loss = -0.5 * torch.mean(kl_elements)
            # kl_loss = kl_elements.mean()/mu.shape[0]
        else:
            raise ValueError("Invalid reduction parameter. Accepted values are 'sum' and 'mean'.")

        return kl_loss


if __name__ == "__main__":
    from .third_party.sena.gaussian_kld_loss import SENA_GaussianVAEKLDivergenceLoss
    torch.manual_seed(42)

    batch_size = 32
    latent_dim = 20

    mu = torch.randn(batch_size, latent_dim, requires_grad=True)
    var = torch.randn(batch_size, latent_dim).abs() + 1e-3

    unit_mu = torch.zeros(batch_size, latent_dim, requires_grad=True)
    unit_var = torch.ones(batch_size, latent_dim, requires_grad=True)

    loss_gkld = GaussianKLDivergenceLoss(reduction='mean')
    loss_sena_kld = SENA_GaussianVAEKLDivergenceLoss(reduction='mean')
    loss_vae = GaussianVAEKLDivergenceVAELoss(reduction='mean')

    # loss_v1_output = loss_v1(mu, var, unit_mu, unit_var)
    #? For loss_gkld_v1_ouptut we need factor -0.5
    loss_gkld_v1_ouptut = loss_gkld(mu, var, unit_mu, unit_var) * -0.5
    loss_sena_kld_output = loss_sena_kld(mu, var.log())
    loss_vae_output = loss_vae(mu, var.log())

    print("=== KL Divergence Loss Comparison ===")
    print(f"Gaussian KLD Version 1 Loss Output: {loss_gkld_v1_ouptut.item()}")
    print(f"VAE Version 1 Loss Output: {loss_sena_kld_output.item()}")
    print(f"VAE Version 2 Loss Output: {loss_vae_output.item()}")