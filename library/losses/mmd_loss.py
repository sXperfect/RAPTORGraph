"""
Optimized Multi-Kernel Maximum Mean Discrepancy (MMD) Implementations.

This module provides memory-efficient implementations for computing MMD,
both as a standalone PyTorch nn.Module and as an Ignite Metric. The primary
optimization is in the Gaussian kernel calculation, which now avoids creating
large intermediate tensors, making it suitable for high-dimensional data.
A flag is provided to switch back to the original, memory-intensive method if needed.

This script can also be run directly to compare the performance and memory
usage of the two approaches.
"""
import typing as t

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import torch
import torch.nn as nn
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from ignite.metrics import MaximumMeanDiscrepancy #? Correct implementation

# =============================================================================
# LOCAL IMPORTS
# =============================================================================
from .third_party.sena.mmd_loss import SENA_MultikernelMaximumMeanDiscrepancy #? From SENA Paper
from .third_party.discrepancy_vae.mmd_loss import DiscrepancyVAE_MultiKernelMaximumMeanDiscrepancy

# =============================================================================
# CONSTANTS
# =============================================================================
DISCREPANCY_VAE_MMD_SIGMA = 1000 #? https://github.com/uhlerlab/discrepancy_vae/blob/4451fdbc9d0aa3a1dee4e7d1b743a434e98fa58a/src/run.py#L29
DISCREPANCY_VAE_KERNEL_NUM = 10 #? https://github.com/uhlerlab/discrepancy_vae/blob/4451fdbc9d0aa3a1dee4e7d1b743a434e98fa58a/src/run.py#L30
DISCREPANCY_VAE_KERNEL_MUL = 2.0 #? https://github.com/uhlerlab/discrepancy_vae/blob/4451fdbc9d0aa3a1dee4e7d1b743a434e98fa58a/src/utils.py#L13

SENA_MMD_SIGMA = 200 #? https://github.com/ML4BM-Lab/SENA/blob/f2dbcc50e2000cd4b4319634ca1fda4028d3f0f1/src/sena_discrepancy_vae/main.py#L40
SENA_KERNEL_NUM = 10 #? https://github.com/ML4BM-Lab/SENA/blob/f2dbcc50e2000cd4b4319634ca1fda4028d3f0f1/src/sena_discrepancy_vae/main.py#L41
SENA_KERNEL_MUL = 2.0 #? https://github.com/ML4BM-Lab/SENA/blob/f2dbcc50e2000cd4b4319634ca1fda4028d3f0f1/src/sena_discrepancy_vae/utils.py#L612

DISCREPANCY_VAE_MMD_KWARGS = {
    "MMD_sigma": DISCREPANCY_VAE_MMD_SIGMA,
    "kernel_num": DISCREPANCY_VAE_KERNEL_NUM,
    "kernel_mul": DISCREPANCY_VAE_KERNEL_MUL,
}

SENA_MMD_KWARGS = {
    "MMD_sigma": DISCREPANCY_VAE_MMD_SIGMA,
    "kernel_num": DISCREPANCY_VAE_KERNEL_NUM,
    "kernel_mul": DISCREPANCY_VAE_KERNEL_MUL,
}


# =============================================================================
# PURE PYTORCH MMD IMPLEMENTATION
# =============================================================================
class MultiKernelMaximumMeanDiscrepancy(nn.Module):
    """
    Computes the Maximum Mean Discrepancy (MMD) loss using a memory-efficient
    method with multiple Gaussian kernels.

    This module calculates the MMD loss by generating a series of Gaussian kernels
    with varying bandwidths. It can compute both the biased and unbiased
    estimators of MMD. The pairwise distance calculation can be done using a
    memory-efficient approach or the original method via a flag.

    Parameters
    ----------
    kernel_mul : float, optional
        The multiplier for generating a geometric progression of bandwidths.
        Defaults to 2.0.
    kernel_num : int, optional
        The number of kernels to use. Defaults to 5.
    fix_sigma : float | None, optional
        A fixed value for the base bandwidth. If None, the bandwidth is
        estimated from the data. Defaults to None.
    unbiased : bool, optional
        If True (default), computes the unbiased MMD estimator. Otherwise,
        computes the biased estimator.
    use_torch_mm : bool, optional
        If True (default), uses a memory-efficient method to calculate pairwise
        distances. If False, uses the original, more memory-intensive method.
    """

    def __init__(
        self,
        #? --- Kernel Configuration ---
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: float | None = None,
        #? --- Estimator Configuration ---
        unbiased: bool = True,
        use_torch_mm: bool = False,
    ):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.unbiased = unbiased
        self.use_torch_mm = use_torch_mm

    def _gaussian_kernel(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the Gaussian kernel matrix.

        Switches between a memory-efficient calculation and the original, more
        memory-intensive approach based on the `use_memory_efficient` flag.
        """
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        if self.use_torch_mm:
            #? --- Memory-efficient pairwise L2 distance calculation ---
            total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
            total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
            l2_distance = ((total0 - total1) ** 2).sum(2)
        else:
            #? --- Using torch mm ---
            gram_matrix = torch.mm(total, total.t())
            sq_norms = torch.diag(gram_matrix)
            l2_distance = sq_norms.unsqueeze(1) - 2.0 * gram_matrix + sq_norms.unsqueeze(0)
            l2_distance = torch.clamp(l2_distance, min=0)

        #? --- Bandwidth estimation (using mean of pairwise distances) ---
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            #? Using l2_distance directly is the modern and correct approach.
            #? The operation is not part of the computation graph for gradients.
            with torch.no_grad():
                bandwidth = torch.sum(l2_distance) / (n_samples**2 - n_samples)

        #? --- Multi-kernel bandwidth calculation ---
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [
            bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)
        ]

        #? Compute the final kernel value by summing over all bandwidths
        kernel_val = [torch.exp(-l2_distance / (bw + 1e-8)) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the MMD loss between source and target distributions.

        Parameters
        ----------
        source : torch.Tensor
            Samples from the source distribution, shape (batch_size, features).
        target : torch.Tensor
            Samples from the target distribution, shape (batch_size, features).

        Returns
        -------
        torch.Tensor
            The computed MMD loss.
        """
        batch_size = source.size(0)
        kernels = self._gaussian_kernel(source, target)

        xx = kernels[:batch_size, :batch_size]
        yy = kernels[batch_size:, batch_size:]
        xy = kernels[:batch_size, batch_size:]

        if not self.unbiased:
            #? --- Biased MMD estimator ---
            loss = torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)
        else:
            #? --- Unbiased MMD estimator ---
            if batch_size <= 1:
                return torch.tensor(0.0, device=source.device)

            #? Exclude diagonal elements from the sums
            sum_xx = torch.sum(xx) - torch.trace(xx)
            sum_yy = torch.sum(yy) - torch.trace(yy)
            sum_xy = torch.sum(xy)

            #? --- Denominators for unbiased estimate ---
            denom_xx_yy = float(batch_size * (batch_size - 1))
            denom_xy = float(batch_size * batch_size)

            loss = (sum_xx / denom_xx_yy) + (sum_yy / denom_xx_yy) - (2 * sum_xy / denom_xy)

        return loss


# =============================================================================
# IGNITE METRIC IMPLEMENTATION
# =============================================================================
class Ignite_MultiKernelMaximumMeanDiscrepancy(Metric):
    """
    Custom Ignite metric to compute MMD with a selectable memory approach.

    This metric calculates the MMD loss by generating a series of Gaussian kernels
    with varying bandwidths. The pairwise distance calculation can be done using a
    memory-efficient approach or the original method via a flag.

    Parameters
    ----------
    kernel_mul : float, optional
        The multiplier for generating a geometric progression of bandwidths.
    kernel_num : int, optional
        The number of kernels to use.
    fix_sigma : float | None, optional
        A fixed value for the base bandwidth.
    unbiased : bool, optional
        If True (default), computes the unbiased MMD estimator.
    use_memory_efficient : bool, optional
        If True (default), uses a memory-efficient method.
    output_transform : callable, optional
        A function to transform the engine's output.
    device : str | torch.device, optional
        The device on which to store the metric's state.
    """

    _state_dict_all_req_keys = ("_sum_mmd", "_num_examples")

    def __init__(
        self,
        #? --- Kernel Configuration ---
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: float | None = None,
        #? --- Estimator Configuration ---
        unbiased: bool = True,
        use_memory_efficient: bool = True,
        #? --- Ignite Configuration ---
        output_transform: t.Callable = lambda x: x,
        device: str | torch.device = "cpu",
    ):
        super().__init__(output_transform, device=device)
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.unbiased = unbiased
        self.use_memory_efficient = use_memory_efficient

    def _gaussian_kernel(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the Gaussian kernel matrix with a selectable memory approach.
        """
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        if self.use_memory_efficient:
            #? --- Memory-efficient pairwise L2 distance calculation ---
            gram_matrix = torch.mm(total, total.t())
            sq_norms = torch.diag(gram_matrix)
            l2_distance = sq_norms.unsqueeze(1) - 2.0 * gram_matrix + sq_norms.unsqueeze(0)
            l2_distance = torch.clamp(l2_distance, min=0)
        else:
            #? --- Original, memory-intensive pairwise L2 distance calculation ---
            total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
            total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
            l2_distance = ((total0 - total1) ** 2).sum(2)


        #? --- Bandwidth estimation ---
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            with torch.no_grad():
                bandwidth = torch.sum(l2_distance) / (n_samples**2 - n_samples)

        #? --- Multi-kernel bandwidth calculation ---
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [
            bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)
        ]

        #? Compute the final kernel value
        kernel_val = [torch.exp(-l2_distance / (bw + 1e-8)) for bw in bandwidth_list]
        return sum(kernel_val)

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_mmd = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: t.Sequence[torch.Tensor]) -> None:
        """
        Updates the metric's state with a new batch of data.
        """
        source, target = output[0].detach(), output[1].detach()
        batch_size = source.size(0)

        kernels = self._gaussian_kernel(source, target)
        xx = kernels[:batch_size, :batch_size]
        yy = kernels[batch_size:, batch_size:]
        xy = kernels[:batch_size, batch_size:]

        if not self.unbiased:
            #? --- Biased MMD estimator ---
            loss = torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)
        else:
            #? --- Unbiased MMD estimator ---
            if batch_size <= 1:
                loss = torch.tensor(0.0, device=source.device)
            else:
                sum_xx = torch.sum(xx) - torch.trace(xx)
                sum_yy = torch.sum(yy) - torch.trace(yy)
                sum_xy = torch.sum(xy)

                denom_xx_yy = float(batch_size * (batch_size - 1))
                denom_xy = float(batch_size * batch_size)

                loss = (sum_xx / denom_xx_yy) + (sum_yy / denom_xx_yy) - (2 * sum_xy / denom_xy)

        self._sum_mmd += loss.to(self._device) * batch_size
        self._num_examples += batch_size

    @sync_all_reduce("_sum_mmd", "_num_examples")
    def compute(self) -> float:
        """
        Computes the final MMD score.
        """
        if self._num_examples == 0:
            raise NotComputableError(
                "Metric must have at least one example before it can be computed."
            )
        return (self._sum_mmd / self._num_examples).item()