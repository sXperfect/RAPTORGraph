from .graph_l1_loss import GraphL1Loss
from .gaussian_kld_loss import (
    GaussianKLDivergenceLoss,
    GaussianVAEKLDivergenceVAELoss, 
    SENA_GaussianVAEKLDivergenceLoss,
)
from .mmd_loss import (
    MultiKernelMaximumMeanDiscrepancy,
    SENA_MultikernelMaximumMeanDiscrepancy,
    DiscrepancyVAE_MultiKernelMaximumMeanDiscrepancy
)