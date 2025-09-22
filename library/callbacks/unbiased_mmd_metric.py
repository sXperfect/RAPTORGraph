# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import typing as t
import warnings

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import torch
import lightning as L

# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
from ..losses.mmd_loss import (
    DISCREPANCY_VAE_MMD_SIGMA,
    DISCREPANCY_VAE_KERNEL_NUM,
    DISCREPANCY_VAE_KERNEL_MUL,
    SENA_MMD_SIGMA,
    SENA_KERNEL_NUM,
    SENA_KERNEL_MUL,
)
from ..losses.crl_loss import (
    MultiKernelMaximumMeanDiscrepancy, 
    MMDStrategy,
)

DISCREPANCY_VAE_MMD_KWARGS = {
    "mmd_sigma": DISCREPANCY_VAE_MMD_SIGMA,
    "mmd_kernel_num": DISCREPANCY_VAE_KERNEL_NUM,
    "mmd_kernel_mul": DISCREPANCY_VAE_KERNEL_MUL,
}

# --- Configuration Dictionary for SENA MMD ---
# Provides a separate configuration set for SENA models, which in this
# case defaults to the same values as the Discrepancy VAE.
SENA_MMD_KWARGS = {
    "mmd_sigma": SENA_MMD_SIGMA,
    "mmd_kernel_num": SENA_KERNEL_NUM,
    "mmd_kernel_mul": SENA_KERNEL_MUL,
}



# =============================================================================
# CALLBACK DEFINITION
# =============================================================================

class UnbiasedMMDMetricCallback(L.Callback):
    """
    A Lightning Callback to compute and log the unbiased Maximum Mean Discrepancy (MMD)
    metric during validation and/or testing phases.

    This callback is designed to work with models that produce interventional
    predictions (`y_hat`) and have corresponding ground truth (`Y`), such as the
    `CausalRepresentationLearningAE` model. It provides a reliable way to monitor
    the discrepancy between the predicted and true distributions for interventional
    data, separate from the main training loss.

    The callback supports multiple MMD calculation strategies (global, per-label, etc.)
    and flexible logging options for both step-wise and epoch-wise reporting.

    Attributes
    ----------
    on_validation : bool
        If True, compute the metric during the validation loop.
    on_test : bool
        If True, compute the metric during the test loop.
    log_on_step : bool
        If True, log the metric at each step.
    log_on_epoch : bool
        If True, log the metric at the end of each epoch.
    metric_name : str
        The base name for the logged metric (e.g., 'unbiased_mmd_cb').
    mmd_func : MultiKernelMaximumMeanDiscrepancy
        The instantiated MMD function used for metric calculation.
    """
    def __init__(
        self,
        #? --- Control when the callback runs ---
        on_train: bool = True,
        on_validation: bool = True,
        on_test: bool = True,
        #? --- Control logging frequency ---
        log_on_step: bool = False,
        log_on_epoch: bool = True,
        log_prog_bar: bool = True,
        #? --- MMD computation parameters ---
        mmd_config_preset: str | None = None,
        mmd_sigma: t.Optional[float] = None,
        mmd_kernel_num: int = 5,
        mmd_kernel_mul: float = 2.0,
        mmd_strategy: str | MMDStrategy = 'dynamic',
        min_sample_per_label: int = 2,
        #? --- Naming for the logged metric ---
        metric_name: str = "unbiased_mmd"
    ):
        """
        Initializes the UnbiasedMMDMetricCallback.

        Parameters
        ----------
        on_validation : bool, default True
            Enable metric computation during the training phase.
        on_validation : bool, default True
            Enable metric computation during the validation phase.
        on_test : bool, default True
            Enable metric computation during the testing phase.
        log_on_step : bool, default False
            Log the metric after each batch.
        log_on_epoch : bool, default True
            Log the metric at the end of an epoch.
        log_prog_bar : bool, default True
            Display the metric on the progress bar.
        mmd_sigma : float | None, optional
            The sigma value for the RBF kernel. If None, it is estimated from data.
        mmd_kernel_num : int, default 5
            The number of kernels to use in the multi-kernel MMD.
        mmd_kernel_mul : float, default 2.0
            The multiplier for the sigma values of the multiple kernels.
        mmd_strategy : str | MMDStrategy, default 'global'
            The strategy for computing MMD ('global', 'per_label', 'weighted', 'dynamic').
        min_sample_per_label : int, default 2
            The minimum number of samples required for a label group to compute
            MMD when using the 'per_label' strategy.
        metric_name : str, default "unbiased_mmd_metric_cb"
            The name used for logging the metric in the logger.
        """
        super().__init__()
        self.on_train = on_train
        self.on_validation = on_validation
        self.on_test = on_test
        self.log_on_step = log_on_step
        self.log_on_epoch = log_on_epoch
        self.log_prog_bar = log_prog_bar
        self.mmd_strategy = MMDStrategy(mmd_strategy)
        self.min_sample_per_label = min_sample_per_label
        self.metric_name = metric_name

        self.mmd_func = MultiKernelMaximumMeanDiscrepancy(
            fix_sigma=mmd_sigma,
            kernel_num=mmd_kernel_num,
            kernel_mul=mmd_kernel_mul,
            unbiased=True,
        )

    def _configure_mmd_params(
        self,
        preset: str | None,
        sigma: float | None,
        kernel_num: int,
        kernel_mul: float,
    ) -> dict:
        """Helper to resolve MMD parameters from a preset or direct arguments."""
        if preset:
            preset_lower = preset.lower()
            if preset_lower == "sena":
                source = SENA_MMD_KWARGS
            elif preset_lower == "dvae":
                source = DISCREPANCY_VAE_MMD_KWARGS
            else:
                raise ValueError(f"Invalid preset '{preset}'. Must be 'sena' or 'dvae'.")
            return {
                "fix_sigma": source["mmd_sigma"],
                "kernel_num": source["mmd_kernel_num"],
                "kernel_mul": source["mmd_kernel_mul"],
            }
        return {"fix_sigma": sigma, "kernel_num": kernel_num, "kernel_mul": kernel_mul}


    def _compute_mmd(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        label: t.Optional[torch.Tensor],
        strategy: MMDStrategy,
        mmd_func: t.Callable,
    ) -> torch.Tensor:
        """
        Internal helper to compute MMD based on the selected strategy.
        This implementation is adapted from CausalRepresentationLearningLoss._compute_mmd.
        """
        if label is None:
            # If no labels are provided, we can only use the global strategy.
            return mmd_func(y_hat, y)

        effective_strategy = strategy
        if strategy == MMDStrategy.DYNAMIC:
            # Switch between PER_LABEL and GLOBAL based on label diversity.
            if (label == label[0]).all():
                effective_strategy = MMDStrategy.GLOBAL
            else:
                effective_strategy = MMDStrategy.PER_LABEL

        if effective_strategy == MMDStrategy.GLOBAL:
            return mmd_func(y_hat, y)

        elif effective_strategy == MMDStrategy.WEIGHTED:
            # --- Inverse frequency weighting ---
            uniq, inv, counts = torch.unique(label, return_inverse=True, return_counts=True)
            weights = 1.0 / counts.float()
            weights = weights / weights.sum()  # Normalize weights
            sample_weights = weights[inv]
            return mmd_func(y_hat, y) * sample_weights.mean()

        elif effective_strategy == MMDStrategy.PER_LABEL:
            # --- Compute MMD for each label group and average ---
            unique_labels = torch.unique(label)
            losses = []
            for lbl in unique_labels:
                mask = label == lbl
                if mask.sum().item() >= self.min_sample_per_label:
                    idx = torch.where(mask)[0]
                    y_hat_grp = torch.index_select(y_hat, 0, idx)
                    y_grp = torch.index_select(y, 0, idx)
                    losses.append(mmd_func(y_hat_grp, y_grp))

            if not losses:
                # Return 0 if no group was large enough
                return torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
            return torch.stack(losses).mean()

        else:
            raise ValueError(f"Invalid MMD strategy: {strategy}")

    def _calculate_and_log(
        self,
        outputs: t.Dict[str, t.Any],
        batch: t.Dict[str, t.Any],
        pl_module: L.LightningModule,
        stage: str
    ):
        """
        Core logic to extract tensors, compute the MMD metric, and log it.
        """
        # The `*_step` methods in `crl_ae.py` return a dict containing "arch_outputs"
        arch_outputs = outputs.get("arch_outputs")
        if arch_outputs is None:
            warnings.warn(
                f"UnbiasedMMDMetricCallback: 'arch_outputs' not found in the step output "
                f"during the '{stage}' phase. Skipping MMD calculation for this batch.",
                UserWarning
            )
            return

        y_hat = getattr(arch_outputs, 'y_hat', None)
        y = batch["Y"]
        label = batch["label"]

        # Ensure we have the necessary tensors to compute the metric
        if y_hat is None or y is None:
            warnings.warn(
                f"UnbiasedMMDMetricCallback: `y_hat` (from arch_outputs) or `Y` (from batch) "
                f"is missing during the '{stage}' phase. Skipping MMD calculation.",
                UserWarning
            )
            return

        self.mmd_func.to(y_hat.device)
        mmd_value = self._compute_mmd(y_hat, y, label, self.mmd_strategy, self.mmd_func)

        log_name = f"{stage}/{self.metric_name}"
        pl_module.log(
            name=log_name,
            value=mmd_value,
            on_step=self.log_on_step,
            on_epoch=self.log_on_epoch,
            prog_bar=self.log_prog_bar,
            sync_dist=True
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: t.Dict[str, t.Any],
        batch: t.Dict[str, t.Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Hook to compute metric at the end of a validation batch."""
        if not self.on_train:
            return
        self._calculate_and_log(outputs, batch, pl_module, "train")
        
    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: t.Dict[str, t.Any],
        batch: t.Dict[str, t.Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Hook to compute metric at the end of a validation batch."""
        if not self.on_validation:
            return
        self._calculate_and_log(outputs, batch, pl_module, "val")

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: t.Dict[str, t.Any],
        batch: t.Dict[str, t.Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Hook to compute metric at the end of a test batch."""
        if not self.on_test:
            return
        self._calculate_and_log(outputs, batch, pl_module, "test")
