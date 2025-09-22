"""
Custom PyTorch Lightning Callbacks for scheduling loss parameters.
"""
# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import torch
import lightning as L
import typing as t

# =============================================================================
# LOCAL IMPORTS
# =============================================================================
from ..losses.crl_loss import CausalRepresentationLearningLoss

# =============================================================================
# SCHEDULER IMPLEMENTATION
# =============================================================================
class DAGMAScheduler(L.Callback):
    """
    A PyTorch Lightning Callback to schedule the `mu` parameter of the
    CausalRepresentationLearningLoss based on the augmented Lagrangian method
    for DAGMA.
    
    This scheduler monitors the `dagma_h_value` (acyclicity constraint) at the
    end of each training batch. If the value exceeds a certain threshold, it
    increases `mu` to place a higher penalty on cycles in the next iteration.
    """
    def __init__(
        self,
        mu_init: float = 1.0,
        mu_update_factor: float = 10.0,
        h_threshold: float = 1e-8,
        mu_max: float = 1e+16,
    ):
        """
        Initializes the DAGMAScheduler callback.

        Parameters
        ----------
        mu_init : float, optional
            The initial value for the mu parameter. Defaults to 1.0.
        mu_update_factor : float, optional
            The factor by which to multiply mu when the acyclicity constraint
            is violated. Defaults to 10.0.
        h_threshold : float, optional
            The threshold for the acyclicity constraint's h-value. If h exceeds
            this, mu is increased. Defaults to 1e-8.
        mu_max : float, optional
            The maximum value for the mu parameter. Defaults to 1e+16.
        """
        super().__init__()
        self.mu_init = mu_init
        self.mu_update_factor = mu_update_factor
        self.h_threshold = h_threshold
        self.mu_max = mu_max

    def on_train_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        """
        Hook called at the beginning of training to set the initial mu value.
        """
        if hasattr(pl_module, 'loss_fn') and isinstance(pl_module.loss_fn, CausalRepresentationLearningLoss):
            if pl_module.loss_fn.graph_loss_type == 'dagma':
                # Initialize mu in the loss function to the scheduler's starting value
                pl_module.loss_fn.dagma_mu = self.mu_init
                print(f"DAGMAScheduler: Initialized 'dagma_mu' to {self.mu_init}")

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor,
        batch: t.Any,
        batch_idx: int,
    ) -> None:
        """
        Hook called after each training step. Updates `mu` based on `h`.
        """
        # Ensure the pl_module has the loss function we need to modify
        if not hasattr(pl_module, 'loss_fn') or not isinstance(pl_module.loss_fn, CausalRepresentationLearningLoss):
            return

        loss_fn = pl_module.loss_fn
        
        #? We only care about DAGMA loss type
        if loss_fn.graph_loss_type != 'dagma':
            return
            
        #? Get the h_value from the logged metrics
        h_value = trainer.callback_metrics.get("dagma_h_value")

        if h_value is not None:
            with torch.no_grad():
                # Use the scheduler's own h_threshold for the check
                if h_value.item() > self.h_threshold:
                    # Use scheduler's parameters to calculate the new mu
                    current_mu = loss_fn.dagma_mu
                    new_mu = min(current_mu * self.mu_update_factor, self.mu_max)
                    # Update the mu parameter directly in the loss function instance
                    if new_mu > current_mu:
                        loss_fn.dagma_mu = new_mu
                        print(f"\nDAGMAScheduler: h_value ({h_value.item():.4f}) > threshold. Updating mu from {current_mu} to {new_mu}.")

