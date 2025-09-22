# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import typing as t
from dataclasses import dataclass
from types import MappingProxyType

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import numpy as np
import torch
import lightning as L

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class ScheduleConfig:
    """Configuration for a single hyperparameter schedule."""
    warmup_epochs: int = 0
    ramp_epochs: int = 1
    start_val: float = 0.0
    max_val: float = 1.0

# =============================================================================
# PRE-DEFINED CONSTANTS FOR EXPERIMENTS
# =============================================================================
#? --- Configuration for DiscrepancyVAE ---
DISC_NUM_EPOCHS = 100
DISC_MX_ALPHA = 10.0
DISC_MX_BETA = 2.0
DISC_LMBDA = 1e-3
DISC_MX_TEMP = 5.0

DISC_SCHEDULER_KWARGS = {
    "alpha": DISC_MX_ALPHA,
    "beta": DISC_MX_BETA,
    "graph_lambda": DISC_LMBDA,
    "temp": DISC_MX_TEMP,
}

#? --- Configuration for SENA ---
SENA_NUM_EPOCHS = 100
SENA_MX_ALPHA = 1.0
SENA_MX_BETA = 1.0
SENA_LMBDA = 0.1
SENA_MX_TEMP = 100.0

SENA_SCHEDULER_KWARGS = {
    "alpha": SENA_MX_ALPHA,
    "beta": SENA_MX_BETA,
    "graph_lambda": SENA_LMBDA,
    "temp": SENA_MX_TEMP,
}


# =============================================================================
# DEFAULT RAMPED SCHEDULE CONFIGURATIONS
# =============================================================================
def get_default_alpha_config(num_epochs: int, max_val: float = 1.0) -> ScheduleConfig:
    """
    Alpha (MMD weight): 5 epochs warmup at 0, ramps to max_val until halfway, then saturates.
    """
    return ScheduleConfig(
        warmup_epochs=5,
        ramp_epochs=max(1, int(num_epochs / 2) - 5),
        start_val=0.0,
        max_val=max_val,
    )

def get_default_beta_config(num_epochs: int, max_val: float = 1.0) -> ScheduleConfig:
    """
    Beta (KLD weight): 10 epochs warmup at 0, then ramps to max_val.
    """
    return ScheduleConfig(
        warmup_epochs=10,
        ramp_epochs=max(1, num_epochs - 10),
        start_val=0.0,
        max_val=max_val,
    )

def get_default_temp_config(num_epochs: int, max_val: float = 100.0) -> ScheduleConfig:
    """
    Temp (Softmax temp): 5 epochs warmup at 1.0, then ramps to max_val.
    """
    return ScheduleConfig(
        warmup_epochs=5,
        ramp_epochs=max(1, num_epochs - 5),
        start_val=1.0,
        max_val=max_val,
    )

def get_default_lmbda_config(num_epochs: int, val: float = 1.0) -> ScheduleConfig:
    """
    Lambda (L1 weight): Stays constant for all epochs.
    """
    return ScheduleConfig(
        warmup_epochs=num_epochs, # Stays at start_val for all epochs
        ramp_epochs=0,
        start_val=val,
        max_val=val,
    )

# =============================================================================
# PRE-DEFINED SCHEDULES FOR DISCREPANCY VAE
# =============================================================================
def get_default_discrepancy_vae_alpha_config(
    num_epochs: int = DISC_NUM_EPOCHS, val: float = DISC_MX_ALPHA
) -> ScheduleConfig:
    """Returns a ramped alpha schedule for the DiscrepancyVAE experiment."""
    return get_default_alpha_config(num_epochs, val)

def get_default_discrepancy_vae_beta_config(
    num_epochs: int = DISC_NUM_EPOCHS, val: float = DISC_MX_BETA
) -> ScheduleConfig:
    """Returns a ramped beta schedule for the DiscrepancyVAE experiment."""
    return get_default_beta_config(num_epochs, val)

def get_default_discrepancy_vae_temp_config(
    num_epochs: int = DISC_NUM_EPOCHS, val: float = DISC_MX_TEMP
) -> ScheduleConfig:
    """Returns a ramped temperature schedule for the DiscrepancyVAE experiment."""
    return get_default_temp_config(num_epochs, val)

def get_default_discrepancy_vae_lmbda_config(
    num_epochs: int = DISC_NUM_EPOCHS, val: float = DISC_LMBDA
) -> ScheduleConfig:
    """Returns a constant lambda schedule for the DiscrepancyVAE experiment."""
    return get_default_lmbda_config(num_epochs, val)

# =============================================================================
# PRE-DEFINED SCHEDULES FOR SENA
# =============================================================================
def get_default_sena_alpha_config(
    num_epochs: int = SENA_NUM_EPOCHS, val: float = SENA_MX_ALPHA
) -> ScheduleConfig:
    """Returns a ramped alpha schedule for the SENA experiment."""
    return get_default_alpha_config(num_epochs, val)

def get_default_sena_beta_config(
    num_epochs: int = SENA_NUM_EPOCHS, val: float = SENA_MX_BETA
) -> ScheduleConfig:
    """Returns a ramped beta schedule for the SENA experiment."""
    return get_default_beta_config(num_epochs, val)

def get_default_sena_temp_config(
    num_epochs: int = SENA_NUM_EPOCHS, val: float = SENA_MX_TEMP
) -> ScheduleConfig:
    """Returns a ramped temperature schedule for the SENA experiment."""
    return get_default_temp_config(num_epochs, val)

def get_default_sena_lmbda_config(
    num_epochs: int = SENA_NUM_EPOCHS, val: float = SENA_LMBDA
) -> ScheduleConfig:
    """Returns a constant lambda schedule for the SENA experiment."""
    return get_default_lmbda_config(num_epochs, val)


# =============================================================================
# PYTORCH LIGHTNING SCHEDULER
# =============================================================================
class CausalRepresentationLearningVAEHyperparameterScheduler(L.Callback):
    """
    A PyTorch Lightning Callback to schedule hyperparameters.
    
    This callback pre-computes the schedules for all hyperparameters upon 
    initialization for efficiency. It supports warmup, ramp-up, and saturation phases.
    """
    def __init__(self, 
        #? --- Scheduling Configurations ---
        num_epochs: int, 
        arch: str | None = 'discrepancy_vae',
        alpha_config: ScheduleConfig | None = None, 
        beta_config: ScheduleConfig | None = None, 
        lmbda_config: ScheduleConfig | None = None, 
        temp_config: ScheduleConfig | None = None,
    ):
        super().__init__()
        self.num_epochs = num_epochs

        if arch is None:
            #? Pre-compute schedules for all hyperparameters
            assert alpha_config is not None
            assert beta_config is not None
            assert lmbda_config is not None
            assert temp_config is not None
            
            self.alpha_schedule = self._precompute_schedule(alpha_config)
            self.beta_schedule = self._precompute_schedule(beta_config)
            self.lmbda_schedule = self._precompute_schedule(lmbda_config)
            self.temp_schedule = self._precompute_schedule(temp_config)
            
        elif arch == 'discrepancy_vae':
            self.alpha_schedule = self._precompute_schedule(
                get_default_discrepancy_vae_alpha_config(num_epochs)
            )
            self.beta_schedule = self._precompute_schedule(
                get_default_discrepancy_vae_beta_config(num_epochs)
            )
            self.lmbda_schedule = self._precompute_schedule(
                get_default_discrepancy_vae_lmbda_config(num_epochs)
            )
            self.temp_schedule = self._precompute_schedule(
                get_default_discrepancy_vae_temp_config(num_epochs)
            )
        elif arch == 'sena':
            self.alpha_schedule = self._precompute_schedule(
                get_default_sena_alpha_config(num_epochs)
            )
            self.beta_schedule = self._precompute_schedule(
                get_default_sena_beta_config(num_epochs)
            )
            self.lmbda_schedule = self._precompute_schedule(
                get_default_sena_lmbda_config(num_epochs)
            )
            self.temp_schedule = self._precompute_schedule(
                get_default_sena_temp_config(num_epochs)
            )
        else:
            raise ValueError("Invalid arch!")


    def _precompute_schedule(self, config: ScheduleConfig) -> np.ndarray:
        """
        Pre-computes a full hyperparameter schedule using NumPy.

        Args:
            config: The schedule configuration for a single hyperparameter.

        Returns:
            A NumPy array of shape (num_epochs,) with the scheduled values.
        """
        warmup_vals = np.full(config.warmup_epochs, config.start_val)
        
        # Handle ramp_epochs > 0 to avoid issues with np.linspace
        if config.ramp_epochs > 0:
            ramp_vals = np.linspace(config.start_val, config.max_val, config.ramp_epochs)
        else:
            ramp_vals = np.array([])
            
        saturation_epochs = self.num_epochs - config.warmup_epochs - config.ramp_epochs
        saturation_vals = np.full(max(0, saturation_epochs), config.max_val)
        
        # Concatenate all parts and ensure it has the correct total length
        full_schedule = np.concatenate([warmup_vals, ramp_vals, saturation_vals])
        return full_schedule[:self.num_epochs]

    def get_hparams_for_epoch(self, epoch: int) -> t.Dict[str, float]:
        """
        Retrieves all pre-computed hyperparameter values for a given epoch.
        This is an O(1) lookup operation.
        """
        if not (0 <= epoch < self.num_epochs):
            raise IndexError(f"Epoch {epoch} is out of the valid range [0, {self.num_epochs-1}]")
            
        return {
            "alpha": self.alpha_schedule[epoch],
            "beta": self.beta_schedule[epoch],
            "lmbda": self.lmbda_schedule[epoch],
            "temp": self.temp_schedule[epoch],
        }

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """
        Automatically called by PyTorch Lightning at the start of each training epoch.
        Sets the hyperparameter values on the model and logs them.
        """
        epoch = trainer.current_epoch
        hparams = self.get_hparams_for_epoch(epoch)
        
        # Update the lightning module with the new hyperparameter values
        pl_module.alpha = hparams["alpha"]
        pl_module.beta = hparams["beta"]
        pl_module.lmbda = hparams["lmbda"]
        pl_module.temp = hparams["temp"]
        
        # pl_module.log_dict(hparams, on_step=False, on_epoch=True, prog_bar=False)

# =============================================================================
# SCRIPT EXECUTION FOR VERIFICATION
# =============================================================================
if __name__ == "__main__":
    print("--- Verifying Scheduler Logic vs. DiscrepancyVAE Snippet Logic ---")

    scheduler_disc = CausalRepresentationLearningVAEHyperparameterScheduler(
        num_epochs=DISC_NUM_EPOCHS,
        alpha_config=get_default_discrepancy_vae_alpha_config(),
        beta_config=get_default_discrepancy_vae_beta_config(),
        lmbda_config=get_default_discrepancy_vae_lmbda_config(),
        temp_config=get_default_discrepancy_vae_temp_config()
    )

    beta_snippet_disc = torch.zeros(DISC_NUM_EPOCHS)
    beta_snippet_disc[10:] = torch.linspace(0, DISC_MX_BETA, DISC_NUM_EPOCHS - 10)
    alpha_snippet_disc = torch.zeros(DISC_NUM_EPOCHS)
    alpha_snippet_disc[5:int(DISC_NUM_EPOCHS/2)] = torch.linspace(0, DISC_MX_ALPHA, int(DISC_NUM_EPOCHS/2) - 5)
    alpha_snippet_disc[int(DISC_NUM_EPOCHS/2):] = DISC_MX_ALPHA

    all_match_disc = True
    for epoch in range(DISC_NUM_EPOCHS):
        hparams_s = scheduler_disc.get_hparams_for_epoch(epoch)
        if not (np.isclose(hparams_s["alpha"], alpha_snippet_disc[epoch].item()) and 
                np.isclose(hparams_s["beta"], beta_snippet_disc[epoch].item())):
            all_match_disc = False
            break
    
    print("\n--- Verification Summary (DiscrepancyVAE) ---")
    if all_match_disc:
        print("All hyperparameter schedules match the DiscrepancyVAE snippet's logic.")
    else:
        print("Mismatch found for DiscrepancyVAE! Please review the logic.")

    print("\n--- Verifying Scheduler Logic vs. SENA Snippet Logic ---")

    scheduler_sena = CausalRepresentationLearningVAEHyperparameterScheduler(
        num_epochs=SENA_NUM_EPOCHS,
        alpha_config=get_default_sena_alpha_config(),
        beta_config=get_default_sena_beta_config(),
        lmbda_config=get_default_sena_lmbda_config(),
        temp_config=get_default_sena_temp_config()
    )

    beta_snippet_sena = torch.cat([torch.zeros(10), torch.linspace(0, SENA_MX_BETA, SENA_NUM_EPOCHS - 10)])
    alpha_snippet_sena = torch.cat([
        torch.zeros(5),
        torch.linspace(0, SENA_MX_ALPHA, int(SENA_NUM_EPOCHS / 2) - 5),
        torch.full((SENA_NUM_EPOCHS - int(SENA_NUM_EPOCHS / 2),), SENA_MX_ALPHA),
    ])
    temp_snippet_sena = torch.cat([torch.ones(5), torch.linspace(1, SENA_MX_TEMP, SENA_NUM_EPOCHS - 5)])
    
    all_match_sena = True
    for epoch in range(SENA_NUM_EPOCHS):
        hparams_s = scheduler_sena.get_hparams_for_epoch(epoch)
        if not (np.isclose(hparams_s["alpha"], alpha_snippet_sena[epoch].item()) and 
                np.isclose(hparams_s["beta"], beta_snippet_sena[epoch].item()) and
                np.isclose(hparams_s["temp"], temp_snippet_sena[epoch].item())):
            all_match_sena = False
            break

    print("\n--- Verification Summary (SENA) ---")
    if all_match_sena:
        print("All hyperparameter schedules match the SENA snippet's logic.")
    else:
        print("Mismatch found for SENA! Please review the logic.")
