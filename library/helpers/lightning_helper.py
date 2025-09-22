"""
Provides helper functions and constants for PyTorch Lightning integration.

This module includes utilities for managing common PyTorch Lightning warnings
and defines standardized status enums for logging purposes.
"""
# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import enum
import warnings
import contextlib

# =============================================================================
# CONFIGURATION ENUMS
# =============================================================================
class FinalizeStatus(enum.StrEnum):
    """
    Enumeration for the final status of a run or trial.
    """
    SUCCESS = "success"
    FAILED = "failed"
    FINISHED = "finished"

    @classmethod
    def _missing_(cls, value):
        """Handles case-insensitive string conversion."""
        if not isinstance(value, str):
            return super()._missing_(value)

        value_lower = value.lower()
        for member in cls:
            if member.value == value_lower:
                return member
        
        valid_options = ", ".join(m.value for m in cls)
        raise ValueError(
            f"'{value}' is not a valid {cls.__name__}. "
            f"Please use one of: {valid_options}"
        )

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def ignore_pl_warnings(
    dataloader_num_workers: bool = True,
    slurm_srun: bool = True,
    mixed_precision: bool = True,
):
    """
    Suppresses common, often noisy, warnings from PyTorch Lightning globally.

    Parameters
    ----------
    dataloader_num_workers : bool, optional
        If True, suppresses the warning about using a small number of workers
        in the DataLoader. Defaults to True.
    slurm_srun : bool, optional
        If True, suppresses the warning about the `srun` command being
        available on the system. Defaults to True.
    mixed_precision : bool, optional
        If True, suppresses the historical usage warning for 16-bit mixed
        precision. Defaults to True.
    """
    if dataloader_num_workers:
        warnings.filterwarnings("ignore", ".*train_dataloader, does not have many workers.*")
    if slurm_srun:
        warnings.filterwarnings("ignore", ".*The `srun` command is available on your system.*")
    if mixed_precision:
        warnings.filterwarnings("ignore", ".*16 is supported for historical reasons but its usage is discouraged.*")


@contextlib.contextmanager
def suppress_pl_warnings(
    dataloader_num_workers: bool = True,
    slurm_srun: bool = True,
    mixed_precision: bool = True,
):
    """
    A context manager to temporarily suppress common PyTorch Lightning warnings.

    Parameters
    ----------
    dataloader_num_workers : bool, optional
        If True, suppresses the warning about using a small number of workers
        in the DataLoader. Defaults to True.
    slurm_srun : bool, optional
        If True, suppresses the warning about the `srun` command being
        available on the system. Defaults to True.
    mixed_precision : bool, optional
        If True, suppresses the historical usage warning for 16-bit mixed
        precision. Defaults to True.
    """
    with warnings.catch_warnings():
        #? Reuse the logic from the global ignore function.
        ignore_pl_warnings(
            dataloader_num_workers=dataloader_num_workers,
            slurm_srun=slurm_srun,
            mixed_precision=mixed_precision,
        )
        yield
