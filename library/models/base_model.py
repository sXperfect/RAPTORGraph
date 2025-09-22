import typing as t
import lightning as L
from torch.optim import AdamW, SGD, RMSprop, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, StepLR
from ..utils import core as my_utils

# =================================================================================
# Base Model (as a separate file)
# =================================================================================

OPTIMIZER_REGISTRY = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
    "rmsprop": RMSprop
}

SCHEDULER_REGISTRY = {
    "cosine_warm_restarts": CosineAnnealingWarmRestarts,
    "linear": LambdaLR,
    "step": StepLR,
    None: None
}

class BaseModel(L.LightningModule):
    # The registries are no longer class attributes to reduce redundancy.
    # They are accessed directly from the module's scope in `configure_optimizers`.

    def __init__(
        self,
        arch_obj,
        learning_rate: float = 1e-5,
        optimizer_name: str = "adam",
        optimizer_kwargs: dict | None = None,
        scheduler_name: str | None = None,
        scheduler_kwargs: dict | None = None,
    ):
        super().__init__()
        # This saves all hyperparameters passed to __init__ (e.g., learning_rate)
        # into self.hparams, which is used by Lightning for checkpointing.
        self.save_hyperparameters(ignore=['arch_obj'])
        self.arch_obj = arch_obj

    def configure_optimizers(self) -> dict:
        """Sets up the optimizer and learning rate scheduler."""
        optimizer_cls = OPTIMIZER_REGISTRY[self.hparams.optimizer_name]
        
        # Use self.parameters() which includes all parameters of the derived class
        # (e.g., DiffusionModelV1) and the arch_obj.
        optimizer = optimizer_cls(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            **my_utils.ensure_dict(self.hparams.optimizer_kwargs)
        )

        if self.hparams.scheduler_name is None:
            return {"optimizer": optimizer}

        scheduler_cls = SCHEDULER_REGISTRY[self.hparams.scheduler_name]
        scheduler = scheduler_cls(
            optimizer,
            **my_utils.ensure_dict(self.hparams.scheduler_kwargs)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
