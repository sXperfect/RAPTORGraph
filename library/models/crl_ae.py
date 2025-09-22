import typing as t
from dataclasses import dataclass

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import torch
import torch.nn as nn
import lightning as L
import numpy as np

# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
from .base_model import BaseModel
from ..structs import CRLForwardPassOutput
from ..losses.crl_loss import (
    CausalRepresentationLearningLoss,
    MMDVersion,
    MMDStrategy,
    MultiKernelMaximumMeanDiscrepancy,
    GraphLossType
)

# =============================================================================
# CONFIGURATION ENUMS AND CONSTANTS
# =============================================================================
#? VAE Configurations
CMVAE_MODEL_CONFIG = {
    "is_variational": True, "use_causal_layer": True, "use_mmd_loss": True, "gamma": 1.0,
}
CVAE_MODEL_CONFIG = {
    "is_variational": True, "use_causal_layer": True, "use_mmd_loss": False, "gamma": 1.0,
}
MVAE_MODEL_CONFIG = {
    "is_variational": True, "use_causal_layer": False, "use_mmd_loss": True, "gamma": 0.0,
}

#? AE Configurations
CMAE_MODEL_CONFIG = {
    "is_variational": False, "use_causal_layer": True, "use_mmd_loss": True, "gamma": 1.0,
}
MAE_MODEL_CONFIG = {
    "is_variational": False, "use_causal_layer": False, "use_mmd_loss": True, "gamma": 0.0,
}


class CausalRepresentationLearningAE(BaseModel):
    """
    Causal and Discrepancy-based AE/VAE implemented as a LightningModule.

    This model can operate in two modes:
    1. Variational Autoencoder (VAE) mode (`is_variational=True`): Learns a
       probabilistic latent space, using a KL divergence penalty.
    2. Standard Autoencoder (AE) mode (`is_variational=False`): Learns a
       deterministic latent representation, disabling the KL divergence loss.

    Two optional, memory-friendly MMD strategies are provided:

    1. Weighted-MMD (Option 1) - compute a single MMD on the whole batch and
       weight it by the inverse size of each label group. This needs only one
       O(N²) kernel matrix regardless of how many labels you have.

    2. Memory-efficient per-label MMD (Option 2) - keep the per-label semantics
       but avoid copying tensors for every label. It still builds a kernel per
       label, but intermediate tensors are freed immediately, dramatically
       reducing peak memory.

    Notes
    -----
    The class inherits from :class:`BaseModel` which provides common Lightning
    functionality such as optimizer creation.  Hyperparameters are saved with
    ``self.save_hyperparameters`` (the heavy ``arch_obj`` is ignored).  The loss
    function is instantiated from :class:`DiscrepancyVAELossFunction`.
    """
    def __init__(
        self,
        #? --- Model Architecture ---
        arch_obj: nn.Module,
        #? --- Optimizer Configuration ---
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        optimizer_kwargs: t.Dict | None = None,
        #? --- Scheduler Configurations ---
        scheduler_name: str | None = None,
        scheduler_kwargs: t.Dict | None = None,
        #? --- Loss Term Coefficients (can be scheduled) ---
        alpha: float = 1.0,
        beta: float = 1.0,
        graph_lambda: float = 0.1,
        triplet_weight: float = 0.1,
        temp: float = 1.0,
        #? --- DAGMA-Specific Hyperparameters ---
        dagma_mu: float = 1.0,
        dagma_lambda1: float = 0.025,
        dagma_score_type: str = "l2",
        #? --- MMD Loss Configuration ---
        ena_mmd_loss: bool = True,
        mmd_version: str | MMDVersion = 'discrepancy_v1',
        mmd_strategy: str | MMDStrategy = 'global',
        mmd_sigma: float | None = None,
        mmd_kernel_num: int = 5,
        mmd_kernel_mul: float = 2.0,
        mmd_min_sample_per_label: int = 2,
        ena_unbiased_mmd_metric: bool = False,
        #? --- Triplet Loss Configuration ---
        ena_triplet_loss: bool = False,
        triplet_margin: float = 0.2,
        triplet_distance_metric: str = "l2",
        triplet_miner_name: str | None = "triplet_margin",
        triplet_miner_margin: float = 0.2,
        #? --- General Loss Behavior Flags ---
        deterministic_intervention: bool = False,
        #? --- Additional MMD ---
        ena_test_mmd: bool = False,
        ena_label_as_test_prefix: bool = False
    ):
        """
        Initializes the CausalRepresentationLearningAE model.

        Parameters
        ----------
        arch_obj : nn.Module
            The neural network architecture (must be a ``torch.nn.Module``).
        learning_rate : float, default 1e-3
            Learning rate for the optimizer.
        optimizer_name : str, default "adam"
            Name of the optimizer to use (e.g., ``"adam"``).
        optimizer_kwargs : Dict | None, optional
            Additional keyword arguments for the optimizer.
        scheduler_name : str | None, optional
            Name of the learning-rate scheduler.
        scheduler_kwargs : Dict | None, optional
            Additional keyword arguments for the scheduler.
        alpha : float, default 1.0
            Weight for the Maximum Mean Discrepancy (MMD) loss.
        beta : float, default 1.0
            Weight for the Kullback-Leibler (KL) divergence (VAE mode only).
        graph_lambda : float, default 0.1
            Weight for the L1 regularisation of the causal graph.
        temp : float, default 1.0
            Temperature for Gumbel-Softmax distribution (VAE mode only).
        mmd_version : str, default "discrepancy_v1"
            Version identifier for the MMD loss implementation.
        MMD_sigma : float, default 200.0
            Sigma parameter for the RBF kernel used in MMD.
        kernel_num : int, default 10
            Number of kernels for the MMD computation.
        matched_IO : bool, default False
            Flag indicating whether input and output should be matched for MMD.
        compute_unbiased_mmd : bool, optional
            If True, compute the unbiased MMD metric.
            Defaults to False.
        compute_mmd_per_label : bool, default True
            * ``True`` - compute MMD per label (default).
            * ``False`` - never compute per-label MMD.
        use_weighted_mmd : bool, default False
            Whether to use the weighted-MMD strategy (Option 1).
        use_mem_efficient_per_label_mmd : bool, default False
            Whether to use the memory-efficient per-label MMD strategy (Option 2).
        min_sample_per_label : int, default 2
            Minimum number of samples required in a label group to compute MMD.

        Returns
        -------
        None
            An instance of ``CausalRepresentationLearningAE`` is created.

        Notes
        -----
        A sanity check ensures that ``use_weighted_mmd`` and
        ``use_mem_efficient_per_label_mmd`` are not both ``True``. The heavy
        ``arch_obj`` is excluded from the saved hyperparameters to keep the
        checkpoint size small.

        Examples
        --------
        """
        super().__init__(
            arch_obj=arch_obj,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_name=scheduler_name,
            scheduler_kwargs=scheduler_kwargs,
        )

        self.save_hyperparameters(ignore=["arch_obj", "knn_metric"])

        self.loss_f = CausalRepresentationLearningLoss(
            #? --- Model & Data Configuration ---
            latent_dim=self.latent_dim, #? Get from arch_obj
            #? --- General Behavior Flags ---
            is_variational=self.is_variational,  #? Get from arch_obj
            is_causal=self.is_causal, #? Get from arch_obj
            deterministic_intervention=self.hparams.deterministic_intervention,
            ret_dict=True,
            #? --- Graph Loss Configuration ---
            graph_loss_type=self.graph_loss_type,
            #? --- DAGMA-Specific Hyperparameters ---
            dagma_mu=self.hparams.dagma_mu,
            dagma_lambda1=self.hparams.dagma_lambda1,
            dagma_score_type=self.hparams.dagma_score_type,
            #? --- MMD Loss Configuration ---
            ena_mmd_loss=self.hparams.ena_mmd_loss,
            mmd_version=self.hparams.mmd_version,
            mmd_strategy=self.hparams.mmd_strategy,
            mmd_sigma=self.hparams.mmd_sigma,
            mmd_kernel_num=self.hparams.mmd_kernel_num,
            mmd_kernel_mul=self.hparams.mmd_kernel_mul,
            mmd_min_sample_per_label=self.hparams.mmd_min_sample_per_label,
            ena_unbiased_mmd_metric=self.hparams.ena_unbiased_mmd_metric,
            #? --- Triplet Loss Configuration ---
            ena_triplet_loss=self.hparams.ena_triplet_loss,
            triplet_margin=self.hparams.triplet_margin,
            triplet_distance_metric=self.hparams.triplet_distance_metric,
            triplet_miner_name=self.hparams.triplet_miner_name,
            triplet_miner_margin=self.hparams.triplet_miner_margin,
        )
        #? These are stored directly as they can be modified during training
        #? by callbacks (e.g., a hyperparameter scheduler).
        self.alpha = alpha
        self.beta = beta
        self.graph_lambda = graph_lambda
        self.temp = temp
        self.triplet_weight = triplet_weight

        if self.hparams.ena_test_mmd:
            self.unbiased_mmd_metric_f = MultiKernelMaximumMeanDiscrepancy(
                fix_sigma=mmd_sigma,
                kernel_num=mmd_kernel_num,
                kernel_mul=mmd_kernel_mul,
                unbiased=True,
            )

    def configure_optimizers(self) -> t.Dict:
        """
        Configures the optimizer and an optional learning-rate scheduler.

        Returns
        -------
        Dict
            A dictionary for the PyTorch Lightning trainer.

        Notes
        -----
        Delegates to the ``BaseModel`` implementation.

        Examples
        --------
        """
        #? The base model handles the creation of optimizer and scheduler
        return super().configure_optimizers()

    def forward(
        self,
        x: torch.Tensor,
        c1: torch.Tensor | None,
        c2: torch.Tensor | None,
        num_interv: int | None,
    ) -> dict:
        """
        Performs the forward pass through the AE/VAE architecture.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.
        c1 : torch.Tensor
            First conditioning variable.
        c2 : torch.Tensor
            Second conditioning variable.
        num_interv : int
            Number of interventions to apply.

        Returns
        -------
        dict
            Dictionary containing the architecture outputs.

        Notes
        -----
        The call is delegated to the encapsulated ``arch_obj``. Key parameters
        like ``is_variational`` and ``temp`` are passed explicitly.

        Examples
        --------
        """
        #? The forward call is delegated to the encapsulated architecture.
        #? It is expected to return a dictionary of tensors.
        if self.is_causal:
            return self.arch_obj(
                x,
                c1,
                c2,
                num_interv=num_interv,
                temp=self.temp,
                return_dict=True,
            )
        else:
            return self.arch_obj(
                x,
            )

    @property
    def is_causal(self) -> bool:
        return getattr(self.arch_obj, "is_causal", False)

    @property
    def is_variational(self) -> bool:
        return getattr(self.arch_obj, "is_variational", False)

    @property
    def latent_dim(self) -> int:
        if hasattr(self.arch_obj, "num_pathways"):
            return self.arch_obj.num_pathways

        raise ValueError("There is no latent dim!")

    @property
    def is_dagma(self) -> bool:
        return getattr(self.arch_obj, "is_dagma", False)

    @property
    def graph_loss_type(self):
        if self.is_causal:
            if self.is_dagma:
                return GraphLossType.DAGMA
            else:
                return GraphLossType.L1
        else:
            return None

    def _calculate_loss(
        self,
        arch_outputs: dict,
        batch: dict,
    ) -> t.Tuple[torch.Tensor, dict]:
        """
        Calculates the total loss and individual loss components in a robust manner.

        Parameters
        ----------
        arch_outputs : dict
            Outputs from the model's forward pass.
        batch : dict
            A dictionary containing the data batch, expected to have keys like
            "X", "Y", and optionally "label".

        Returns
        -------
        Tuple[torch.Tensor, dict]
            - Total loss as a `torch.Tensor`.
            - Dictionary of individual loss components for logging.

        Notes
        -----
        This function safely handles cases where optional loss components (like KL,
        L1, or MMD) are not computed by the loss function, preventing KeyErrors.

        Examples
        --------
        >>> # Assuming self.loss_f is properly initialized
        >>> # total_loss, loss_log_dict = self._calculate_loss(outputs, batch)
        """
        #? --- Call the main loss function ---
        #? This returns a dictionary where keys are only present if the
        #? corresponding loss was actually computed.
        loss_components = self.loss_f.compute_loss(
            outputs=arch_outputs,
            x=batch["X"],
            y=batch.get("Y"),
            label=batch.get("label"),
        )

        #? --- Safely retrieve each loss component, defaulting to 0.0 if not present ---
        #? This prevents KeyError if a loss (e.g., kl_loss) was not computed.
        recon_loss = loss_components.get("recon_loss", 0.0)
        mmd_loss = loss_components.get("mmd_loss", 0.0)
        kl_loss = loss_components.get("kl_loss", 0.0)
        triplet_loss = loss_components.get("triplet_loss", 0.0)

        graph_loss = loss_components.get("graph_loss", 0.0)

        #? --- Combine the components using the current hyper-parameters ---
        #? This calculation is now safe even if some components are zero.
        total_loss = (
            recon_loss
            + self.alpha * mmd_loss
            + self.beta * kl_loss
            + self.graph_lambda * graph_loss
            + self.triplet_weight + triplet_loss
        )

        #? --- Prepare the dictionary for logging ---
        #? It contains all computed losses and the final total.
        loss_log_dict = loss_components
        loss_log_dict["total_loss"] = total_loss

        return total_loss, loss_log_dict

    def _shared_step(
        self,
        batch: t.Tuple | t.Dict,
        batch_idx: int,
        stage: str,
    ) -> t.Tuple[torch.Tensor, dict, CRLForwardPassOutput]:
        """
        Common logic for train/val/test steps.

        Parameters
        ----------
        batch : Tuple | Dict
            Batch of data from the dataloader.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Tuple[torch.Tensor, dict]
            Total loss tensor and a dictionary of logged metrics.

        Examples
        --------
        """
        if isinstance(batch, dict):

            batch["X"] = batch["X"].float()
            batch["Y"] = batch["Y"].float()

            x = batch["X"]
            # y = batch["Y"]
            if self.is_causal:
                batch["c1"] = batch["c1"].float()
                batch["c2"] = batch["c2"].float()
                c1 = batch["c1"]
                c2 = batch["c2"]
                num_int = batch["num_int"]
                num_interv = num_int[0].item()
            else:
                c1 = None
                c2 = None
                num_int = None
                num_interv = None
        elif isinstance(batch, list):
            #? Ensure your list dataloader provides these in this order
            if len(batch) < 5:
                raise ValueError(f"List batch expected at least 5 elements, but got {len(batch)}")
            x = batch[0]
            # y = batch[1]
            # label = batch[5] if len(batch) > 5 else None

            if self.is_causal:
                c1 = batch[2]
                c2 = batch[3]
                num_int = batch[4]
                num_interv = num_int[0].item()
            else:
                c1 = None
                c2 = None
                num_int = None
                num_interv = None
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        arch_outputs = self.forward(x, c1, c2, num_interv=num_interv)
        total_loss, losses = self._calculate_loss(
            arch_outputs,
            batch
        )

        return total_loss, losses, arch_outputs

    def training_step(
        self,
        batch: t.Tuple | t.Dict,
        batch_idx: int,
    ) -> dict:
        """
        Performs a single training step.

        Parameters
        ----------
        batch : Tuple | Dict
            Data batch from the dataloader.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Computed loss for this training step.

        Notes
        -----
        Logs metrics under the ``train/`` namespace.

        Examples
        --------
        """
        loss, losses, arch_outputs = self._shared_step(batch, batch_idx, "train")
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            # on_step=False,
            # on_epoch=True,
            prog_bar=True
        )
        return {
            "loss": loss,
            "arch_outputs": arch_outputs,
            "losses": losses,
        }

    def validation_step(
        self,
        batch: t.Tuple | t.Dict,
        batch_idx: int,
    ) -> dict:
        """
        Performs a single validation step.

        Parameters
        ----------
        batch : Tuple | Dict
            Data batch from the dataloader.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        None

        Notes
        -----
        Logs metrics under the ``val/`` namespace.

        Examples
        --------
        """
        _, losses, arch_outputs = self._shared_step(batch, batch_idx, "val")
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return {
            "arch_outputs": arch_outputs,
            "losses": losses,
        }

    def test_step(
        self,
        batch: t.Tuple | t.Dict,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> dict:
        """
        Performs a single test step.

        Parameters
        ----------
        batch : Tuple | Dict
            Data batch from the dataloader.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        None

        Notes
        -----
        Logs metrics under the ``test/`` namespace.

        Examples
        --------
        """
        _, losses, arch_outputs = self._shared_step(batch, batch_idx, "test")

        if self.hparams.ena_test_mmd:
            labels = batch["label"]
            assert (labels == labels[0]).all(), \
                "Batch must contain same label!"

            unbiased_mmd_metric = self.unbiased_mmd_metric_f(
                arch_outputs.y_hat,
                batch["Y"]
            )
            losses["unbiased_mmd_metric"] = unbiased_mmd_metric

        if self.hparams.ena_label_as_test_prefix:
            labels = batch["label"]
            label = labels[0]
            assert (labels == label).all(), \
                "Batch must contain same label!"

            losses = {f"label_{label}/{k}": v for k, v in losses.items()}
            add_dataloader_idx = False
        else:
            add_dataloader_idx = True

        self.log_dict(
            {f"test/{k}": v for k, v in losses.items()},
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=add_dataloader_idx
        )

        return {
            "arch_outputs": arch_outputs,
            "losses": losses,
        }

    @torch.no_grad()
    def predict_step(
        self,
        batch: t.Any,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> dict:
        """
        Performs a single prediction step.

        This method is designed to be flexible and can handle various input
        formats for the `batch` argument, including dictionaries from a
        DataLoader, lists, raw PyTorch tensors, or NumPy arrays.

        Parameters
        ----------
        batch : t.Any
            The input data for prediction. Can be a dict, list, tensor, or numpy array.
        batch_idx : int
            The index of the batch.
        dataloader_idx : int, default 0
            The index of the dataloader.

        Returns
        -------
        dict
            A dictionary containing all the output tensors from the model's
            forward pass (e.g., reconstructions, latent variables).

        Raises
        ------
        TypeError
            If the `batch` is of an unsupported type.
        ValueError
            If the primary input tensor 'X' cannot be extracted from the batch.
        """
        x, c1, c2, num_interv = None, None, None, None

        #? --- 1. Flexible Input Parsing ---
        if isinstance(batch, dict):
            x = batch.get("X")
            if self.is_causal:
                c1 = batch.get("c1")
                c2 = batch.get("c2")
                num_int = batch["num_int"]
                num_interv = num_int[0].item()

        elif isinstance(batch, (list, tuple)):
            x = batch[0]
            if self.is_causal and len(batch) >= 5:
                c1, c2, num_int = batch[2], batch[3], batch[4]
                if num_int is not None:
                    num_interv = num_int[0].item()

        elif isinstance(batch, (torch.Tensor, np.ndarray)):
            x = batch #? Assume the raw batch is the input `x`

        else:
            raise TypeError(f"Unsupported batch type for prediction: {type(batch)}")

        if x is None:
            raise ValueError("Input 'x' could not be found or extracted from the batch.")

        #? --- 2. Data Conversion and Device Placement ---
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float().to(self.device)

        if c1 is not None:
            if isinstance(c1, np.ndarray): c1 = torch.from_numpy(c1)
            c1 = c1.float().to(self.device)
        if c2 is not None:
            if isinstance(c2, np.ndarray): c2 = torch.from_numpy(c2)
            c2 = c2.float().to(self.device)

        #? --- 3. Run Forward Pass in Inference Mode ---
        arch_outputs = self.forward(x, c1, c2, num_interv=num_interv)

        return arch_outputs

