"""Causal Representation Learning loss utilities.

This module implements a flexible loss function intended for Autoencoder (AE)
and Variational Autoencoder (VAE). It supports several MMD-based strategies,
optional per-label weighting, an optional triplet margin loss for metric learning,
and provides an unbiased MMD metric for validation. The KL-divergence term can 
be disabled to support a standard Autoencoder model.
"""
import typing as t

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import LpDistance

# =============================================================================
# LOCAL IMPORTS
# =============================================================================
from ..structs import CRLForwardPassOutput, BaseStrEnum
from .mmd_loss import (
    DiscrepancyVAE_MultiKernelMaximumMeanDiscrepancy,
    MultiKernelMaximumMeanDiscrepancy,
)
from .dagma_loss import (
    DAGMALoss, 
)

# =============================================================================
# ENUMS AND TYPE ALIASES
# =============================================================================
LossDict = t.Dict[str, torch.Tensor]
LossTuple = t.Tuple[torch.Tensor, ...]


class MMDVersion(BaseStrEnum):
    """
    Specifies the type of MMD implementation for the training loss.

    Members
    -------
    DISCREPANCY_V1
        An MMD implementation from the Discrepancy VAE paper.
    BIASED
        A standard biased estimator for the multi-kernel MMD.
    UNBIASED
        A standard unbiased estimator for the multi-kernel MMD.
    """
    DISCREPANCY_V1 = "discrepancy_v1"
    BIASED = "biased"
    UNBIASED = "unbiased"

class MMDStrategy(BaseStrEnum):
    """
    Defines the strategy for applying MMD loss across a batch.

    Members
    -------
    GLOBAL
        Computes a single MMD loss over the entire batch.
    WEIGHTED
        Computes a global MMD loss, but weights each sample by the inverse
        frequency of its class.
    PER_LABEL
        Computes a separate MMD loss for each unique label present in the
        batch and averages the results.
    DYNAMIC
        Acts as 'PER_LABEL' if multiple labels are present in the batch,
        otherwise falls back to 'GLOBAL'.
    """
    GLOBAL = "global"
    WEIGHTED = "weighted"
    PER_LABEL = "per_label"
    DYNAMIC = "dynamic"
    
class GraphLossType(BaseStrEnum):
    """
    Defines the type of loss to apply for graph regularization.

    Members
    -------
    L1
        A simple L1 penalty on the graph weights to encourage sparsity.
    DAGMA
        The full DAGMA objective, combining a score function, L1 penalty,
        and an acyclicity constraint.
    """
    L1 = "l1"
    DAGMA = "dagma"

# =============================================================================
# LOSS FUNCTION
# =============================================================================
class CausalRepresentationLearningLoss:
    """Causal Representation Learning loss utilities.

    This module implements a flexible composite loss function for Causal Representation 
    Learning models, supporting both Autoencoder (AE) and Variational Autoencoder (VAE) 
    architectures. The loss function integrates multiple components for representation 
    learning with causal structure.

    Key Features:
    - Reconstruction loss (MSE) for input data
    - KL-divergence loss for variational inference (optional)
    - Graph regularization using L1 penalty or DAGMA loss
    - Maximum Mean Discrepancy (MMD) loss with multiple strategies:
    * Global: single MMD across the entire batch
    * Weighted: MMD weighted by inverse class frequency
    * Per-label: separate MMD for each label, then averaged
    * Dynamic: per-label when multiple labels present, otherwise global
    - Optional triplet margin loss for metric learning
    - Unbiased MMD metric for validation purposes

    The KL-divergence term can be disabled to support standard Autoencoder models, 
    and the graph regularization can be configured for causal structure learning.

    This loss function is designed to work with the ```CRLForwardPassOutput``` structure 
    that contains all necessary components for the various loss terms.
    """

    def __init__(
        self,
        #? --- Model & Data Configuration ---
        latent_dim: int,
        #? --- General Behavior Flags ---
        is_variational: bool = False,
        is_causal: bool = False,
        deterministic_intervention: bool = False,
        ret_dict: bool = False,
        #? --- External Metrics ---
        knn_metric: t.Any | None = None,
        #? --- Graph Loss Configuration ---
        graph_loss_type: str | GraphLossType | None = 'l1',
        #? --- DAGMA-Specific Hyperparameters ---
        dagma_mu: float = 1.0,
        dagma_lambda1: float = 0.025,
        dagma_score_type: str = "l2",
        # #? --- DAGMA Augmented Lagrangian Parameters for Stability ---
        # dagma_mu_init: float = 1.0,
        # dagma_mu_update_factor: float = 10.0,
        # dagma_h_threshold: float = 1e-8,
        # dagma_mu_max: float = 1e+16,
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
    ) -> None:
        """Composite loss for Causal Representation Learning Autoencoders/VAEs.

        This class implements a flexible composite loss function that combines
        multiple components for training causal representation learning models.
        It supports both standard Autoencoders (AE) and Variational Autoencoders (VAE)
        through configurable options.

        The loss components include:
        1. Reconstruction loss (MSE) - always included
        2. KL-divergence loss - for VAEs (optional)
        3. Graph regularization loss - for causal structure learning (optional)
        - L1 penalty on graph weights
        - DAGMA loss (score function + L1 penalty + acyclicity constraint)
        4. Maximum Mean Discrepancy (MMD) loss - for distribution matching (optional)
        - Multiple strategies: global, weighted, per-label, dynamic
        5. Triplet margin loss - for metric learning (optional)
        6. Unbiased MMD metric - for validation (optional)

        The class provides flexibility through initialization parameters and runtime
        overrides, allowing for different training regimes and evaluation metrics.

        Attributes
        ----------
        mse_loss_fn : nn.MSELoss
            Mean Squared Error loss function used for reconstruction.
        is_variational : bool
            Whether KL-divergence computation is enabled for VAEs.
        is_causal : bool
            Whether causal graph regularization is enabled.
        ena_mmd_loss : bool
            Whether MMD loss computation is enabled.
        mmd_strategy : MMDStrategy
            Strategy for MMD calculation ('global', 'weighted', 'per_label', 'dynamic').
        deterministic_intervention : bool
            Whether to use MSE instead of MMD for interventional loss.
        ena_unbiased_mmd_metric : bool
            Whether to compute unbiased MMD metric for validation.
        ret_dict : bool
            Whether to return losses as a dictionary or tuple.
        min_sample_per_label : int
            Minimum group size for per-label MMD calculations.
        mmd_version : MMDVersion
            Version of MMD implementation used.
        knn_metric : Any or None
            External metric for KNN accuracy calculation.
        ena_triplet_loss : bool
            Whether triplet margin loss is enabled.
        triplet_margin : float
            Margin for triplet loss.
        graph_loss_type : GraphLossType
            Type of graph regularization loss ('l1' or 'dagma').
        graph_loss_f : DAGMALoss or None
            Graph loss function (if causal learning is enabled).
        matching_function_interv_f : Callable
            Function for computing interventional loss (MMD or MSE).
        unbiased_mmd_metric_f : MultiKernelMaximumMeanDiscrepancy or None
            Unbiased MMD metric function for validation.
        triplet_loss_f : TripletMarginLoss or None
            Triplet margin loss function.
        triplet_miner : Any or None
            Miner for selecting triplets.

        Notes
        -----
        - If `deterministic_intervention` is True, the interventional loss uses MSE
        instead of MMD, regardless of other MMD configurations.
        - The `compute_loss` method allows runtime overrides of loss components
        through parameters like `force_kl`, `force_causal`, etc.
        - For causal models with DAGMA loss, observational latent representation
        ('z_obs') must be provided in the forward pass output.
        """
        self.mse_loss_fn = nn.MSELoss()
        self.is_variational = is_variational
        self.is_causal = is_causal
        self.ena_mmd_loss = ena_mmd_loss
        self.mmd_strategy = MMDStrategy(mmd_strategy) if mmd_strategy else None
        self.deterministic_intervention = deterministic_intervention
        self.ena_unbiased_mmd_metric = ena_unbiased_mmd_metric
        self.ret_dict = ret_dict
        self.min_sample_per_label = mmd_min_sample_per_label
        self.mmd_version = MMDVersion(mmd_version)
        self.knn_metric = knn_metric

        self.ena_triplet_loss = ena_triplet_loss
        self.triplet_margin = triplet_margin

        #? Enforce the consistency rule
        if self.ena_mmd_loss and self.mmd_version == MMDVersion.UNBIASED:
            if self.ena_unbiased_mmd_metric:
                warnings.warn(
                    "`ena_unbiased_mmd_metric` is set to False because "
                    "`ena_mmd_loss=True` with `mmd_version='unbiased'`. "
                    "The unbiased loss already provides the desired statistic.",
                    UserWarning,
                )
            self.ena_unbiased_mmd_metric = False

        #? --- Graph Loss Setup ---
        self.graph_loss_type = GraphLossType(graph_loss_type)
        if self.is_causal:
            if self.graph_loss_type == GraphLossType.L1:
                self.graph_loss_f = None
            elif self.graph_loss_type == GraphLossType.DAGMA:
                # self.dagma_mu_init = dagma_mu_init
                # self.dagma_mu_update_factor = dagma_mu_update_factor
                # self.dagma_h_threshold = dagma_h_threshold
                # self.dagma_mu_max = dagma_mu_max
                # self.dagma_mu = dagma_mu_init
                self.dagma_alpha = 0.0
                self.graph_loss_f = DAGMALoss(
                    d=latent_dim,
                    score_type=dagma_score_type,
                    mu=dagma_mu,
                    lambda1=dagma_lambda1,
                )
            elif self.graph_loss_type is None:
                self.graph_loss_f = None
            else:
                raise ValueError(f"Unsupported graph loss type: {self.graph_loss_type}")

        #? --- MMD Loss Setup ---
        if self.deterministic_intervention:
            self.matching_function_interv_f = self.mse_loss_fn
        else:
            mmd_config = {
                "fix_sigma": mmd_sigma, 
                "kernel_num": mmd_kernel_num,
                "kernel_mul": mmd_kernel_mul,
            }
            if mmd_version == MMDVersion.DISCREPANCY_V1:
                self.matching_function_interv_f = (
                    DiscrepancyVAE_MultiKernelMaximumMeanDiscrepancy(**mmd_config)
                )
            elif mmd_version in (MMDVersion.BIASED, MMDVersion.UNBIASED):
                self.matching_function_interv_f = MultiKernelMaximumMeanDiscrepancy(
                    **mmd_config, unbiased=(mmd_version == MMDVersion.UNBIASED)
                )

        #? --- Unbiased MMD metric (used for validation) ---
        if self.ena_unbiased_mmd_metric:
            self.unbiased_mmd_metric_f = MultiKernelMaximumMeanDiscrepancy(
                fix_sigma=mmd_sigma, 
                kernel_num=mmd_kernel_num, 
                kernel_mul=mmd_kernel_mul,
                unbiased=True,
            )

        #? --- Triplet Loss and Miner Setup ---
        if self.ena_triplet_loss:
            dist_f = LpDistance(p=2) if triplet_distance_metric.lower() == 'l2' else None
            
            self.triplet_loss_f = losses.TripletMarginLoss(
                margin=triplet_margin,
                distance=dist_f, #? default use MSE
            )
            if triplet_miner_name:
                if triplet_miner_name.lower() == "triplet_margin":
                    self.triplet_miner = miners.TripletMarginMiner(
                        margin=triplet_miner_margin,
                        distance=dist_f,
                        type_of_triplets="hard"
                    )
                else:
                    raise ValueError(f"Unsupported triplet miner: {triplet_miner_name}")
            else:
                self.triplet_miner = None

    def _compute_mmd(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        label: torch.Tensor,
        strategy: MMDStrategy,
        mmd_func: t.Callable,
    ) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy (MMD) loss based on the selected strategy.

        This internal method calculates the MMD loss between predicted and target
        distributions using different strategies for handling labeled data.

        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted tensor (e.g., reconstructed or interventional representation).
        y : torch.Tensor
            Target tensor (e.g., original input or interventional target).
        label : torch.Tensor
            Integer labels for each sample, used for per-label or weighted strategies.
        strategy : MMDStrategy
            Strategy for MMD calculation:
            - GLOBAL: single MMD across the entire batch
            - WEIGHTED: MMD weighted by inverse class frequency
            - PER_LABEL: separate MMD for each label, then averaged
            - DYNAMIC: per-label when multiple labels present, otherwise global
        mmd_func : Callable
            Function that computes the MMD between two tensors.

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the computed MMD loss.

        Notes
        -----
        - For WEIGHTED strategy, weights are normalized to sum to 1.
        - For PER_LABEL strategy, groups with fewer samples than `min_sample_per_label`
          are skipped.
        - DYNAMIC strategy automatically switches between PER_LABEL and GLOBAL based
          on label diversity in the batch.
        """
        if strategy == MMDStrategy.DYNAMIC:
            if (label == label[0]).all():
                strategy = MMDStrategy.GLOBAL
            else:
                strategy = MMDStrategy.PER_LABEL            
        
        if strategy in [MMDStrategy.GLOBAL]:
            return mmd_func(y_hat, y)

        elif strategy == MMDStrategy.WEIGHTED:
            #? --- Correctly compute weights for inverse frequency weighting ---
            uniq, inv, counts = torch.unique(label, return_inverse=True, return_counts=True)
            weights = 1.0 / counts.float()
            weights = weights / weights.sum()  #? Normalize weights for each class
            sample_weights = weights[inv]  #? Get weight for each sample
            return mmd_func(y_hat, y) * sample_weights.mean()

        elif strategy == MMDStrategy.PER_LABEL:
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
                return torch.tensor(0.0, device=y_hat.device)
            return torch.stack(losses).mean()

        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        #? --- Fallback to global MMD if dynamic strategy results in single label ---
        return mmd_func(y_hat, y)

    def compute_mmd_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        label: torch.Tensor | None = None,
        strategy: MMDStrategy | None = MMDStrategy.GLOBAL,
    ) -> torch.Tensor:
        """Compute the MMD term for the training loss.

        This method calculates the Maximum Mean Discrepancy between predicted and
        target distributions, with options for different calculation strategies
        when labels are available.

        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted tensor (e.g., reconstructed or interventional representation).
        y : torch.Tensor
            Target tensor (e.g., original input or interventional target).
        label : torch.Tensor or None, optional
            Integer labels for each sample, used for per-label or weighted MMD.
            If None, global MMD is computed across the entire batch.
            Default: None
        strategy : MMDStrategy or None, optional
            Strategy for MMD calculation. If None, uses the strategy set during
            initialization. Default: MMDStrategy.GLOBAL

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the computed MMD loss.

        Notes
        -----
        - If `deterministic_intervention` is True, this method returns MSE loss
          instead of MMD, regardless of other configurations.
        - For PER_LABEL strategy, groups with fewer samples than `min_sample_per_label`
          are skipped in the calculation.
        - The method handles all MMD strategies internally, making it flexible for
          different training regimes.

        Examples
        --------
        """
        if self.deterministic_intervention:
            return self.mse_loss_fn(y_hat, y)
        if label is None:
            return self.matching_function_interv_f(y_hat, y)

        #? Use the self.matching_function_interv
        return self._compute_mmd(
            y_hat, 
            y, 
            label, 
            strategy, 
            self.matching_function_interv_f
        )

    def calculate_unbiased_mmd_metric(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        label: torch.Tensor | None = None,
        strategy: MMDStrategy | None = MMDStrategy.GLOBAL,
    ) -> torch.Tensor:
        """Compute the unbiased MMD metric for validation.

        This method calculates an unbiased estimate of the Maximum Mean Discrepancy
        between predicted and target distributions, suitable for validation and
        monitoring model performance.

        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted tensor (e.g., reconstructed or interventional representation).
        y : torch.Tensor
            Target tensor (e.g., original input or interventional target).
        label : torch.Tensor or None, optional
            Integer labels for each sample. If None, global MMD is computed.
            Default: None
        strategy : MMDStrategy or None, optional
            Strategy for MMD calculation. If None, uses the strategy set during
            initialization. Default: MMDStrategy.GLOBAL

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the unbiased MMD metric value.

        Notes
        -----
        - This method is typically used for validation, not for training.
        - The unbiased estimator provides a more accurate measure of distribution
          discrepancy compared to the biased version used in training.
        - If `ena_unbiased_mmd_metric` is False (set during initialization), this
          method will return 0.

        Examples
        --------
        """
        if not hasattr(self, "ena_unbiased_mmd_metric"):
            return torch.tensor(0.0, device=y_hat.device)
        elif label is None:
            return self.unbiased_mmd_metric_f(y_hat, y)

        #? Use the self.unbiased_mmd_metric
        return self._compute_mmd(
            y_hat, 
            y, 
            label, 
            strategy, 
            self.unbiased_mmd_metric_f
        )

    def compute_loss(
        self,
        outputs: CRLForwardPassOutput,
        x: torch.Tensor,
        y: torch.Tensor | None,
        *,
        label: torch.Tensor | None = None,
        mmd_strategy: MMDStrategy | str | None = None,
        #? --- Runtime Overrides ---
        force_kl: bool | None = None,
        force_causal: bool | None = None,
        force_mmd: bool | None = None,
        force_unbiased_mmd_metric: bool | None = None,
        force_triplet_loss: bool | None = None,
    ) -> LossDict | LossTuple:
        """Compute all loss components based on model configuration and outputs.

        This method calculates the complete loss for training or evaluation,
        combining multiple components based on the model's configuration and
        the provided forward pass outputs.

        Parameters
        ----------
        outputs : CRLForwardPassOutput
            Structured output from the model's forward pass containing:
            - `x_recon`: reconstructed input
            - `y_hat`: predicted interventional representation (if applicable)
            - `kl_mu`, kl_log_var: distribution parameters for VAEs
            - `G`: causal graph matrix
            - `z_obs`: observational latent representation
            - `z`: latent representation for metric learning
        x : torch.Tensor
            Original input data tensor.
        y : torch.Tensor or None
            Target data tensor for intervention/reconstruction. Can be None if
            no interventional loss is needed.
        label : torch.Tensor or None, optional
            Integer labels for each sample, used for per-label or weighted MMD
            and triplet loss. Default: None
        mmd_strategy : MMDStrategy, str, or None, optional
            Strategy for MMD calculation. If None, uses the strategy set during
            initialization. Default: None
        force_kl : bool or None, optional
            If not None, overrides the instance's `is_variational` setting for
            this computation. Default: None
        force_causal : bool or None, optional
            If not None, overrides the instance's `is_causal` setting for this
            computation. Default: None
        force_mmd : bool or None, optional
            If not None, overrides the instance's `ena_mmd_loss` setting for
            this computation. Default: None
        force_unbiased_mmd_metric : bool or None, optional
            If not None, overrides the instance's `ena_unbiased_mmd_metric`
            setting for this computation. Default: None
        force_triplet_loss : bool or None, optional
            If not None, overrides the instance's `ena_triplet_loss` setting for
            this computation. Default: None

        Returns
        -------
        LossDict or LossTuple
            If `ret_dict` is True (set during initialization):
                Dictionary mapping loss component names to their values.
            Otherwise:
                Tuple containing the values of the loss dictionary.

            The dictionary always includes 'recon_loss'. Additional keys may
            include:
            - 'kl_loss': KL divergence loss for VAEs
            - 'graph_loss': Causal graph regularization loss
            - 'dagma_score': DAGMA score component (if using DAGMA loss)
            - 'dagma_h_value': DAGMA acyclicity constraint value
            - 'mmd_loss' or 'interv_mse_loss': Interventional loss
            - 'unbiased_mmd': Unbiased MMD metric value
            - 'triplet_loss': Triplet margin loss
            - 'true_triplet_loss': Triplet loss minus margin
            - 'knn_acc', 'knn_acc_recon', 'knn_acc_interv': KNN accuracy metrics

        Raises
        ------
        ValueError
            If required components are missing for enabled loss terms:
            - KL loss requires mu and log_var
            - Causal loss requires G and possibly z_obs
            - Triplet loss requires z and label

        Notes
        -----
        - The method provides flexibility through runtime overrides, allowing
          different loss configurations during training, validation, or testing.
        - For causal models with DAGMA loss, the observational latent representation
          ('z_obs') must be provided in the outputs.
        - The 'true_triplet_loss' is computed as 'triplet_loss - triplet_margin'
          to provide a measure of how much the loss exceeds the margin.

        Examples
        --------
        """
        device = x.device
        loss_values: LossDict = {}

        #? --- Determine the active MMD strategy (override or default) ---
        active_mmd_strategy = MMDStrategy(mmd_strategy) if mmd_strategy is not None else self.mmd_strategy

        #? --- Always compute reconstruction loss ---
        loss_values["recon_loss"] = self.mse_loss_fn(outputs.x_recon, x)

        #? --- Decide whether to compute each loss component ---
        compute_kl = force_kl if force_kl is not None else self.is_variational
        compute_causal = force_causal if force_causal is not None else self.is_causal
        compute_mmd = (force_mmd if force_mmd is not None else self.ena_mmd_loss) and outputs.y_hat is not None
        compute_unbiased = force_unbiased_mmd_metric if force_unbiased_mmd_metric is not None else self.ena_unbiased_mmd_metric
        compute_triplet = force_triplet_loss if force_triplet_loss is not None else self.ena_triplet_loss

        #? --- Conditional KL Divergence Loss ---
        if compute_kl:
            mu, log_var = outputs.kl_mu, outputs.kl_log_var
            if mu is None or log_var is None:
                raise ValueError("A valid distribution (`mu`, `log_var`) must be available for KL divergence in variational mode.")
            loss_values["kl_loss"] = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


        #? --- Conditional L1 Graph Regularization ---
        if compute_causal:
            if outputs.G is None:
                raise ValueError("Graph `G` must be provided in causal mode.")
            
            if self.graph_loss_type == GraphLossType.DAGMA:
                #? For DAGMA, the 'X' data is the observational latent representation 'z_obs'
                if not hasattr(outputs, 'z_obs') or outputs.z_obs is None:
                    raise ValueError("Observational latent variable 'z_obs' must be provided for DAGMA loss.")
                # self.graph_loss_f.mu, self.graph_loss_f.alpha = self.dagma_mu, self.dagma_alpha
                objective, score, h = self.graph_loss_f(outputs.G, outputs.z_obs)
                loss_values.update({
                    "graph_loss": objective, 
                    "dagma_score": score, 
                    "dagma_h_value": h.detach()
                })
                # with torch.no_grad():
                #     if h.item() > self.dagma_h_threshold:
                #         self.dagma_mu = min(self.dagma_mu * self.dagma_mu_update_factor, self.dagma_mu_max)
                #     self.dagma_alpha += self.dagma_mu * h.item()
            #? L1
            elif self.graph_loss_type == GraphLossType.L1:
                dag_matrix = torch.triu(outputs.G, diagonal=1)
                num_possible_edges = torch.sum(torch.triu(torch.ones_like(outputs.G), diagonal=1))
                l1_loss = torch.tensor(0.0, device=device)
                if num_possible_edges > 0:
                    l1_loss = torch.norm(dag_matrix, p=1) / num_possible_edges
                loss_values["graph_loss"] = l1_loss

        #? --- MMD Loss ---
        if compute_mmd:
            # y_hat = t.cast(torch.Tensor, outputs.y_hat)
            #? If matched_IO is on, the MMD loss is actually an MSE loss.
            #? We use a more descriptive key name for clarity in logging.
            loss_key = "interv_mse_loss" if self.deterministic_intervention else "mmd_loss"
            loss_values[loss_key] = self.compute_mmd_loss(
                outputs.y_hat, 
                y, 
                label, 
                active_mmd_strategy
            )

        #? --- Unbiased MMD Metric ---
        if compute_unbiased:
            #! Does not make great sense for the "_X" because num samples per label is too low
            # loss_values["unbiased_mmd_X"] = self.calculate_unbiased_mmd_metric(
            #     outputs.x_recon, x, label, MMDStrategy.PER_LABEL
            # )
            if outputs.y_hat is not None:
                # y_hat = t.cast(torch.Tensor, outputs.y_hat)
                loss_values["unbiased_mmd"] = self.calculate_unbiased_mmd_metric(
                    outputs.y_hat, 
                    y, 
                    label, 
                    active_mmd_strategy
                )

        #? --- Triplet Margin Loss ---
        if compute_triplet:
            if outputs.z is None or label is None:
                raise ValueError("Embeddings 'z' and 'label' must be provided for triplet loss.")
            
            if self.triplet_miner:
                indices_tuple = self.triplet_miner(outputs.z, label)
                loss_values["triplet_loss"] = self.triplet_loss_f(outputs.z, label, indices_tuple)
            else:
                loss_values["triplet_loss"] = self.triplet_loss_f(outputs.z, label)

            loss_values["true_triplet_loss"] = loss_values["triplet_loss"] - self.triplet_margin

        #? --- Compute KNN Accuracy if metric is provided ---
        if self.knn_metric is not None and label is not None:
            if self.is_causal:
                #? For causal models, we have two distinct reconstructions
                if outputs.x_recon is not None:
                    # Observational reconstruction is compared against the 'control' class (label 0)
                    recon_labels = torch.zeros(outputs.x_recon.shape[0], dtype=torch.long, device=outputs.x_recon.device)
                    loss_values["knn_acc_recon"] = self.knn_metric.get_accuracy(outputs.x_recon, recon_labels)
                if outputs.y_hat is not None:
                    #? Interventional reconstruction is compared against the actual intervention label
                    loss_values["knn_acc_interv"] = self.knn_metric.get_accuracy(outputs.y_hat, label)
            else:
                #? For non-causal models, there's only one reconstruction
                if outputs.x_recon is not None:
                    loss_values["knn_acc"] = self.knn_metric.get_accuracy(outputs.x_recon, label)
                
        #? --- Assemble final return value ---
        if self.ret_dict:
            return loss_values

        return tuple(loss_values.values())
