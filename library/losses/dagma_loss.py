"""
PyTorch implementation of the DAGMA algorithm for learning Directed Acyclic Graphs (DAGs)
from observational data, structured with modular loss components.
"""
# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import torch
import torch.nn as nn

# ----------------- Loss Modules -----------------

class BaseLoss(nn.Module):
    """Base class for all loss components."""
    def __init__(self):
        super().__init__()

    def forward(
        self,
        W: nn.Parameter,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

class L2Score(BaseLoss):
    """Calculates the L2 score (least squares loss)."""
    def forward(
        self,
        W: nn.Parameter,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the score function (negative log-likelihood).

        Equivalent to 1/2n ||X(I-W)||_F^2

        Parameters
        ----------
        W : nn.Parameter
            The (d, d) weighted adjacency matrix.
        X : torch.Tensor
            The (n, d) data matrix.

        Returns
        -------
        torch.Tensor
            The scalar loss value.
        """
        n, d = X.shape
        #? Dynamically create Id with the same dtype and device as W
        Id = torch.eye(d, device=W.device, dtype=W.dtype)
        residuals = X @ (Id - W)
        return 0.5 * torch.sum(residuals ** 2) / n

class LogisticScore(BaseLoss):
    """Calculates the logistic score for binary data."""
    def forward(
        self,
        W: nn.Parameter,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the score function using binary cross-entropy.

        Parameters
        ----------
        W : nn.Parameter
            The (d, d) weighted adjacency matrix.
        X : torch.Tensor
            The (n, d) data matrix (expected to be binary).

        Returns
        -------
        torch.Tensor
            The scalar loss value.
        """
        n = X.shape[0]
        R = X @ W
        #? Use binary cross-entropy with logits for numerical stability
        return nn.functional.binary_cross_entropy_with_logits(R, X, reduction='sum') / n

class AcyclicityConstraint(BaseLoss):
    """Calculates the log-determinant acyclicity constraint h(W)."""
    def __init__(
        self,
        d: int,
        s: float = 1.0,
    ):
        super().__init__()
        self.d = d
        self.s = s
        self.register_buffer('Id', torch.eye(d))

    def forward(
        self,
        W: nn.Parameter,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculates h(W) = -log|sI - W◦W| + d*log(s).

        Parameters
        ----------
        W : nn.Parameter
            The (d, d) weighted adjacency matrix.
        X : torch.Tensor | None, optional
            Data matrix, not used in this calculation. Defaults to None.

        Returns
        -------
        torch.Tensor
            The scalar value of the acyclicity constraint.
        """
        #? Ensure the buffer's dtype and device match the parameter's
        M = self.s * self.Id.to(W.device, W.dtype) - W * W
        #? Use slogdet for numerical stability with logarithms
        _sign, logabsdet = torch.linalg.slogdet(M)
        h = -logabsdet + self.d * torch.log(torch.tensor(self.s, device=W.device))
        return h

class L1Penalty(BaseLoss):
    """Calculates the L1 penalty for sparsity."""
    def forward(
        self,
        W: nn.Parameter,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the L1 norm of the matrix W.

        Parameters
        ----------
        W : nn.Parameter
            The (d, d) weighted adjacency matrix.
        X : torch.Tensor | None, optional
            Data matrix, not used in this calculation. Defaults to None.

        Returns
        -------
        torch.Tensor
            The scalar L1 penalty.
        """
        return torch.sum(torch.abs(W))

class DAGMALoss(BaseLoss):
    """Combines all DAGMA loss components into a single objective function."""
    def __init__(
        self,
        #? --- Model & Data Configuration ---
        d: int,
        score_type: str = 'l2',
        #? --- Hyperparameters ---
        mu: float = 1.0,
        lambda1: float = 0.025,
    ):
        super().__init__()
        #? --- Loss Components ---
        if score_type == 'l2':
            self.score_loss = L2Score()
        elif score_type == 'logistic':
            self.score_loss = LogisticScore()
        else:
            raise ValueError(f"Unsupported score_type: {score_type}. Choose 'l2' or 'logistic'.")
        
        self.acyclicity_constraint = AcyclicityConstraint(d)
        self.l1_penalty = L1Penalty()
        
        #? --- Hyperparameters ---
        self.mu = mu
        self.lambda1 = lambda1

    def forward(
        self,
        W: nn.Parameter,
        X: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the total DAGMA objective and its individual components.
        objective = mu * (score + lambda1 * l1_reg) + h

        Parameters
        ----------
        W : nn.Parameter
            The (d, d) weighted adjacency matrix.
        X : torch.Tensor
            The (n, d) data matrix.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the total objective, the score, and the h-value.
        """
        #? Detach X from the computation graph. This prevents gradients from flowing
        #? back to the encoder that produced X (e.g., in a VAE). This ensures
        #? that this loss function is ONLY responsible for updating the graph weights W.
        X_detached = X.detach()
        
        score = self.score_loss(W, X_detached)
        h = self.acyclicity_constraint(W)
        l1_reg = self.l1_penalty(W)
        objective = self.mu * (score + self.lambda1 * l1_reg) + h
        return objective, score, h