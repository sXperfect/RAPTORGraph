"""
PyTorch implementation of the DAGMA algorithm for learning Directed Acyclic Graphs (DAGs)
from observational data, structured with modular loss components.
"""
# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import enum
import typing as t

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import torch
import torch.nn as nn

# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
from ..losses.dagma_loss import DAGMALoss, L2Score, LogisticScore, AcyclicityConstraint, L1Penalty

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================
class MatrixInverseError(RuntimeError):
    """Custom exception for when both matrix inversion methods fail."""
    def __init__(self, message):
        super().__init__(message)

# ----------------- Main Model -----------------

class DAGMALinear(nn.Module):
    """
    PyTorch implementation of DAGMA for linear models.

    This module learns the weighted adjacency matrix W of a DAG.
    The forward pass solves the linear system X = WX + E for X, given noise E.
    """

    def __init__(
        self,
        #? --- Model Configuration ---
        d: int,
        w_threshold: float | None = 0.3,
        force_dag: bool = False,
        #? --- General Settings ---
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        d : int
            Number of variables.
        w_threshold : float | None, optional
            Threshold for pruning edges. If a float, edges with absolute
            weight below this value are removed. Defaults to 0.3.
        force_dag : bool, optional
            If True, iteratively prunes the weakest edges after thresholding
            to guarantee the final graph is a DAG. If `w_threshold` is None,
            this is automatically set to True. Defaults to False.
        verbose : bool, optional
            If true, prints progress during the optimization. Defaults to False.
        """
        super().__init__()
        self.d = d
        self.w_threshold = w_threshold
        self.force_dag = force_dag or (w_threshold is None)
        self.verbose = verbose
        self.vprint = print if verbose else lambda *a, **k: None

        self.weight = nn.Parameter(torch.zeros(d, d))
        self.enforce_zero_diagonal()

        self.register_buffer('Id', torch.eye(d))
        self.register_buffer('W_thresholded', torch.zeros(d, d))

    @property
    def W(self):
        return self.weight

    def forward(
        self,
        E: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solves the structural equation X = XW^T + E for X.
        In training mode, uses the full W. In eval mode, uses the thresholded W.

        Parameters
        ----------
        E : torch.Tensor
            The (n, d) exogenous noise tensor.

        Returns
        -------
        torch.Tensor
            The (n, d) observed variables X.
            
        Raises
        ------
        MatrixInverseError
            If both `torch.linalg.solve` and `torch.linalg.pinv` fail due to
            a singular or ill-conditioned matrix.
        """
        if E.shape[1] != self.d:
            raise ValueError(f"Input E has {E.shape[1]} variables, but the model is configured for {self.d}.")

        W_operative = self.weight if self.training else self.W_thresholded
        #? Ensure the buffer matches the parameter's dtype and device at runtime
        I_minus_W = self.Id.to(device=W_operative.device, dtype=W_operative.dtype) - W_operative
        try:
            X = torch.linalg.solve(I_minus_W.T, E.T).T
        except torch._C._LinAlgError:
            self.vprint("Warning: (I - W) is singular. Falling back to pseudo-inverse.")
            try:
                pinv_I_minus_W = torch.linalg.pinv(I_minus_W)
                X = E @ pinv_I_minus_W
            except torch._C._LinAlgError as e:
                raise MatrixInverseError(
                    "Failed to compute both `solve` and `pinv` for (I - W). "
                    "The matrix is likely severely ill-conditioned."
                ) from e
        return X

    def get_adjacency(self) -> torch.Tensor:
        """Returns the learned adjacency matrix."""
        return self.weight.detach().clone()

    def enforce_zero_diagonal(self):
        """Enforces the zero-diagonal constraint on the adjacency matrix."""
        with torch.no_grad():
            self.weight.fill_diagonal_(0)

    def train(
        self,
        mode: bool = True,
    ) -> t.Self:
        """Sets the module in training mode.

        Parameters
        ----------
        mode : bool, optional
            Whether to set training mode (`True`) or evaluation
            mode (`False`). Defaults to True.

        Returns
        -------
        t.Self
            The module instance.
        """
        self.training = mode
        return self
        
    def _is_dag_dfs(self, B: torch.Tensor) -> bool:
        """
        Checks if a binary matrix B corresponds to a DAG using Depth First Search.
        This is much more efficient than the matrix power method for large graphs.
        """
        d = B.shape[0]
        #? Adjacency list representation for efficient neighbor lookup
        adj = [[] for _ in range(d)]
        for i in range(d):
            for j in range(d):
                if B[i, j] > 0:
                    adj[i].append(j)

        path = [False] * d
        visited = [False] * d

        def _dfs_util(u):
            visited[u] = True
            path[u] = True
            for v in adj[u]:
                if path[v]: #? Node is in the current recursion stack -> cycle
                    return True
                if not visited[v]:
                    if _dfs_util(v):
                        return True
            path[u] = False
            return False

        for node in range(d):
            if not visited[node]:
                if _dfs_util(node):
                    return False #? Cycle detected
        return True #? No cycles found

    def _break_cycles_iteratively(self, W_adj: torch.Tensor) -> torch.Tensor:
        """
        Iteratively removes the weakest edge in any detected cycles
        until the graph is a DAG. Implemented purely in PyTorch.
        """
        B = (W_adj != 0).long()
        
        while not self._is_dag_dfs(B):
            #? Find the weakest edge in the entire graph.
            non_zero_abs_weights = torch.abs(W_adj[B != 0])
            if len(non_zero_abs_weights) == 0:
                break #? Should not happen if not a DAG, but as a safeguard
            
            min_weight = torch.min(non_zero_abs_weights)
            
            #? Find all edges with this minimum weight
            rows, cols = torch.where(torch.abs(W_adj) == min_weight)
            
            #? Remove the first one found to break a cycle
            if len(rows) > 0:
                r, c = rows[0], cols[0]
                W_adj[r, c] = 0
                B[r, c] = 0
            else:
                break #? No edges left to remove

        return W_adj

    def eval(self) -> t.Self:
        """
        Sets the module in evaluation mode. Applies an optional threshold
        and, if `force_dag` is True, iteratively breaks any cycles to
        guarantee the final graph is a DAG.
        """
        self.train(False)
        W_adj = self.get_adjacency()
        
        #? 1. Apply the fixed threshold if one is provided.
        if self.w_threshold is not None:
            W_adj[torch.abs(W_adj) < self.w_threshold] = 0
        
        #? 2. Guarantee a DAG if forced.
        if self.force_dag:
            self.W_thresholded = self._break_cycles_iteratively(W_adj)
        else:
            self.W_thresholded = W_adj
            
        return self

class DAGMAPathFollowing(nn.Module):
    """
    Implements the full DAGMA algorithm with the path-following optimization strategy.
    """
    def __init__(
        self,
        d: int,
        score_type: str = 'l2',
        lambda1: float = 0.025,
        w_threshold: float | None = 0.3,
        force_dag: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.d = d
        self.verbose = verbose
        self.vprint = print if verbose else lambda *a, **k: None

        self.linear_model = DAGMALinear(d, w_threshold, force_dag, verbose)
        if score_type == 'l2': self.score_loss = L2Score()
        elif score_type == 'logistic': self.score_loss = LogisticScore()
        else: raise ValueError(f"Unsupported score_type: {score_type}.")
        self.acyclicity_constraint = AcyclicityConstraint(d)
        self.l1_penalty = L1Penalty()
        self.lambda1 = lambda1

    @property
    def W(self) -> nn.Parameter:
        return self.linear_model.weight

    def _calculate_objective(self, W: nn.Parameter, X: torch.Tensor, mu: float) -> tuple:
        score = self.score_loss(W, X.detach()) #? Detach X
        h = self.acyclicity_constraint(W)
        l1_reg = self.l1_penalty(W)
        objective = mu * (score + self.lambda1 * l1_reg) + h
        return objective, score, h

    def fit(
        self, X: torch.Tensor, 
        mu_init: float = 1.0, 
        mu_factor: float = 0.1,
        mu_steps: int = 4, 
        lr: float = 0.01, 
        inner_steps: int = 500,
    ):
        self.vprint("Starting DAGMA path-following optimization...")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mu_schedule = [mu_init * (mu_factor ** i) for i in range(mu_steps)]
        
        self.linear_model.train()
        for step, mu in enumerate(mu_schedule):
            self.vprint(f"\n--- Path-following step {step + 1}/{mu_steps} (mu = {mu:.1e}) ---")
            for i in range(inner_steps):
                optimizer.zero_grad()
                X_device = X.to(self.W.device, self.W.dtype)
                objective, score, h = self._calculate_objective(self.W, X_device, mu)
                if torch.isinf(objective) or torch.isnan(objective):
                    self.vprint(f"Warning: Objective is {objective.item()}. Halting.")
                    break
                objective.backward()
                optimizer.step()
                self.linear_model.enforce_zero_diagonal()
                if self.verbose and i > 0 and i % (inner_steps // 4) == 0:
                    self.vprint(f"  Inner iter {i}: Obj={objective.item():.4f}, Score={score.item():.4f}, h(W)={h.item():.4f}")
            else: continue # Continue if the inner loop wasn't broken
            break # Break outer loop if inner loop was broken
        
        self.vprint("\nOptimization finished.")
        self.linear_model.eval()
        return self.linear_model.W_thresholded

    def get_adjacency(self) -> torch.Tensor:
        return self.linear_model.W_thresholded.detach().clone()