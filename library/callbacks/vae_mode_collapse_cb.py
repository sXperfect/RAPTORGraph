import typing as t
import lightning as L
import torch
import numpy as np
from scipy import stats
from typing import Optional

class VAEModeCollapseError(Exception):
    """Specialized error for VAE mode collapse detection"""
    def __init__(self, message: str, details: dict):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self):
        return f"{self.message}\nDetails:\n{self.details}"

class VAEModeCollapseCallback(L.Callback):
    def __init__(
        self,
        kl_window: int = 5,  # Epoch window for KL trend analysis
        recon_kl_ratio_threshold: float = 100.0,  # Ratio of recon_loss/kl_loss
        diversity_batch_size: int = 256,  # Samples for diversity check
        p_value_threshold: float = 0.01,  # Statistical significance level
        patience: int = 3,  # Allow N consecutive violations before error
        ena_z_stat_test: bool = False,
        z_stat_test_min_epoch = 0,
        z_mu_min_name: t.Optional[str] = "z_mu_min",
        z_mu_max_name: t.Optional[str] = "z_mu_max",
        z_mu_min_diff_thres: t.Optional[float ] = 0.5,
        z_var_min_name: t.Optional[str] = "z_var_min",
        z_var_max_name: t.Optional[str] = "z_var_max",
        z_var_min_diff_thres: t.Optional[float ] = 0.2,
        ena_kl_trend_test: bool = False,  # New flag for KL trend test
        ena_recon_kl_test: bool = False,  # New flag for reconstruction-KL ratio test
        ena_diversity_test: bool = False,
    ):
        super().__init__()
        self.kl_history = []
        self.recon_history = []
        self.diversity_scores = []
        self._violation_count = 0

        #? Initialize parameters with validation
        assert kl_window > 1, "KL window must be at least 2 epochs"
        self.kl_window = kl_window
        self.recon_kl_ratio_threshold = recon_kl_ratio_threshold
        self.patience = patience

        self.ena_z_stat_test = ena_z_stat_test
        self.z_stat_test_min_epoch = z_stat_test_min_epoch
        self.z_mu_min_name = z_mu_min_name
        self.z_mu_max_name = z_mu_max_name
        self.z_mu_min_diff_thres = z_mu_min_diff_thres
        self.z_var_min_name = z_var_min_name
        self.z_var_max_name = z_var_max_name
        self.z_var_min_diff_thres = z_var_min_diff_thres

        self.ena_kl_trend_test = ena_kl_trend_test
        self.p_value_threshold = p_value_threshold

        self.ena_diversity_test = ena_diversity_test
        self.diversity_batch_size = diversity_batch_size

        self.ena_recon_kl_test = ena_recon_kl_test

    def on_train_epoch_end(self,
        trainer: L.Trainer,
        pl_module: L.LightningModule
    ) -> None:

        callback_metrics = trainer.callback_metrics
        current_epoch = trainer.current_epoch

        current_kl = callback_metrics.get('train/kl_loss').detach().cpu().numpy()
        current_recon = callback_metrics.get('train/recon_loss').detach().cpu().numpy()

        # Store metrics with exponential smoothing
        self.kl_history.append(current_kl.item() if current_kl else float('nan'))
        self.recon_history.append(current_recon.item() if current_recon else float('nan'))

        # Check for sufficient history
        if len(self.kl_history) < self.kl_window:
            return

        violations = False
        if self.ena_z_stat_test and current_epoch >= self.z_stat_test_min_epoch:
            is_z_stat_collapse = self._detect_z_stat_collapse(callback_metrics)
            violations += is_z_stat_collapse
        else:
            is_z_stat_collapse = False

        if self.ena_kl_trend_test:
            kl_trend = self._detect_kl_collapse()
            violations += kl_trend
        else:
            kl_trend = False

        if self.ena_recon_kl_test:
            ratio_violation = self._check_recon_kl_ratio()
            violations += ratio_violation
        else:
            ratio_violation = False

        if self.ena_diversity_test:
            diversity_violation = self._check_sample_diversity(pl_module)
            violations += diversity_violation
        else:
            diversity_violation = False

        if violations > 0:
            self._violation_count += 1
            if self._violation_count >= self.patience:
                self._raise_collapse_error(
                    callback_metrics,
                    is_z_stat_collapse=is_z_stat_collapse,
                    kl_trend=kl_trend,
                    ratio_violation=ratio_violation,
                    diversity_violation=diversity_violation
                )
        else:
            self._violation_count = max(0, self._violation_count - 1)

    def _detect_z_stat_collapse(self, callback_metrics) -> bool:
        """Checks latent variable statistics for signs of mode collapse."""
        #? Get latent variable statistics from callback metrics
        z_mu_min = callback_metrics[self.z_mu_min_name].detach().cpu().numpy()
        z_mu_max = callback_metrics[self.z_mu_max_name].detach().cpu().numpy()
        z_var_min = callback_metrics[self.z_var_min_name].detach().cpu().numpy()
        z_var_max = callback_metrics[self.z_var_max_name].detach().cpu().numpy()

        #? Calculate key metrics
        z_mu_diff = z_mu_max - z_mu_min
        z_var_diff = z_var_max - z_var_min

        #? Check for low diversity in latent means
        if z_mu_diff < self.z_mu_min_diff_thres:
            return True

        #? Check for low variance in latent variables
        if z_var_diff < self.z_var_min_diff_thres:
            return True

        return False

    def _detect_kl_collapse(self) -> bool:
        """Detects KL collapse using statistical trend analysis."""
        recent_kl = np.array(self.kl_history[-self.kl_window:])
        slope, _, p_value = self._linear_trend_test(recent_kl)
        return slope < 0 and p_value < self.p_value_threshold

    def _check_recon_kl_ratio(self) -> bool:
        """Checks if reconstruction loss dominates KL loss unnaturally."""
        recent_kl = np.nanmean(self.kl_history[-self.kl_window:])
        recent_recon = np.nanmean(self.recon_history[-self.kl_window:])

        if recent_kl < 1e-8:  # Avoid division by zero
            return True

        ratio = recent_recon / recent_kl
        return ratio > self.recon_kl_ratio_threshold

    def _check_sample_diversity(self, pl_module) -> bool:
        """Performs statistical test on generated samples."""
        try:
            with torch.no_grad():
                # Generate two independent batches
                samples1 = pl_module.sample(self.diversity_batch_size)
                samples2 = pl_module.sample(self.diversity_batch_size)

                if self.ena_diversity_test:
                    return self._permutation_test(samples1, samples2)
                else:
                    return self._distance_test(samples1, samples2)
        except AttributeError:
            pl_module.print("Warning: sample() method not found - skipping diversity check")
            return False

    def _permutation_test(self, samples1: torch.Tensor, samples2: torch.Tensor) -> bool:
        """Non-parametric permutation test for distribution similarity."""
        combined = torch.cat([samples1, samples2])
        n = len(combined)

        # Calculate original distance
        original_dist = torch.cdist(samples1, samples2).mean()

        # Permutation test
        num_permutations = 100
        extreme_count = 0

        for _ in range(num_permutations):
            permuted = combined[torch.randperm(n)]
            p1 = permuted[:len(samples1)]
            p2 = permuted[len(samples1):]
            perm_dist = torch.cdist(p1, p2).mean()

            if perm_dist <= original_dist:
                extreme_count += 1

        p_value = extreme_count / num_permutations
        return p_value < self.p_value_threshold

    def _linear_trend_test(self, data: np.ndarray) -> tuple[float, float, float]:
        """Calculates linear trend significance using Theil-Sen estimator."""
        x = np.arange(len(data))
        slope = stats.theilslopes(data, x)[0]
        _, p_value = stats.kendalltau(x, data)
        return slope, 0.0, p_value

    def _raise_collapse_error(self,
        callback_metrics,
        is_z_stat_collapse: bool,
        kl_trend: bool,
        ratio_violation: bool,
        diversity_violation: bool
    ) -> None:
        """Raises an error with details about the detected mode collapse."""
        error_msg = "Mode collapse detected:\n"
        error_details = {}

        if is_z_stat_collapse:
            z_mu_min = float(callback_metrics[self.z_mu_min_name].detach().cpu().numpy())
            z_mu_max = float(callback_metrics[self.z_mu_max_name].detach().cpu().numpy())
            z_var_min = float(callback_metrics[self.z_var_min_name].detach().cpu().numpy())
            z_var_max = float(callback_metrics[self.z_var_max_name].detach().cpu().numpy())
            z_mu_diff = z_mu_max - z_mu_min
            z_var_diff = z_var_max - z_var_min
        
            error_msg += "- Latent variable statistics indicate potential collapse\n"
            error_details["latent_stats"] = {
                self.z_mu_min_name: z_mu_min,
                self.z_mu_max_name: z_mu_max,
                self.z_var_min_name: z_var_min,
                self.z_var_max_name: z_var_max,
                "z_mu_diff": z_mu_diff,
                "z_var_diff": z_var_diff
            }
        if kl_trend:
            error_msg += "- Significant negative trend in KL divergence\n"
            error_details["kl_trend"] = {
                "window_size": self.kl_window,
                "kl_values": self.kl_history[-self.kl_window:]
            }
        if ratio_violation:
            error_msg += "- Abnormally high reconstruction loss to KL loss ratio\n"
            error_details["recon_kl_ratio"] = {
                "recon_loss": np.nanmean(self.recon_history[-self.kl_window:]),
                "kl_loss": np.nanmean(self.kl_history[-self.kl_window:]),
                "threshold": self.recon_kl_ratio_threshold
            }
        if diversity_violation:
            error_msg += "- Low sample diversity detected\n"
            error_details["diversity"] = {
                "batch_size": self.diversity_batch_size,
                "p_value_threshold": self.p_value_threshold
            }

        error_msg += f"Violations persisted for {self.patience} consecutive checks."

        raise VAEModeCollapseError(error_msg, error_details)

    @staticmethod
    def _distance_test(samples1: torch.Tensor, samples2: torch.Tensor) -> bool:
        """Distance-based diversity test using Wasserstein distance."""
        # Flatten samples for distance calculation
        flat1 = samples1.flatten(start_dim=1)
        flat2 = samples2.flatten(start_dim=1)

        # Compute Wasserstein distance
        dist = torch.mean(torch.norm(flat1[:, None] - flat2[None, :], dim=-1))
        return dist.item() < 1e-3  # Threshold should be dataset-dependent