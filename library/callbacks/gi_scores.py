"""
A Lightning Callback to compute and log Genetic Interaction (GI) scores.

This callback orchestrates a multi-step evaluation protocol at the end of the
test phase by calling specialized functions from the `gi_score` module.
"""
# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import logging
import typing as t

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import lightning as L
from torch.utils.data import DataLoader

# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
from ..datasets import ControlPerturbDataModule
from ..metrics import (
    comp_gt_gi_scores,
    comp_pred_gi_scores,
    compute_gi_scores,
)

# =============================================================================
# CALLBACK DEFINITION
# =============================================================================

class GIScoreCallback(L.Callback):
    """
    A Lightning Callback to compute Genetic Interaction (GI) scores post-training.

    This callback activates at the end of the test epoch. It uses imported
    functions to perform a series of prediction steps, calculate interaction
    scores, and compare them to the ground truth to derive a final precision metric.
    """

    def __init__(
        self,
        #? --- Data Sources ---
        datamodule: ControlPerturbDataModule,
        #? --- Configuration ---
        batch_size: int| None = None,
        k: int = 10,
        perc: float = 0.75,
        #? --- Logging Control ---
        log_prog_bar: bool = True,
        metric_prefix: str = "gi_precision",
        verbose: bool = False,
    ):
        """
        Initializes the GIScoreCallback.

        Parameters
        ----------
        datamodule : ControlPerturbDataModule
            The datamodule containing the necessary data, such as control samples
            and perturbation mappings.
        batch_size : int
            The batch size to use for the internal prediction dataloaders.
        k : int, default 10
            The number of top predictions to consider for precision calculation.
        perc : float, default 0.75
            The percentile used to set the threshold for identifying true positives.
        log_prog_bar : bool, default True
            If True, displays the final computed metrics on the progress bar.
        metric_prefix : str, default "gi_precision"
            The prefix for the names of the logged metrics.
        verbose : bool, default True
            If True, enables detailed logging messages during execution.
        """
        super().__init__()
        self.datamodule = datamodule
        self.test_dataloader = datamodule.test_dataloader()
        self.batch_size = batch_size if batch_size is not None else self.test_dataloader.batch_size
        self.k = k
        self.perc = perc
        self.log_prog_bar = log_prog_bar
        self.metric_prefix = metric_prefix
        self.verbose = verbose

        self.gt_gi_scores_df = comp_gt_gi_scores(self.datamodule.cond_gene_exp_data)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Reset results at the beginning of a test run."""
        self.results = None

    def on_test_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ):
        """
        Hook to run the GI score protocol at the end of the test epoch.

        This method orchestrates the full GI score evaluation pipeline:
        1. Computes ground truth scores from the provided datamodule.
        2. Computes predicted scores by running new prediction loops with the
           fully trained model.
        3. Compares the two sets of scores to calculate final precision metrics.
        4. Logs the final metrics to the Lightning logger.

        Parameters
        ----------
        trainer : L.Trainer
            The Lightning Trainer instance.
        pl_module : L.LightningModule
            The LightningModule that was tested.
        """
        if self.verbose:
            logging.info("--- Starting GI Score Callback ---")
        try:
            #? --- Pre-computation checks ---
            #? Ensure that the necessary data sources were provided during initialization.
            if self.datamodule is None or self.test_dataloader is None:
                if self.verbose:
                    logging.warning(
                        "Datamodule or test_dataloader not provided to GIScoreCallback. "
                        "Skipping GI score computation."
                    )
                return

            #? --- Step 1: Compute ground truth scores from the datamodule ---
            if self.verbose:
                logging.info("Calculating ground truth GI scores...")
            

            #? --- Step 2: Compute predicted scores using the trained model ---
            if self.verbose:
                logging.info("Calculating predicted GI scores from model outputs...")
            pred_gi_scores_df = comp_pred_gi_scores(
                trainer=trainer,
                model=pl_module,
                datamodule=self.datamodule,
                test_dl=self.test_dataloader,
                batch_size=self.batch_size,
            )

            #? --- Step 3: Compare predictions to ground truth ---
            if self.verbose:
                logging.info("Computing final precision scores...")
            final_scores = compute_gi_scores(
                gt_gi_scores_df=self.gt_gi_scores_df,
                pred_gi_scores_df=pred_gi_scores_df,
                k=self.k,
                perc=self.perc,
            )
            final_scores = {k.split("_")[0]: v for k,v in final_scores.items()}

            self.results = final_scores

            if self.verbose:
                logging.info(
                    f"--- GI Score Callback Finished. Final scores: {final_scores} ---"
                )

        except Exception as e:
            #? Catch any exception during the complex GI score pipeline to
            #? prevent it from crashing the entire testing process.
            if self.verbose:
                logging.error(
                    f"GI Score callback failed with an unexpected error: {e}",
                    exc_info=True,
                )

