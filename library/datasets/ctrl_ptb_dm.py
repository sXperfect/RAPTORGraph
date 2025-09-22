import typing as t
import random
import hashlib
import json
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import ot
import lightning as L
import torch
from scipy.sparse import spmatrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler, Subset


# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
from ..consts import TestSet
from ..structs import (
    BaseIntEnum,
    BaseStrEnum,
    ConditionedGeneExpressionData,
)

from .perturb_data_handler import (
    process_perturb_adata,
)
from .tensor_dict import TensorDictDataset

# =============================================================================
# CONSTANTS & ENUMS
# =============================================================================
DEFAULT_STRAT_PERTRUBS = [
    "ETS2", "SGK1", "POU3F2", "TBX2", "CBL", "MAPK1",
    "CDKN1C", "S1PR2", "PTPN1", "MAP2K6", "COL1A1",
]

class CellPairingStrategy(BaseIntEnum):
    """Determines the pairing strategy between control and perturbed cells."""
    FIND_CONTROL = 0  #? For each perturbed cell, find the best-matching control cell.
    FIND_PERTURB = 1  #? For each control cell, find the best-matching perturbed cell.

class DataMode(BaseStrEnum):
    """Determines the type of dataset to be created."""
    CONTROL_PERTURB = "control_perturb"

def generate_cache_filename(
    mode: str,
    val_size: float,
    test_size: float | None,
    train_val_pairing_mode: CellPairingStrategy | None,
    test_pairing_mode: CellPairingStrategy | None,
    trainval_add_identity_pairs: bool,
    test_add_identity_pairs: bool,
    split_unseen_datasets: bool = False,
    sort_by_perturbation_status: bool= True,
    cache_version: str = "1.0",
) -> str:
    mode = DataMode(mode)
    train_val_pairing_mode = CellPairingStrategy(train_val_pairing_mode) if train_val_pairing_mode is not None else None
    test_pairing_mode = CellPairingStrategy(test_pairing_mode) if test_pairing_mode is not None else None

    if f"v{cache_version}" == "v1.0":
        train_val_pairing_mode = CellPairingStrategy(train_val_pairing_mode) if train_val_pairing_mode is not None else None
        test_pairing_mode = CellPairingStrategy(test_pairing_mode) if test_pairing_mode is not None else None

        params = {
            "mode": mode,
            "val_size": val_size,
            "test_size": test_size,
            "train_val_pairing_mode": train_val_pairing_mode.name if train_val_pairing_mode is not None else None,
            "test_pairing_mode": test_pairing_mode.name if test_pairing_mode is not None else None,
            "trainval_add_identity_pairs": trainval_add_identity_pairs,
            "test_add_identity_pairs": test_add_identity_pairs,
            "split_unseen_datasets": split_unseen_datasets,
            "ot_numItermax": 1e6,
            "ot_numThreads": 16,
            "cache_version": cache_version,
            "sort_by_perturbation_status": str(sort_by_perturbation_status),
        }

        param_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        return f"ctrl_ptb_dm-{param_hash}"

    else:
        raise ValueError(f"Invalid cache version: {cache_version}")

# ----------------------------------------------------------------------
# LABEL‑BALANCED BATCH SAMPLER
# ----------------------------------------------------------------------
class LabelBalancedBatchSampler(Sampler[t.List[int]]):
    """
    Returns batches that contain samples of a single label while shuffling the
    order of the batches across labels. Works with a plain Dataset **or** with a
    ``torch.utils.data.Subset``. When a Subset is supplied the sampler yields
    *local* indices (the positions inside the subset) so that
    ``Subset.__getitem__`` can resolve them correctly.
    """

    def __init__(
        self,
        dataset: t.Any,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        dataset : Dataset or Subset
            The dataset to sample from.
        batch_size : int
            Number of samples per batch.
        drop_last : bool, optional
            Whether to drop the last incomplete batch. Default is False.
        shuffle : bool, optional
            Whether to shuffle samples within each label and batches across labels.
            Default is True.
        """
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        #? Resolve the underlying dataset and the list of *effective* indices
        if isinstance(dataset, Subset):
            #? ``dataset`` is a Subset → keep a reference to the parent and the
            #? indices that belong to the subset.
            self.parent_dataset = dataset.dataset
            self.effective_indices = np.asarray(dataset.indices, dtype=int)
            #? Mapping: global index → position inside the subset
            self._global_to_local = {
                g: i for i, g in enumerate(self.effective_indices)
            }
        else:
            #? Plain dataset
            self.parent_dataset = dataset
            self.effective_indices = np.arange(len(dataset), dtype=int)
            self._global_to_local = {
                g: i for i, g in enumerate(self.effective_indices)
            }

        #? Verify that the parent dataset provides ``ptb_label``
        if not hasattr(self.parent_dataset, "ptb_label"):
            raise AttributeError(
                "The underlying dataset must have a 'ptb_label' attribute."
            )

        #? Build a mapping: label → list of *global* indices that belong to the
        #? subset (or to the whole dataset if no Subset was given)
        labels = np.asarray(self.parent_dataset.ptb_label)[self.effective_indices]

        self.label_to_global: t.Dict[int, t.List[int]] = defaultdict(list)
        for g_idx, lbl in zip(self.effective_indices, labels):
            self.label_to_global[int(lbl)].append(int(g_idx))

        #? Convert global indices to *local* (subset) indices and create batches
        self.batches_per_label: t.List[t.List[int]] = []
        for lbl, global_idx_list in self.label_to_global.items():
            if self.shuffle:
                random.shuffle(global_idx_list)

            # Full batches
            n_full = len(global_idx_list) // self.batch_size
            for i in range(n_full):
                start = i * self.batch_size
                batch_global = global_idx_list[start : start + self.batch_size]

                # Translate to local indices that the Subset expects
                batch_local = [
                    self._global_to_local[g] for g in batch_global
                ]
                self.batches_per_label.append(batch_local)

            # Possible leftover batch
            leftover = len(global_idx_list) % self.batch_size
            if leftover and not self.drop_last:
                batch_global = global_idx_list[-leftover:]
                batch_local = [
                    self._global_to_local[g] for g in batch_global
                ]
                self.batches_per_label.append(batch_local)

        #? Randomise the *order* of the batches across labels
        random.shuffle(self.batches_per_label)

    def __iter__(self) -> t.Iterator[t.List[int]]:
        """Yield batches in the pre‑shuffled order.

        Returns
        -------
        Iterator[List[int]]
            Iterator over batches of local indices.
        """
        return iter(self.batches_per_label)

    def __len__(self) -> int:
        """Total number of batches that will be produced.

        Returns
        -------
        int
            Number of batches.
        """
        return len(self.batches_per_label)

class ControlPerturbDataset(Dataset):
    """
    A PyTorch Dataset for single-cell control and perturbation data.

    This class holds control and perturbed expression profiles and maps them
    to perturbation identities using optimal transport pairing.
    """

    def __init__(
        self,
        perturbed_expression: spmatrix | np.ndarray,
        perturb_gene_names_per_sample: np.ndarray,
        control_expression: spmatrix | np.ndarray,
        perturb_label: np.ndarray,
        perturb_matrix: np.ndarray,
        pairing_mode: CellPairingStrategy | int | str | None = CellPairingStrategy.FIND_CONTROL,
        add_identity_pairs: bool = False,
        use_double_precision: bool = True,
        #? --- OT Parameters ---
        ot_num_iter_max: float = 1e6,
        ot_num_threads: int | None = None,
        #? --- Output format config ---
        return_dict: bool = False,
    ):
        """
        Parameters
        ----------
        perturbed_expression : spmatrix | np.ndarray
            Expression matrix of perturbed cells (samples x genes).
        perturb_gene_names_per_sample : np.ndarray
            Array of perturbation gene names for each perturbed sample.
        control_expression : spmatrix | np.ndarray
            Expression matrix of control (unperturbed) cells.
        perturb_label : np.ndarray
            Integer label for each perturbed sample.
        perturb_matrix : np.ndarray
            Binary matrix (samples x perturbation genes) indicating which
            perturbations are present in each sample.
        pairing_mode : CellPairingStrategy | int | str | None, optional
            Strategy for pairing control and perturbed cells.  ``None`` disables
            pairing and falls back to random subsampling.  Default is
            ``CellPairingStrategy.FIND_CONTROL``.
        add_identity_pairs : bool, optional
            If ``True`` and ``pairing_mode`` is ``FIND_CONTROL``, unpaired
            control cells are added as identity pairs (control → control).  Default
            is ``False``.
        use_double_precision : bool, optional
            Whether to store tensors as ``torch.double`` (True) or ``torch.float``.
            Default is ``True``.
        ot_num_iter_max : float, optional
            Maximum number of iterations for the optimal transport solver. Default is 1e6.
        ot_num_threads : int | None, optional
            Number of threads to use for optimal transport computation. If None,
            automatically calculates based on available CPU cores. Default is None.
        return_dict : bool, optional
            If ``True`` each ``__getitem__`` call returns a dictionary instead of a
            list/tuple.  Default is ``False``.
        """
        super().__init__()
        self.use_double_precision = use_double_precision
        self.dtype = torch.double if use_double_precision else torch.float
        self.ot_num_iter_max = ot_num_iter_max

        #? Calculate optimal OT thread configuration
        total_cpu_cores = cpu_count()

        #? If ot_num_threads is not specified, set a reasonable default
        if ot_num_threads is None:
            # Default: use 4 threads per OT computation
            self.ot_num_threads = 4
        else:
            self.ot_num_threads = ot_num_threads

        #? Calculate number of processes based on OT threads
        #? This ensures we don't oversubscribe CPU resources:
        #? num_processes * ot_num_threads <= total_cpu_cores
        self.num_processes = max(1, total_cpu_cores // self.ot_num_threads)

        self.total_cpu_cores = total_cpu_cores

        #? Convert int to PairingMode enum if needed
        if isinstance(pairing_mode, int):
            pairing_mode = CellPairingStrategy(pairing_mode)

        self.ctrl_samples = control_expression
        self.ptb_samples = perturbed_expression
        self.ptb_names = perturb_gene_names_per_sample
        self.ptb_label = perturb_label
        self.ptb_matrix = perturb_matrix
        self.return_dict = return_dict

        self.cond_ids = (
            pd.DataFrame(np.argwhere(perturb_matrix))
            .groupby(0)
            .apply(lambda x: list(x[1].to_numpy()))
        )
        unique_perturb_label = np.unique(perturb_label)

        if pairing_mode is not None and pairing_mode in [CellPairingStrategy.FIND_CONTROL, CellPairingStrategy.FIND_PERTURB]:
            #? Parallel processing of each perturb label using multiprocessing
            #? Now using calculated num_processes instead of hard-coded cpu_count()//4
            with Pool(processes=self.num_processes) as pool:
                ids_list = pool.map(
                    self._pairing,
                    [
                        (control_expression, perturbed_expression[perturb_label == v], perturb_label == v, pairing_mode) for v in unique_perturb_label
                    ]
                )

            if pairing_mode == CellPairingStrategy.FIND_CONTROL:
                self.ctrl_ids = np.concatenate(ids_list)

                if add_identity_pairs:
                    mask = np.ones(self.ctrl_samples.shape[0], dtype=bool)
                    mask[np.unique(self.ctrl_ids)] = 0

                    num_new_samples = mask.sum()
                    additional_ctrl_ids = np.argwhere(mask).flatten()
                    unpaired_ctrl_samples = self.ctrl_samples[mask, :]
                    new_ptb_label = np.zeros(num_new_samples, dtype=self.ptb_label.dtype)
                    new_ptb_matrix = np.zeros((num_new_samples, self.ptb_matrix.shape[1]), dtype=self.ptb_matrix.dtype)
                    #TODO: ptb_names
                    # new_ptb_names = self.ptb_names[:num_new_samples].copy()
                    # new_ptb_names

                    self.ctrl_ids = np.concatenate([self.ctrl_ids, additional_ctrl_ids])
                    self.ptb_samples = np.concatenate([self.ptb_samples, unpaired_ctrl_samples], axis=0) #? Include control as "output"
                    self.ptb_label = np.concatenate([self.ptb_label, new_ptb_label])
                    self.ptb_matrix = np.concatenate([self.ptb_matrix, new_ptb_matrix], axis=0)
                    #TODO: ptb_names
                    # self.ptb_names = np.concat([self.ptb_names, new_ptb_names])

                self.ptb_ids = np.arange(self.ptb_samples.shape[0])

            elif pairing_mode == CellPairingStrategy.FIND_PERTURB:
                self.ptb_ids = np.concatenate(ids_list)
                self.ctrl_ids = np.repeat(np.arange(self.ctrl_samples.shape[0]).reshape(1, -1), perturb_matrix.shape[1], axis=0).flatten()

        else:
            #? Subsample control cells to match the number of perturbed cells
            if control_expression.shape[0] > 0 and self.ptb_samples.shape[0] > 0:
                #? Use np.random, which is seeded by lightning.seed_everything()
                rand_indices = np.random.choice(
                    control_expression.shape[0],
                    self.ptb_samples.shape[0],
                    replace=True,
                )
                self.ctrl_ids = rand_indices
                self.ptb_ids = np.arange(perturb_matrix.shape[0])
            else:
                self.ctrl_samples = np.array([])

    def _pairing(self, args):
        """
        Compute optimal-transport pairing for a single perturbation label.

        Parameters
        ----------
        args : tuple
            ``(control_expression, curr_perturbed_expression, mask, pairing_mode)``

        Returns
        -------
        np.ndarray
            Array of paired indices (control or perturbed depending on
            ``pairing_mode``).
        """
        control_expression, curr_perturbed_expression, mask, pairing_mode = args
        M = ot.dist(control_expression, curr_perturbed_expression)
        normalized_ctrl_exps = np.ones(control_expression.shape[0]) / control_expression.shape[0]
        normalized_pert_exps = np.ones(curr_perturbed_expression.shape[0]) / curr_perturbed_expression.shape[0]
        gamma = ot.emd(
            normalized_ctrl_exps,
            normalized_pert_exps,
            M,
            numItermax=self.ot_num_iter_max,
            numThreads=self.ot_num_threads
        )

        #? Convert enum to int for numpy operations
        axis = int(pairing_mode)

        if pairing_mode == CellPairingStrategy.FIND_CONTROL:
            ctrl_ids = np.argmax(gamma, axis=axis)  #? 0: pair based on perturb find control, 1: pair based on control, find perturb
            return ctrl_ids

        elif pairing_mode == CellPairingStrategy.FIND_PERTURB:
            rel_ptb_ids = np.argmax(gamma, axis=axis)  #? 0: pair based on perturb find control, 1: pair based on control, find perturb
            ptb_ids = np.argwhere(mask).flatten()[rel_ptb_ids]
            return ptb_ids

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return self.ctrl_ids.shape[0]

    def __getitem__(self,
        item: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        """
        Retrieves a single data point.

        Parameters
        ----------
        item : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]
            If return_dict is False, returns a tuple containing:
            - X: Control cell expression profile.
            - Y: Perturbed cell expression profile.
            - label: Perturbation embedding indices.
            If return_dict is True, returns a dictionary with keys 'X', 'Y', and 'label'.
        """
        ctrl_id = self.ctrl_ids[item]
        ptb_id = self.ptb_ids[item]
        label = self.ptb_label[ptb_id]
        ptb_vec = self.ptb_matrix[ptb_id].astype(np.float32)

        pert_gene_ids = np.argwhere(ptb_vec).flatten()
        num_int = len(pert_gene_ids)

        c1 = np.zeros_like(ptb_vec)
        if num_int > 0:
            c1[pert_gene_ids[0]] = 1

        c2 = np.zeros_like(ptb_vec)
        if num_int > 1:
            c2[pert_gene_ids[1]] = 1
        else:
            c2 = c1

        ctrl_exps = self.ctrl_samples[ctrl_id, :]
        perturb_exps = self.ptb_samples[ptb_id, :]

        #? Convert to dense array if it's sparse
        if isinstance(ctrl_exps, spmatrix):
            ctrl_exps = ctrl_exps.toarray()
        if isinstance(perturb_exps, spmatrix):
            perturb_exps = perturb_exps.toarray()

        ctrl_exps = torch.from_numpy(ctrl_exps.flatten())
        perturb_exps = torch.from_numpy(perturb_exps.flatten())
        label = torch.tensor(label, dtype=torch.long).flatten()
        ptb_vec = torch.from_numpy(ptb_vec)
        c1 = torch.from_numpy(c1)
        c2 = torch.from_numpy(c2)

        out_dict =  {
            "X": ctrl_exps.to(self.dtype),
            "Y": perturb_exps.to(self.dtype),
            "label": label.to(int),
            "ptb_vec": ptb_vec.to(self.dtype)
        }
        out_dict["c1"] = c1.to(self.dtype)
        out_dict["c2"] = c2.to(self.dtype)
        out_dict["num_int"] = num_int

        return out_dict

# =============================================================================
# PYTORCH LIGHTNING DATAMODULE
# =============================================================================
class ControlPerturbDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for perturbation-response data.

    This class handles all stages of data loading and processing, from reading
    the H5AD file to creating DataLoaders for training, validation, and testing.
    """

    def __init__(
        self,
        #? --- Data & Mode Configuration ---
        cond_gene_exp_data: ConditionedGeneExpressionData,
        mode: DataMode | str = 'control_perturb',
        #? --- Splitting & Pairing ---
        val_size: float = 0.1,
        test_size: float | None = None,
        val_stratification_perturbs: list[str] | None = DEFAULT_STRAT_PERTRUBS,
        trainval_pairing_mode: CellPairingStrategy | int | None = CellPairingStrategy.FIND_CONTROL,
        test_pairing_mode: CellPairingStrategy | int | None = CellPairingStrategy.FIND_CONTROL,
        trainval_add_identity_pairs: bool = True,
        test_add_identity_pairs: bool = False,
        split_unseen_datasets: bool = False,
        #? --- DataLoader Configuration ---
        batch_size: int = 32,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = False,
        #? --- Output Format Configuration ---
        label_balanced: bool = False,
        train_label_balanced: bool | None = None,
        val_label_balanced: bool | None = None,
        test_label_balanced: bool | None = None,
    ):
        """
        Parameters
        ----------
        cond_gene_exp_data : ConditionedGeneExpressionData
            Processed gene-expression data object.
        mode : DataMode | str, optional
            ``'control_perturb'`` or ``'autoencoder'``.  Default is
            ``DataMode.CONTROL_PERTURB``.
        val_size : float, optional
            Fraction of the training data to use for validation.  Must be in
            ``(0, 1)``.  Default ``0.1``.
        test_size : float | None, optional
            Fraction of the data to reserve for testing when ``mode`` is
            ``CONTROL_PERTURB``.  If ``None`` an automatic split based on the
            number of single- vs. multi-perturbation samples is used.
        val_stratification_perturbs : list[str] | None, optional
            List of perturbations to stratify for the validation set.  If ``None``
            a random split is performed.  Default is ``DEFAULT_STRAT_PERTRUBS``.
        train_val_pairing_mode, test_pairing_mode : CellPairingStrategy | int | None, optional
            Pairing strategy for training/validation and test sets respectively.
        trainval_add_identity_pairs, test_add_identity_pairs : bool, optional
            Whether to add identity (control-control) pairs when using
            ``FIND_CONTROL`` pairing.
        split_unseen_datasets : bool, optional
            If ``True`` and ``mode`` is ``AUTOENCODER``, creates separate test
            datasets for each unseen perturbation.
        batch_size, val_batch_size, test_batch_size : int, optional
            DataLoader batch sizes.
        num_workers, shuffle, pin_memory, drop_last, persistent_workers : optional
            Standard DataLoader arguments.
        label_balanced : bool, optional
            If ``True`` use :class:`LabelBalancedBatchSampler` for training.
        """

        super().__init__()

        self.mode = DataMode(mode)
        self.cond_gene_exp_data = cond_gene_exp_data
        self.val_size = val_size
        self.test_size = test_size
        self.val_stratification_perturbs = val_stratification_perturbs
        self.trainval_pairing_mode = CellPairingStrategy(trainval_pairing_mode) if trainval_pairing_mode else None
        self.test_pairing_mode = CellPairingStrategy(test_pairing_mode) if test_pairing_mode else None
        self.trainval_add_identity_pairs = trainval_add_identity_pairs
        self.test_add_identity_pairs = test_add_identity_pairs
        self.split_unseen_datasets = split_unseen_datasets
        self.label_balanced = label_balanced
        self.train_label_balanced = train_label_balanced if train_label_balanced else label_balanced
        self.val_label_balanced = val_label_balanced if val_label_balanced else label_balanced
        self.test_label_balanced = test_label_balanced if test_label_balanced else label_balanced
        # self.return_dict = return_dict

        # #? --- Winsorizing attributes ---
        # if not (0.0 <= winsorize_min_percentile <= 1.0 and 0.0 <= winsorize_max_percentile <= 1.0):
        #     raise ValueError("Winsorizing percentiles must be between 0.0 and 1.0.")
        # if winsorize_min_percentile >= winsorize_max_percentile:
        #     raise ValueError("winsorize_min_percentile must be less than winsorize_max_percentile.")

        # self.ena_winsorzing = ena_winsorzing
        # self.winsorize_mode = WinsorizeMode(winsorize_mode)
        # self.winsorize_metric = WinsorizeMetric(winsorize_metric)
        # self.winsorize_min_percentile = winsorize_min_percentile
        # self.winsorize_max_percentile = winsorize_max_percentile
        # self.winsorizing_mask: torch.Tensor | None = None

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else max(1, self.batch_size // 4)
        self.test_batch_size = test_batch_size if test_batch_size is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._is_set = False
        self.curr_test_set = TestSet.SEEN

    @classmethod
    def from_adata(
        cls,
        adata_fpath: str,
        #? --- Data & Mode Configuration ---
        mode: DataMode | str = 'control_perturb',
        #? --- Splitting & Pairing ---
        val_size: float = 0.1,
        test_size: float | None = None,
        val_stratification_perturbs: list[str] | None = DEFAULT_STRAT_PERTRUBS,
        trainval_pairing_mode: CellPairingStrategy | int | None = CellPairingStrategy.FIND_CONTROL,
        test_pairing_mode: CellPairingStrategy | int | None = CellPairingStrategy.FIND_CONTROL,
        trainval_add_identity_pairs: bool = True,
        test_add_identity_pairs: bool = False,
        #? --- DataLoader Configuration ---
        batch_size: int = 32,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = False,
        #? --- Output Format Configuration ---
        label_balanced: bool = False,
        train_label_balanced: bool | None = None,
        val_label_balanced: bool | None = None,
        test_label_balanced: bool | None = None,
        **process_kwargs,
    ):
        """
        Creates the DataModule by processing a perturbation dataset from an H5AD file.
        """
        processed_data = process_perturb_adata(
            adata_fpath=adata_fpath,
            **process_kwargs
        )

        assert 0 < val_size < 1

        return cls(
            cond_gene_exp_data=processed_data,
            mode=mode,
            val_size=val_size,
            test_size=test_size,
            val_stratification_perturbs=val_stratification_perturbs,
            trainval_pairing_mode=trainval_pairing_mode,
            test_pairing_mode=test_pairing_mode,
            trainval_add_identity_pairs=trainval_add_identity_pairs,
            test_add_identity_pairs=test_add_identity_pairs,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            label_balanced=label_balanced,
            train_label_balanced=train_label_balanced,
            val_label_balanced=val_label_balanced,
            test_label_balanced=test_label_balanced,
        )

    @classmethod
    def from_cond_gene_exp_data(
        cls,
        cond_gene_exp_data: ConditionedGeneExpressionData,
        #? --- Data & Mode Configuration ---
        mode: DataMode | str = 'control_perturb',
        #? --- Splitting & Pairing ---
        val_size: float = 0.1,
        test_size: float | None = None,
        val_stratification_perturbs: list[str] | None = DEFAULT_STRAT_PERTRUBS,
        trainval_pairing_mode: CellPairingStrategy | int | None = CellPairingStrategy.FIND_CONTROL,
        test_pairing_mode: CellPairingStrategy | int | None = CellPairingStrategy.FIND_CONTROL,
        trainval_add_identity_pairs: bool = True,
        test_add_identity_pairs: bool = False,
        split_unseen_datasets: bool = False,
        #? --- DataLoader Configuration ---
        batch_size: int = 32,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = False,
        #? --- Output Format Configuration ---
        label_balanced: bool = False,
        train_label_balanced: bool | None = None,
        val_label_balanced: bool | None = None,
        test_label_balanced: bool | None = None,
    ):
        """
        Creates the DataModule directly from a ConditionedGeneExpressionData object.
        """
        assert 0 < val_size < 1, "val_size must be between 0 and 1"

        return cls(
            cond_gene_exp_data=cond_gene_exp_data,
            mode=mode,
            val_size=val_size,
            test_size=test_size,
            val_stratification_perturbs=val_stratification_perturbs,
            trainval_pairing_mode=trainval_pairing_mode,
            test_pairing_mode=test_pairing_mode,
            trainval_add_identity_pairs=trainval_add_identity_pairs,
            test_add_identity_pairs=test_add_identity_pairs,
            split_unseen_datasets=split_unseen_datasets,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            label_balanced=label_balanced,
            train_label_balanced=train_label_balanced,
            val_label_balanced=val_label_balanced,
            test_label_balanced=test_label_balanced,
        )

    @property
    def gene_ids(self):
        return self.cond_gene_exp_data.gene_ids

    @property
    def gene_names(self):
        """Get the gene names from the conditioned gene expression data.

        Returns
        -------
        list or array-like
            The gene names present in the dataset.
        """
        return self.cond_gene_exp_data.gene_names

    @property
    def num_genes(self) -> int:
        """
        Returns the number of genes in the dataset.

        Returns
        -------
        int
            The total number of genes.
        """
        return len(self.gene_names)

    @property
    def perturb_gene_names(self) -> list[str]:
        """
        Returns the list of unique perturbation names present in the dataset.

        Returns
        -------
        list[str]
            List of unique perturbation gene names.
        """
        return self.cond_gene_exp_data.perturb_gene_names

    @property
    def num_perturb_genes(self) -> int:
        """
        Returns the number of unique perturbations in the dataset.

        Returns
        -------
        int
            The number of unique perturbation genes.
        """
        return self.cond_gene_exp_data.num_perturb_genes

    @property
    def dim(self) -> int:
        """
        Returns the dimensionality of the gene expression data.

        Returns
        -------
        int
            The number of genes (dimensionality).
        """
        return self.num_genes

    @property
    def cdim(self) -> int:
        """
        Returns the dimensionality of the perturbation space.

        Returns
        -------
        int
            The number of perturbation genes (condition dimensionality).
        """
        return self.num_perturb_genes

    @property
    def ena_val_stratification(self):
        """
        Check if validation stratification is enabled.

        Returns
        -------
        bool
            True if validation stratification perturbations are specified, False otherwise.
        """
        return self.val_stratification_perturbs is not None
    
    @property
    def sort_by_perturbation_status(self):
        return self.cond_gene_exp_data.sort_by_perturbation_status

    @property
    def cache_filename(self) -> str:
        """
        Generate cache filename using the datamodule's configuration.

        Returns
        -------
        str
            Cache filename based on current configuration
        """
        return generate_cache_filename(
            mode=self.mode.value,
            val_size=self.val_size,
            test_size=self.test_size,
            train_val_pairing_mode=self.trainval_pairing_mode,
            test_pairing_mode=self.test_pairing_mode,
            trainval_add_identity_pairs=self.trainval_add_identity_pairs,
            test_add_identity_pairs=self.test_add_identity_pairs,
            split_unseen_datasets=self.split_unseen_datasets,
            sort_by_perturbation_status=self.sort_by_perturbation_status,
        )

    def setup(self, stage: str | None = None):
        """
        Data operations to perform on every GPU.
        Splits data into train, validation, and test sets and applies winsorizing.
        """
        if self._is_set:
            return

        data = self.cond_gene_exp_data
        n_perturbs = data.perturb_gene_names_per_sample.apply(len)

        perturb_label = data.perturb_label

        #? Masks for different perturbation cardinalities
        control_mask = (n_perturbs == 0).to_numpy()
        single_mask = (n_perturbs == 1).to_numpy()
        multi_mask = (n_perturbs > 1).to_numpy()

        #? Expression data
        expression_data = data.expression_data
        if isinstance(expression_data, spmatrix):
            expression_data = expression_data.toarray()

        num_control_exp = control_mask.sum()
        control_expression = expression_data[control_mask]
        single_expression = expression_data[single_mask]
        multi_expression = expression_data[multi_mask]

        #? Perturbation lists
        single_perturb_per_sample = data.perturb_gene_names_per_sample[single_mask]
        multi_perturb_per_sample = data.perturb_gene_names_per_sample[multi_mask]

        single_perturb_matrix = data.perturb_matrix[single_mask, :]
        single_perturb_label = perturb_label[single_mask]

        multi_perturb_matrix = data.perturb_matrix[multi_mask, :]
        multi_perturb_label = perturb_label[multi_mask]

        #? Split control expression for train/val and test datasets
        if self.test_size is not None:
            single_count = single_expression.shape[0]
            multi_count = multi_expression.shape[0]
            if single_count + multi_count > 0:
                automatic_test_size = multi_count / (single_count + multi_count)
            else:
                automatic_test_size = 0.0

            self.test_size = automatic_test_size

        trainval_control_ids, test_control_ids = train_test_split(
            np.arange(num_control_exp),
            test_size=self.test_size,
            random_state=None, #! RELIES ON GLOBAL SEED
        )

        trainval_control_expression = control_expression[trainval_control_ids, :]
        test_control_expression = control_expression[test_control_ids, :]


        #? --- Train/Validation Dataset Creation ---
        if self.mode == DataMode.CONTROL_PERTURB:
            trainval_dataset = ControlPerturbDataset(
                perturbed_expression=single_expression,
                perturb_gene_names_per_sample=single_perturb_per_sample.to_numpy(),
                control_expression=trainval_control_expression,
                perturb_label=single_perturb_label,
                perturb_matrix=single_perturb_matrix,
                pairing_mode=self.trainval_pairing_mode,
                add_identity_pairs=self.trainval_add_identity_pairs,
            )

            all_indices = np.arange(len(trainval_dataset))
            val_ids = np.array([], dtype=int)

            #? If a list of perturbations is provided, perform partial stratification
            assert 0 < self.val_size < 1

            if self.ena_val_stratification:
                val_indices_list = []

                #? Create a map from perturbation name to indices
                ptb_to_indices = defaultdict(list)
                for i, ptb_list in enumerate(trainval_dataset.ptb_names):
                    if len(ptb_list) == 1: #? Only consider single perturbations
                        ptb_to_indices[ptb_list[0]].append(all_indices[i])

                #? For each specified perturbation, sample for the validation set
                for ptb_name in self.val_stratification_perturbs:
                    indices_for_ptb = ptb_to_indices.get(ptb_name, [])
                    if len(indices_for_ptb) > 1:
                        val_count = int(len(indices_for_ptb) * self.val_size)
                        if val_count > 0:
                            chosen_val_indices = np.random.choice(
                                indices_for_ptb, size=val_count, replace=False
                            )
                            val_indices_list.extend(chosen_val_indices)

                if val_indices_list:
                    val_ids = np.unique(val_indices_list)

            #? If not stratifying, perform a simple random split
            else:
                train_ids, val_ids = train_test_split(
                    all_indices,
                    test_size=self.val_size,
                    random_state=None, #! RELIES ON GLOBAL SEED
                )

            train_ids = np.setdiff1d(all_indices, val_ids, assume_unique=True)

            self.train_dataset = Subset(trainval_dataset, train_ids)
            self.val_dataset = Subset(trainval_dataset, val_ids)

            #? --- Test Dataset Creation ---
            self.test_dataset = ControlPerturbDataset(
                perturbed_expression=multi_expression,
                perturb_gene_names_per_sample=multi_perturb_per_sample.to_numpy(),
                control_expression=test_control_expression,
                perturb_label=multi_perturb_label,
                perturb_matrix=multi_perturb_matrix,
                pairing_mode=self.test_pairing_mode,
                add_identity_pairs=self.test_add_identity_pairs,
            )

        elif self.mode == DataMode.AUTOENCODER:
            trainval_expression = np.concatenate([
                trainval_control_expression,
                single_expression
            ], axis=0)

            # if self.ena_winsorzing:
            #     target_expression = trainval_expression

            #     if self.winsorize_mode == WinsorizeMode.ALL:
            #         genes_to_consider = target_expression
            #     elif self.winsorize_mode == WinsorizeMode.NON_PERTURBED_ONLY:
            #         genes_to_consider = target_expression[:, self.num_perturb_genes:]

            #     #? Calculate the chosen metric for the target genes.
            #     if self.winsorize_metric == WinsorizeMetric.SPARSITY:
            #         metric_values = (genes_to_consider > 0).mean(axis=0)
            #     elif self.winsorize_metric == WinsorizeMetric.MEAN:
            #         metric_values = genes_to_consider.mean(axis=0)

            #     #? Calculate percentile thresholds from the metric values.
            #     #? The input percentiles are 0-1, but np.percentile expects 0-100.
            #     min_threshold = np.percentile(metric_values, self.winsorize_min_percentile * 100)
            #     max_threshold = np.percentile(metric_values, self.winsorize_max_percentile * 100)

            #     #? Create a boolean mask based on these thresholds.
            #     metric_mask = (metric_values >= min_threshold) & (metric_values <= max_threshold)

            #     if self.winsorize_mode == WinsorizeMode.ALL:
            #         self.winsorizing_mask = torch.from_numpy(metric_mask)
            #     else: #? NON_PERTURBED_ONLY
            #         #? Build the full mask: perturbed genes are always included (True).
            #         full_mask = np.ones(self.num_genes, dtype=bool)
            #         full_mask[self.num_perturb_genes:] = metric_mask
            #         self.winsorizing_mask = torch.from_numpy(full_mask)
            # else:
            #     self.winsorizing_mask = torch.from_numpy(np.ones(self.num_genes, dtype=bool))

            trainval_perturb_label = np.concatenate([
                perturb_label[control_mask][trainval_control_ids],
                single_perturb_label,
            ])

            trainval_dataset = TensorDictDataset({
                "X": torch.from_numpy(trainval_expression),
                "label": torch.from_numpy(trainval_perturb_label),
            })

            trainval_num_data = len(trainval_perturb_label)

            if self.ena_val_stratification:
                raise NotImplementedError("Val stratification is not implemented for AUTOENCDDER mode!")
            else:
                train_ids, val_ids = train_test_split(
                    np.arange(trainval_num_data),
                    test_size=self.val_size,
                    random_state=None, #! RELIES ON GLOBAL SEED
                    stratify=trainval_perturb_label,
                )

            self.train_dataset = Subset(trainval_dataset, train_ids)
            self.val_dataset = Subset(trainval_dataset, val_ids)

            self._test_seen_dataset = TensorDictDataset({
                "X": torch.from_numpy(test_control_expression),
                "label" : torch.from_numpy(perturb_label[control_mask][test_control_ids])
            })

            if self.split_unseen_datasets:
                self._test_unseen_dataset = []
                unseen_label_values = np.unique(multi_perturb_label)
                for l in unseen_label_values:
                    mask = multi_perturb_label == l

                    _test_unseen_dataset = TensorDictDataset({
                        "X": torch.from_numpy(multi_expression[mask, :]),
                        "label" : torch.from_numpy(multi_perturb_label[mask])
                    })

                    self._test_unseen_dataset.append(_test_unseen_dataset)

            else:
                self._test_unseen_dataset = TensorDictDataset({
                    "X": torch.from_numpy(multi_expression),
                    "label" : torch.from_numpy(multi_perturb_label)
                })

            self.test_dataset = self._test_seen_dataset

        self._is_set = True
        self.curr_test_set = TestSet.SEEN

    def swap_test(
        self,
        test_set: str | TestSet | None = None
    ):
        """Swaps the current test dataset between seen and unseen perturbations.

        Parameters
        ----------
        test_set : str or TestSet, optional
            The desired test set to switch to ('seen' or 'unseen'). If None,
            it toggles to the other set.
        """
        assert self.mode == DataMode.AUTOENCODER, \
            "swap_test only available for DataMode AUTOENCODER!"

        is_unseen = (test_set is None and self.curr_test_set == TestSet.SEEN) or \
                     test_set == TestSet.UNSEEN

        if is_unseen:
            self.curr_test_set = TestSet.UNSEEN
            self.test_dataset = self._test_unseen_dataset
        else:
            self.curr_test_set = TestSet.SEEN
            self.test_dataset = self._test_seen_dataset

    def _build_loader(
        self,
        dataset: t.Any,
        batch_sz: int,
        shuffle: bool,
        drop_last: bool,
        label_balanced: bool,
    ) -> DataLoader:
        """
        Builds a DataLoader with specified configuration.

        Notes
        -----
        This method handles both label-balanced and regular data loading.
        For label-balanced mode, it uses LabelBalancedBatchSampler instead of standard batching.

        Parameters
        ----------
        dataset : t.Any
            The dataset to load data from.
        batch_sz : int
            The batch size to use for the DataLoader.
        shuffle : bool
            Whether to shuffle the data.
        drop_last : bool
            Whether to drop the last partial batch.
        label_balanced : bool
            Whether to use the label-balanced sampler.

        Returns
        -------
        DataLoader
            A DataLoader configured according to the specified parameters.
        """

        if not dataset or len(dataset) == 0:
            return None

        dl_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            dl_kwargs["persistent_workers"] = self.persistent_workers

        if label_balanced:
            if self.mode != DataMode.CONTROL_PERTURB:
                raise ValueError("Label balancing is only supported in 'CONTROL_PERTURB' mode.")

            #? When label_balanced is True we ignore the ``shuffle`` argument
            #? because the sampler already randomises the batch order.
            sampler = LabelBalancedBatchSampler(
                dataset,
                batch_size=batch_sz,
                drop_last=drop_last,
                shuffle=True,
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                **dl_kwargs,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_sz,
                shuffle=shuffle,
                drop_last=drop_last,
                **dl_kwargs,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training set.

        Notes
        -----
        The training DataLoader uses the specified batch size and shuffling behavior.
        If label_balanced is True, it uses a label-balanced sampler.

        Returns
        -------
        DataLoader
            The training DataLoader.
        """

        return self._build_loader(
            dataset=self.train_dataset,
            batch_sz=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            label_balanced=self.train_label_balanced,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation set.

        Notes
        -----
        The validation DataLoader is configured with shuffle=False and drop_last=False.
        It uses the val_batch_size specified in the DataModule configuration.

        Returns
        -------
        DataLoader
            The validation DataLoader.
        """

        return self._build_loader(
            dataset=self.val_dataset,
            batch_sz=self.val_batch_size,
            shuffle=False,
            drop_last=False,
            label_balanced=self.val_label_balanced,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the test set.

        Notes
        -----
        The test DataLoader is configured with shuffle=False, drop_last=False, and uses
        the test_batch_size specified in the DataModule.

        Returns
        -------
        DataLoader
            The test DataLoader.
        """

        if isinstance(self.test_dataset, list):
            dls = []
            for ds in self.test_dataset:
                dl = self._build_loader(
                    dataset=ds,
                    batch_sz=self.test_batch_size,
                    shuffle=False,
                    drop_last=False,
                    label_balanced=self.test_label_balanced,
                )
                dls.append(dl)

            return dls

        else:
            return self._build_loader(
                dataset=self.test_dataset,
                batch_sz=self.test_batch_size,
                shuffle=False,
                drop_last=False,
                label_balanced=self.test_label_balanced,
            )

