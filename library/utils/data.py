"""
Utility functions for data processing and collation in machine learning pipelines,
focusing on single-cell genomics data.
"""



# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import typing as t
from collections.abc import Callable
from dataclasses import dataclass

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import numpy as np
import pandas as pd

def conv_vec_to_cat(
    vects: np.ndarray,
) -> tuple[np.ndarray, dict[tuple, int]]:
    """
    Converts arrays of values into categorical values (labels).

    Parameters
    ----------
    vects : np.ndarray
        A 2D numpy array of values, with shape (num_samples, num_features).

    Returns
    -------
    tuple[np.ndarray, dict[tuple, int]]
        A tuple containing the 1D array of category indices and the dictionary
        mapping unique vector tuples to category indices.

    Notes
    -----
    This function identifies unique vectors, sorts them based on the sum of their
    elements, and maps each unique vector to an integer category. The control
    condition (where all features are 0s) gets the smallest ID.

    Examples
    --------

    """
    unique_vecs = np.unique(vects, axis=0)
    
    #? Sort unique vectors by the sum of elements to ensure control condition (all zeros) gets smallest ID
    sort_ids = np.argsort(unique_vecs.sum(axis=1))
    unique_vecs = unique_vecs[sort_ids, :]
    assert ~unique_vecs[0, :].any(), "First unique vector must be all-false"

    num_perturbs = unique_vecs.sum(axis=1)
    label_to_vec_dict = dict()

    # control_mask = num_perturbs == 0
    control_vec = unique_vecs[0, :]
    assert not control_vec.any()
    label_to_vec_dict[0] = tuple(control_vec)

    for i in range(unique_vecs.shape[1]):
        single_pert_vec = control_vec.copy()
        single_pert_vec[i] = True
        label_to_vec_dict[i+1] = tuple(single_pert_vec)

    for i_perturbs in np.unique(num_perturbs[num_perturbs > 1]):
        mult_pert_mask = num_perturbs == i_perturbs
        mult_pert_mask_vecs = unique_vecs[mult_pert_mask, :]

    unique_indices_tuples = [tuple(np.where(vec)[0]) for vec in mult_pert_mask_vecs]
    sorted_unique_indices = sorted(unique_indices_tuples)

    #? Convert each unique vector to a tuple for hashable dictionary keys
    unique_tuples = [tuple(vec) for vec in unique_vecs]
    label_to_vec_dict = {t: idx for idx, t in enumerate(unique_tuples)}
    
    #? Map each original vector to its category index
    labels = np.array(
        [label_to_vec_dict[tuple(vec)] for vec in vects]
    )
    
    return labels, label_to_vec_dict

def _get_apply_dispatcher(
) -> Callable[[pd.Series, Callable], pd.Series]:
    """
    Returns the appropriate apply method based on available parallel processing libraries.

    Returns
    -------
    Callable[[pd.Series, Callable], pd.Series]
        A function that takes a pandas Series and a function, and applies
        the function to the series, either in parallel or sequentially.
        
    Notes
    -----
    Checks for parallel-pandas and returns the appropriate apply method.
    If parallel-pandas is installed, it initializes it and returns a function
    that uses `p_apply`. Otherwise, it returns a function for the standard
    `apply` method.
    """
    try:
        from parallel_pandas import ParallelPandas
        ParallelPandas.initialize(disable_pr_bar=True)
        print("ParallelPandas found. Using parallel apply.")
        return lambda s, func: s.p_apply(func)
    except ImportError:
        print("ParallelPandas not found. Using standard apply.")
        return lambda s, func: s.apply(func)

def lists_to_boolean_matrix(
    series_of_lists: pd.Series,
    valid_items: list[str]
) -> np.ndarray:
    """
    Creates a boolean matrix from a Series of lists.

    Parameters
    ----------
    series_of_lists : pd.Series
        A Series where each element is a list of strings.
    valid_items : list[str]
        An ordered list of all unique possible items. This list defines
        the columns of the output matrix.

    Returns
    -------
    np.ndarray
        A boolean numpy array of shape (n_samples, n_items).
        `matrix[i, j]` is True if the list for sample `i` contains item `j`.
    """
    n_samples = len(series_of_lists)
    n_items = len(valid_items)
    
    #? Initialize boolean matrix with zeros (False values)
    bool_matrix = np.zeros((n_samples, n_items), dtype=bool)
    
    #? Use .to_list() for potentially faster iteration over Series values
    for i, item_list in enumerate(series_of_lists.to_list()):
        bool_matrix[i, :] = np.isin(valid_items, item_list)
    
    return bool_matrix

def bool_matrix_to_embed_ids(
    cond_mask: np.ndarray,
    padding_idx: int | None = 0,
    sample_cond_ids_mapping: np.ndarray | None = None,
    dataset_idx: int | None = None,
    max_conds: int | None = None,
) -> np.ndarray:
    """
    Converts boolean vectors to embedding indices.

    Parameters
    ----------
    cond_mask : np.ndarray
        A 2D numpy array of boolean values.
    padding_idx : int | None, optional
        The index to use for padding. Must be 0 or None. Default is 0.
    sample_cond_ids_mapping : np.ndarray | None, optional
        An optional mapping of sample condition IDs. Not implemented.
    dataset_idx : int | None, optional
        An optional index to prepend to each sample's embedding indices.
        Must be None. Default is None.
    max_conds : int | None, optional
        Maximum number of conditions per sample. If None, uses the maximum
        found in the data.

    Returns
    -------
    np.ndarray
        A 2D integer array of embedding indices.

    Notes
    -----
    Converts a 2D boolean array into a 2D integer array of embedding
    indices, with support for optional padding and dataset indexing.
    
    Raises
    ------
    NotImplementedError
        If padding_idx is not 0 or None, or if dataset_idx is not None, or
        if sample_cond_ids_mapping is provided.
    """
    IDS_DTYPE = np.int32  # Use int32 for compatibility with torch.long

    if padding_idx not in [0, None]:
        raise NotImplementedError("Currently supports only padding_idx=0 or None.")
    if dataset_idx not in [None]:
        raise NotImplementedError("Currently supports only None.")
    if sample_cond_ids_mapping is not None:
        raise NotImplementedError("sample_cond_ids_mapping is not yet implemented.")

    num_samples, _ = cond_mask.shape
    num_cond_per_sample = np.sum(cond_mask, axis=1)
    if len(num_cond_per_sample) > 0:
        true_max_conds = np.max(num_cond_per_sample)
    else:
        true_max_conds = 0

    if max_conds is None:
        max_conds = true_max_conds
    else:
        assert max_conds >= true_max_conds

    embed_ids = np.full((num_samples, max_conds), padding_idx, dtype=IDS_DTYPE)
    embed_offset = 1 if padding_idx == 0 else 0

    for i in range(num_samples):
        sample_cond_vec = cond_mask[i, :]
        sample_embed_ids = np.argwhere(sample_cond_vec).flatten()
        sample_embed_ids += embed_offset
        num_sample_cond = len(sample_embed_ids)
        embed_ids[i, :num_sample_cond] = sample_embed_ids

    return embed_ids