"""
Utility functions for data processing and collation in machine learning pipelines, 
focusing on single-cell genomics data.
"""
import typing as t

def remove_ctrl(perturb_list: t.List[str]) -> t.List[str]:
    """Removes the string 'ctrl' from a list of perturbations.

    Parameters
    ----------
    perturb_list : t.List[str]
        The list from which 'ctrl' should be removed.

    Returns
    -------
    t.List[str]
        A new list with 'ctrl' removed.
        
    Examples
    --------
    """
    return [p for p in perturb_list if p != 'ctrl']