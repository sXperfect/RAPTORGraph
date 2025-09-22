# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
from dataclasses import dataclass, field
from functools import cached_property

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
from ..utils import (
    lists_to_boolean_matrix,
    bool_matrix_to_embed_ids,
    conv_vec_to_cat
)

# =============================================================================
# OUTPUT DATA STRUCTURE
# =============================================================================
@dataclass
class ConditionedGeneExpressionData:
    """
    Stores gene expression data with lazily-computed, cacheable attributes.

    Attributes
    ----------
    expression_data : np.ndarray
        The gene expression matrix (cells x genes).
    gene_ids : np.ndarray
        The stable gene identifiers (from `adata.var.index`).
    gene_names : np.ndarray
        The common gene names.
    perturb_gene_names_per_sample : pd.Series, optional
        A Series where each entry is a sorted list of perturbations for a sample.
    cell_types : np.ndarray, optional
        An array of cell types for each cell, if available.
    """
    #? --- Core Data Fields with Descriptions ---
    expression_data: np.ndarray = field(
        metadata={"description": "The gene expression matrix (cells x genes)."}
    )
    gene_ids: np.ndarray = field(
        metadata={"description": "The stable gene identifiers (from `adata.var.index`)."}
    )
    gene_names: np.ndarray = field(
        metadata={"description": "The common gene names."}
    )
    perturb_gene_names_per_sample: pd.Series | None = field(
        default=None,
        metadata={"description": "A Series where each entry is a sorted list of perturbations for a sample."}
    )
    cell_types: np.ndarray | None = field(
        default=None,
        metadata={"description": "An array of cell types for each cell, if available."}
    )

    def __init__(
        self,
        expression_data: np.ndarray,
        gene_ids: np.ndarray,
        gene_names: np.ndarray,
        cell_types: np.ndarray | None = None,
        perturb_gene_names_per_sample: pd.Series | None = None,
        perturb_gene_names: list[str] | None = None,
        perturb_matrix: np.ndarray | None = None,
        perturb_embed_ids_per_sample: np.ndarray | None = None,
    ):
        self.expression_data = expression_data
        self.gene_ids = gene_ids
        self.gene_names = gene_names
        self.perturb_gene_names_per_sample = perturb_gene_names_per_sample
        self.cell_types = cell_types

        if perturb_gene_names is not None:
            self.__dict__['perturb_gene_names'] = perturb_gene_names
        if perturb_matrix is not None:
            self.__dict__['perturb_matrix'] = perturb_matrix
        if perturb_embed_ids_per_sample is not None:
            self.__dict__['perturb_embed_ids_per_sample'] = perturb_embed_ids_per_sample

    @cached_property
    def sort_by_perturbation_status(self) -> bool:
        unperturb_gene_mask = ~np.isin(self.gene_names, self.perturb_gene_names)
        unperturb_gene_names = np.sort(self.gene_names[unperturb_gene_mask])
        
        #? Not all perturbed genes exist in gene expression data
        found_perturb_gene_names = self.expressed_perturb_gene_names
        found_perturb_gene_names = np.sort(found_perturb_gene_names)
        
        ret = True
        ret &= (self.perturb_gene_names == np.sort(self.perturb_gene_names)).all()
        ret &= (found_perturb_gene_names == self.gene_names[:len(found_perturb_gene_names)]).all()
        ret &= (unperturb_gene_names == self.gene_names[len(found_perturb_gene_names):]).all()
        
        return ret
    
    @cached_property
    def num_genes(self) -> int:
        return len(self.gene_names)

    @cached_property
    def num_perturb_genes(self) -> int:
        return len(self.perturb_gene_names)

    @cached_property
    def perturb_gene_names(self) -> list[str] | None:
        """
        Get a sorted list of unique perturbation gene names.

        Returns
        -------
        list of str or None
            A sorted list of unique perturbation gene names, or None if
            perturb_gene_names_per_sample is not available.
        """
        if self.perturb_gene_names_per_sample is not None:
            return sorted(self.perturb_gene_names_per_sample.explode().dropna().unique())
        return None


    @cached_property
    def expressed_perturb_gene_names(self):
        perturb_gene_mask = np.isin(self.gene_names, self.perturb_gene_names)
        found_perturb_gene_names = self.gene_names[perturb_gene_mask]
        
        return found_perturb_gene_names

    @cached_property
    def perturb_matrix(self) -> np.ndarray | None:
        """
        Generate a boolean matrix of perturbations.

        Returns
        -------
        np.ndarray or None
            Boolean matrix of perturbations with shape (samples x perturbations),
            or None if required data is not available.
        """
        if self.perturb_gene_names_per_sample is not None and self.perturb_gene_names is not None:
            return lists_to_boolean_matrix(
                self.perturb_gene_names_per_sample, self.perturb_gene_names
            )
        return None

    @cached_property
    def perturb_embed_ids_per_sample(self) -> np.ndarray | None:
        """
        Generate padded integer matrix of embedding IDs for each sample.

        Returns
        -------
        np.ndarray or None
            Padded integer matrix of embedding IDs, or None if perturb_matrix is not available.
        """
        if self.perturb_matrix is not None:
            return bool_matrix_to_embed_ids(self.perturb_matrix)
        return None

    @cached_property
    def gene_names_to_ids(self) -> dict[str, str]:
        """
        Create a mapping from gene names to their stable IDs.

        Returns
        -------
        dict
            Dictionary mapping gene names (str) to gene IDs (str).
        """
        return {k: v for k, v in zip(self.gene_names, self.gene_ids)}

    @cached_property
    def perturb_gene_ids(self) -> np.ndarray | None:
        """
        Get array of stable IDs for the perturbed genes.

        Returns
        -------
        np.ndarray or None
            Array of stable gene IDs for perturbed genes, or None if mapping data is not available.
        """
        gene_map = self.gene_names_to_ids
        perturb_names = self.perturb_gene_names
        if gene_map is not None and perturb_names is not None:
            return pd.Series(perturb_names).map(gene_map).to_numpy()
        return None

    @cached_property
    def perturb_label_and_mapping(self) -> tuple[np.ndarray | None, dict[int, list[int]] | None]:
        """
        Compute perturbation labels and mappings from the perturbation matrix.

        Returns
        -------
        tuple[np.ndarray | None, dict[int, list[int]] | None]
            A tuple containing:
            - perturb_label: Integer label for each sample.
            - perturb_mapping: Mapping from label integers to perturbation indices.
        """
        if self.perturb_matrix is not None:
            return conv_vec_to_cat(self.perturb_matrix)
        return (None, None)

    @cached_property
    def perturb_label(self) -> np.ndarray | None:
        """
        Get integer labels for each sample based on perturbation combinations.

        Returns
        -------
        np.ndarray or None
            Integer label for each sample, or None if perturb_matrix is not available.
        """
        label, _ = self.perturb_label_and_mapping
        return label

    @cached_property
    def perturb_mapping(self) -> dict[int, list[int]] | None:
        """
        Get mapping from label integers to perturbation indices.

        Returns
        -------
        dict[int, list[int]] or None
            Mapping from label integers to perturbation indices, or None if
            perturb_matrix is not available.
        """
        _, mapping = self.perturb_label_and_mapping
        return mapping
        
    @cached_property
    def inv_perturb_mapping(self) -> dict[int, tuple] | None:
        """
        Get mapping from a unique integer label back to its perturbation vector (tuple).

        This is the inverse of the `perturb_mapping` dictionary.

        Returns
        -------
        dict[int, tuple] or None
            Mapping from integer labels to perturbation vectors (as tuples),
            or None if the original mapping is not available.
        """
        if self.perturb_mapping:
            return {v: k for k, v in self.perturb_mapping.items()}
        return None

    @cached_property
    def num_perturbs(self):
        return self.perturb_matrix.sum(axis=1)
