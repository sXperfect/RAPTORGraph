# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import scanpy as sc

# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
from ..utils import (
    _get_apply_dispatcher,
)
from ..structs import ConditionedGeneExpressionData

# =============================================================================
# IMMUTABLE CONFIGURATIONS
# =============================================================================
NORMAN_CPA_CONFIG = {
    "gene_var_col": "gene_symbols",
    "cond_col": "guide_ids",
    "cond_delimiter": ",",
    "control_cond_name": None,
    "cell_types_col": None,
    #! Not all perturbed genes are in the expressed genes
    "expressed_perturbation_consistency": False
}

NORMAN_SENA_CONFIG = NORMAN_CPA_CONFIG


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================
def process_perturb_adata(
    #? --- Configuration for data paths ---
    adata_fpath: str,
    #? --- Author specific adata config ---
    gene_var_col: str = "gene_name",
    cond_col: str = "condition",
    cond_delimiter: str = "+",
    control_cond_name: str | None = "ctrl",
    cell_types_col: str | None = "cell_type",
    #? --- Perturbation related config ---
    expressed_perturbation_consistency: bool = True,
    single_perturbation_consistency: bool = True,
    #? --- Gene Filter ---
    # ena_hvg_filter: bool = False,
    # hvg_flavor: str = "seurat",
    # hvg_n_top_genes: int | None = None,
    # hvg_min_mean: float = 0.0125,
    # hvg_max_mean: float = 3.0,
    # hvg_min_disp: float = 0.5,
    # hvg_max_disp: float | None = None,
    # select_n_densest_genes: int | None = None,
    drop_zero_expressed_genes: bool = False,
    #? --- Post Processing ---
    sort_by_perturbation_status: bool = True,
) -> ConditionedGeneExpressionData:
    """
    Processes an AnnData file to extract perturbation and optional pathway data.

    This function reads single-cell data, cleans perturbation labels, validates
    consistency, applies gene filtering, and, if paths are provided, generates
    a gene-pathway membership mask.

    Parameters
    ----------
    adata_fpath : str or pathlib.Path
        Path to the input AnnData file (`.h5ad`).
    gene_var_col : str, optional
        Column in `adata.var` with gene symbols, by default "gene_name".
    cond_col : str, optional
        Column in `adata.obs` with perturbation conditions, by default "condition".
    cond_delimiter : str, optional
        Character separating multiple perturbations, by default "+".
    control_cond_name : str or None, optional
        Name of the control condition to exclude, by default "ctrl".
    cell_types_col : str or None, optional
        Column in `adata.obs` for cell types, by default "cell_type".
    expressed_perturbation_consistency : bool, optional
        If True, ensures that all genes specified in the perturbation conditions
        are present in the dataset's gene names. This validation step checks for
        consistency between the perturbation labels and the actual genes available
        in the data. If any perturbed gene is not found in the gene names, an assertion
        error is raised. Default is True.
    single_perturbation_consistency : bool, optional
        If True, validates that all genes in higher-order perturbations
        (e.g., combinations of multiple genes) are part of the valid single-gene
        perturbations. This ensures no unseen or invalid genes are included in
        multi-gene perturbation conditions. Default is True.
    ena_hvg_filter : bool, optional
        If True, apply highly variable gene (HVG) filtering, by default False.
    hvg_flavor : str, optional
        Flavor of HVG selection ('seurat', etc.), by default "seurat".
    hvg_n_top_genes : int or None, optional
        Number of top HVGs to select, by default None.
    hvg_min_mean : float, optional
        Min mean expression for HVG, by default 0.0125.
    hvg_max_mean : float, optional
        Max mean expression for HVG, by default 3.0.
    hvg_min_disp : float, optional
        Min dispersion for HVG, by default 0.5.
    hvg_max_disp : float or None, optional
        Max dispersion for HVG, by default None.
    select_n_densest_genes : int or None, optional
        If set, selects top N genes by non-zero counts, by default None.
    drop_zero_expressed_genes : bool, optional
        If True, removes genes with zero counts, by default False.
    sort_by_perturbation_status : bool, optional
        If True, sorts the final gene list to have perturbed genes first,
        followed by unperturbed genes, by default **True**.

    Returns
    -------
    ConditionedGeneExpressionData
        A dataclass instance containing the processed data, including
        optional pathway information.

    Raises
    ------
    AssertionError
        If expressed_perturbation_consistency is True and there are invalid perturbed genes.
    ValueError
        If single_perturbation_consistency is True and there are unseen perturbed genes.

    Notes
    -----
    - This function asserts that 'ena_hvg_filter' and 'select_n_densest_genes' are not
      used simultaneously.
    - It also validates that all perturbed genes are present in the gene names.
    - If `single_perturbation_consistency` is True, it ensures that all genes in
      higher-order perturbations are valid perturbed genes.

    """
    #? --- 1. SETUP: Conditional Parallelism Utility ---
    dispatch_apply = _get_apply_dispatcher()

    # assert not (ena_hvg_filter and select_n_densest_genes is not None), (
    #     "Cannot use both 'ena_hvg_filter' and 'select_n_densest_genes' simultaneously."
    # )

    adata = sc.read_h5ad(adata_fpath)
    gene_ids_names_ps = adata.var[gene_var_col]
    gene_names = gene_ids_names_ps.to_numpy()
    gene_ids = gene_ids_names_ps.index.to_numpy()
    cell_types = (
        adata.obs[cell_types_col].to_numpy()
        if cell_types_col and cell_types_col in adata.obs
        else None
    )

    #? --- 2. PARALLELIZED DATA CLEANING ---
    perturb_gene_names_per_sample = (
        adata.obs[cond_col]
        .astype(str)
        .fillna("")
        .str.split(cond_delimiter)
    )

    perturb_gene_names_per_sample = dispatch_apply(
        perturb_gene_names_per_sample, lambda x: sorted([p for p in x if p != ""])
    )
    if control_cond_name:
        perturb_gene_names_per_sample = dispatch_apply(
            perturb_gene_names_per_sample,
            lambda x: [p for p in x if p != control_cond_name],
        )

    num_perturbs_per_sample = dispatch_apply(perturb_gene_names_per_sample, len)
    unique_num_perturbs = np.unique(num_perturbs_per_sample)
    num_perturbs_mask_dict = {n: (num_perturbs_per_sample == n).to_numpy() for n in unique_num_perturbs}

    #? Valid perturbed genes are based on the single perturbations
    single_perturb_per_sample = perturb_gene_names_per_sample[num_perturbs_mask_dict[1]]
    perturb_gene_names_set = set(
        p for sublist in single_perturb_per_sample for p in sublist
    )
    perturb_gene_names = sorted(list(perturb_gene_names_set))

    if expressed_perturbation_consistency:
        #? Ensure every perturbation gene appears in the expression matrix.
        valid_perturb_gene_mask = np.isin(perturb_gene_names, gene_names)
        if not valid_perturb_gene_mask.all():
            missing_genes = np.setdiff1d(perturb_gene_names, gene_names)
            raise ValueError(
                f"Perturbation genes not found in dataset: {missing_genes.tolist()}"
            )

    #? Filter the validity of higher order of perturbation
    if single_perturbation_consistency:
        for num_perturb in unique_num_perturbs[unique_num_perturbs > 1]:
            curr_sample_mask = num_perturbs_mask_dict[num_perturb]
            curr_perturb_per_sample = perturb_gene_names_per_sample[curr_sample_mask]

            def check_validity(ptb_list: list[str]) -> bool:
                return all(gene in perturb_gene_names_set for gene in ptb_list)

            has_valid_perturbs = dispatch_apply(curr_perturb_per_sample, check_validity)
            if not has_valid_perturbs.all():
                raise ValueError("There is unseen perturbed gene!")

    #? --- 3. GENE FILTERING ---
    final_gene_mask = np.ones(adata.shape[1], dtype=bool)

    if drop_zero_expressed_genes:
        nz_mask = np.array(adata.X.getnnz(axis=0)) > 0
        final_gene_mask &= nz_mask

    if not final_gene_mask.all():
        adata = adata[:, final_gene_mask]
        gene_ids = gene_ids[final_gene_mask]
        gene_names = gene_names[final_gene_mask]

    #? Create a single reordering index to group samples by perturbation count
    original_indices = np.arange(adata.shape[0])
    sample_reorder_ids = np.concatenate([original_indices[mask] for n, mask in sorted(num_perturbs_mask_dict.items())])

    #? Apply the reordering to all sample-based data to ensure consistency
    expression_data = adata.X[sample_reorder_ids, :].toarray()
    perturb_gene_names_per_sample = perturb_gene_names_per_sample.iloc[sample_reorder_ids].reset_index(drop=True)
    if cell_types is not None:
        cell_types = cell_types[sample_reorder_ids]

    #? --- Gene Reordering ---
    if sort_by_perturbation_status:
        unperturb_gene_mask = ~np.isin(gene_names, perturb_gene_names)
        unperturb_gene_names = np.sort(gene_names[unperturb_gene_mask])
        
        #? Not all perturbed genes exist in gene expression data
        found_perturb_gene_names = gene_names[~unperturb_gene_mask]
        found_perturb_gene_names = np.sort(found_perturb_gene_names)

        new_gene_names = np.concatenate([found_perturb_gene_names, unperturb_gene_names])

        #? Create a mapping from gene name to its index in the original filtered gene_names
        gene_name_to_index = {name: idx for idx, name in enumerate(gene_names)}
        gene_sort_ids = np.array([gene_name_to_index[name] for name in new_gene_names])

        expression_data = expression_data[:, gene_sort_ids]
        gene_ids = gene_ids[gene_sort_ids]
        gene_names = gene_names[gene_sort_ids]

        assert (gene_names == new_gene_names).all()

    #? --- 5. CONSTRUCT AND RETURN DATACLASS OUTPUT ---
    return ConditionedGeneExpressionData(
        expression_data=expression_data,
        gene_ids=gene_ids,
        gene_names=gene_names,
        perturb_gene_names=perturb_gene_names,
        perturb_gene_names_per_sample=perturb_gene_names_per_sample,
        cell_types=cell_types,
    )