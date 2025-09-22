# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
# No local imports in this module.

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================
def create_pathway_mask(
    gene_ids: np.ndarray,
    pathway_fpath: str | None,
    pathway_filter_fpath: str | None,
    #? --- Gene Filter ---
    min_valid_genes_in_pathways: int | None = None
) -> tuple[np.ndarray | None, np.ndarray | None, dict | None]:
    """
    Creates a boolean mask of gene membership in specified pathways.

    This function reads pathway and filter files, aggregates genes for each
    pathway, and constructs a binary matrix where rows correspond to pathways
    and columns correspond to genes from the provided gene list.

    Parameters
    ----------
    all_genes : np.ndarray
        An array of all gene symbols/IDs in the dataset.
    pathway_fpath : str | Path | None
        Path to the main pathway file (whitespace-separated). Must have
        'PathwayID' and 'Symbol' columns.
    pathway_filter_fpath : str | Path | None
        Path to a file containing a list of 'PathwayID's to keep.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None, dict | None]
        A tuple containing:
        - pathway_mask (np.ndarray | None): The boolean mask of shape
          (n_pathways, n_genes). None if input paths are not provided.
        - pathway_ids (np.ndarray | None): The array of pathway IDs
          corresponding to the rows of the mask. None if input paths
          are not provided.
        - pathway_gene_map (dict | None): A dictionary mapping each pathway
          ID to a list of its constituent genes.

    """
    
    if not (pathway_fpath and pathway_filter_fpath):
        return None, None, None

    #? Read pathway data and the filter list
    pathway_df = pd.read_csv(pathway_fpath, sep=r"\s+")
    pathway_filter_df = pd.read_csv(pathway_filter_fpath, sep=r"\s+")

    #? Keep only the pathways specified in the filter file
    mask = pathway_df["PathwayID"].isin(pathway_filter_df["PathwayID"])
    pathway_df = pathway_df[mask]

    #? Group genes by pathway ID
    agg_pathway_df = pathway_df.groupby("PathwayID")["Symbol"].agg(list)
    pathway_gene_map = agg_pathway_df.to_dict()

    #? Initialize the boolean mask: (n_pathways, n_all_genes)
    pathway_mask = np.zeros((len(agg_pathway_df), len(gene_ids)), dtype=bool)

    #? Populate the mask
    for i, genes_in_pathway in enumerate(agg_pathway_df):
        #? For each pathway (row), mark the genes that are present
        pathway_mask[i, :] = np.isin(gene_ids, genes_in_pathway)

    #? Filter out pathways that have no gene overlap with the dataset
    valid_pathway_mask = pathway_mask.any(axis=1)

    #? Filter out pathways with less valid genes than the thresholds
    if min_valid_genes_in_pathways is not None:
        tmp_mask = pathway_mask.sum(axis=1) >= min_valid_genes_in_pathways
        valid_pathway_mask &= tmp_mask
    
    pathway_mask = pathway_mask[valid_pathway_mask, :]
    pathway_ids = np.array(agg_pathway_df[valid_pathway_mask].index)

    filtered_pathway_gene_map = dict()
    for pathway_id, gene_ids_in_pathway in pathway_gene_map.items():
        if pathway_id in pathway_ids:
            filtered_pathway_gene_map[pathway_id] = gene_ids_in_pathway

    return pathway_mask, pathway_ids, filtered_pathway_gene_map

def create_deconfounding_pathway_mask(
    gene_names: np.ndarray,
    perturb_gene_names: np.ndarray,
    num_pathways: int,
) -> np.ndarray:
    """
    Creates a gene-pathway mask for deconfounding causal effects.

    This mask sets up a specific structure where a subset of pathways
    (equal to the number of perturbed genes) are uniquely associated with
    those perturbed genes. The remaining pathways are connected to all other
    (non-perturbed) genes.

    Parameters
    ----------
    gene_names : np.ndarray
        An array of all gene names in the dataset.
    perturb_gene_nameshtop
    : np.ndarray
        An array of the names of the perturbed genes.
    num_pathways : int
        The total number of pathways (must be >= number of perturbed genes).

    Returns
    -------
    np.ndarray
        The generated (num_pathways, num_genes) deconfounding mask.

    Notes
    -----
    The goal of this masking strategy is to achieve a **disentangled
    representation** in the latent space. By creating a one-to-one mapping
    between a perturbed gene and a dedicated latent variable (pathway), we
    force the model to encode all information about that specific perturbation
    into its assigned latent dimension. This prevents the "spill-out" effect,
    where a perturbation's signal might influence multiple latent variables.
    As a result, the remaining latent variables are free to model the
    background cell state, allowing for a cleaner separation of causal effects
    from confounding factors.
    """
    num_genes = len(gene_names)
    num_perturb_genes = len(perturb_gene_names)

    data_perturb_gene_mask = np.isin(gene_names, perturb_gene_names)
    data_perturb_gene_names = gene_names[data_perturb_gene_mask]

    assert num_pathways >= num_perturb_genes, \
        "Number of pathways must be >= number of perturbed genes."
    # assert np.all(gene_names[:num_perturb_genes] == perturb_gene_names), \
    #     "Perturbed genes must be the first genes in the `gene_names` array."
    assert len(np.argwhere(np.diff(data_perturb_gene_mask)).flatten()) == 1, \
        "Perturbed genes must be the first genes in the `gene_names` array."

    #? Not all perturb genes are in the gene expression data
    perturb_in_data_mask = np.isin(perturb_gene_names, data_perturb_gene_names)
    
    gp_mask = np.zeros((num_pathways, num_genes))
    eye_mask = np.eye(num_perturb_genes)
    diag_row_ids, diag_col_ids = np.diag_indices(num_perturb_genes)
    eye_mask[diag_row_ids, diag_col_ids] = perturb_in_data_mask

    gp_mask[:num_perturb_genes, :num_perturb_genes] = eye_mask
    gp_mask[:, num_perturb_genes:] = 1.0

    return gp_mask