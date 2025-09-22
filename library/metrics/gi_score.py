"""
Computes and classifies genetic interaction (GI) scores from gene expression data.

This module provides functions to calculate ground truth GI scores using an additive
model and subsequently classify those interactions into categories such as synergy,
redundancy, suppression, neomorphism, and epistasis, based on the methodology
described in the GEARS paper.
"""
# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
from ..structs import ConditionedGeneExpressionData
from ..datasets.tensor_dict import TensorDictDataset

DEF_COLUMNS = [
    "synergy_ratio", 
    "neomorphism_score", 
    "redundancy_score", 
    "epistasis_score"
]

def calculate_interaction_scores(
    delta_A,
    delta_B,
    delta_AB,
    epsilon:float = 1e-8
) -> dict[str, float]:
    """
    Calculates genetic interaction scores based on vector effects.

    Parameters
    ----------
    delta_A, delta_B, delta_AB : np.ndarray
        The effect vectors for perturbation A, B, and the double perturbation AB.
    epsilon : float, optional
        A small value to prevent division by zero, by default 1e-8.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the calculated interaction scores.
    """
    delta_A = np.asarray(delta_A).flatten()
    delta_B = np.asarray(delta_B).flatten()
    delta_AB = np.asarray(delta_AB).flatten()

    #? Create the naive additive baseline vector
    additive_delta = delta_A + delta_B

    #? Calculate magnitudes (L2 norm)
    mag_AB = np.linalg.norm(delta_AB)
    mag_additive = np.linalg.norm(additive_delta)

    #? Calculate cosine similarities
    cosine_sim_additive = cosine_similarity(
        delta_AB.reshape(1, -1), additive_delta.reshape(1, -1) + epsilon
    )[0, 0]

    scores = {}

    #? Synergy & Suppression calculated as a ratio (ratio > 1 indicates synergy; < 1 indicates suppression)
    if mag_additive > epsilon:
        scores["synergy_ratio"] = mag_AB / mag_additive
    else:
        scores["synergy_ratio"] = 1.0  # Avoid division by zero; treat as additive

    #? Neomorphism Calculation
    scores["neomorphism_score"] = 1 - cosine_sim_additive

    #? Redundancy & Epistasis Calculation
    sim_A_AB = cosine_similarity(
        delta_A.reshape(1, -1), delta_AB.reshape(1, -1) + epsilon
    )[0, 0]
    sim_B_AB = cosine_similarity(
        delta_B.reshape(1, -1), delta_AB.reshape(1, -1) + epsilon
    )[0, 0]

    scores["redundancy_score"] = min(sim_A_AB, sim_B_AB)
    scores["epistasis_score"] = abs(sim_A_AB - sim_B_AB)

    return scores

def comp_gt_gi_scores(
    processed_data: ConditionedGeneExpressionData,
    epsilon=1e-8,
):
    """Computes Ground Truth GI scores and intermediate data for classification.

    This function calculates GI scores based on the additive model and also returns
    key intermediate data structures needed for downstream classification, such as
    the mean expression profiles.

    Parameters
    ----------
    processed_data : ConditionedGeneExpressionData
        An object containing expression data, perturbation labels, and other
        metadata. The data must be sorted by perturbation status.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing:
        - gi_scores_df: DataFrame of GI scores.
        - pseudo_gene_exp_df: DataFrame of mean expression profiles.
    """
    assert processed_data.sort_by_perturbation_status, \
        "Processed data must be sorted by perturbation status"

    perturb_mapping = processed_data.perturb_mapping
    inv_perturb_mapping = processed_data.inv_perturb_mapping

    exp_data_df = pd.DataFrame(
        columns=processed_data.gene_names,
        data=processed_data.expression_data,
    )
    exp_data_df["label"] = processed_data.perturb_label
    num_perturbs = processed_data.num_perturbs

    pseudo_gene_exp_df = exp_data_df.groupby("label").mean()

    double_pert_mask = num_perturbs == 2
    double_pert_labels = np.unique(
        processed_data.perturb_label[double_pert_mask]
    )

    ctrl_label = 0
    gi_table_df = pd.DataFrame(
        columns=["label", *DEF_COLUMNS]
    )

    for d_label in double_pert_labels:
        double_pert_vec = np.array(inv_perturb_mapping[d_label])

        c1_val, c2_val = np.argwhere(double_pert_vec).flatten()

        p1_vec = np.zeros_like(double_pert_vec, dtype=bool)
        p1_vec[c1_val] = 1
        single_pert_label1 = perturb_mapping[tuple(p1_vec)]

        p2_vec = np.zeros_like(double_pert_vec, dtype=bool)
        p2_vec[c2_val] = 1
        single_pert_label2 = perturb_mapping[tuple(p2_vec)]

        #? Get the expression profiles
        pseudo_double = pseudo_gene_exp_df.loc[d_label].values
        pseudo_single1 = pseudo_gene_exp_df.loc[single_pert_label1].values
        pseudo_single2 = pseudo_gene_exp_df.loc[single_pert_label2].values
        pseudo_ctrl = pseudo_gene_exp_df.loc[ctrl_label].values

        delta_A = (pseudo_single1-pseudo_ctrl)
        delta_B = (pseudo_single2-pseudo_ctrl)
        delta_AB = (pseudo_double-pseudo_ctrl)
        
        label = tuple(sorted([c1_val, c2_val]))

        scores = calculate_interaction_scores(
            delta_A,
            delta_B,
            delta_AB,
            epsilon=epsilon,
        )
        scores["label"] = label

        gi_table_df.loc[len(gi_table_df)+1, :] = pd.Series(scores)

    gi_table_df.set_index("label", inplace=True)
    gi_table_df = gi_table_df.astype(float)
    
    return gi_table_df

def comp_pred_gi_scores(
    trainer,
    model,
    datamodule,
    test_dl,
    batch_size=None,
    use_test_control=False,
):
    #? GI Score Computation
    if use_test_control:
        ctrl_exps = test_dl.dataset.ctrl_samples

    else:
        all_exps = datamodule.cond_gene_exp_data.expression_data
        label = datamodule.cond_gene_exp_data.perturb_label
        ctrl_exps = all_exps[label == 0]
    
    unique_double_pert_mat = np.unique(test_dl.dataset.ptb_matrix, axis=0)
    pert_gene_ids = np.argwhere(unique_double_pert_mat.any(axis=0)).flatten()

    num_samples = ctrl_exps.shape[0]
    num_perturb_genes = datamodule.num_perturb_genes

    num_interv = np.full((num_samples, 1), 2)
    mean_double_pert_exp_dict = dict()
    for double_pert_vec in unique_double_pert_mat:
        #? Identify the two perturbed gene indices (non‑zero entries)
        pert_idxs = np.argwhere(double_pert_vec).flatten()
        if pert_idxs.size != 2:
            #? Skip malformed rows (should contain exactly two 1’s)
            continue

        c1 = np.zeros((num_samples, num_perturb_genes))
        c2 = np.zeros((num_samples, num_perturb_genes))
        c1[:, pert_idxs[0]] = 1.0
        c2[:, pert_idxs[1]] = 1.0

        predict_ds = TensorDictDataset({
            "X": torch.from_numpy(ctrl_exps),
            "c1": torch.from_numpy(c1),
            "c2": torch.from_numpy(c2),
            "num_int": torch.from_numpy(num_interv),
        })  # type: ignore

        predict_dl = DataLoader(
            predict_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        pred_outputs = trainer.predict(
            model,
            dataloaders=predict_dl,
            return_predictions=True,
        )

        mean_double_pert_exp = (
            torch.concat([batch["y_hat"] for batch in pred_outputs], axis=0).mean(axis=0)
        )

        label = tuple(sorted([pert_idxs[0], pert_idxs[1]]))
        mean_double_pert_exp_dict[label] = mean_double_pert_exp

    num_interv = np.ones((num_samples, 1))
    mean_single_pert_exp_dict = dict()
    for pert_ID in pert_gene_ids:
        c1 = np.zeros((num_samples, num_perturb_genes))
        c1[:, pert_ID] = 1.0
        c2 = c1.copy()

        predict_ds = TensorDictDataset({
            "X": torch.from_numpy(ctrl_exps),
            "c1": torch.from_numpy(c1),
            "c2": torch.from_numpy(c2),
            "num_int":  torch.from_numpy(num_interv),
        }) # type: ignore

        predict_dl = DataLoader(
            predict_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )

        pred_outputs = trainer.predict(
            model,
            dataloaders=predict_dl,
            return_predictions=True,
        )

        mean_single_pert_exp = (
            torch.concat([pred_outputs_batch["y_hat"] for pred_outputs_batch in pred_outputs], axis=0)
            .mean(axis=0)
        )

        mean_single_pert_exp_dict[pert_ID] = mean_single_pert_exp

    gi_table_df = pd.DataFrame(
        columns=["label", *DEF_COLUMNS]
    )


    pseudo_ctrl = ctrl_exps.mean(axis=0)
    for label, pseudo_double in mean_double_pert_exp_dict.items():
        # delta_AB = pseudo_double - ctrl_exps

        pert_ID1, pert_ID2 = label

        #? Get the expression profiles
        pseudo_single1 = mean_single_pert_exp_dict[pert_ID1]
        pseudo_single2 = mean_single_pert_exp_dict[pert_ID2]

        pseudo_double = pseudo_double.numpy()
        pseudo_single1 = pseudo_single1.numpy()
        pseudo_single2 = pseudo_single2.numpy()

        delta_A = (pseudo_single1-pseudo_ctrl)
        delta_B = (pseudo_single2-pseudo_ctrl)
        delta_AB = (pseudo_double-pseudo_ctrl)

        scores = calculate_interaction_scores(
            delta_A,
            delta_B,
            delta_AB,
        )
        scores["label"] = label

        gi_table_df.loc[len(gi_table_df)+1, :] = pd.Series(scores)

    gi_table_df.set_index("label", inplace=True)
    gi_table_df = gi_table_df.astype(float)

    return (
        gi_table_df,
        mean_single_pert_exp_dict,
        mean_double_pert_exp_dict,
    )

def compute_gi_scores(
    gt_gi_scores_df,
    pred_gi_scores_df,
    k=10,
    perc=0.75
):
    precision_summary = dict()

    pred_aligned = pred_gi_scores_df.reindex(gt_gi_scores_df.index).dropna()
    gt_aligned = gt_gi_scores_df.loc[pred_aligned.index]
    
    for val_name in ["neomorphism_score", "redundancy_score", "epistasis_score"]:
        threshold = gt_gi_scores_df[val_name].quantile(perc)
        top_k_preds = pred_aligned[val_name].nlargest(k)
        true_positives = (gt_aligned.loc[top_k_preds.index, val_name] >= threshold).sum()

        precision_summary[val_name.split('_')[0]] = true_positives / k

    #? Handle Synnergy
    synergy_threshold = gt_aligned["synergy_ratio"].quantile(perc)
    top_k_synergy = pred_aligned["synergy_ratio"].nlargest(k)
    tp_synergy = (gt_aligned.loc[top_k_synergy.index, "synergy_ratio"] >= synergy_threshold).sum()
    precision_summary["synergy"] = tp_synergy / k

    #? Handle Suppression
    suppression_threshold = gt_aligned["synergy_ratio"].quantile(1 - perc)
    top_k_suppression = pred_aligned["synergy_ratio"].nsmallest(k)
    tp_suppression = (gt_aligned.loc[top_k_suppression.index, "synergy_ratio"] <= suppression_threshold).sum()
    precision_summary["suppression"] = tp_suppression / k

    return precision_summary