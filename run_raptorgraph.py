"""
Main script for optimizing and analyzing the RaptorDiff_VAE model.

This script uses Hydra for configuration management
optimization. It includes advanced plotting functionalities for analyzing
optimization results, including categorical distributions and Pareto frontiers.
"""
# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import typing as t
import logging
import os
import pickle
import gc
import datetime as dt
import subprocess as sp
import pathlib
import json

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import pandas as pd
import hydra
import torch
import lightning as L
from omegaconf import DictConfig, ListConfig

# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
from library import helpers as my_helpers
from library.configs.run_cfg import RunConfig

from library.datasets import (
    ControlPerturbDataModule,
    generate_cache_filename,
    NORMAN_CPA_CONFIG,
)
from library.models.crl_ae import CausalRepresentationLearningAE
from library.archs.vae.rdvae import (
    RaptorGraphVAEArch,
)
from library.callbacks import (
    get_default_alpha_config,
    get_default_beta_config,
    get_default_lmbda_config,
    get_default_temp_config,
    CausalRepresentationLearningVAEHyperparameterScheduler,
    SENA_MMD_KWARGS,
    UnbiasedMMDMetricCallback,
)
from library.metrics.gi_score import (
    comp_gt_gi_scores,
    comp_pred_gi_scores,
    compute_gi_scores,
)

#? Suppress excessive logging from third-party libraries for a cleaner output.
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("pydantic").setLevel(logging.ERROR)

def _dump_json(data: dict, fpath: pathlib.Path) -> None:
    """
    Write *data* to *fpath* as a human‑readable JSON file.
    The parent directory is created automatically.
    """
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with fpath.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, sort_keys=True)
    logging.info(f"Metrics saved to {fpath}")

# =============================================================================
# OBJECTIVE FUNCTION FOR OPTUNA
# =============================================================================
def objective_f(
    hydra_cfg,
    exp_params: DictConfig | dict,
    cfg: DictConfig,
) -> float | t.List[float]:
    """
    The objective function for an Optuna trial or a single run.
    """

    run_cfg = RunConfig(
        params_cfg=exp_params
    )

    if cfg.get("model") and cfg.model.get("kwargs"):
        model_kwargs = my_helpers.resolve_cfg(cfg.model.kwargs)
    else:
        model_kwargs = dict()

    optimized_model_kwargs = run_cfg.get_dict_param("model-kwargs")
    model_kwargs.update(optimized_model_kwargs)

    learning_rate = run_cfg.get_param("learning_rate")
    optimizer_name = run_cfg.get_param("optimizer")
    optimizer_kwargs = run_cfg.get_dict_param(f"{optimizer_name}-kwargs")

    core_kwargs = run_cfg.get_dict_param(
        namespace="arch-core",
    )

    interacting_kwargs = run_cfg.get_dict_param(
        namespace="arch-interact",
    )

    modulator_kwargs = run_cfg.get_dict_param(
        namespace="arch-modulator",
    )

    dagma_kwargs = run_cfg.get_dict_param(
        namespace="arch-dagma",
        ena_def_dict=True,
    )

    misc_kwargs = run_cfg.get_dict_param(
        namespace="arch-misc",
    )

    arch_kwargs = {
        **core_kwargs,
        **interacting_kwargs,
        **modulator_kwargs,
        **dagma_kwargs,
        **misc_kwargs,
    }

    num_pathways_mult = run_cfg.get_param("num_pathways_mult", 1.0, True)

    arch_kwargs = {
        **core_kwargs,
        **interacting_kwargs,
        **modulator_kwargs,
        **dagma_kwargs,
        **misc_kwargs,
    }

    alpha = run_cfg.get_param("alpha") #! Mandatory
    kl_beta = run_cfg.get_param("kl_beta") #! Mandatory
    loss_kwargs = {
        "dagma_mu": run_cfg.get_param("dagma_mu"),
        "dagma_lambda1": run_cfg.get_param("dagma_lambda1"),
    }

    #? --- 2. Run Training and Evaluation ---
    run_seeds = cfg.args.get('seeds', 42)
    if not isinstance(run_seeds, (list, ListConfig)):
        run_seeds = [run_seeds]

    is_multi_seed = len(run_seeds) > 1

    all_seed_metrics = []

    for seed in run_seeds:
        try:
            L.seed_everything(seed, workers=True)

            dm_params = cfg.datamodule.kwargs
            dataset_name = cfg.datamodule.get("dataset_name", "norman-cpa_raw")

            #? --- Data Loading (Moved inside seed loop) ---
            PICKLE_PATH = cfg.path.pickle

            #? Use the datamodule's cache_filename property
            #? Generate cache filename using the independent function
            hash_name = generate_cache_filename(
                mode=dm_params.mode,
                val_size=dm_params.val_size,
                test_size=dm_params.test_size,
                train_val_pairing_mode=dm_params.trainval_pairing_mode,
                test_pairing_mode=dm_params.test_pairing_mode,
                trainval_add_identity_pairs=dm_params.trainval_add_identity_pairs,
                test_add_identity_pairs=dm_params.test_add_identity_pairs,
                split_unseen_datasets=dm_params.get("split_unseen_datasets", False),
                sort_by_perturbation_status=dm_params.get("sort_by_perturbation_status", True),
            )

            cache_name = f"{dataset_name}-seed_{seed}-{hash_name}.pkl"
            cache_fpath = pathlib.Path(cfg.path.pickle) / cache_name

            if os.path.exists(cache_fpath):
                with open(cache_fpath, 'rb') as f:
                    datamodule = pickle.load(f)

            else:
                cache_datamodule(cfg)

            datamodule.batch_size = dm_params.get("batch_size")
            datamodule.val_batch_size = dm_params.get("val_batch_size", datamodule.batch_size)
            datamodule.test_batch_size = dm_params.get("test_batch_size", datamodule.batch_size)

            datamodule.label_balanced = dm_params.get("label_balanced", False)
            datamodule.train_label_balanced = dm_params.get("train_label_balanced", datamodule.label_balanced)
            datamodule.val_label_balanced = dm_params.get("val_label_balanced", datamodule.label_balanced)
            datamodule.test_label_balanced = dm_params.get("test_label_balanced", datamodule.label_balanced)

            #? Informations from the dataloaders
            num_perturb_genes = datamodule.num_perturb_genes

            #? Patch the data-dependent parameters
            num_pathways = int(num_perturb_genes * num_pathways_mult)

            arch = RaptorGraphVAEArch(
                gene_names=datamodule.gene_names,
                perturb_gene_names=datamodule.perturb_gene_names,
                num_pathways=num_pathways,
                **arch_kwargs
            )

            model = CausalRepresentationLearningAE(
                arch_obj=arch,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                optimizer_kwargs=optimizer_kwargs,
                alpha=alpha,
                beta=kl_beta,
                graph_lambda=1.0,
                temp=1.0,
                **SENA_MMD_KWARGS,
                **model_kwargs,
                **loss_kwargs,
            )

            callbacks = []

            ena_on_train = run_cfg.get_param('model-kwargs-deterministic_intervention', False, True)
            ena_on_train &= cfg.model.callbacks.get("ena_train_unbiased_mmd_metric", False)
            ena_on_val = run_cfg.get_param('model-kwargs-deterministic_intervention', False, True)
            ena_on_val &= cfg.model.callbacks.get("ena_val_unbiased_mmd_metric", False)
            ena_on_test = cfg.model.callbacks.get("ena_test_unbiased_mmd_metric", False)

            if ena_on_train or ena_on_val:
                mmd_callback_val_only = UnbiasedMMDMetricCallback(
                    on_train=ena_on_train,
                    on_validation=ena_on_val,
                    on_test=False,
                    log_prog_bar=True,
                    log_on_step=True,
                    log_on_epoch=False,
                    mmd_strategy='dynamic',
                    metric_name="mmd_loss",
                    **SENA_MMD_KWARGS
                )
                callbacks.append(mmd_callback_val_only)

            if ena_on_test:
                mmd_callback_test_only = UnbiasedMMDMetricCallback(
                    on_train=False,
                    on_validation=False,
                    on_test=True,
                    log_prog_bar=False,
                    log_on_step=True,
                    log_on_epoch=False,
                    mmd_strategy='global',
                    metric_name="unbiased_mmd",
                    **SENA_MMD_KWARGS
                )
                callbacks.append(mmd_callback_test_only)

            #? --- Configure Scheduler Callback ---
            key = "ena_crl_vae_hp_cb"
            ena_crl_vae_hp_cb = run_cfg.get_param(
                f"cb-{key}",
                cfg.model.callbacks.get(key, False),
                True,
            )
            if ena_crl_vae_hp_cb:
                num_epochs = cfg.pl.trainer.max_epochs
                alpha_config = get_default_alpha_config(num_epochs=num_epochs, max_val=alpha)
                beta_config = get_default_beta_config(num_epochs=num_epochs, max_val=kl_beta)
                lmbda_config = get_default_lmbda_config(num_epochs=num_epochs, val=1.0)
                temp_config = get_default_temp_config(num_epochs=num_epochs, max_val=1.0)

                cb = CausalRepresentationLearningVAEHyperparameterScheduler(
                    num_epochs=num_epochs,
                    arch=None, #? Disable presets to use manual config
                    alpha_config=alpha_config,
                    beta_config=beta_config,
                    lmbda_config=lmbda_config,
                    temp_config=temp_config
                )
                callbacks.append(cb)

            trainer_kwargs = my_helpers.resolve_cfg(cfg.pl.trainer)
            save_dir = f"{cfg.path.logs}/tensorboard"
            os.makedirs(save_dir, exist_ok=True)

            if not cfg.args.dry_run:
                tb_logger = L.pytorch.loggers.tensorboard.TensorBoardLogger(
                    save_dir=save_dir,
                    name=f"{cfg.args.python_fname}/{hydra_cfg.runtime.choices['exp/run_raptorgraph']}/seed_{seed}",
                    version=dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                )
            else:
                tb_logger = None

            trainer = L.Trainer(
                logger=tb_logger,
                callbacks=callbacks,
                **trainer_kwargs
            )

            arch_savedir = pathlib.Path(cfg.path.models)
            arch_savedir = arch_savedir / f"{cfg.args.python_fname}/{hydra_cfg.runtime.choices['exp/run_raptorgraph']}/seed_{seed}"
            arch_savedir.mkdir(parents=True, exist_ok=True)
            arch_fpath = arch_savedir / "best_model.pt"

            if arch_fpath.exists():
                arch_state_dict = torch.load(arch_fpath, map_location=torch.device("cpu"))
                arch.load_state_dict(arch_state_dict)
                metrics = dict()

            else:
                trainer.fit(model, datamodule=datamodule)
                metrics = {k: v.item() for k, v in trainer.callback_metrics.items()}
                torch.save(arch.state_dict(), arch_fpath)

            for key in trainer.logged_metrics:
                if key not in metrics:
                    metrics[key] = trainer.logged_metrics[key].item()

            test_dl = datamodule.test_dataloader()
            test_results = trainer.test(
                model,
                dataloaders=test_dl
            )
            metrics.update({f"{k}": v for k, v in test_results[0].items()})

            for key in trainer.logged_metrics:
                if key not in metrics:
                    metrics[key] = trainer.logged_metrics[key].item()

            if cfg.model.callbacks.get("ena_gi_scores_metric", False):
                gt_gi_scores_df = comp_gt_gi_scores(datamodule.cond_gene_exp_data)

                out = comp_pred_gi_scores(
                    trainer,
                    model,
                    datamodule,
                    test_dl,
                    batch_size=datamodule.batch_size,
                    use_test_control=cfg.model.callbacks.get("gi_scores_metric-use_test_control", False)
                )

                pred_gi_scores_df, _, _ = out

                precision_summary = compute_gi_scores(
                    gt_gi_scores_df,
                    pred_gi_scores_df,
                )

                for key, val in precision_summary.items():
                    metrics[f"test/{key}"] = val

            #? Reduce memory consumptions due to multiple seeds
            trainer = None
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            del datamodule

            all_seed_metrics.append(metrics)
            tb_logger.log_hyperparams(
                run_cfg.hparams,
                {f"final/{k}": v for k,v in metrics.items()},
            )
            result_path = pathlib.Path(cfg.path.results)
            result_path.mkdir(parents=True, exist_ok=True)

            seed_result_path = result_path / f"{cfg.args.python_fname}" / f"{hydra_cfg.runtime.choices['exp/run_raptorgraph']}/seed_{seed}"
            metrics_txt = json.dumps(
                metrics,
                indent=2,
            )
            print(metrics_txt)
            _dump_json(metrics, seed_result_path /  "metrics.json")

        except Exception as e:
            logging.error(f"Run failed on seed {seed} with an unexpected error: {e}")
            raise

    #? --- 3. Aggregate Metrics if Multi-Seed ---
    if is_multi_seed:
        df = pd.DataFrame(all_seed_metrics)
        final_metrics = {}
        for metric_name in df.columns:
            values = df[metric_name]
            final_metrics[f"agg/mean/{metric_name}"] = values.mean()
            final_metrics[f"agg/max/{metric_name}"] = values.max()
            final_metrics[f"agg/min/{metric_name}"] = values.min()
            if values.nunique() > 1:
                final_metrics[f"agg/var/{metric_name}"] = values.var()

        result_path = result_path / f"{cfg.args.python_fname}" / f"{hydra_cfg.runtime.choices['exp/run_raptorgraph']}"
        _dump_json(final_metrics, seed_result_path /  "metrics.json")
    else:
        #? For a single run, the metrics are not aggregated.
        final_metrics = all_seed_metrics[0]

    # final_metrics.update({f"param/{k}": v for k, v in run_cfg.hparams.items()})
    print(final_metrics)


def cache_datamodule(
    # hydra_cfg,
    cfg,
):
    run_seeds = cfg.args.get('seeds', 42)
    if not isinstance(run_seeds, (list, ListConfig)):
        run_seeds = [run_seeds]

    is_multi_seed = len(run_seeds) > 1

    for seed in run_seeds:
        L.seed_everything(seed=seed, workers=True)

        dataset_name = cfg.datamodule.dataset_name

        dm = ControlPerturbDataModule.from_adata(
            adata_fpath=cfg.path.dataset,
            **cfg.datamodule.kwargs
        )

        cache_name = f"{dataset_name}-seed_{seed}-{dm.cache_filename}.pkl"
        cache_path = pathlib.Path(cfg.path.pickle) / cache_name

        if cache_path.exists():
            continue

        dm.setup() #? Preprocess data

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with cache_path.open("wb") as f:
            pickle.dump(dm, f, protocol=pickle.HIGHEST_PROTOCOL)

        del dm

@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="base_run"
)
def run(
    cfg: DictConfig,
):

    my_helpers.ignore_pl_warnings()
    hydra_cfg = my_helpers.init_hydra_and_check_config(
        cfg,
        check_script_name=True,
        check_paths=False
    )
    try:
        mode = cfg.args.mode.lower()
    except (AttributeError, ValueError):
        raise ValueError("`args.mode` not found or invalid. Use 'run'.")

    if mode == 'run':
        exp_params = cfg.run.parameters

        objective_f(
            hydra_cfg=hydra_cfg,
            exp_params=exp_params,
            cfg=cfg,
        )
    elif mode == 'cache':
        cache_datamodule(
            cfg=cfg,
        )
    else:
        raise ValueError("Invalid mode: {mode}!")

if __name__ == "__main__":
    run()
