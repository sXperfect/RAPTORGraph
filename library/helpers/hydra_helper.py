"""
Provides helper functions for working with Hydra configurations.

This module includes utilities for initializing and validating Hydra
configurations and for resolving OmegaConf objects into standard Python dicts.
"""
# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import typing as t
from os import makedirs
from os.path import exists

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def init_hydra_and_check_config(
    cfg: DictConfig,
    script_name: str | None = None,
    check_script_name: bool = True,
    check_paths: bool = True,
    allow_unresolved_keys: bool = False
) -> HydraConfig:
    """
    Initializes and validates the Hydra configuration for an experiment.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object provided by Hydra.
    script_name : str | None, optional
        The expected name of the main script. If None, it's inferred from
        the Hydra config. Defaults to None.
    check_script_name : bool, optional
        If True, validates that the experiment config matches the script name.
        Defaults to True.
    check_paths : bool, optional
        If True, ensures that all paths in `cfg.path` exist. If False, it
        creates them instead. Defaults to True.
    allow_unresolved_keys : bool, optional
        If False, raises an error if any keys in the config are missing
        (i.e., have a value of '???'). Defaults to False.

    Returns
    -------
    HydraConfig
        The active Hydra configuration object.

    Raises
    ------
    ValueError
        If the experiment configuration is invalid for the current script.
    RuntimeError
        If `allow_unresolved_keys` is False and missing keys are found.
    """
    hydra_cfg = HydraConfig.get()

    #? Validate that the loaded experiment config matches the running script.
    if check_script_name:
        #? Find the experiment name and the script name from the Hydra choices.
        #? The key is expected to be in the format 'exp/<script_name>'.
        exp_name = None
        detected_script_name = None
        for key, value in hydra_cfg.runtime.choices.items():
            if key.startswith("exp/"):
                exp_name = value
                #? The script name is the part of the key after "exp/".
                detected_script_name = key.split('/', 1)[1]
                break #? Found it, no need to continue looping.

        if not exp_name or not detected_script_name:
             raise ValueError(
                 "Could not determine experiment name from Hydra config. "
                 "Expected a key in `hydra.runtime.choices` starting with 'exp/'."
             )

        #? This assumes a config key `args.python_fname` exists for validation.
        #? It compares the script name from the config with the one detected in the choices.
        if cfg.args.python_fname != detected_script_name:
            raise ValueError(
                f"Invalid config for this script. The config is for "
                f"'{cfg.args.python_fname}', but the running script "
                f"appears to be '{detected_script_name}'."
            )

    #? Validate or create paths defined in the configuration.
    if "path" in cfg:
        for key, path in cfg.path.items():
            if key.startswith('_'):
                continue
            if check_paths:
                if not exists(path):
                    raise FileNotFoundError(f"Path for '{key}' does not exist: {path}")
            # else:
            #     makedirs(path, exist_ok=True)

    #? Check for any unresolved ('???') keys in the configuration.
    if not allow_unresolved_keys:
        missing_keys = OmegaConf.missing_keys(cfg)
        if missing_keys:
            raise RuntimeError(f"The following keys are missing from the config: {missing_keys}")

    return hydra_cfg


def resolve_cfg(
    cfg: DictConfig | None,
    default_to_empty_dict: bool = False,
) -> dict | None:
    """
    Resolves an OmegaConf DictConfig object into a standard Python dictionary.

    Parameters
    ----------
    cfg : DictConfig | None
        The configuration object to resolve.
    default_to_empty_dict : bool, optional
        If True, returns an empty dict if `cfg` is None. If False,
        returns None. Defaults to False.

    Returns
    -------
    dict | None
        The resolved configuration as a dictionary, or None.
    """
    if cfg is None:
        return {} if default_to_empty_dict else None
    
    return OmegaConf.to_container(cfg, resolve=True)

