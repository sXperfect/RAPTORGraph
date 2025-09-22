"""
Provides configuration handling for a single, non-optimized run.
"""
# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import typing as t

# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
from ._base_cfg import BaseConfig

# =============================================================================
# CONFIGURATION CLASS
# =============================================================================
class RunConfig(BaseConfig):
    """
    A wrapper for a fixed dictionary of hyperparameters for a single run.

    This class implements the `BaseConfig` interface for a non-optimization
    scenario, where all parameter values are predefined.
    """
    def __init__(self,
        params_cfg: t.Dict[str, t.Any]
    ):
        """
        Initializes the RunConfig instance.

        Parameters
        ----------
        params_cfg : dict
            A dictionary containing the fixed key-value pairs for all
            hyperparameters for the run.
        """
        self.params_cfg = params_cfg
        #? For a single run, the hyperparameters are simply the provided parameters.
        self.hparams = params_cfg

    def suggest_param(
        self,
        name: str,
        def_val: t.Any = None,
        ena_def_val: bool = False,
    ) -> t.Any:
        """
        Retrieves a parameter value from the configuration dictionary.

        This method performs a simple dictionary lookup. If the parameter is
        not found and `ena_def_val` is True, it returns the default value.

        Parameters
        ----------
        name : str
            The name (or key) of the parameter to retrieve.
        def_val : t.Any, optional
            The default value to return if the parameter is not found.
            Defaults to None.
        ena_def_val : bool, optional
            If True, enables the default value mechanism. Defaults to False.

        Returns
        -------
        t.Any
            The fixed or default value for the parameter.
        """
        if name in self.params_cfg:
            return self.params_cfg[name]

        if ena_def_val:
            return def_val

        raise KeyError(f"Parameter '{name}' not found in the run configuration.")

    get_param = suggest_param
    get_dict_param = BaseConfig.suggest_dict_params
