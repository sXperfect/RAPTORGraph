"""
Defines the abstract base class for configuration wrappers.

This module provides the `BaseConfig` class, which serves as an interface
for different configuration handlers (e.g., for Optuna, MLFlow, or a simple run)
to suggest and manage hyperparameters.
"""
# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import abc
import typing as t

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
#? No third-party imports are needed for this module.

# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
#? No local application imports are needed for this module.


class BaseConfig(abc.ABC):
    """
    Abstract base class for handling experiment configurations.

    This class defines a common interface for suggesting parameters, ensuring
    that different configuration sources (like Optuna trials or static dicts)
    can be used interchangeably.

    Attributes
    ----------
    params_cfg : dict
        A dictionary containing the parameter configurations for the experiment.
    """
    params_cfg: t.Dict[str, t.Any]

    @abc.abstractmethod
    def suggest_param(
        self,
        name: str,
        def_val: t.Any = None,
        ena_def_val: bool = False,
    ) -> t.Any:
        """
        Suggests or retrieves a single parameter value.

        This method must be implemented by subclasses to handle the logic
        of fetching a parameter, either from a hyperparameter optimization
        trial or a static configuration file.

        Parameters
        ----------
        name : str
            The name (or key) of the parameter to suggest.
        def_val : t.Any, optional
            The default value to return if the parameter is not found in the
            configuration. Defaults to None.
        ena_def_val : bool, optional
            A flag that must be set to True to enable the default value
            mechanism. Defaults to False.

        Returns
        -------
        t.Any
            The suggested, fixed, or default value for the parameter.
        """
        pass

    def suggest_dict_params(
        self,
        namespace: str,
        dest_dict: t.Dict[str, t.Any] | None = None,
        def_dict: t.Dict[str, t.Any] | None = None,
        ena_def_dict: bool = False,
    ) -> t.Dict[str, t.Any] | None:
        """
        Suggests a dictionary of parameters based on a common namespace.

        This method intelligently combines parameters defined in the main
        configuration with a provided default dictionary.

        Parameters
        ----------
        namespace : str
            The prefix used to identify related parameters in the main config.
        dest_dict : dict | None, optional
            If provided, the suggested parameters will be added to this
            dictionary in-place. If `None`, a new dictionary is returned.
            Defaults to `None`.
        def_dict : dict | None, optional
            A dictionary of default values for the parameters in this namespace.
            Defaults to `None`.
        ena_def_dict : bool, optional
            If True, enables the use of the `def_dict`. Defaults to False.

        Returns
        -------
        dict | None
            A dictionary of parameters, or `None` if `dest_dict` was provided.
        """
        _namespace = f"{namespace}-"
        params = {}

        #? 1. Get all keys that are explicitly defined in the main config for this namespace.
        config_keys = {
            key.replace(_namespace, "", 1) for key in self.params_cfg if key.startswith(_namespace)
        }

        #? 2. Get all keys from the default dictionary, if provided and enabled.
        default_keys = set(def_dict.keys()) if ena_def_dict and def_dict else set()

        #? 3. The union of these two sets represents all possible parameters for this namespace.
        all_keys = config_keys.union(default_keys)

        for key in all_keys:
            full_key = f"{_namespace}{key}"
            
            #? Get the default value for this specific key, if it exists.
            default_value = def_dict.get(key) if ena_def_dict and def_dict else None
            
            #? Call suggest_param. It will:
            #? - Use the definition from `params_cfg` if `full_key` exists.
            #? - Fall back to `default_value` if `full_key` does not exist.
            params[key] = self.suggest_param(
                name=full_key,
                def_val=default_value,
                ena_def_val=(ena_def_dict and def_dict is not None)
            )

        if dest_dict is None:
            return params
        else:
            #? Update the user-provided destination dictionary in-place.
            dest_dict.update(params)
            return None
