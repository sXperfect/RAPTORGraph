"""Utility functions for core application logic and data handling."""

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import sys
import typing as t
import copy as cp
import math
import sys
import typing as t

def exists(
    x: t.Any
) -> bool:
    """
    Checks if the given input is not None.

    Parameters
    ----------
    x : Any
        The input to check.

    Returns
    -------
    bool
        True if x is not None, otherwise False.

    Notes
    -----
    This function is used to check if a variable is not None.

    Examples
    --------

    """
    
    return x is not None

def default(
    val: t.Any,
    d: t.Any | t.Callable,
    copy: bool = False
) -> t.Any:
    """
    Returns the value if it exists, otherwise returns the default value.

    Parameters
    ----------
    val : Any
        The value to check.
    d : Any | Callable
        The default value or a callable that returns the default value.
    copy : bool, optional
        If True, returns a deep copy of the default value. Default is False.

    Returns
    -------
    Any
        The value if it exists, otherwise the default value.

    Notes
    -----
    If `val` is None, it returns the result of calling `d` if `d` is callable, 
    otherwise it returns `d`.

    Examples
    --------

    """
    
    if exists(val):
        return val

    if callable(d):
        return d()
    else:
        if copy:
            return cp.deepcopy(d)
        return d

def identity(
    t: t.Any,
    *args: t.Any,
    **kwargs: t.Any
) -> t.Any:
    """
    Returns the input value as-is.

    Parameters
    ----------
    t : Any
        The input value.
    *args : Any
        Additional positional arguments (ignored).
    **kwargs : Any
        Additional keyword arguments (ignored).

    Returns
    -------
    Any
        The input value.

    Notes
    -----
    This function is used to return the input value without any modifications.

    Examples
    --------

    """
    
    return t

def cycle(
    dl: t.Iterable
) -> t.Iterator:
    """
    Creates an infinite iterator that cycles through the input iterable.

    Parameters
    ----------
    dl : Iterable
        The iterable to cycle through.

    Returns
    -------
    Iterator
        An infinite iterator that cycles through the input iterable.

    Notes
    -----
    This function can be used to repeatedly iterate over an iterable indefinitely.

    """
    
    while True:
        for data in dl:
            yield data

def ensure_dict(
    d: dict | None
) -> dict:
    """
    Returns an empty dictionary if input is None, otherwise returns the input as-is.

    Parameters
    ----------
    d : dict | None
        The dictionary to check.

    Returns
    -------
    dict
        An empty dict if d is None, otherwise d.

    Notes
    -----
    Ensures that the output is always a dictionary, even if the input is None.

    Examples
    --------

    """
    return default(d, {})

def has_int_squareroot(
    num: int
) -> bool:
    """
    Checks if a number has an integer square root.

    Parameters
    ----------
    num : int
        The number to check.

    Returns
    -------
    bool
        True if the number has an integer square root, otherwise False.

    Notes
    -----
    This function checks if the square of the square root of the number is equal 
    to the number itself.

    Examples
    --------

    """
    
    return (math.sqrt(num) ** 2) == num

def num_to_groups(
    num: int,
    divisor: int
) -> list[int]:
    """
    Divides a number into groups of a given size and returns a list of group sizes.

    Parameters
    ----------
    num : int
        The number to divide.
    divisor : int
        The size of each group.

    Returns
    -------
    list[int]
        A list of group sizes.

    Notes
    -----
    The last group may contain a remainder if the number is not perfectly 
    divisible by the divisor.

    Examples
    --------

    """
    
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
