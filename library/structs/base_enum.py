"""
Base Enum Utilities.

This module provides custom base classes for StrEnum to handle common
case-insensitive and separator-insensitive matching for creating enum members,
leveraging modern Python 3.11+ features while properly supporting "none" values
through a dedicated NONE enum member.
"""


# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import enum
from typing import Self, TypeVar

# =============================================================================
# TYPE VARIABLES
# =============================================================================
T = TypeVar("T", bound="BaseStrEnum")

# =============================================================================
# BASE ENUM IMPLEMENTATION
# =============================================================================
@enum.verify(enum.UNIQUE)
class BaseStrEnum(enum.StrEnum):
    """
    A base class for StrEnum that provides case-insensitive and separator-insensitive
    matching for enum creation.
    
    This class handles common variations like different cases and separators
    (spaces/underscores/dashes), but does NOT attempt partial matching that
    could lead to ambiguous results.
    
    Examples
    --------
    >>> class Color(BaseStrEnum):
    ...     DARK_BLUE = "dark_blue"
    ...     LIGHT_GREEN = "light green"
    ...
    >>> Color("DARK_BLUE") is Color.DARK_BLUE
    True
    >>> Color("dark-blue") is Color.DARK_BLUE
    True
    >>> Color("light green") is Color.LIGHT_GREEN
    True
    >>> Color("lightgreen")  # This will NOT match (too aggressive)
    Traceback (most recent call last):
        ...
    ValueError: 'lightgreen' is not a valid Color. Please use one of: 'dark_blue', 'light green'
    """
    @classmethod
    def _missing_(cls, value: object) -> Self:
        """
        Handles case-insensitive and separator-insensitive matching for enum creation.
        
        Parameters
        ----------
        value : object
            The value to match against enum members.
            
        Returns
        -------
        Self
            The matching enum member.
            
        Raises
        ------
        ValueError
            If no matching enum member is found.
        """
        if not isinstance(value, str):
            return super()._missing_(value)
        
        #? Normalize the input string by standardizing separators and case
        normalized_value = value.lower().replace("-", "_").replace(" ", "_")
        
        #? Find the first member whose normalized value matches
        for member in cls:
            member_value = member.value.lower().replace("-", "_").replace(" ", "_")
            if member_value == normalized_value:
                return member
        
        #? If no match is found, raise a helpful error
        valid_options = ", ".join(f"'{m.value}'" for m in cls)
        raise ValueError(
            f"'{value}' is not a valid {cls.__name__}. " 
            f"Please use one of: {valid_options}"
        )


@enum.verify(enum.UNIQUE)
class OptionalBaseStrEnum(BaseStrEnum):
    """
    A base class for StrEnum that treats None as the NONE enum member.
    
    This class requires subclasses to define a NONE member, which will be
    returned when None is provided to the constructor.
    
    IMPORTANT: Subclasses MUST define a NONE member.
    
    Examples
    --------
    >>> class Status(OptionalBaseStrEnum):
    ...     ACTIVE = "active"
    ...     INACTIVE = "inactive"
    ...     PENDING = "pending"
    ...     NONE = "none"  # Required NONE member
    ...
    >>> Status("ACTIVE") is Status.ACTIVE
    True
    >>> Status(None) is Status.NONE
    True
    >>> Status("none") is Status.NONE
    True
    """
    
    def __init_subclass__(cls, **kwargs) -> None:
        """Check that subclasses define a NONE member after class creation."""
        super().__init_subclass__(**kwargs)
        
        #? Check if NONE member is defined
        if "NONE" not in cls.__members__:
            raise TypeError(
                f"Class {cls.__name__} must define a NONE member when inheriting "
                f"from {OptionalBaseStrEnum.__name__}. Example: NONE = 'none'"
            )
    
    @classmethod
    def _missing_(cls, value: object) -> Self:
        """
        Handles None values and provides case-insensitive, separator-insensitive matching.
        
        Parameters
        ----------
        value : object
            The value to match against enum members.
            
        Returns
        -------
        Self
            The matching enum member.
            
        Raises
        ------
        ValueError
            If no matching enum member is found.
        """
        #? Handle None input by returning the NONE member
        if value is None:
            return cls.NONE
        
        #? Delegate to BaseStrEnum's _missing_ implementation for string values
        return super()._missing_(value)

@enum.verify(enum.UNIQUE)
class BaseIntEnum(enum.IntEnum):
    """
    A base class for IntEnum that provides case-insensitive string matching
    for enum creation. This allows creating enum members from strings
    that match their names.
    
    Examples
    --------
    >>> class ErrorCode(BaseIntEnum):
    ...     NOT_FOUND = 404
    ...     FORBIDDEN = 403
    ...
    >>> ErrorCode("NOT_FOUND") is ErrorCode.NOT_FOUND
    True
    >>> ErrorCode("not_found") is ErrorCode.NOT_FOUND
    True
    >>> ErrorCode(404) is ErrorCode.NOT_FOUND
    True
    >>> ErrorCode(None) is None
    True
    """

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        """
        Handles conversion from None or case-insensitive strings.
        
        Parameters
        ----------
        value : None, str, or int
            The value to convert to an enum member.
            
        Returns
        -------
        Self or None
            The corresponding enum member, or None if the input value is None.
            
        Raises
        ------
        ValueError
            If the string value doesn't match any enum member name.
        """
        if value is None:
            return None

        if isinstance(value, str):
            value_upper = value.upper()
            for member in cls:
                if member.name == value_upper:
                    return member
        
        #? Let the default handler attempt to convert from int or raise an error
        return super()._missing_(value)