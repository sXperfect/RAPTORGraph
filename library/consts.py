"""
This module defines project-wide constants and enumerations.

Using enums ensures that configuration values are type-safe, validated,
and constrained to a set of allowed options.
"""
import enum

# =============================================================================
# ENUMERATIONS
# =============================================================================

class TestSet(enum.StrEnum):
    """Enumeration for test sets."""
    SEEN = "seen"
    UNSEEN = "unseen"

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return super()._missing_(value)
        
        value_lower = value.lower()
        for member in cls:
            if member.value == value_lower:
                return member
        
        valid_options = ", ".join(m.value for m in cls)
        raise ValueError(
            f"'{value}' is not a valid {cls.__name__}. "
            f"Please use one of: {valid_options}"
        )

class CondType(enum.StrEnum):
    """Enumeration for conditioning types in models."""
    STACK = 'stack'
    FILM = 'film'
    ATTENTION = 'attention'
    SELF_ATTENTION = 'self_attention'
    CROSS_ATTENTION = 'cross_attention'

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return super()._missing_(value)
        
        value_lower = value.lower()
        for member in cls:
            if member.value == value_lower:
                return member
        
        valid_options = ", ".join(m.value for m in cls)
        raise ValueError(
            f"'{value}' is not a valid {cls.__name__}. "
            f"Please use one of: {valid_options}"
        )

class AttentionBackend(enum.StrEnum):
    """Enum for available attention backends."""
    PYTORCH = "pytorch"
    FLASH_ATTENTION = "flash_attention"

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return super()._missing_(value)
        
        value_lower = value.lower().replace("_", "").replace("-", "")
        if value_lower == "flash":
            return cls.FLASH_ATTENTION

        for member in cls:
            if member.value.replace("_", "") == value_lower:
                return member
        
        valid_options = ", ".join(m.value for m in cls)
        raise ValueError(
            f"'{value}' is not a valid {cls.__name__}. "
            f"Please use one of: {valid_options}"
        )

class NormPlacement(enum.StrEnum):
    """
    Enum to define the placement of normalization layers within a block.
    """
    PRE = "pre"
    MID = "mid"
    POST = "post" # Added for completeness
    ALL = "all"

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return super()._missing_(value)
        
        value_lower = value.lower()
        for member in cls:
            if member.value == value_lower:
                return member
        
        valid_options = ", ".join(m.value for m in cls)
        raise ValueError(
            f"'{value}' is not a valid {cls.__name__}. "
            f"Please use one of: {valid_options}"
        )