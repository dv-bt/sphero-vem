"""
Functions and utilities for config classes
"""

from pathlib import Path
import json
from dataclasses import asdict
from typing import ClassVar, Self, Any
from sphero_vem.utils.misc import CustomJSONEncoder
import dacite


def to_serializable(input_dict) -> dict:
    """Convert dictionary keys to a JSON serializable ones"""
    json_string = json.dumps(input_dict, cls=CustomJSONEncoder)
    return json.loads(json_string)


class BaseConfig:
    """Base config class providing JSON serialization for config dataclasses."""

    # Fields that cannot be serialized
    EXCLUDED_JSON_FIELDS: ClassVar[set[str]] = set()
    # Fields that are not relevant for scientific reproducibility
    EXCLUDED_PROCESSING_FIELDS: ClassVar[set[str]] = set()
    DACITE_CONFIG = dacite.Config(type_hooks={Path: Path})

    def to_json(self, filepath: str | Path) -> None:
        """Saves the dataclass instance to a JSON file."""
        with open(filepath, "w") as file:
            json.dump(self.full_config(), file, indent=4)

    @classmethod
    def from_json(cls, filepath: str | Path) -> Self:
        """Loads a dataclass instance from a JSON file with type coercion."""
        with open(filepath, "r") as file:
            config_dict = json.load(file)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        """Loads a dataclass instance from a dict with type coercion."""
        return dacite.from_dict(cls, config_dict, config=cls.DACITE_CONFIG)

    def full_config(self) -> dict:
        """Returns full serializable config for complete reproducibility."""
        config_dict = asdict(self)
        for key in self.EXCLUDED_JSON_FIELDS:
            config_dict.pop(key, None)
        return to_serializable(config_dict)

    def processing_metadata(self) -> dict:
        """Returns only scientifically relevant processing parameters."""
        config_dict = asdict(self)
        excluded = self.EXCLUDED_JSON_FIELDS | self.EXCLUDED_PROCESSING_FIELDS
        for key in excluded:
            config_dict.pop(key, None)
        return to_serializable(config_dict)
