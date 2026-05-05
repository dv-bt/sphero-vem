"""
Functions and utilities for config classes
"""

from pathlib import Path
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from typing import ClassVar, Self, Any
from sphero_vem.utils.misc import CustomJSONEncoder
import dacite


def to_serializable(input_dict) -> dict:
    """Convert all dictionary values to JSON-serializable types.

    Handles non-standard types such as numpy scalars, numpy arrays, and
    ``Path`` objects by round-tripping through ``json.dumps``/``json.loads``
    with ``CustomJSONEncoder``.

    Parameters
    ----------
    input_dict : dict
        Dictionary whose values may contain non-serializable types.

    Returns
    -------
    dict
        A new dictionary where all values are JSON-native types (str, int,
        float, list, dict, bool, or None).
    """
    json_string = json.dumps(input_dict, cls=CustomJSONEncoder)
    return json.loads(json_string)


def _list_to_tuple(value: Any) -> tuple:
    """Convert a list to a tuple, leaving non-list values unchanged.

    Used as a dacite type hook so that tuple fields survive JSON round-trips
    (JSON deserializes tuples as lists).

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    tuple | Any
        ``tuple(value)`` if *value* is a list, otherwise *value* unchanged.
    """
    if isinstance(value, list):
        return tuple(value)
    return value


class BaseConfig:
    """Base class for pipeline configuration dataclasses.

    Provides JSON serialization, deserialization with type coercion, and a
    two-tier parameter view (full config vs. scientifically relevant metadata).
    Subclasses should be ``@dataclass`` and may override the two class
    variables below to control which fields are exposed in each tier.

    Class Variables
    ---------------
    EXCLUDED_JSON_FIELDS : ClassVar[set[str]]
        Field names omitted from ``to_json`` / ``full_config``. Use this for
        fields that cannot be JSON-serialized at all (e.g. live ``zarr.Array``
        handles or ``torch.device`` objects).
    EXCLUDED_PROCESSING_FIELDS : ClassVar[set[str]]
        Field names omitted from ``processing_metadata`` *in addition to*
        those in ``EXCLUDED_JSON_FIELDS``. Use this for fields that are
        serializable but irrelevant to scientific reproducibility — such as
        file paths, verbosity flags, worker counts, or derived runtime values.

    Notes
    -----
    Deserialization uses ``dacite`` with ``DACITE_CONFIG``, which applies
    ``Path``, ``tuple``,``float``, and ``int`` type coercions so that configs survive a
    JSON round-trip without losing type information.
    """

    # Fields that cannot be serialized
    EXCLUDED_JSON_FIELDS: ClassVar[set[str]] = set()
    # Fields that are not relevant for scientific reproducibility
    EXCLUDED_PROCESSING_FIELDS: ClassVar[set[str]] = set()
    DACITE_CONFIG = dacite.Config(
        type_hooks={
            Path: Path,
            tuple: _list_to_tuple,
            float: float,
            int: int,
        },
        cast=[tuple],
    )

    def to_json(self, filepath: str | Path) -> None:
        """Serialize the config to a JSON file.

        Parameters
        ----------
        filepath : str | Path
            Destination file path. Created or overwritten.
        """
        with open(filepath, "w") as file:
            json.dump(self.full_config(), file, indent=4)

    @classmethod
    def from_json(cls, filepath: str | Path) -> Self:
        """Load a config instance from a JSON file.

        Parameters
        ----------
        filepath : str | Path
            Path to a JSON file previously written by ``to_json``.

        Returns
        -------
        Self
            A new instance of the calling class with fields populated from
            the JSON file, with type coercion applied via dacite.
        """
        with open(filepath, "r") as file:
            config_dict = json.load(file)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        """Instantiate a config from a plain dictionary.

        Type coercion (e.g. ``list`` → ``tuple``, ``str`` → ``Path``) is
        applied via dacite using ``DACITE_CONFIG``.

        Parameters
        ----------
        config_dict : dict[str, Any]
            Dictionary mapping field names to values.

        Returns
        -------
        Self
            A new instance of the calling class.
        """
        return dacite.from_dict(cls, config_dict, config=cls.DACITE_CONFIG)

    def full_config(self) -> dict:
        """Return a fully serializable representation of the config.

        Excludes fields listed in ``EXCLUDED_JSON_FIELDS``.

        Returns
        -------
        dict
            JSON-serializable dictionary of all config fields except those
            excluded by ``EXCLUDED_JSON_FIELDS``.
        """
        config_dict = asdict(self)
        for key in self.EXCLUDED_JSON_FIELDS:
            config_dict.pop(key, None)
        return to_serializable(config_dict)

    def processing_metadata(self) -> dict:
        """Return the subset of config parameters relevant for scientific reproducibility.

        Excludes fields listed in both ``EXCLUDED_JSON_FIELDS`` and
        ``EXCLUDED_PROCESSING_FIELDS``.

        Returns
        -------
        dict
            JSON-serializable dictionary of scientifically relevant parameters.
        """
        config_dict = asdict(self)
        excluded = self.EXCLUDED_JSON_FIELDS | self.EXCLUDED_PROCESSING_FIELDS
        for key in excluded:
            config_dict.pop(key, None)
        return to_serializable(config_dict)


@dataclass
class ProcessingStep:
    """Represents a single processing step in the pipeline.

    Can be created from a config or manually for manual steps.
    """

    step_name: str
    timestamp: str
    parameters: dict
    version: str | None = None

    @classmethod
    def from_config(
        cls, step_name: str, config: "BaseConfig", version: str | None = None
    ) -> Self:
        """Create a processing step from a config object.

        Parameters
        ----------
        step_name: str
            Name of the processing step
        config: BaseConfig
            Configuration dataclass instance, it should be a subclass of BaseConfig
            that inherits its BaseConfig.processing_metadata() method.
        version: str | None
            Optional software version string. Default is None.
        """
        return cls(
            step_name=step_name,
            timestamp=datetime.now().isoformat(),
            parameters=config.processing_metadata(),
            version=version,
        )

    @classmethod
    def manual(
        cls, step_name: str, parameters: dict, version: str | None = None
    ) -> Self:
        """Create a manual processing step (no config).

        Parameters
        ----------
        step_name: str
            Name of the processing step
        parameters: dict
            Dictionary of parameters for this step. Take care that non-serializable
            objects are not passed as a parameter value.
        version: str | None
            Optional software version string. Default is None.
        """
        return cls(
            step_name=step_name,
            timestamp=datetime.now().isoformat(),
            parameters=parameters,
            version=version,
        )

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Load a processing step from a dictionary (e.g., from zarr attrs)."""
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert to a serializable dictionary for storage (e.g., in zarr attrs)."""
        return to_serializable(asdict(self))
