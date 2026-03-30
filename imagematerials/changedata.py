"""Module to modify input to the models."""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
import warnings

from imagematerials.factory import Sector


@dataclass
class ChangeAction(ABC):
    """Abstract base class for all ChangeActions to inherit from.
    """

    @abstractmethod
    def apply(self, value: Any):
        """Applies the change defined by this ChangeAction object to the
        `value` passed.

        Parameters
        ----------
        value
            The element the change should be applied to.
        """
        pass

@dataclass
class ChangeReplace(ChangeAction):
    """ChangeAction for replacing a value with an entirely `new_value`.
    """
    new_value: Any

    def apply(self, value: Any):
        """Replaces `value` with `self.new_value`.

        Parameters
        ----------
        value
            The element the change should be applied to.
        """
        if type(value) != type(self.new_value):
            warnings.warn(
                f"Replacing a value of type {type(value)} with one of type {type(self.new_value)}."
                f" Replacing {value}  with {self.new_value} will cause its type to change.",
                RuntimeWarning
            )
        return deepcopy(self.new_value)


def change_input(
    data: dict[str, Any],
    change_definition: dict[str, Any],
    inplace: bool = False,
    recursive: bool = False
):
    """Changes the data in `data` according to the `change_definition`.

    Parameters
    ----------
    data
        Dictionary containing data.
    change_definition
        Dictionary (partially) mirroring the structure of `data`,
        ultimately containing `ChangeAction` objects that define how
        the data in the corresponding key of `data` should be changed.
    inplace
        True if `data` is changed inplace, False if a modified copy is
        returned.
    recursive
        True if recursion into dictionaries within the main `data`
        dictionary is required, False otherwise.
    """
    if inplace:
        changed_data = data
    else:
        changed_data = deepcopy(data)

    return _change_input_recursive(changed_data, change_definition, inplace, recursive)


def _change_input_recursive(
    data: dict[str, Any],
    change_definition: dict[str, Any],
    inplace: bool = False,
    recursive: bool = False
):
    """Changes the data in `data` according to the `change_definition`.

    Parameters
    ----------
    data
        Dictionary containing data.
    change_definition
        Dictionary (partially) mirroring the structure of `data`,
        ultimately containing `ChangeAction` objects that define how
        the data in the corresponding key of `data` should be changed.
    inplace
        True if `data` is changed inplace, False if a modified copy is
        returned.
    recursive
        True if recursion into dictionaries within the main `data`
        dictionary is required, False otherwise.
    """
    for key, change in change_definition.items():
        assert key in data, f"Change cannot be applied to non-existent key '{key}'."
        if recursive and isinstance(change, dict):
            data[key] = _change_input_recursive(data, change, inplace, recursive)
        else:
            data[key] = change.apply(data[key])
    return data


def change_sector(orig_sector: Sector, change_definition: dict[str, Any], inplace: bool = False):
    """Changes the data in `orig_sector` according to the
    `change_definition`.

    Parameters
    ----------
    orig_sector
        Sector containing data.
    change_definition
        Dictionary (partially) mirroring the structure of
        `orig_sector.prep_data`, ultimately containing `ChangeAction`
        objects that define how the data in the corresponding key of
        `orig_sector.prep_data` and `orig_sector.all_data` should be
        changed.
    inplace
        True if `orig_sector` is changed inplace, False if a modified
        copy is returned.
    """
    if inplace:
        # Modify the prep_data and all_data member variables
        orig_sector.prep_data = change_input(
            orig_sector.prep_data,
            change_definition,
            inplace = inplace,
            recursive = False
        )
        orig_sector.all_data = change_input(
            orig_sector.all_data,
            change_definition,
            inplace = inplace,
            recursive = False
        )
        return orig_sector
    else:
        # Create a new Sector object from scratch
        new_data = change_input(
            orig_sector.prep_data,
            change_definition,
            inplace = inplace,
            recursive = False
        )
        return Sector(
            orig_sector.name,
            new_data
        )
