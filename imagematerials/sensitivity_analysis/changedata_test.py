"""Module to modify input to the models."""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
import warnings

from imagematerials.factory import Sector

# change_definition = {
#     "material_fractions": ChangeReplace(value),
#     "old_key":            ChangeRename("new_key"),
#     "obsolete_key":       ChangeDelete(),
#     "brand_new_key":      ChangeAdd(some_value),
# }

@dataclass
class ChangeAction(ABC):
    """Abstract base class for all ChangeActions to inherit from.
    """

    # ### ADDED ###
    # Class attribute (NOT a dataclass field, so it does not interfere with
    # the field order of subclasses).
    #   True  -> the key must already exist in the data.
    #   False -> the key must NOT already exist (used by ChangeAdd).
    requires_existing_key = True

    # ### CHANGED ###
    # `apply` is no longer abstract. Actions that work on the value itself
    # (e.g. ChangeReplace) implement it; actions that work on the key itself
    # (delete/add/rename) override `apply_to_dict` instead.
    def apply(self, value: Any):
        """Applies the change defined by this ChangeAction object to the
        `value` passed.

        Parameters
        ----------
        value
            The element the change should be applied to.
        """
        raise NotImplementedError

    # ### ADDED ###
    @abstractmethod
    def apply_to_dict(self, data: dict[str, Any], key: str):
        """Applies the change to `data` at `key`, modifying `data` in place.

        This is the entry point used by `change_input`. It exists because
        `apply` only ever sees the value, so it cannot add, delete or
        rename keys.

        Parameters
        ----------
        data
            The dictionary containing the element to change.
        key
            The key in `data` the change applies to.
        """
        pass


# ### ADDED ###
@dataclass
class ChangeValueAction(ChangeAction):
    """Base class for ChangeActions that only modify the value of an
    existing key. Subclasses only need to implement `apply`.
    """

    def apply_to_dict(self, data: dict[str, Any], key: str):
        data[key] = self.apply(data[key])


# ### CHANGED ###
# Now inherits from ChangeValueAction instead of ChangeAction.
# The body of the class is unchanged.
@dataclass
class ChangeReplace(ChangeValueAction):
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


# ### ADDED ###
@dataclass
class ChangeDelete(ChangeAction):
    """ChangeAction for deleting a key entirely."""

    def apply_to_dict(self, data: dict[str, Any], key: str):
        del data[key]


# ### ADDED ###
@dataclass
class ChangeAdd(ChangeAction):
    """ChangeAction for adding a new key that does not exist yet."""
    new_value: Any

    requires_existing_key = False

    def apply_to_dict(self, data: dict[str, Any], key: str):
        data[key] = deepcopy(self.new_value)


# ### ADDED ###
@dataclass
class ChangeRename(ChangeAction):
    """ChangeAction for renaming a key, keeping its value.

    Note: the renamed key is moved to the end of the dictionary.
    """
    new_key: str

    def apply_to_dict(self, data: dict[str, Any], key: str):
        assert self.new_key not in data, f"Cannot rename to already existing key '{self.new_key}'."
        data[self.new_key] = data.pop(key)


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
        if recursive and isinstance(change, dict):
            assert key in data, f"Change cannot be applied to non-existent key '{key}'."
            # ### FIXED ### was `_change_input_recursive(data, ...)`, which
            # passed the whole dict instead of the sub-dict at `key`.
            data[key] = _change_input_recursive(data[key], change, inplace, recursive)
        else:
            # ### CHANGED ###
            # The key-existence check now depends on the action:
            # ChangeAdd requires the key to be absent, all others require it.
            if change.requires_existing_key:
                assert key in data, f"Change cannot be applied to non-existent key '{key}'."
            else:
                assert key not in data, f"Cannot add already existing key '{key}'."
            # ### CHANGED ###
            # was `data[key] = change.apply(data[key])`
            change.apply_to_dict(data, key)
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
