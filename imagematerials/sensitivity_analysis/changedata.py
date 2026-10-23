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

    "abstract" means this class is never used directly -- it just defines what every ChangeAction 
    must be able to do. Each specific kind of change (replace, delete, add, rename) is written as 
    its own subclass below.
    """
    # This is a plain class attribute (not a dataclass field, so it doesn't show up as a constructor
    # argument). It tells `_change_input_recursive` whether the key this action is applied to must 
    # already exist in the data.
    # - True  (default): the key must already exist (replace, delete, rename).
    # - False: the key must NOT already exist yet (used by ChangeAdd).
    requires_existing_key = True

    @abstractmethod
    def apply_to_dict(self, data: dict[str, Any], key: str):
        """Applies the change defined by this ChangeAction object to `data` at `key`, in place.

        Every subclass must implement this, because it's the one method that 
        `_change_input_recursive` actually calls to perform the change. It receives the whole 
        dictionary (not just the value) because some changes -- like delete, add, or rename -- need 
        to affect the dictionary's keys, not just the value stored under a key.

        Parameters
        ----------
        data
            The dictionary containing the element to change.
        key
            The key in `data` that the change applies to.
        """
        pass

@dataclass
class ChangeReplace(ChangeAction):
    """ChangeAction for replacing a value with an entirely `new_value`.

    """
    new_value: Any

    def apply_to_dict(self, data: dict[str, Any], key: str):
        """Replaces `data[key]` with `self.new_value`.

        Parameters
        ----------
        data
            The dictionary containing the element to change.
        key
            The key in `data` whose value should be replaced.
        """
        old_value = data[key]
        if type(old_value) != type(self.new_value):
            # Just a check, not an error: replacing e.g. a number with a string is allowed, but it's 
            # often a mistake, so a warning is issued instead of silently letting it happen.
            warnings.warn(
                f"Replacing a value of type {type(old_value)} with one of type {type(self.new_value)}."
                f" Replacing {old_value} with {self.new_value} will cause its type to change.",
                RuntimeWarning
            )
        # deepcopy so that changing new_value later doesn't accidentally also change the value we 
        # just stored in `data`.
        data[key] = deepcopy(self.new_value)

    def apply(self, value: Any):
        """Replaces `value` with `self.new_value`. With apply() also usable outside of changing a 
        dictionary.

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

@dataclass
class ChangeDelete(ChangeAction):
    """ChangeAction for deleting a key (and its value) entirely."""

    def apply_to_dict(self, data: dict[str, Any], key: str):
        """Removes `key` from `data`.

        Parameters
        ----------
        data
            The dictionary to delete the key from.
        key
            The key to remove.
        """
        del data[key]

@dataclass
class ChangeAdd(ChangeAction):
    """ChangeAction for adding a brand new key that does not exist yet.

    Use this instead of ChangeReplace when the key is not already present in the data 
    -- `_change_input_recursive` checks this and will raise a clear error if the key already exists.
    """
    new_value: Any

    # Unlike the other actions, adding requires the key to be ABSENT, not present. This overrides 
    # the default set on ChangeAction.
    requires_existing_key = False

    def apply_to_dict(self, data: dict[str, Any], key: str):
        """Adds `key` to `data` with value `self.new_value`.

        Parameters
        ----------
        data
            The dictionary to add the new key to.
        key
            The new key to add.
        """
        data[key] = deepcopy(self.new_value)

@dataclass
class ChangeRename(ChangeAction):
    """ChangeAction for renaming a key while keeping its value.

    Note: because of how Python dictionaries work, the renamed key ends up at the END of the 
    dictionary (its original position is not preserved).
    """
    new_key: str

    def apply_to_dict(self, data: dict[str, Any], key: str):
        """Renames `key` to `self.new_key` in `data`, keeping the value.

        Parameters
        ----------
        data
            The dictionary containing the key to rename.
        key
            The existing key to rename.
        """
        assert self.new_key not in data, f"Cannot rename to already existing key '{self.new_key}'."
        # .pop(key) removes `key` and returns its value in one step, so we can immediately store 
        # that value under the new name.
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
            # `change` is itself a dictionary of changes, meaning we need to go one level deeper 
            # into `data[key]` and apply those changes there instead of at this level.
            assert key in data, f"Change cannot be applied to non-existent key '{key}'."
            data[key] = _change_input_recursive(data[key], change, inplace, recursive)
        else:
            # Whether the key needs to already exist depends on the specific action: ChangeAdd 
            # expects it to be ABSENT, everything else expects it to already be there.
            if change.requires_existing_key:
                assert key in data, f"Change cannot be applied to non-existent key '{key}'."
            else:
                assert key not in data, f"Cannot add already existing key '{key}'."
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
            new_data,
            check_coordinates = orig_sector.check_coordinates,
        )
