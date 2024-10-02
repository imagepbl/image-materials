"""Module containing classes and methods to create a survival matrix."""
from functools import cached_property
from itertools import islice
from typing import Optional

import numpy as np
import xarray as xr

from imagematerials.constants import SUBTYPE_SEPARATOR
from imagematerials.distribution import NAME_TO_DIST


class SurvivalMatrix:
    """Container class to create a survival matrix.

    The survival matrix itself is oblivious to the dimensions of the matrix to keep
    it flexible. In some cases we might have a regional dependence, in other we might
    not. This will depend on the input data for the model.
    """

    def __init__(self, survival):
        """Initialize the survival matrix.

        Parameters
        ----------
        survival
            The survival object is a class that knows how to compute the survival matrix.
            It contains the information on which dimensions the life times of the stocks
            depend, which could be the "mode", "region", both or neither.

        """
        # TODO: Add docstrings
        self.survival_matrix = survival.new_matrix()
        self._cached_timesteps = set()
        self.num_timesteps = len(survival.time_series)
        self.survival = survival

    def __getitem__(self, idx):
        """Use the survival matrix as a numpy array."""
        # TODO: make the computation dependent on t_idx
        # We know that if t_idx < cohort_idx, the result is 0,
        # So if we compute s[t, :], we know that we only need to compute
        # the columns until cohort_idx == t
        t, cohort = idx

        # TODO: Check if this actually creates the cohort indices properly.
        if isinstance(cohort, slice):
            comp_list = islice(self.survival_matrix.indexes["cohort"],
                               *cohort.indices(self.num_timesteps))
        else:
            comp_list = [int(cohort)]

        # Is the column already computed?
        for cur_cohort in comp_list:
            if cur_cohort not in self._cached_timesteps:
                # Compute the survival matrix for the current cohort
                # The survival coeficients are only relevant for future times t
                # So the values are created in column cohort, for t=cohort, cohort+1, ...
                self.survival_matrix.loc[cur_cohort:, cur_cohort] = self.survival.compute_survival(
                    cur_cohort)
                self._cached_timesteps.add(cur_cohort)
        return self.survival_matrix.loc[t, cohort]


class ScipySurvival():
    """Method based on scipy distributions to compute the survival matrix.

    This version has no region dependency for the survival matrix. This would need
    to be implemented still. This version for computing the survival matrix is
    static, where all values are already known beforehand.
    """

    def __init__(self, lifetime_parameters: dict[str, xr.DataArray],
                 output_modes: Optional[xr.DataArray] = None):
        """Initialize scipysurvival class.

        Parameters
        ----------
        lifetime_parameters
            Output from convert_life_time_vehicles function.
        output_modes:
            To allow for sub types that have the same lifetime as the super type.
            By default, this value is None, in which case it is assumed that all sub types
            (if any) have their lifetimes specified.

        """
        self.lifetime_parameters = lifetime_parameters
        self._output_modes = output_modes.values

    def new_matrix(self):
        """Create a new data array with zeros everywhere.

        Returns
        -------
            A DataArray with the correct dimensions and zeros everywhere.

        """
        return xr.DataArray(
            0.0, dims=("time", "cohort", "mode"),
            coords={"time": self.time_series.to_numpy(),
                    "cohort": self.time_series.to_numpy(),
                    "mode": self.modes})

    def compute_survival(self, cohort: int) -> xr.DataArray:
        """Compute the survival fractions for one cohort, but all modes.

        Parameters
        ----------
        cohort
            The cohort for which to compute the survival matrix. This should
            not be the cohort index.

        Returns
        -------
            An array with the survival fractions for the current cohort at all
            future times.

        """
        n_coords_left = len(self.time_series.loc[cohort:])
        res_arrays = []
        for dist_name, param_array in self.lifetime_parameters.items():
            n_modes = param_array.shape[1]
            # The index array signifies the relative time delta from the cohort to the
            # future.
            # TODO: Fix this so that this works for dt != 1
            index_array = np.empty((n_coords_left, n_modes))
            index_array[:, :] = np.arange(n_coords_left).reshape(-1, 1)
            param_dict = {}
            for param_name in ["c", "scale", "loc"]:
                if param_name in param_array.attrs:
                    param_dict[param_name] = param_array.attrs[param_name]
                else:
                    param_dict[param_name] = param_array.loc[cohort, :, param_name]
            method = NAME_TO_DIST[dist_name].method
            res_numpy_array = method(index_array, **param_dict)
            res_arrays.append(xr.DataArray(res_numpy_array, dims=("time", "mode"),
                                           coords={"time": self.time_series.loc[cohort:],
                                                   "mode": param_array.coords["mode"]}))
        base_array = xr.concat(res_arrays, dim="mode", coords="minimal")
        if self._output_modes is None:
            return base_array
        new_array = xr.DataArray(0.0, dims=("time", "mode"),
                                 coords={"time": base_array.coords["time"],
                                         "mode": self.modes})
        base_modes = base_array.coords["mode"].values
        for mode in self.modes:
            base_mode = mode.split(SUBTYPE_SEPARATOR)[0]
            if mode in base_modes:
                new_array.loc[:, mode] = base_array.loc[:, mode]
            elif base_mode in base_modes:
                new_array.loc[:, mode] = base_array.loc[:, base_mode].to_numpy()
            else:
                raise ValueError(f"Unknown mode '{mode}' needed for survival matrix, "
                                 "but lifetime unknown.")
        return new_array

    @cached_property
    def modes(self) -> list[str]:
        """Return all the modes in survival matrix.

        Cache it for performance.

        Returns
        -------
            All the modes for which there is a survival distribution in list form.

        """
        if self._output_modes is None:
            all_modes = []
            for param_dict_array in self.lifetime_parameters.values():
                all_modes.extend(str(x) for x in param_dict_array.coords["mode"].to_numpy())
        else:
            all_modes = self._output_modes
        return all_modes

    @property
    def time_series(self) -> xr.DataArray:
        """Get all the time values in the simulation."""
        first_array = list(self.lifetime_parameters.values())[0]
        return first_array.coords["time"]

