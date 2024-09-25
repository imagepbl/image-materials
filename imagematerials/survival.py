from functools import cached_property
from itertools import islice

import numpy as np
import xarray as xr

from imagematerials.distribution import NAME_TO_DIST


class SurvivalMatrix:
    def __init__(self, survival):
        # TODO: Add docstrings
        self.survival_matrix = survival.new_matrix()
        self._cached_timesteps = set()
        self.num_timesteps = len(survival.time_series)
        self.survival = survival

    def __getitem__(self, idx):
        # TODO: make the computation dependent on t_idx
        # We know that if t_idx < cohort_idx, the result is 0,
        # So if we compute s[t, :], we know that we only need to compute
        # the columns until cohort_idx == t
        t, cohort = idx

        # TODO: Check if this actually creates the cohort indices properly.
        if isinstance(cohort, slice):
            comp_list = islice(self.survival_matrix.indexes["cohort"], *cohort.indices(self.num_timesteps))
        else:
            comp_list = [int(cohort)]

        # Is the column already computed?
        for cur_cohort in comp_list:
            if cur_cohort not in self._cached_timesteps:
                # Compute the survival matrix for the current cohort_idx
                # The survival coeficients are only relevant for future times t
                # So the values are created in column cohort_idx, for t_idx=cohort_idx, cohort_idx+1, ...
                self.survival_matrix.loc[cur_cohort:, cur_cohort] = self.survival.compute_survival(cur_cohort)
                self._cached_timesteps.add(cur_cohort)
        return self.survival_matrix.loc[t, cohort]


class ScipySurvival():
    """Method based on scipy distributions to compute the survival matrix.
    
    This version has no region dependency for the survival matrix. This would need
    to be implemented still. This version for computing the survival matrix is
    static, where all values are already known beforehand.
    """

    def __init__(self, lifetime_parameters: dict[str, xr.DataArray]):
        """Initialize scipysurvival class

        Parameters
        ----------
        lifetime_parameters
            Output from convert_life_time_vehicles function.
        """
        self.lifetime_parameters = lifetime_parameters

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
            # params = {param_name: (full_array.loc[cohort, :] if isinstance(full_array, xr.DataArray) else full_array)
                    #   for param_name, full_array in param_dict_array.items()}
            res_numpy_array = method(index_array, **param_dict)
            res_arrays.append(xr.DataArray(res_numpy_array, dims=("time", "mode"),
                                           coords={"time": self.time_series.loc[cohort:],
                                                   "mode": param_array.coords["mode"]}))
        return xr.concat(res_arrays, dim="mode")

    @cached_property
    def modes(self) -> list[str]:
        """Return all the modes in survival matrix.

        Cache it for performance.

        Returns
        -------
            All the modes for which there is a survival distribution in list form.

        """
        all_modes = []
        for param_dict_array in self.lifetime_parameters.values():
            all_modes.extend(str(x) for x in param_dict_array.coords["mode"].to_numpy())
        return all_modes

    @property
    def time_series(self) -> xr.DataArray:
        """Get all the time values in the simulation."""
        first_array = list(self.lifetime_parameters.values())[0]
        return first_array.coords["time"]

