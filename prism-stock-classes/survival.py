import numpy as np
import scipy
from itertools import islice
import xarray as xr
from functools import cached_property
from collections import defaultdict


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
        for param_dict_array, method in self.lifetime_parameters.values():
            first_array = list(param_dict_array.values())[0]
            n_modes = first_array.shape[1]
            # The index array signifies the relative time delta from the cohort to the
            # future.
            # TODO: Fix this so that this works for dt != 1
            index_array = np.empty((n_coords_left, n_modes))
            index_array[:, :] = np.arange(n_coords_left).reshape(-1, 1)
            params = {param_name: (full_array.loc[cohort, :] if isinstance(full_array, xr.DataArray) else full_array)
                      for param_name, full_array in param_dict_array.items()}
            res_numpy_array = method(index_array, **params)
            res_arrays.append(xr.DataArray(res_numpy_array, dims=("time", "mode"),
                                           coords={"time": self.time_series.loc[cohort:],
                                                   "mode": first_array.coords["mode"]}))
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
        for param_dict_array, _ in self.lifetime_parameters.values():
            first_array = list(param_dict_array.values())[0]
            all_modes.extend(str(x) for x in first_array.coords["mode"].to_numpy())
        return all_modes

    @property
    def time_series(self) -> xr.DataArray:
        """Get all the time values in the simulation."""
        first_dict = list(self.lifetime_parameters.values())[0][0]
        first_array = list(first_dict.values())[0]
        return first_array.coords["time"]

    @classmethod
    def from_lifetime_vehicles(cls, lifetime_vehicles):
        """Convenience class method to quicly create the survival instance."""
        lifetime_parameters = convert_life_time_vehicles(lifetime_vehicles)
        return cls(lifetime_parameters)


class WeibullDistribution():
    """Weibull Distribution with parameters shape and scale"""
    name = "weibull"
    method = scipy.stats.weibull_min.sf
    params = ["shape", "scale"]

    @staticmethod
    def get_param(param_dict):
        """Get the parameters into the right format for Scipy."""
        return {
            "c": param_dict["shape"],
            "scale": param_dict["scale"],
            "loc": 0
        }

    @staticmethod
    def has_param(param):
        """Used to check whether the parameters are compatible with the Weibull dist."""
        if "shape" in param and "scale" in param:
            return True
        return False


class FoldedNormalDistribution():
    """Folded Normal distirbution with parameters mean and stdev"""
    name = "folded_norm"
    method = scipy.stats.foldnorm.sf
    params = ["mean", "stdev"]

    @staticmethod
    def get_param(param_dict):
        """Get the parameters into the right format for Scipy."""
        mean, stdev = param_dict["mean"], param_dict["stdev"]
        return {"c": mean/stdev, "scale": stdev, "loc": 0}

    @staticmethod
    def has_param(param):
        """Used to check whether the parameters are compatible with the fold_norm dist."""
        if "mean" in param and "stdev" in param:
            return True
        return False



ALL_DISTRIBUTIONS = [WeibullDistribution, FoldedNormalDistribution]
NAME_TO_DIST = {dist.name: dist for dist in ALL_DISTRIBUTIONS}


def convert_life_time_vehicles(life_time_vehicles: xr.Dataset) -> dict[str, xr.DataArray]:
    """Convert lifetime vehicles dataset to a more appropriate data format.

    This conversion should probably move to the preprocessing stage after we figure out
    the exact details of what the output should look like.

    Parameters
    ----------
    life_time_vehicles
        The input life_time_vehicles xarray dataset. It is supposed to be in a very particular format:
        it contains the parameters for each of the modes. However, the distribution types are not
        the same for each of the modes. Thus, the distribution types need to be inferred from the
        names of the parameters that are given. If multiple parameter sets for multiple distributions
        are given, the Weibull distribution is given preference over the FoldedNormal distribution.

    Returns
    -------
        A dictionary that contains a data array for each of the distributions. Given the setup there is
        an implicit assumption that only one distribution is used for each of the modes. If distribution
        types change over time, this data structure needs to be adjusted.
    """

    # Create a dictionary to find which parameters are available for which mode.
    mode_param = defaultdict(list)
    for mode, par in life_time_vehicles.data_vars:
        mode_param[mode].append(par)

    # Create a dictionary that says which modes are tied to which distribution.
    dist_mode = defaultdict(list)
    modes_done = set()  # temporary
    for dist in ALL_DISTRIBUTIONS:
        for mode, param in mode_param.items():
            if mode not in modes_done and dist.has_param(param):
                dist_mode[dist.name].append(mode)
                modes_done.add(mode)

    # Iterate over all distributions to create a data array for each of them.
    scipy_params = {}
    for dist_name, mode_list in dist_mode.items():
        dist = NAME_TO_DIST[dist_name]

        param_arrays = {}
        for param in dist.params:
            array = xr.DataArray(
                0.0, dims=("time", "mode"),
                coords={
                    "time": life_time_vehicles.coords["year"].to_numpy(),
                    "mode": mode_list})
            for mode in mode_list:
                array.loc[:, str(mode)] = life_time_vehicles.data_vars[str(mode), param].to_numpy()
            param_arrays[param] = array
        scipy_params[dist_name] = (dist.get_param(param_arrays), dist.method)
    return scipy_params
