"""Module containing classes and methods to create a survival matrix."""
from functools import cached_property
from itertools import islice
from typing import Optional

import numpy as np
import xarray as xr

from imagematerials.constants import SUBTYPE_SEPARATOR
from imagematerials.distribution import NAME_TO_DIST
from imagematerials.concepts import vehicle_knowledge_graph


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
            depend, which could be the "Type", "region", both or neither.

        """
        # The XArray.Datarray that contains the values is part of the SurvivalMatrix object.
        # Method to create this matrix is part of the survival object,
        # because it know the dimensions.
        self.survival_matrix = survival.new_matrix()
        self._cached_timesteps = set()  # Cohorts that are already computed.
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
            # Convert the slice into a list.
            comp_list = islice(self.survival_matrix.indexes["Cohort"],
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
                 output_modes: Optional[list, xr.DataArray] = None):
        """Initialize scipysurvival class.

        Parameters
        ----------
        lifetime_parameters
            Output from convert_life_time_vehicles function. This should be a dictionary, with
            the keys the name of the distribution, and the values a xr.DataArray with the scipy
            parameters. E.g. weibull or folded_normal.
            mandatory dimensions for the arrays: Cohort, Type, ScipyParam
            optional dimension: Region
        output_modes:
            To allow for sub types that have the same lifetime as the super type.
            By default, this value is None, in which case it is assumed that all sub types
            (if any) have their lifetimes specified.

        """
        self.lifetime_parameters = lifetime_parameters
        if output_modes is not None:
            try:
                # Xarray coordinates, convert to np.NDArray
                self._output_modes = output_modes.values
            except AttributeError:
                # Lists, and other
                self._output_modes = output_modes
        else:
            self._output_modes = None

    def new_matrix(self):
        """Create a new data array with zeros everywhere.

        Returns
        -------
            A DataArray with the correct dimensions and zeros everywhere.

        """
        coords = {"Time": self.time_series.to_numpy(), "Cohort": self.time_series.to_numpy()}
        coords.update(self.extra_coords)
        return xr.DataArray(
            0.0, dims=("Time", "Cohort", *self.extra_dims),
            coords=coords
        )

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
        # Get the number of time steps in the future including the current year.
        n_coords_left = len(self.time_series.loc[cohort:])
        res_arrays = []
        for dist_name, param_array in self.lifetime_parameters.items():
            var_shape = [len(param_array.coords[dim]) for dim in param_array.dims if dim not in ["Time", "ScipyParam"]]
            # n_modes = len(param_array.coords["Type"])
            # The index array signifies the relative time delta from the cohort to the
            # future.
            index_array = np.broadcast_to(self.dt*np.arange(n_coords_left), reversed((n_coords_left, *var_shape))).T
            param_dict = {}
            for param_name in ["c", "scale", "loc"]:
                if param_name in param_array.attrs:
                    param_dict[param_name] = param_array.attrs[param_name]
                else:
                    param_dict[param_name] = param_array.loc[{"Time": cohort, "ScipyParam": param_name}]
            method = NAME_TO_DIST[dist_name].method
            res_numpy_array = method(index_array, **param_dict)
            array_coords = {"Time": self.time_series.loc[cohort:]}
            array_coords.update({dim: param_array.coords[dim] for dim in param_array.dims
                                 if dim not in ["Time", "ScipyParam"]})
            res_arrays.append(xr.DataArray(res_numpy_array, dims=list(array_coords),
                                           coords=array_coords))
        base_array = xr.concat(res_arrays, dim="Type", coords="minimal")

        if self._output_modes is None:
            # Not needed to deal with subtypes
            return base_array

        vehicle_knowledge_graph.rebroadcast_xarray(base_array, self._output_modes)

        # # Deal with subtypes/submodes of the form "{mode} - {submode}"
        # base_modes = base_array.coords["Type"].values
        # keep_modes = []
        # coords = {coord.name: coord for coord in base_array.coords.values()}
        # coords["Type"] = self._output_modes
        # new_array = xr.DataArray(0.0, dims=base_array.dims, coords=coords)
        # for mode in self.modes:
        #     base_mode = mode.split(SUBTYPE_SEPARATOR)[0]
        #     if mode in base_modes:
        #         keep_modes.append(mode)
        #     elif base_mode in base_modes:
        #         new_array.loc[{"Type": mode}] = base_array.loc[{"Type": base_mode}]
        #     else:
        #         raise ValueError(f"Unknown mode '{mode}' needed for survival matrix, "
        #                          "but lifetime unknown.")
        # new_array.loc[{"Type": keep_modes}] = base_array.loc[{"Type": keep_modes}]
        # return new_array

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
                all_modes.extend(str(x) for x in param_dict_array.coords["Type"].to_numpy())
        else:
            all_modes = self._output_modes
        return all_modes

    @property
    def time_series(self) -> xr.DataArray:
        """Get all the time values in the simulation."""
        first_array = list(self.lifetime_parameters.values())[0]
        return first_array.coords["Time"]

    @cached_property
    def extra_dims(self) -> list[str]:
        dims = []
        first_array = list(self.lifetime_parameters.values())[0]
        for dim in first_array.dims:
            if dim not in ["Time", "ScipyParam"]:
                dims.append(dim)
        return dims

    @cached_property
    def extra_coords(self) -> list[list]:
        first_array = list(self.lifetime_parameters.values())[0]
        coords = {}
        for dim in self.extra_dims:
            if dim == "Type":
                coords[dim] = self.modes
            else:
                coords[dim] = first_array.coords[dim]
        return coords

    @cached_property
    def dt(self) -> int:
        first_array = list(self.lifetime_parameters.values())[0]
        dt = first_array.coords["Time"].values[1] - first_array.coords["Time"].values[0]
        assert dt == 1
        return dt
