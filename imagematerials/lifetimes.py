from typing import Union

import numpy as np
import xarray as xr
from imagematerials.distribution import NAME_TO_DIST


def lifetimes_to_matrix(
        lifetime_parameters: dict[str, xr.DataArray],
        output_modes: Union[None, list, xr.DataArray] = None,
        knowledge_graph = None) -> xr.DataArray:

    if output_modes is not None:
        try:
            # Xarray coordinates, convert to np.NDArray
            modes = output_modes.values
        except AttributeError:
            # Lists, and other
            modes = output_modes
    else:
        modes = []
        for param_dict_array in lifetime_parameters.values():
            modes.extend(str(x) for x in param_dict_array.coords["Type"].to_numpy())

    first_array = list(lifetime_parameters.values())[0]
    time_series = first_array.coords["Time"]
    extra_dims = [dim for dim in first_array.dims if dim not in ["Time", "ScipyParam"]]
    extra_coords = {dim: modes if dim == "Type" else first_array.coords[dim] for dim in extra_dims}
    dt = first_array.coords["Time"].values[1] - first_array.coords["Time"].values[0]

    coords = {"Time": time_series.to_numpy(), "Cohort": time_series.to_numpy()}
    coords.update(extra_coords)
    matrix = xr.DataArray(
        0.0, dims=("Time", "Cohort", *extra_dims),
        coords=coords
    )

    for t in time_series.values:
        # print(matrix.coords["Type"], modes)
        matrix.loc[t:, t] = compute_cohort(t, lifetime_parameters, time_series, dt, output_modes,
                                           knowledge_graph)
    return matrix


def compute_cohort(cohort: int, lifetime_parameters, time_series, dt, output_modes, knowledge_graph
                   ) -> xr.DataArray:
    """Compute the survival fractions for one cohort, but all modes.

    Parameters
    ----------
    cohort : int
        The cohort for which to compute the survival matrix. This should
        not be the cohort index.

    Returns
    -------
    xr.DataArray
        An array with the survival fractions for the current cohort at all
        future times.

    """
    # Get the number of time steps in the future including the current year.
    n_coords_left = len(time_series.loc[cohort:])
    res_arrays = []
    for dist_name, param_array in lifetime_parameters.items():
        var_shape = [len(param_array.coords[dim]) for dim in param_array.dims if dim not in ["Time", "ScipyParam"]]
        # n_modes = len(param_array.coords["Type"])
        # The index array signifies the relative time delta from the cohort to the
        # future.
        index_array = np.broadcast_to(dt*np.arange(n_coords_left), reversed((n_coords_left, *var_shape))).T
        param_dict = {}
        for param_name in ["c", "scale", "loc"]:
            if param_name in param_array.attrs:
                param_dict[param_name] = param_array.attrs[param_name]
            else:
                param_dict[param_name] = param_array.loc[{"Time": cohort, "ScipyParam": param_name}]
        method = NAME_TO_DIST[dist_name].method
        res_numpy_array = method(index_array, **param_dict)
        array_coords = {"Time": time_series.loc[cohort:]}
        array_coords.update({dim: param_array.coords[dim] for dim in param_array.dims
                                if dim not in ["Time", "ScipyParam"]})
        res_arrays.append(xr.DataArray(res_numpy_array, dims=list(array_coords),
                                        coords=array_coords))
    base_array = xr.concat(res_arrays, dim="Type", coords="minimal")

    if output_modes is None or list(output_modes) == list(base_array.coords["Type"].to_numpy()):
        # Not needed to deal with subtypes
        return base_array

    if knowledge_graph is None:
        if len(set(output_modes) - set(base_array.coords["Type"].to_numpy())) == 0:
            return base_array.reindex({"Type": output_modes})
        raise ValueError(f"Need knowledge graph for broadcasting to {output_modes} from "
                         f"{base_array.coords.values()}.")
    return knowledge_graph.rebroadcast_xarray(base_array, output_modes)
