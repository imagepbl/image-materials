"""Test survival classes."""
import numpy as np
import xarray as xr
from pytest import mark

from imagematerials.lifetimes import lifetimes_to_matrix


def create_life_times(prefix, more_life=0, time_size=30):
    time = np.arange(time_size)+1900
    mode = [f"{prefix}_{i}" for i in range(4)]
    array = xr.DataArray(0.0, dims=["Time", "Type", "ScipyParam"],
                         coords={"Time": time, "Type": mode, "ScipyParam": ["c", "scale"]})
    for i_mode, mode_name in enumerate(mode):
        array.loc[:, mode[i_mode], "c"] = 4
        array.loc[:, mode[i_mode], "scale"] = i_mode + 1
    array.attrs["loc"] = 0
    return array


def _check_part(sm_part, i_cohort, n_cohort, n_modes):
    assert sm_part.shape == (n_cohort-i_cohort, n_modes)
    assert np.all(sm_part[i_cohort+1:, :] >= 0 )
    assert np.all(np.isclose(sm_part[0, :], 1))
    assert not np.all(np.isclose(sm_part[1:, :], 1))


@mark.parametrize("time_size", [5, 10, 30])
def test_scipy_survival(time_size):
    lifetime_vehicles = {
        "weibull": create_life_times("weibull", time_size=time_size),
        "folded_norm": create_life_times("folded_norm", time_size=time_size)
    }
    survival_matrix = lifetimes_to_matrix(lifetime_vehicles)
    assert len(survival_matrix.coords["Type"]) == 8
    assert len(survival_matrix.coords["Time"]) == time_size
    assert len(survival_matrix.coords["Cohort"]) == time_size
    sm_numpy = survival_matrix.to_numpy().mean(axis=2)

    # Diagonal should be 1
    assert np.all(np.isclose(np.diag(sm_numpy), 1))

    # Below diagonal should be between 0 and 1
    assert np.all(np.tril(sm_numpy, k=-1) < 1)
    assert np.all(sm_numpy[np.tril_indices(time_size, k=-1)] > 0)

    # Above diagonal should be 0
    assert np.all(np.isclose(np.triu(sm_numpy, k=1), 0))

    # Check whether a reordering of the output coordinates works without a knowledge graph
    new_output_modes = list(reversed(survival_matrix.coords["Type"].to_numpy()))
    survival_matrix_2 = lifetimes_to_matrix(
        lifetime_vehicles,
        output_modes=new_output_modes)

    assert list(survival_matrix_2.coords["Type"].to_numpy()) == new_output_modes
