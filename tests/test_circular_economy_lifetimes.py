"""Tests for the building lifetime-extension circular-economy measure.

Covers:
* ``FlagLifetimeExtensionSlow`` - the config-driven Weibull-scale rescaling in
  ``imagematerials.buildings.preprocessing.circular_economy_measures``.
* the mutual-exclusion check between the two building lifetime-extension flags in
  ``imagematerials.util.validate_resource_efficiency_flags``.
"""
from math import gamma

import numpy as np
import pytest
import xarray as xr

from imagematerials.buildings.preprocessing.circular_economy_measures import (
    apply_lifetime_extension_buildings,
)
from imagematerials.util import validate_resource_efficiency_flags

ANCHOR_YEAR = 2020
TARGET_YEAR = 2060

# SSP2_CP baseline used by _make_lifetimes_array: shape 2, scale 50 everywhere.
BASE_SHAPE = 2.0
BASE_SCALE = 50.0


def _weibull_mean(c, scale):
    return scale * gamma(1.0 + 1.0 / c)


BASE_MEAN = _weibull_mean(BASE_SHAPE, BASE_SCALE)


def _make_lifetimes_array():
    """A minimal (time, Region, Type, Parameter) Weibull-parameter array.

    Two regions (codes "1" = Canada, "20" = China), one residential and one
    commercial type, constant shape=2 and scale=50 across 2000..2100 - i.e. the
    already-interpolated SSP2_CP database as ``compute_lifetimes`` produces it.
    """
    times = np.arange(2000, 2101, 10)
    regions = np.array([1, 20])
    types = ["Detached - Urban", "Office"]
    params = ["Shape", "Scale"]

    data = np.zeros((len(times), len(regions), len(types), len(params)))
    data[..., 0] = BASE_SHAPE
    data[..., 1] = BASE_SCALE
    return xr.DataArray(
        data,
        dims=["time", "Region", "Type", "Parameter"],
        coords={"time": times, "Region": regions, "Type": types, "Parameter": params},
    )


@pytest.fixture
def flag_config():
    return {
        "base_year": 2025,
        "target_year": TARGET_YEAR,
        "residential": {
            "lifetime_increase_percent": {"Canada": 90, "China": 60},
        },
        "non_residential": {
            "lifetime_increase_percent": {"Canada": 90, "China": 90},
        },
    }


def test_anchor_and_target_mean_lifetime(flag_config):
    out = apply_lifetime_extension_buildings(_make_lifetimes_array(), flag_config)

    def mean_at(region, typ, year):
        c = float(out.sel(Region=region, Type=typ, Parameter="Shape", time=year))
        s = float(out.sel(Region=region, Type=typ, Parameter="Scale", time=year))
        return _weibull_mean(c, s)

    # 2020 mean == the SSP2_CP baseline mean (config never overrides it)
    assert mean_at(1, "Detached - Urban", 2020) == pytest.approx(BASE_MEAN)
    assert mean_at(20, "Detached - Urban", 2020) == pytest.approx(BASE_MEAN)
    assert mean_at(1, "Office", 2020) == pytest.approx(BASE_MEAN)
    assert mean_at(20, "Office", 2020) == pytest.approx(BASE_MEAN)

    # target_year mean == baseline * (1 + pct/100)
    assert mean_at(1, "Detached - Urban", TARGET_YEAR) == pytest.approx(BASE_MEAN * 1.9)
    assert mean_at(20, "Detached - Urban", TARGET_YEAR) == pytest.approx(BASE_MEAN * 1.6)
    assert mean_at(20, "Office", TARGET_YEAR) == pytest.approx(BASE_MEAN * 1.9)

    # held flat after target_year
    assert mean_at(20, "Office", 2100) == pytest.approx(mean_at(20, "Office", TARGET_YEAR))


def test_linear_ramp_between_anchor_and_target(flag_config):
    out = apply_lifetime_extension_buildings(_make_lifetimes_array(), flag_config)
    c = float(out.sel(Region=20, Type="Office", Parameter="Shape", time=2040))
    s = float(out.sel(Region=20, Type="Office", Parameter="Scale", time=2040))
    # 2040 is exactly half-way from 2020 to 2060: factor = 1 + 0.5 * 0.9
    assert _weibull_mean(c, s) == pytest.approx(BASE_MEAN * (1 + 0.5 * 0.9))


def test_shape_parameter_is_untouched(flag_config):
    arr = _make_lifetimes_array()
    out = apply_lifetime_extension_buildings(arr, flag_config)
    xr.testing.assert_allclose(out.sel(Parameter="Shape"), arr.sel(Parameter="Shape"))


def test_pre_anchor_years_are_untouched(flag_config):
    arr = _make_lifetimes_array()
    out = apply_lifetime_extension_buildings(arr, flag_config)
    # Every year up to and including 2020 keeps the SSP2_CP scale.
    xr.testing.assert_allclose(
        out.sel(Parameter="Scale", time=slice(None, 2020)),
        arr.sel(Parameter="Scale", time=slice(None, 2020)),
    )


def test_dims_and_coords_preserved(flag_config):
    arr = _make_lifetimes_array()
    out = apply_lifetime_extension_buildings(arr, flag_config)
    assert out.dims == arr.dims
    for dim in arr.dims:
        np.testing.assert_array_equal(out.coords[dim].values, arr.coords[dim].values)


def test_region_absent_from_config_is_unchanged():
    arr = _make_lifetimes_array()
    config = {
        "base_year": 2025,
        "target_year": TARGET_YEAR,
        "residential": {
            "lifetime_increase_percent": {"Canada": 90},
        },
        # no non_residential block -> Office untouched everywhere
    }
    out = apply_lifetime_extension_buildings(arr, config)
    # China residential not in config -> unchanged
    xr.testing.assert_allclose(
        out.sel(Region=20, Type="Detached - Urban"),
        arr.sel(Region=20, Type="Detached - Urban"),
    )
    # Office (commercial) unchanged for both regions
    xr.testing.assert_allclose(out.sel(Type="Office"), arr.sel(Type="Office"))


def test_mutually_exclusive_lifetime_flags_raise():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_resource_efficiency_flags(
            {"buildings": {"FlagLifetimeExtensionSlow": True,
                           "FlagLifetimeExtension2D_RE": True}}
        )


@pytest.mark.parametrize("flags", [
    {"buildings": {"FlagLifetimeExtensionSlow": True}},
    {"buildings": {"FlagLifetimeExtension2D_RE": True}},
    {"buildings": {"FlagLifetimeExtensionSlow": False, "FlagLifetimeExtension2D_RE": False}},
])
def test_single_lifetime_flag_is_allowed(flags):
    validate_resource_efficiency_flags(flags)  # must not raise
