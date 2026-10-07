# DUMMY VALUES!!: mean lifetimes (years) are loosely based on published sources

# Lifetime parameters for appliances (for now only Weibull), following the
# same ScipyParam convention as imagematerials.distribution / imagematerials.lifetimes.

# (US DOE / AHAM appliance-lifespan data: fridge ~14, washing machine ~13,
# dishwasher ~15, dryer ~14, AC ~10); TV/PC/small-electronics lifetimes are a
# rough estimate as no authoritative figure was found.
# The Weibull shape parameter (2.34) is a national-survey-based average shape
# factor for residential appliances, from DOE/OSTI "Using National Survey Data
# to Estimate Lifetimes of Residential Appliances" (osti.gov/servlets/purl/1182737).
# TODO: replace with real / literature-based per-type lifetime distributions.

import xarray as xr
import prism

def define_lifetimes(count_appliances, appliance_types):

    mean_lifetime_years = {
        "Fan": 10, "Air Cooler": 10, "Air Conditioner": 12, "Refrigerator": 14,
        "Microwave": 9, "Washing Machine": 13, "Clothes Dryer": 14,
        "Dish Washer": 15, "TV": 8, "VCR/DVD": 7, "PC/Other": 6,
    }

    WEIBULL_SHAPE = 2.34  # DOE national survey average shape factor, applied to all types

    cohorts = count_appliances.coords["time"].values
    regions = count_appliances.coords["Region"].values

    mean_da = xr.DataArray(
        [[mean_lifetime_years[a] for a in appliance_types] for _ in regions],
        dims=["Region", "Type"],
        coords={"Region": regions, "Type": appliance_types},
    ).expand_dims({"time": cohorts}).transpose("time", "Region", "Type")

    # Weibull: shape held constant at the DOE national-survey average (2.34) across
    # all appliance types, scale = mean lifetime per type (loc = 0).
    weibull_shape = xr.full_like(mean_da, WEIBULL_SHAPE)
    weibull_scale = mean_da

    weibull_lifetimes = xr.concat((weibull_shape, weibull_scale), dim="ScipyParam")
    weibull_lifetimes.coords["ScipyParam"] = ["c", "scale"]
    weibull_lifetimes = weibull_lifetimes.transpose("time", "Region", "Type", "ScipyParam")
    weibull_lifetimes.attrs["loc"] = 0


    lifetime_parameters = {
        "weibull": weibull_lifetimes,
    }

    return lifetime_parameters