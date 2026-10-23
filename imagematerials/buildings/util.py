import xarray as xr

def scale_to_target(dataarray, target_values, year=2020):
    """
    Scales the entire time series for selected regions so that the 2020 values match the targets.
    
    Parameters:
    - dataarray: xarray.DataArray with dimensions Region, Area, Type, time
    - target_values_2020: dict of {region_name: (region_id, target_value)}
    - year: year to match to target (default = 2020)
    
    Returns:
    - xarray.DataArray with scaled values
    """
    scaled = dataarray.copy(deep=True)

    for region_name, (region_id, target) in target_values.items():
        region_str = str(region_id)

        # Get original 2020 mean value
        original = dataarray.sel(Region=region_str, time=year)
        original_mean = original.sum(dim="Type").mean(dim="Area").item()

        # Compute scale factor
        scale_factor = target / original_mean

        # Scale all years for this region
        scaled.loc[{"Region": region_str}] *= scale_factor

    return scaled


def _as_string_regions(data_array: xr.DataArray) -> xr.DataArray:
    return data_array.assign_coords(Region=[str(r) for r in data_array.coords["Region"].values])


def _normalize_shares(shares: xr.DataArray, dim: str = "Area", fallback: float = 0.0) -> xr.DataArray:
    shares = shares.clip(min=0)
    denom = shares.sum(dim=dim)
    return xr.where(denom > 0, shares / denom, fallback)