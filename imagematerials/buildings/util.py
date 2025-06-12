def scale_to_target(dataarray, target_values, year=2020):
    """
    Scales the entire time series for selected regions so that the 2020 values match the targets.
    
    Parameters:
    - dataarray: xarray.DataArray with dimensions Region, Area, Type, Time
    - target_values_2020: dict of {region_name: (region_id, target_value)}
    - year: year to match to target (default = 2020)
    
    Returns:
    - xarray.DataArray with scaled values
    """
    scaled = dataarray.copy(deep=True)

    for region_name, (region_id, target) in target_values.items():
        region_str = str(region_id)

        # Get original 2020 mean value
        original = dataarray.sel(Region=region_str, Time=year)
        original_mean = original.sum(dim="Type").mean(dim="Area").item()

        # Compute scale factor
        scale_factor = target / original_mean

        # Scale all years for this region
        scaled.loc[{"Region": region_str}] *= scale_factor

    return scaled
