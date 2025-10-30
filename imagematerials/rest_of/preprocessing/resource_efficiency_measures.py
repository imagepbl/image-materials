import numpy as np
import xarray as xr


def adapt_gompertz_regional(xr_gompertz, scenario: str, 
                            start_implementation_year=2030,
                            end_implementation_year=2100):
    """
    Adapt the 'a' (alpha) coefficient in a Gompertz parameter xarray DataArray for resource efficiency scenarios.

    For each material and region:
    - The region with the lowest 'a' value at year 2030 remains unchanged.
    - All other regions:
        - If their 'a' value at 2030 is less than twice the lowest value, it is linearly reduced by 20% from 2030 to 2100,
          but never below the lowest value.
        - If their 'a' value at 2030 is greater than or equal to twice the lowest value, it is linearly reduced to
          exactly twice the lowest value from 2030 to 2100, but never below the lowest value.
    - For years before 2030, values remain unchanged.
    - For years after 2100, the final target value is held constant.

    Parameters
    ----------
    xr_gompertz : xarray.DataArray
        Gompertz coefficients with dimensions including 'coef', 'material', 'Region', and 'Time'.
        The 'a' coefficient is adapted according to the rules above.
    scenario : str
        Name of the scenario for which the adaptation is applied. Used for logging.
    start_implementation_year : int
        The year when the adaptation starts (default is 2030).
    end_implementation_year : int
        The year when the adaptation ends (default is 2100).

    Returns
    -------
    xarray.DataArray
        The adapted Gompertz coefficients DataArray with updated 'a' values for each region and material.

    Notes
    -----
    - This function modifies the input DataArray in-place.
    - Only the 'a' coefficient is adapted; other coefficients remain unchanged.
    - The adaptation is applied for years 2030 to 2100.
    """
    
    years = xr_gompertz.coords['Time'].values
    for material in xr_gompertz.coords['material'].values:
        a_2030 = xr_gompertz.sel(coef='a', material=material, Time=start_implementation_year).values
        regions = xr_gompertz.coords['Region'].values
        valid = ~np.isnan(a_2030)
        sorted_a = np.sort(a_2030[valid])
        lowest_a = sorted_a[0]
        target_a = 2 * lowest_a

        for i, region in enumerate(regions):
            current_a_2030 = a_2030[i]
            if np.isnan(current_a_2030):
                continue
            # Lowest region stays at its value
            if current_a_2030 == lowest_a:
                continue
            # If current value is lower than 2x lowest, reduce by 20%, but not below lowest_a
            if current_a_2030 < target_a:
                reduce_target = max(current_a_2030 * 0.8, lowest_a)
                for year in years:
                    if year < start_implementation_year:
                        continue
                    elif year >= start_implementation_year and year <= end_implementation_year:
                        frac = (year - start_implementation_year) / (end_implementation_year - start_implementation_year)
                        new_a = current_a_2030 + frac * (reduce_target - current_a_2030)
                        new_a = max(new_a, lowest_a)
                        xr_gompertz.loc[dict(coef='a', material=material, Region=region, Time=year)] = new_a
                    elif year > end_implementation_year:
                        xr_gompertz.loc[dict(coef='a', material=material, Region=region, Time=year)] = reduce_target
            # Otherwise, reduce to 2x lowest value, but not below lowest_a
            else:
                for year in years:
                    if year < start_implementation_year:
                        continue
                    elif year >= start_implementation_year and year <= end_implementation_year:
                        frac = (year - start_implementation_year) / (end_implementation_year - start_implementation_year)
                        new_a = current_a_2030 + frac * (target_a - current_a_2030)
                        new_a = max(new_a, lowest_a)
                        xr_gompertz.loc[dict(coef='a', material=material, Region=region, Time=year)] = new_a
                    elif year > end_implementation_year:
                        xr_gompertz.loc[dict(coef='a', material=material, Region=region, Time=year)] = max(target_a, lowest_a)
    
    print("gompertz scaling applied for scenario:", scenario)
    return xr_gompertz