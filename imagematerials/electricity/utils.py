
from statistics import mode
import numpy as np
import pandas as pd
import xarray as xr
import math
import scipy.stats
import pint
import prism
from typing import Optional
from pathlib import Path
import warnings
from pint.errors import UnitStrippedWarning


from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.concepts import create_electricity_graph

from imagematerials.constants import (
    IMAGE_REGIONS,
)

from imagematerials.electricity.constants import (
    YEAR_FIRST_GRID,
    YEAR_SWITCH,
    REGIONS,
    EPG_TECHNOLOGIES,
    EPG_TECHNOLOGIES_VRE,
    STD_LIFETIMES_ELECTR,
)
# from imagematerials.vehicles.constants import END_YEAR, FIRST_YEAR, REGIONS

# from read_scripts.dynamic_stock_model_BM import DynamicStockModel as DSM
idx = pd.IndexSlice

#####################################################################################################
# General functions for electricity materials model
#####################################################################################################

# In order to calculate inflow & outflow smoothly (without peaks for the initial years), we calculate a historic tail to the stock, 
# by adding a 0 value for first year of operation (=1926), then interpolate values towards 1971
def stock_tail(stock, YEAR_OUT):
    zero_value = [0 for i in range(0,REGIONS)]
    stock_used = pd.DataFrame(stock).reindex_like(stock)
    stock_used.loc[YEAR_FIRST_GRID] = zero_value  # set all regions to 0 in the year of initial operation
    stock_new  = stock_used.reindex(list(range(YEAR_FIRST_GRID,YEAR_OUT+1))).interpolate() # only interpolates missing values, so existing values (from 1971 onwards) are kept
    return stock_new


def add_historic_stock(da_stock, year_start=1920, interp_method="linear"):
    """
    Calculates historic (pre TIMER simulation time = before 1971) stock values.
    In year_start the stock is 0, then it (linearly) increases to the first existing year in da.

    Parameters
    ----------
    da_stock : xarray.DataArray
        Input DataArray with dims ('Time', 'Region', 'Type').
    year_start : int
        The first year to start historic stock from (will be filled with zeros).
    interp_method : str, optional
        Interpolation method for historic stock growth. Supported options:
        - "linear": linear increase from 0 to first year value
        - "quadratic": quadratic increase from 0 to first year value
        Default is "linear".

    Returns
    -------
    xarray.DataArray
        DataArray extended backwards to year_start, with historic values
        interpolated according to `interp_method` up to the first year in `da`
    """

    t_first = int(da_stock.Time.min())
    if year_start >= t_first:
        return da_stock
    
    unit = prism.U_(da_stock)

    t_hist = np.arange(year_start, t_first)
    n_hist = len(t_hist)

    # Interpolate from 0 to first existing value - use this approach as it is faster than using xr.interp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnitStrippedWarning)
        first_values = da_stock.sel(Time=t_first).values.astype(float)
    if interp_method == "linear":
        stock_hist = np.linspace(0, 1, n_hist)[:, None, None] * first_values[None, :, :]
    elif interp_method == "quadratic":
        stock_hist = np.linspace(0, 1, n_hist)[:, None, None]**2 * first_values[None, :, :]
    else:
        raise ValueError(f"Unknown method: {interp_method}. Choose 'linear' or 'quadratic'.")

    # Build historic DataArray
    da_stock_hist = xr.DataArray(
        stock_hist,
        coords={"Time": t_hist, "Region": da_stock.Region, "Type": da_stock.Type},
        dims=("Time", "Region", "Type")
    )

    da_stock_hist = prism.Q_(da_stock_hist, unit)

    # Concatenate with input array
    da_stock_extended = xr.concat([da_stock_hist, da_stock], dim="Time")

    return da_stock_extended

def _extrap_to_zero(times, first_value, method='linear'):
    """ Generate values ramping from 0 at times[0] to first_value at times[-1].
    
    Parameters
    ----------
    times : np.ndarray
        Array of time steps for the ramp.
    first_value : float
        Target value at the end of the ramp.
    method : str
        'linear'      : straight line
        'exponential' : exponential growth, slow start then accelerating
        'logistic'    : S-curve, slow start, fast middle, slow end
    """
    n = len(times)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([first_value])  # only one point, just return the target value

    # normalized x from 0 to 1
    x = np.linspace(0, 1, n)

    if method == 'linear':
        return first_value * x

    elif method == 'exponential':
        # exponential: y = first_value * (e^(kx) - 1) / (e^k - 1)
        # k controls steepness; higher k = more convex (slower start)
        k = 5
        return first_value * (np.exp(k * x) - 1) / (np.exp(k) - 1)

    elif method == 'logistic':
        # S-curve centered at midpoint, scaled to pass through (0,0) and (1, first_value)
        k = 10  # steepness
        x0 = 0.5  # midpoint
        s = 1 / (1 + np.exp(-k * (x - x0)))
        # rescale so it passes exactly through 0 and first_value
        s = (s - s[0]) / (s[-1] - s[0])
        return first_value * s

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'linear', 'exponential', or 'logistic'.")

def interpolate_xr(data_array: xr.DataArray, 
                   t_start: int | float | dict,
                   t_end: int | float,
                   interp_method: str='linear',
                   extrap_before: str='constant',
                   extrap_before_method: str='linear'):
    """ Interpolate an xarray.DataArray over a continuous time range and
    extend its boundary values beyond the available data to span t_start - t_end.
    The function performs (linear) interpolation between all existing time
    coordinates in the input DataArray and fills values outside the
    original time range with either the first and last available data, respectively,
    or performs an extrapolation to 0 at t_start.
    If t_start is a dict, different start years can be specified per Region,
    with values linearly interpolated to 0 at the region-specific start year,
    and zero-filled between the global t_start and the region-specific start year.

    Parameters
    ----------
    data_array : xarray.DataArray
        Input DataArray with a 'Time' coordinate containing numeric values (e.g. 2020, 2050).
    t_start : int, float, or dict
        Start year for the interpolation range. If a dict, keys are Region names
        and values are the year at which that region's data reaches 0. Regions not
        in the dict fall back to the minimum of the dict values.
    t_end : int or float
        End year for the interpolation range.
    interp_method : str, optional
        Interpolation method to use (default is 'linear'). See xarray documentation
        for available methods.
    extrap_before : str, optional
        How to handle values before the first available data point.
        - 'constant' : fill with the first available value (default)
        - 'zero'     : linearly ramp down to 0 at t_start (or region-specific start if t_start is a dict)
    extrap_before_method : str, optional
        If extrap_before is 'zero', this controls the shape of the ramp:
        - 'linear'      : straight line (default)
        - 'exponential' : exponential growth, slow start then accelerating
        - 'logistic'    : S-curve, slow start, fast middle, slow end

    Returns
    -------
    xarray.DataArray
        DataArray interpolated across the full range from global t_start to t_end.

    Note:
    Units are temporarily stripped during interpolation but are reattached
    before returning the result. The corresponding warning is suppressed.
    """
    
    dim = 'Time' if 'Time' in data_array.dims else 'Cohort'
    unit = prism.U_(data_array)
    coord_values = data_array[dim].values

    # --- Determine global t_start ---
    if isinstance(t_start, dict):
        global_t_start = min(t_start.values())
    else:
        global_t_start = t_start

    new_range = np.arange(global_t_start, t_end + 1)

    # --- Interpolate over full time range ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnitStrippedWarning)
        da_interp = data_array.interp({dim: new_range}, method=interp_method)

    # --- Fill beyond last data point (always constant) ---
    da_interp.loc[{dim: slice(coord_values.max(), None)}] = da_interp.sel({dim: coord_values.max()})

    # --- Handle extrapolation before first data point ---
    if isinstance(t_start, dict) and 'Region' in data_array.dims:
        # Region-specific start years — must iterate over regions
        regions = data_array.Region.values
        fallback_year = global_t_start #if not isinstance(t_start, dict) else min(t_start.values())

        for region in regions:
            region_start = t_start.get(region, fallback_year) if isinstance(t_start, dict) else t_start
            first_value = float(da_interp.sel(Region=region, **{dim: coord_values.min()}).values)

            if extrap_before == 'zero':
                # ramp from 0 at region_start to first_value at coord_values.min()
                ramp_times = np.arange(region_start, coord_values.min() + 1)
                # ramp_values = np.linspace(0, first_value, len(ramp_times))
                ramp_values = _extrap_to_zero(ramp_times, first_value, method=extrap_before_method)
                da_interp.loc[{dim: ramp_times, 'Region': region}] = ramp_values

                # zero-fill between global_t_start and region_start if there's a gap
                if region_start > global_t_start:
                    zero_times = np.arange(global_t_start, region_start)
                    da_interp.loc[{dim: zero_times, 'Region': region}] = 0.0

            else:  # 'constant'
                da_interp.loc[{dim: slice(None, coord_values.min()), 'Region': region}] = first_value

    elif isinstance(t_start, dict):
        raise ValueError("t_start is a dict but 'Region' is not a dimension in the DataArray.")
    
    else:
        # Uniform t_start — broadcast directly, no need to loop over regions       
        first_slice = da_interp.sel({dim: coord_values.min()})
        last_slice = da_interp.sel({dim: coord_values.max()})
        
        if extrap_before == 'zero':
            ramp_times = np.arange(global_t_start, coord_values.min() + 1)
            # Compute ramp for each element, then reassemble into the right shape
            # first_slice has shape (2, 34) — apply _extrap_to_zero per scalar value
            ramp_values = xr.apply_ufunc(
                lambda val: _extrap_to_zero(ramp_times, val, method=extrap_before_method),
                first_slice,
                vectorize=True,
                output_core_dims=[[dim]],
                dask='parallelized',
            )
            ramp_values[dim] = ramp_times
            da_interp.loc[{dim: ramp_times}] = ramp_values.transpose(*da_interp.dims)
        else:  # 'constant'
            da_interp.loc[{dim: slice(None, coord_values.min())}] = first_slice
            da_interp.loc[{dim: slice(coord_values.max(), None)}] = last_slice

    # --- Reattach unit ---
    if unit != prism.Unit('dimensionless'):
        da_interp = prism.Q_(da_interp, unit)

    return da_interp

# def interpolate_xr(data_array, t_start, t_end, interp_method = 'linear'):
#     """ Interpolate an xarray.DataArray over a continuous time range and 
#     extend its boundary values beyond the available data to span t_start - t_end.

#     The function performs (linear) interpolation between all existing time 
#     coordinates in the input DataArray and fills values outside the 
#     original time range with the first and last available data, respectively.

#     Parameters
#     ----------
#     data_array : xarray.DataArray
#         Input DataArray with a 'Time' coordinate containing numeric values (e.g. 2020, 2050).
#     t_start : int or float
#         Start year for the interpolation range.,
#     t_end : int or float
#         End year for the interpolation range.
#     interp_method : str, optional
#         Interpolation method to use (default is 'linear'). See xarray documentation for available methods.

#     Returns
#     -------
#     xarray.DataArray
#         DataArray interpolated across the full range from `t_start` to `t_end`
    
#     Note:
#     Units are temporarily stripped during interpolation but are reattached
#     before returning the result. The corresponding warning is suppressed.
#     """

#     # Determine which dimension to use
#     dim = 'Time' if 'Time' in data_array.dims else 'Cohort'

#     # The interpolations strips the unit, so save it here and reattach later
#     unit = prism.U_(data_array)
    
#     # Get the coordinate values along that dimension
#     coord_values = data_array[dim].values
#     # Define new full range
#     new_range = np.arange(t_start, t_end + 1)

#     # Interpolate linearly
#     with warnings.catch_warnings(): # suppress warning
#         warnings.simplefilter("ignore", UnitStrippedWarning)
#         da_interp = data_array.interp({dim: new_range}, method = interp_method)

#     # Fill values outside original range
#     da_interp.loc[{dim: slice(None, coord_values.min())}] = da_interp.sel({dim: coord_values.min()})
#     da_interp.loc[{dim: slice(coord_values.max(), None)}] = da_interp.sel({dim: coord_values.max()})

#     # Reattach the unit if it existed
#     if unit != prism.Unit('dimensionless'):
#         da_interp = prism.Q_(da_interp, unit)

#     return da_interp


def MNLogit(data: xr.DataArray | pd.DataFrame, 
            logitpar: float, 
            dim_type: str | None =None
            ) -> xr.DataArray | pd.DataFrame:
    """ Multinomial Logit function to calculate market shares from technology prices.

    Works with:
      - pandas.DataFrame (index = years, columns = technologies)
      - xarray.DataArray (dims: Cohort, Type)

    Returns the same type as the input, containing market shares.

    Note:
    logitpar: calibrated Logit parameter (usually a negative number between 0 and 1)
    - represents the price sensitivity or substitution elasticity between different technologies.
    -> how responsive consumers/markets are to price differences between technologies.
    - More negative values (e.g., -2, -5) = higher price sensitivity -> Small price differences lead to large changes in market share = Technologies are highly substitutable
    - Less negative values (e.g., -0.1, -0.5) = lower price sensitivity -> Price differences have less impact on market share = Technologies are less substitutable (perhaps due to quality differences, switching costs, etc.)

    - mathematically: exp(logitpar * price) means: when logitpar is negative and prices are positive, higher prices get exponentially smaller weights
    """

    # ---------- xarray ----------
    if isinstance(data, xr.DataArray):

        if dim_type is None:
            # detect type dimension (Type / SuperType / BatteryType)
            dim_type = next(
                d for d in ["Type", "BatteryType", "SuperType"]
                if d in data.dims
            )
        
        # strip physical units (required for exp)
        values = data.pint.dequantify() if hasattr(data, "pint") else data

        weights = np.exp(logitpar * values)

        # normalize over technologies
        shares = weights / weights.sum(dim=dim_type)

        # attach new unit
        if hasattr(data, "pint"):
            shares = prism.Q_(shares, prism.Unit('shares'))

        return shares

    # ---------- pandas ----------
    elif isinstance(data, pd.DataFrame):
        weights = np.exp(logitpar * data)

        shares = weights.div(weights.sum(axis=1), axis=0)

        return shares

    else:
        raise TypeError("Input must be pandas.DataFrame or xarray.DataArray")


def create_prep_data(results_dict, conversion_table, unit_mapping):
    # Convert the DataFrames to xarray Datasets and apply units
    prep_data = {}
    for df_name, df in results_dict.items():
        if df_name in conversion_table and isinstance(df, pd.DataFrame): # convert to xarray
            # print(f"{df_name} to xarray Dataset")
            data_xar_dataset = pandas_to_xarray(df, unit_mapping)
            data_xarray = dataset_to_array(data_xar_dataset, *conversion_table[df_name])
        elif df_name in conversion_table and isinstance(df, xr.DataArray): # already xarray
            data_xarray = df 
        else:
            # print(f"{df_name} not in conversion_table")
            # lifetimes_vehicles does not need to be converted in the same way.
            data_xarray = pandas_to_xarray(df, unit_mapping)
        prep_data[df_name] = data_xarray

    for df_name in list(prep_data.keys()):
        if "lifetime" in df_name:
            prep_data["lifetimes"] = convert_lifetime(prep_data.pop(df_name))
        elif "stock" in df_name:
            prep_data["stocks"] = prep_data.pop(df_name)
        elif "material" in df_name:
            prep_data["material_intensities"] = prep_data.pop(df_name)
        elif "share" in df_name:
            prep_data["shares"] = prep_data.pop(df_name)

    prep_data["knowledge_graph"] = create_electricity_graph()
    # prep_data["shares"] = None

    return prep_data


# for testing, move later to a plotting utils file in analysis repository
def flexible_plot_1panel(
    da: xr.DataArray,
    x_dim: str,
    varying_dims: list,
    fixed: dict = None,
    figsize=(8, 5),
    plot_type = 'line' # 'line' or 'scatter'
):
    """
    da          : xarray.DataArray
    x_dim       : dimension to use on the x axis (e.g. 'Time' or 'Cohort')
    varying_dims: list of dims that define separate lines (e.g. ['Type', 'Region'])
    fixed       : dict of {dim: value or list} to filter (e.g. {'Type': [1, 2], 'Region': 5})

    use as e.g.:
    flexible_plot_1panel(
        da=grid_length,
        x_dim="Time",
        varying_dims=["Type", "Region"],
        fixed={"Type": [1, 2], "Region": [0, 3]},
        plot_type='scatter'
    )
    """
    
    # 1. Apply filtering
    if fixed:
        for dim, sel in fixed.items():
            # da = da.sel({dim: sel})
            # Convert to list
            if not isinstance(sel, (list, tuple)):
                sel = [sel]

            coord_vals = da.coords[dim].values

            # If the requested values exist as labels → use sel
            if all(v in coord_vals for v in sel):
                da = da.sel({dim: sel})
            else:
                # Otherwise interpret as positional integers → use isel
                da = da.isel({dim: sel})
    
    # 2. Ensure requested dims exist
    for d in varying_dims + [x_dim]:
        if d not in da.dims:
            raise ValueError(f"Dimension '{d}' missing in DataArray")

    # 3. Collapse all varying dims into a combined index
    if varying_dims:
        da_plot = da.stack(curve=varying_dims)
    else:
        da_plot = da

    # 4. Prepare y-axis label with units (if present)
    units = None
    if hasattr(da.data, "units"):               # pint quantity
        units = str(da.data.units)
    else:                                       # normal xarray units
        units = da.attrs.get("units", None)

    if units:
        y_label = f"{da.name or ''} [{units}]"
    else:
        y_label = da.name or ""

    # 5. Plot
    plt.figure(figsize=figsize)
    for curve in da_plot.curve.values:
        sub = da_plot.sel(curve=curve)
        label = ", ".join(f"{dim}={val}" for dim, val in zip(varying_dims, curve))
        if plot_type == 'scatter':
            plt.scatter(sub[x_dim], sub.values, label=label)
        else:
            plt.plot(sub[x_dim], sub.values, label=label)

    plt.xlabel(x_dim)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.show()




def logistic(x, L, x0=None):
    """ Compute logistic-curve values for given x values.

    Parameters
    ----------
    x : array-like
        Input x values (list, numpy array, pandas Series).
    L : float or np.array (matching shape to x)
        Maximum value (upper asymptote). If x has multiple columns, L can be either a float (applied to all columns) 
        or an array with one value per column.
    k : float
        Growth rate (steepness of the curve).
    x0 : float
        Midpoint (x value where y = L/2). If None, defaults to the mit point of x.

    Returns
    -------
    numpy.ndarray
        Logistic-curve y values corresponding to x.
    """

    x0 = x0 if x0 is not None else x.iloc[[int(len(x)/2)],:].index[0]

    x_fct = x.iloc[1:-1] # exclude first and last point to keep them fixed
    delta_x = x_fct.iloc[-1] - x_fct.iloc[0]
    k = 9 / delta_x # 9 was set by try and error to get a good steepness
    z = np.clip(-k * (x_fct - x_fct.loc[x0]), -700, 700)  # limit range to avoid overflow
    y_logistic = L / (1 + np.exp(z))
    y_logistic.loc[x.index[0]] = x.iloc[0]
    y_logistic.loc[x.index[-1]] = L

    return y_logistic


def quadratic(x):
    """ Compute quadratic values for given x values.

    Parameters
    ----------
    x : array-like
        Input x values (list, numpy array, pandas Series).

    Returns
    -------
    numpy.ndarray
        Quadratic y values corresponding to x.
    """
    return x**2


def print_df_info(df, name):
    """
    Prints basic information about a DataFrame, including its shape, columns, and first few index values.
    Just for testing and debugging purposes.   
    """
    print(f"""{name}
          Shape: {df.shape}, 
          Columns: {df.columns.tolist()},
          Index name(s): {df.index.names},
          Index: {df.index.tolist()[:5]}...""")
    

def sanitize_attrs(da): # for saving xarray objects to netcdf
	""" Sanitize the attributes of a DataArray and its coordinates for safe serialization. This function 
    converts all attribute values that are not of type str, int, or float into strings. It applies this 
    transformation to both the DataArray's `.attrs` and each coordinate's `.attrs`. This is useful when 
    saving xarray objects to formats like NetCDF, which require attribute values to be basic serializable 
    types.
	
	Parameters
	----------
	da : xarray.DataArray
		The input DataArray whose attributes need to be sanitized.
	
	Returns
	-------
	xarray.DataArray
		A copy of the input DataArray with sanitized attributes.
	
	Notes
	-----
	- This function does not modify the original DataArray in-place; it returns a copy.
	- It preserves the data, coordinates, and dimensions of the original DataArray.
	"""
	
	# use as:
	# da_example = sanitize_attrs(model_lines.inflow.to_array())
	# da_example.to_netcdf(path_test / "grid_lines_inflow_v0.nc")
	
	da = da.copy()
	da.attrs = {k: str(v) if not isinstance(v, (str, int, float)) else v
				for k, v in da.attrs.items()}
	for c in da.coords:
		da.coords[c].attrs = {
			k: str(v) if not isinstance(v, (str, int, float)) else v
			for k, v in da.coords[c].attrs.items()
		}
	return da

def compare_da(da_new: xr.DataArray, 
               da_old: xr.DataArray = None, 
               path_to_saved_da: Optional[str | Path] = None): # for testing xarray objects
    """ Compare a (saved) DataArray to a new one.
	
	Parameters
	----------
	path : str or Path
		Path to the saved DataArray file.
	da_new : xarray.DataArray
		The new DataArray to compare.
	
	Returns
	-------
	equal : bool
		True if the DataArrays match numerically (after removing units in case of a saved da).
	diff_nonzero : xarray.DataArray or None
		Differences where values differ; None if equal.
	"""
	# use as:
	# compare_da(model_lines.inflow.to_array(), path_test / "grid_lines_inflow_v0.nc")

    if path_to_saved_da is not None and da_old is None:
        da_old = xr.open_dataarray(str(path_to_saved_da))
        da_old_clean = da_old.pint.dequantify()
        da_new_clean = da_new.pint.dequantify()
    elif da_old is not None and path_to_saved_da is None:
        da_old_clean = da_old
        da_new_clean = da_new
    else:
        raise ValueError("Either da_old or path_to_saved_da must be provided, but not both or neither.")
	
    equal = da_old_clean.equals(da_new_clean)

    if not equal:
        diff = da_new_clean - da_old_clean
        diff_nonzero = diff.where(diff != 0, drop=True)
        return equal, diff_nonzero

    return equal, None



#####################################################################################################
# Functions for grid preprocessing
#####################################################################################################


def calculate_grid_growth(gcap, grid_lines):
    """ Calculate grid line growth factors over time based on regional generation capacity development.

    Total (peak) generation capacity is used as a proxy for grid expansion. Growth factors are
    defined relative to a base year (2016) and applied uniformly to all voltage levels for
    overhead lines. Underground line growth factors are included only to match the shape of
    the grid length DataArray and are set to NaN, as underground lines are calculated separately.

    For high-voltage (HV) overhead lines, additional growth adjustments are applied from 2020
    onwards based on the share of variable renewable energy (VRE; solar and wind) capacity.
    Line additions and reductions are gradually introduced using a linear ramp between 2020
    and 2050.

    Parameters
    ----------
    gcap : xarray.DataArray
        Regional generation capacity by technology and time. Must include dimensions
        ``Type`` and ``Time``.
    grid_lines : xarray.DataArray
        Grid line lengths by voltage level, type, region, and time. Used to define the target
        shape and coordinates of the returned growth factor DataArray.

    Returns
    -------
    grid_growth_expanded : xarray.DataArray
        Growth factors for grid line lengths with the same dimensions and coordinates as
        ``grid_lines``. Values represent multiplicative factors relative to 2016.
    """

    # regional total (peak) generation capacity is used as a proxy for the grid growth
    gcap_total = gcap.sum(dim='Type')
    gcap_growth = gcap_total / gcap_total.loc[2016]        # define growth according to 2016 as base year

    # copy growth factor for all voltage levels (overhead)
    grid_growth = gcap_growth.expand_dims(Type=["HV - Lines - Overhead","MV - Lines - Overhead","LV - Lines - Overhead"]).copy()
    # add coordinates for underground lines to match grid length dataarray (set to NaN, as underground lines are later calculated based on aboveground lines & fixed ratios)
    grid_growth_expanded = grid_growth.broadcast_like(grid_lines).copy().rename("GridGrowthFactor")

    # for HV lines: additional growth is presumed after 2020 based on the fraction of variable renewable energy (vre) generation capacity (solar & wind) (used to be only in the sensitivity variant, but now used in the base case as well)
    vre_fraction = gcap.sel(Type=EPG_TECHNOLOGIES_VRE).sum(dim='Type') / gcap_total
    # Compute additional/reduced growth
    add_growth = vre_fraction * 1             # if value is e.g. 0.2 = 20% additional HV lines per doubling of vre gcap
    red_growth = (1 - vre_fraction) * 0.7     
    add_growth = add_growth.where(add_growth.Time >= 2020, 0)  # Set pre-2020 values to 0
    red_growth = red_growth.where(red_growth.Time >= 2020, 0)
    # Create a ramp factor (line length addition/reduction is gradually introduced from 2020 towards 2050)
    ramp_factor = xr.DataArray(
        np.clip((add_growth.Time - 2020) / 30, 0, 1),
        coords={"Time": add_growth.Time},
        dims=["Time"]
    )
    add_growth = add_growth * ramp_factor # Apply ramp factor
    red_growth = red_growth * ramp_factor
    grid_growth_expanded.loc[{"Type": "HV - Lines - Overhead"}] = grid_growth_expanded.loc[{"Type": "HV - Lines - Overhead"}] + add_growth - red_growth

    return grid_growth_expanded


def calculate_fraction_underground(grid_lines, gdp_pc, ratio_underground):
    """ Calculate fractions of underground and overhead grid lines by region, voltage level, and time.

    Underground fractions are estimated as linear functions of GDP per capita, with separate
    parameterizations for European and non-European regions. Fractions are bounded between
    0 and 1 and expanded to match the dimensions of the grid line DataArray. Overhead fractions
    are derived as the complement of underground fractions.

    Parameters
    ----------
    grid_lines : xarray.DataArray
        Grid line lengths by region, type, and time; used to define target shape and coordinates.
    gdp_pc : xarray.DataArray
        GDP per capita by region and time.
    ratio_underground : pandas.DataFrame or xarray object
        Coefficients (multipliers and offsets) defining the GDP-based underground line fractions
        by region group and voltage level.

    Returns
    -------
    fraction_lines_above_below : xarray.DataArray
        Fractions of underground and overhead grid lines with the same dimensions as
        ``grid_lines``.
    """
    fraction_lines_above_below = xr.full_like(grid_lines, np.nan).rename("FractionUndergroundAboveground")

    gdp_pc_unitless = gdp_pc.pint.dequantify() #work-around for now
    for region in IMAGE_REGIONS:
        if region in ['WEU','CEU']:
            select_proxy = 'Europe'
        else:
            select_proxy = 'Other'

        fraction_lines_above_below.loc[{"Region": region, "Type": "HV - Lines - Underground"}] = (gdp_pc_unitless.sel(Region=region) * ratio_underground.loc[idx[select_proxy,'mult'],'HV'] + ratio_underground.loc[idx[select_proxy,'add'],'HV'])/100 # /100 to convert from % to fraction
        fraction_lines_above_below.loc[{"Region": region, "Type": "MV - Lines - Underground"}] = (gdp_pc_unitless.sel(Region=region) * ratio_underground.loc[idx[select_proxy,'mult'],'MV'] + ratio_underground.loc[idx[select_proxy,'add'],'MV'])/100
        fraction_lines_above_below.loc[{"Region": region, "Type": "LV - Lines - Underground"}] = (gdp_pc_unitless.sel(Region=region) * ratio_underground.loc[idx[select_proxy,'mult'],'LV'] + ratio_underground.loc[idx[select_proxy,'add'],'LV'])/100
    # fraction must be between 0 and 1
    fraction_lines_above_below = fraction_lines_above_below.clip(min=0, max=1)

    # MIND! the HV lines found in OSM (+national sources) are considered as the total of the aboveground line length + the underground line length
    # currently the total is saved in xarray grid_lines as the aboveground fraction only -> need to split into aboveground & underground based on the calculated ratios
    # for this, we copy the aboveground length into the underground length & then multiply both with the coresponding fraction to get the correct lengths
    for level in ["HV", "MV", "LV"]:
        over = f"{level} - Lines - Overhead"
        under = f"{level} - Lines - Underground"
        # grid_lines.loc[dict(Type=under)] = grid_lines.sel(Type=over) # copy the aboveground length into the underground length
        fraction_lines_above_below.loc[dict(Type=over)] = 1-fraction_lines_above_below.loc[dict(Type=under)] # above = 1 - under
    
    return fraction_lines_above_below




##########################################################################################
# Storage preprocessing functions
##########################################################################################


def calculate_storage_market_shares(
    storage_costs: xr.DataArray,
    costs_correction: xr.DataArray,
    cost_decline_longterm_correction: xr.DataArray,
    mnlogit_param: float,
    t_start_interpolation: int = 1970,
    t_end_interpolation: int = 2050,
    dim_type: str | None = None) -> xr.DataArray:
    """ Calculate technology market shares for energy storage based on cost
    developments and a multinomial logit model.

    The function interpolates storage costs and correction factors over a
    specified time range, applies cost decline assumptions for future and past
    years, and computes market shares based on costs using a multinomial logit formulation.

    Parameters
    ----------
    storage_costs : xr.DataArray
        Storage technology costs indexed by year (dimension ``Cohort``)
        and technology type (e.g. ``Type``). Values represent base costs
        before corrections.
    costs_correction : xr.DataArray
        Multiplicative correction factors for storage costs, aligned with
        ``storage_costs`` in dimensions.
    cost_decline_longterm_correction : xr.DataArray
        Scalar correction factor applied to the annual cost decline rate
        derived from the historical period (2018-2030) to represent
        long-term cost decline after 2030.
    mnlogit_param : float
        Logit parameter used in the multinomial logit model. Typically a
        negative value controlling price sensitivity.
    t_start_interpolation : int, optional
        First year of interpolation and extrapolation (default: YEAR_START).
    t_end_interpolation : int, optional
        Last year of interpolation and extrapolation (default: 2050).

    Returns
    -------
    storage_market_share: xr.DataArray
        Market shares of storage technologies over time, indexed by year
        (``Cohort``) and technology type.

    Notes
    -----
    - Past costs assume twice the average long-term annual decline rate,
      except for deep-cycle lead-acid technology, which is held constant
      at its 2018 level.
    """

    # if dim_type not specified as input: detect from data. Raise warning if multiple candidates found.
    if dim_type is None:
        candidates = [d for d in ["Type", "BatteryType", "SuperType"] if d in storage_costs.dims]
        if len(candidates) == 0:
            raise ValueError("No valid type dimension found.")
        if len(candidates) > 1:
            warnings.warn(f"Multiple type dimensions found: {candidates}. Using {candidates[0]}.")
        dim_type = candidates[0]

    t_start = storage_costs.Cohort.values[0]
    t_end   = storage_costs.Cohort.values[-1]

    # interpolate from first to last vailable year within the data, then extend to  YEAR_START and 2050 (keep values constant before first and after last year)
    storage_costs      = interpolate_xr(storage_costs, t_start_interpolation, t_end_interpolation)
    costs_correction   = interpolate_xr(costs_correction, t_start_interpolation, t_end_interpolation)

    # determine the annual % decline of the costs based on the 2018-2030 data (original, before applying the malus)
    xr_cost_decline = (((storage_costs.loc[t_start,:]-storage_costs.loc[t_end,:])/(t_end-t_start))/storage_costs.loc[t_start,:]).drop_vars('Cohort')
    xr_cost_decline_longterm = xr_cost_decline*cost_decline_longterm_correction # cost decline after 2030 = cost decline 2018-2030 * correction factor
    # cost_decline_longterm_correction is a single number and should describe the long-term decline after 2030 relative to the 2018-2030 decline

    # storage_costs_new = storage_costs_interpol * costs_correction_interpol
    storage_costs_cor = storage_costs * costs_correction

    # ---------- future development ----------
    # calculate the development from 2030 to 2050 (using annual price decline)
    # vectorized approach for "for t in range(t_end, 2050+1): ..."
    years_fwd = storage_costs_cor.Cohort.where(
        storage_costs_cor.Cohort > t_end, drop=True
    )
    n_fwd = years_fwd - t_end
    factors_fwd = (1 - xr_cost_decline_longterm) ** n_fwd

    storage_costs_cor.loc[dict(Cohort=years_fwd)] = (
        storage_costs_cor.sel(Cohort=t_end) * factors_fwd
    )

    # ---------- past development ----------
    # for historic price development, assume 2x AVERAGE annual price decline on all technologies
    years_bwd = storage_costs_cor.Cohort.where(
        storage_costs_cor.Cohort < t_start, drop=True
    )
    n_bwd = t_start - years_bwd
    factors_bwd = (1 + 2 * xr_cost_decline_longterm.mean()) ** n_bwd

    storage_costs_cor.loc[dict(Cohort=years_bwd)] = (
        storage_costs_cor.sel(Cohort=t_start) * factors_bwd
    )
    # restore values for lead-acid (set to constant 2018 values) -> exception: so that lead-acid gets a relative price advantage from 1970-2018
    storage_costs_cor.loc[dict(Cohort=years_bwd, **{dim_type: "Deep-cycle Lead-Acid"})] = ( # **{dim_type: "Deep-cycle Lead-Acid"} dynamically creates a keyword argument 
        storage_costs_cor.sel(Cohort=t_start, **{dim_type: "Deep-cycle Lead-Acid"})         # if dim_type == "Type" -> Type="Deep-cycle Lead-Acid"
    )

    # market shares ---
    # use the storage price development in the logit model to get market shares
    storage_market_share = MNLogit(storage_costs_cor, mnlogit_param, dim_type=dim_type) #assumes input of an ordered dataframe with rows as years and columns as technologies, values as prices. Logitpar is the calibrated Logit parameter (usually a nagetive number between 0 and 1)

    return storage_market_share


def normalize_selected_techs(market_share: xr.DataArray | pd.DataFrame, 
                             techs: list[str], 
                             dim_type: str | None = None
                             ) -> xr.DataArray | pd.DataFrame:
    """ Select technologies and renormalize their market shares to sum to 1 per year.

    Works with:
    - xarray.DataArray (dimension: Type)
    - pandas.DataFrame (columns: technologies)

    Parameters
    ----------
    market_share : xarray.DataArray or pandas.DataFrame
        Market shares by year and technology.
    techs : list of str
        Technologies to select and normalize.

    Returns
    -------
    Same type as input
        Normalized market shares of the selected technologies.
    """
    if isinstance(market_share, xr.DataArray):
        # if type dimension not specified as input: detect from data (Type / SuperType / BatteryType)
        if dim_type is None:
            dim_type = next(
                d for d in ["Type", "BatteryType", "SuperType"]
                if d in market_share.dims
            )
        sel = market_share.sel(**{dim_type: techs}) # **{...} is dictionary unpacking -> handles flexible keyword arguments
        return sel / sel.sum(dim=dim_type)
    else:  # pandas
        sel = market_share[techs]
        return sel.div(sel.sum(axis=1), axis=0)





#####################################################################################################
# Circular economy implementations
#####################################################################################################


def apply_ce_measures_to_elc(arr: xr.DataArray, base_year: int, target_year: int, change: dict,
                    implementation_rate: str, data_sector: Optional[str]=None, data_type: Optional[str]=None, steepness: float=0.5) -> xr.DataArray:
        """ Apply CE measures to an xarray DataArray over time for (technology) types.

        The function modifies values by applying a percentage change according to a chosen
        implementation pathway (immediate, linear, or s-curve) between a base year and a target year.
        Changes are applied relative to the same-year baseline and preserve the original data structure.

        Parameters
        ----------
        arr : xr.DataArray
            Input DataArray containing a time dimension ('Time' or 'Cohort') and a type
            classification dimension ('Type' or 'SuperType').
        base_year : int
            Year at which the implementation of lightweighting begins.
        target_year : int
            Year by which the full lightweighting effect is achieved.
        change : dict
            Mapping of type (matching entries in the type dimension) to percentage change
            to apply (e.g., {\"Solar PV\": -10} for a 10% reduction).
        implementation_rate : str
            Implementation pathway. One of {'immediate', 'linear', 's-curve'}.
        data_sector : str, optional
            To which sector data belong (e.g., 'electricity grid').
        data_type : str, optional
            Type of data being processed. Needed for the case of modifying lifetime distributions.
            Acccepted value is 'lifetime'.
        steepness : float, default 0.5
            Steepness parameter for the logistic function when using the 's-curve' implementation.

        Returns
        -------
        xr.DataArray
            A new DataArray with ce measure applied along the time dimension.

        Raises
        ------
        ValueError
            If required dimensions are missing, a specified vehicle type is not found,
            or an unsupported implementation_rate is provided.

        Notes
        -----
        - Supports DataArrays with either 'Time' or 'Cohort' as the temporal dimension.
        - Supports either 'Type' or 'SuperType' as the vehicle classification dimension.
        - A pandas-based implementation exists for vehicle-level data.
        - A general xarray-based version is available in general utilities for regional data.
        """

        result = arr.copy()

        # Determine time and type dimensions (to support different types of DataArrays)
        if "Time" in arr.dims:
            time_dim = "Time"
        elif "Cohort" in arr.dims:
            time_dim = "Cohort"
        else:
            raise ValueError("Input DataArray must have either 'Time' or 'Cohort' dimension.")
        
        if "Type" in arr.dims:
            type_dim = "Type"
        elif "SuperType" in arr.dims:
            type_dim = "SuperType"
        else:
            raise ValueError("Input DataArray must have either 'Type' or 'SuperType' dimension.")
        
        # If CE measures are only specified for aggregated grid items, expand to all relevant types
        if data_sector == 'electricity grid':
            change = expand_change_dict_for_grid_items(change)
        elif data_sector is not None:
            raise ValueError(f"Unknown data_sector: '{data_sector}'. Supported types are 'electricity grid' or None.")
        
        # If data_type is 'lifetime', we need to determine which distribution parameter to modify
        if data_type == "lifetime":
            dist_param = 'mean'
        elif data_type is None:
            pass
        else:
            raise ValueError(f"Unknown data_type: '{data_type}'. Supported types are 'lifetime' or None.")

        for type_stock, increase in change.items():
            if type_stock in result[type_dim]:
                if implementation_rate == 'linear':
                    # ramp progress: 0 at base_year, 1 at target_year, held at 1 after
                    span = max(1, target_year - base_year)
                    # apply to each year explicitly to preserve structure
                    for year in range(base_year + 1, target_year + 1):
                        progress = (year - base_year) / span
                        if data_type == "lifetime":
                            result.loc[{time_dim: year, type_dim: type_stock, "DistributionParams": dist_param}] = (
                                arr.loc[{time_dim: year, type_dim: type_stock, "DistributionParams": dist_param}] * (1 + ((increase / 100.0) * progress))
                            )
                        else:
                            result.loc[{time_dim: year, type_dim: type_stock}] = (
                                arr.loc[{time_dim: year, type_dim: type_stock}] * (1 + ((increase / 100.0) * progress))
                            )
                    # after target_year, full effect but still relative to same-year baseline
                    if data_type == "lifetime":
                        result.loc[{time_dim: slice(target_year + 1, None), type_dim: type_stock, "DistributionParams": dist_param}] = (
                            arr.loc[{time_dim: slice(target_year + 1, None), type_dim: type_stock, "DistributionParams": dist_param}] * ((1 + increase / 100.0))
                        )
                    else:
                        result.loc[{time_dim: slice(target_year + 1, None), type_dim: type_stock}] = (
                            arr.loc[{time_dim: slice(target_year + 1, None), type_dim: type_stock}] * ((1 + increase / 100.0))
                        )

                elif implementation_rate == 'immediate':
                    # unchanged up to base_year; full step from base_year+1 onward, relative to same-year baseline
                    if data_type == "lifetime":
                        result.loc[{time_dim: slice(None, base_year), type_dim: type_stock, "DistributionParams": dist_param}] = \
                            arr.loc[{time_dim: slice(None, base_year), type_dim: type_stock, "DistributionParams": dist_param}]
                        result.loc[{time_dim: slice(base_year + 1, None), type_dim: type_stock, "DistributionParams": dist_param}] = \
                            arr.loc[{time_dim: slice(base_year + 1, None), type_dim: type_stock, "DistributionParams": dist_param}] * (1 + increase / 100.0)
                    else:
                        result.loc[{time_dim: slice(None, base_year), type_dim: type_stock}] = \
                            arr.loc[{time_dim: slice(None, base_year), type_dim: type_stock}]
                        result.loc[{time_dim: slice(base_year + 1, None), type_dim: type_stock}] = \
                            arr.loc[{time_dim: slice(base_year + 1, None), type_dim: type_stock}] * (1 + increase / 100.0)

                elif implementation_rate == 's-curve':
                    years = list(range(base_year, target_year + 1))
                    mid_year = (base_year + target_year) / 2
                    # normalize logistic so progress(base)=0 and progress(target)=1
                    s0 = 1.0 / (1.0 + np.exp(-steepness * (base_year - mid_year)))
                    s1 = 1.0 / (1.0 + np.exp(-steepness * (target_year - mid_year)))
                    for year in years:
                        s = 1.0 / (1.0 + np.exp(-steepness * (year - mid_year)))
                        progress = np.clip((s - s0) / (s1 - s0), 0.0, 1.0)
                        if data_type == "lifetime":
                            result.loc[{time_dim: year, type_dim: type_stock, "DistributionParams": dist_param}] = (
                                arr.loc[{time_dim: year, type_dim: type_stock, "DistributionParams": dist_param}] * (1 + (increase / 100.0) * progress)
                            )
                        else:
                            result.loc[{time_dim: year, type_dim: type_stock}] = (
                                arr.loc[{time_dim: year, type_dim: type_stock}] * (1 + (increase / 100.0) * progress)
                            )
                    # after target_year, full effect relative to same-year baseline
                    if data_type == "lifetime":
                        result.loc[{time_dim: slice(target_year + 1, None), type_dim: type_stock, "DistributionParams": dist_param}] = (
                            arr.loc[{time_dim: slice(target_year + 1, None), type_dim: type_stock, "DistributionParams": dist_param}] * (1 + increase / 100.0)
                        )
                    else:
                        result.loc[{time_dim: slice(target_year + 1, None), type_dim: type_stock}] = (
                            arr.loc[{time_dim: slice(target_year + 1, None), type_dim: type_stock}] * (1 + increase / 100.0)
                        )
                else: 
                    raise ValueError(f"Unknown implementation method: '{implementation_rate}'. "
                                    "Supported methods are 'immediate', 'linear', and 's-curve'.")
            else:
                raise ValueError(f"{type_stock} not found in DataArray.")
            
        nan_mask = result.isnull()
        if nan_mask.any():
            years_with_nans = result[time_dim].where(nan_mask.any(dim=[d for d in result.dims if d != time_dim]), drop=True)
            warnings.warn(
                f"NaNs present in years: {years_with_nans.values}",
                RuntimeWarning
            )

        return result

def expand_change_dict_for_grid_items(dict_change: dict) -> dict:
    """Expand generic grid component categories into voltage-specific categories.

    This function takes a dictionary describing changes for grid components and
    expands generic keys ("Transformers", "Substations") into their corresponding
    voltage-level-specific keys:
        - Transformers → "HV - Transformers", "MV - Transformers", "LV - Transformers"
        - Substations → "HV - Substations", "MV - Substations", "LV - Substations"

    Expansion only occurs if:
        - The generic key (e.g. "Transformers") is present, AND
        - None of the corresponding expanded keys are already present.

    If expansion is performed, the generic key is removed and replaced by the
    expanded keys, all assigned the same value.

    Parameters
    ----------
    dict_change : dict
        Dictionary mapping grid component categories (str) to values.

    Returns
    -------
    dict
        A new dictionary where generic categories are expanded into
        voltage-specific categories when applicable.
    """

    transformer_expanded = {
        "HV - Transformers",
        "MV - Transformers",
        "LV - Transformers",
    }
    substation_expanded = {
        "HV - Substations",
        "MV - Substations",
        "LV - Substations",
    }

    has_transformer_expanded = any(k in dict_change for k in transformer_expanded)
    has_substation_expanded = any(k in dict_change for k in substation_expanded)

    dict_change_expanded = dict(dict_change)

    if not has_transformer_expanded and "Transformers" in dict_change:
        val = dict_change["Transformers"]
        for k in transformer_expanded:
            dict_change_expanded[k] = val
        dict_change_expanded.pop("Transformers")

    if not has_substation_expanded and "Substations" in dict_change:
        val = dict_change["Substations"]
        for k in substation_expanded:
            dict_change_expanded[k] = val
        dict_change_expanded.pop("Substations")

    return dict_change_expanded

