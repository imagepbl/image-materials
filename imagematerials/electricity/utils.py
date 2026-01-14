
import numpy as np
import pandas as pd
import xarray as xr
import math
import matplotlib.pyplot as plt
import scipy.stats
import prism
from typing import Optional
from pathlib import Path
import warnings
from pint.errors import UnitStrippedWarning


from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.concepts import create_electricity_graph
from imagematerials.electricity.constants import (
    YEAR_FIRST,
    YEAR_FIRST_GRID,
    YEAR_SWITCH,
    REGIONS,
    MEGA_TO_TERA,
    PKMS_TO_VKMS,
    TONNES_TO_KGS,
    LOAD_FACTOR,
    BEV_CAPACITY_CURRENT,
    PHEV_CAPACITY_CURRENT
)
# from imagematerials.vehicles.constants import END_YEAR, FIRST_YEAR, REGIONS

# from read_scripts.dynamic_stock_model_BM import DynamicStockModel as DSM
idx = pd.IndexSlice


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



def interpolate_xr(data_array, t_start, t_end, interp_method = 'linear'):
    """ Interpolate an xarray.DataArray over a continuous time range and 
    extend its boundary values beyond the available data to span t_start - t_end.

    The function performs (linear) interpolation between all existing time 
    coordinates in the input DataArray and fills values outside the 
    original time range with the first and last available data, respectively.

    Parameters
    ----------
    data_array : xarray.DataArray
        Input DataArray with a 'Time' coordinate containing numeric values (e.g. 2020, 2050).
    t_start : int or float
        Start year for the interpolation range.,
    t_end : int or float
        End year for the interpolation range.
    interp_method : str, optional
        Interpolation method to use (default is 'linear'). See xarray documentation for available methods.

    Returns
    -------
    xarray.DataArray
        DataArray interpolated across the full range from `t_start` to `t_end`
    
    Note:
    Units are temporarily stripped during interpolation but are reattached
    before returning the result. The corresponding warning is suppressed.
    """

    # Determine which dimension to use
    dim = 'Time' if 'Time' in data_array.dims else 'Cohort'

    # The interpolations strips the unit, so save it here and reattach later
    unit = prism.U_(data_array)
    
    # Get the coordinate values along that dimension
    coord_values = data_array[dim].values
    # Define new full range
    new_range = np.arange(t_start, t_end + 1)

    # Interpolate linearly
    with warnings.catch_warnings(): # suppress warning
        warnings.simplefilter("ignore", UnitStrippedWarning)
        da_interp = data_array.interp({dim: new_range}, method = interp_method)

    # Fill values outside original range
    da_interp.loc[{dim: slice(None, coord_values.min())}] = da_interp.sel({dim: coord_values.min()})
    da_interp.loc[{dim: slice(coord_values.max(), None)}] = da_interp.sel({dim: coord_values.max()})

    # Reattach the unit if it existed
    if unit != prism.Unit('dimensionless'):
        da_interp = prism.Q_(da_interp, unit)

    return da_interp

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
            print(f"{df_name} to xarray Dataset")
            data_xar_dataset = pandas_to_xarray(df, unit_mapping)
            data_xarray = dataset_to_array(data_xar_dataset, *conversion_table[df_name])
        elif df_name in conversion_table and isinstance(df, xr.DataArray): # already xarray
            data_xarray = df 
        else:
            print(f"{df_name} not in conversion_table")
            # lifetimes_vehicles does not need to be converted in the same way.
            data_xarray = pandas_to_xarray(df, unit_mapping)
        prep_data[df_name] = data_xarray

    for df_name in list(prep_data.keys()):
        if "lifetime" in df_name:
            prep_data["lifetimes"] = convert_lifetime(prep_data[df_name])
        elif "stock" in df_name:
            prep_data["stocks"] = prep_data.pop(df_name)
        elif "material" in df_name:
            prep_data["material_intensities"] = prep_data.pop(df_name)
        elif "share" in df_name:
            prep_data["shares"] = prep_data.pop(df_name)

    prep_data["knowledge_graph"] = create_electricity_graph()
    # prep_data["shares"] = None

    return prep_data


# def materials_grid_additions_to_kgperkm(materials_df, additions_df):
#     """
#     Vectorized approach to transform the materials DataFrame by multiplying each row with the corresponding values from the additions DataFrame.
#     materials_df: DataFrame with MultiIndex (year, technology) and columns for materials. - units in kg/unit (unit = 1 substation or transformer)
#     additions_df: DataFrame with index for components (Substations, Transformers) and columns for voltage levels (HV, MV, LV), values are the number of units per km of grid length.

#     NOT USED ANYMORE (?) -> separate stock modeling needed for lines vs substations & transformers

#     """
    
#     # Create mapping series
#     mapping_dict = {}
    
#     for voltage in ['HV', 'MV', 'LV']:
#         for component in ['Substations', 'Transformers']:
#             tech_key = f"{voltage} {component}"
            
#             if voltage in additions_df.columns and component in additions_df.index:
#                 mapping_dict[tech_key] = additions_df.loc[component, voltage]
    
#     # Create a series to map each row to its multiplier
#     multipliers = materials_df.index.get_level_values(1).map(mapping_dict)
    
#     # Convert to DataFrame for broadcasting
#     multipliers_df = pd.DataFrame(
#         np.outer(multipliers, np.ones(len(materials_df.columns))),
#         index=materials_df.index,
#         columns=materials_df.columns
#     )
    
#     # Multiply
#     result_df = materials_df * multipliers_df
    
#     return result_df


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
    xr.DataArray
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


# STOCK MODELLING OLD -----------------------------------------------------------------------------------------------------

# weibull_shape = 1.89  # for what was this used??
# weibull_scale = 10.3
# stdev_mult = 0.214      # multiplier that defines the standard deviation (standard deviation = mean * multiplier)

# calculate inflow & outflow with market shares in inflow (generation, storage other)
def stock_share_calc(stock, market_share, init_tech, techlist):
    YEAR_START = 1971  # start year of the simulation period
    YEAR_END = 2060    # end year of the calculations
    YEAR_OUT = 2060    # year of output generation = last year of reporting

    # Here, we define the market share of the stock based on a pre-calculation with several steps: 
    # 1) use Global total stock development, the market shares of the inflow and technology specific lifetimes to derive 
    # the shares in the stock, assuming that pre-1990 100% of the stock was determined by Lead-acid batteries. 
    # 2) Then, apply the global stock-related market shares to disaggregate the stock to technologies in all regions 
    # (assuming that battery markets are global markets)
    # As the development of the total stock of dedicated electricity storage is known, but we don't know the inflow, 
    # and only the market share related to the inflow we need to calculate the inflow first. 

    # first we define the survival of the 1990 stock (assumed 100% Lead-acid, for each cohort in 1990)
    pre_time = 20                           # the years required for the pre-calculation of the Lead-acid stock
    cohorts = YEAR_OUT-YEAR_SWITCH            # nr. of cohorts in the stock calculations (after YEAR_SWITCH)
    timeframe = np.arange(0,cohorts+1)      # timeframe for the pre-calcluation
    pre_year_list = list(range(YEAR_SWITCH-pre_time,YEAR_SWITCH+1))   # list of years to pre-calculate the Lead-acid stock for

    #define new dataframes
    stock_cohorts = pd.DataFrame(index=pd.MultiIndex.from_product([stock.columns,range(YEAR_SWITCH,YEAR_OUT+1)]), columns=pd.MultiIndex.from_product([techlist, range(YEAR_SWITCH-pre_time,YEAR_END+1)])) # year, by year, by tech
    inflow_by_tech = pd.DataFrame(index=pd.MultiIndex.from_product([stock.columns,range(YEAR_SWITCH,YEAR_OUT+1)]), columns=techlist)
    outflow_cohorts = pd.DataFrame(index=pd.MultiIndex.from_product([stock.columns,range(YEAR_SWITCH,YEAR_OUT+1)]), columns=pd.MultiIndex.from_product([techlist, range(YEAR_SWITCH-pre_time,YEAR_END+1)])) # year by year by tech
    
    #specify lifetime & other settings
    mean = storage_lifetime_interpol.loc[YEAR_SWITCH,init_tech] # select the mean lifetime for Lead-acid batteries in 1990
    stdev = mean * stdev_mult               # we use thesame standard-deviation as for generation technologies, given that these apply to 'energy systems' more generally
    survival_init = scipy.stats.foldnorm.sf(timeframe, mean/stdev, 0, scale=stdev)
    techlist_new = techlist
    techlist_new.remove(init_tech)          # techlist without Lead-acid (or other init_tech)
    
    # actual inflow & outflow calculations, this bit takes long!
    # loop over regions, technologies and years to calculate the inflow, stock & outflow of storage technologies, given their share of the inflow.
    for region in stock.columns:
        
        # pre-calculate the stock by cohort of the initial stock of Lead-acid
        multiplier_pre = stock.loc[YEAR_SWITCH,region]/survival_init.sum()   # the stock is subdivided by the previous cohorts according to the survival function (only allowed when assuming steady stock inflow) 
        
        #pre-calculate the stock as lists (for efficiency)
        initial_stock_years = [np.flip(survival_init[0:pre_time+1]) * multiplier_pre]
            
        for year in range(1, (YEAR_OUT-YEAR_SWITCH)+1):      # then fill the columns with the remaining fractions
            initial_stock_years.append(initial_stock_years[0] * survival_init[year])
    
        stock_cohorts.loc[idx[region,:],idx[init_tech, list(range(YEAR_SWITCH-pre_time,YEAR_SWITCH+1))]] = initial_stock_years       # fill the stock dataframe according to the pre-calculated stock 
        outflow_cohorts.loc[idx[region,:],idx[init_tech, list(range(YEAR_SWITCH-pre_time,YEAR_SWITCH+1))]] = stock_cohorts.loc[idx[region,:],idx[init_tech, list(range(YEAR_SWITCH-pre_time,YEAR_SWITCH+1))]].shift(1, axis=0) - stock_cohorts.loc[idx[region,:],idx[init_tech, list(range(YEAR_SWITCH-pre_time,YEAR_SWITCH+1))]]
    
        # set the other stock cohorts to zero
        stock_cohorts.loc[idx[region,:],idx[techlist_new, pre_year_list]] = 0
        outflow_cohorts.loc[idx[region,:],idx[techlist_new, pre_year_list]] = 0
        inflow_by_tech.loc[idx[region, YEAR_SWITCH], techlist_new] = 0                                                   # inflow of other technologies in 1990 = 0
        
        # except for outflow and inflow in 1990 (YEAR_SWITCH), which can be pre-calculated for Deep-cycle Lead Acid (@ steady state inflow, inflow = outflow = stock/lifetime)
        outflow_cohorts.loc[idx[region, YEAR_SWITCH], idx[init_tech,:]] = outflow_cohorts.loc[idx[region, YEAR_SWITCH+1], idx[init_tech,:]]     # given the assumption of steady state inflow (pre YEAR_SWITCH), we can determine that the outflow is the same in switchyear as in switchyear+1                                        
        inflow_by_tech.loc[idx[region, YEAR_SWITCH], init_tech] =   stock.loc[YEAR_SWITCH,region]/mean                                          # given the assumption of steady state inflow (pre YEAR_SWITCH), we can determine the inflow to be the same as the outflow at a value of stock/avg. lifetime                                                          
        
        # From YEAR_SWITCH onwards, define a stock-driven model with a known market share (by tech) of the new inflow 
        for year in range(YEAR_SWITCH+1,YEAR_OUT+1):
            
            # calculate the remaining stock as the sum of all cohorts in a year, for each technology
            remaining_stock = 0 # reset remaining stock
            for tech in inflow_by_tech.columns:
                remaining_stock += stock_cohorts.loc[idx[region,year],idx[tech,:]].sum()
               
            # total inflow required (= required stock - remaining stock);    
            inflow = max(0, stock.loc[year, region] - remaining_stock)   # max 0 avoids negative inflow, but allows for idle stock surplus in case the size of the required stock is declining more rapidly than it's natural decay
            
            stock_cohorts_list = []
                   
            # enter the new inflow & apply the survival rate, which is different for each technology, so calculate the surviving fraction in stock for each technology  
            for tech in inflow_by_tech.columns:
                # apply the known market share to the inflow
                inflow_by_tech.loc[idx[region,year],tech] = inflow * market_share.loc[year,tech]
                # first calculate the  (based on lifetimes specific to the year of inflow)
                survival = scipy.stats.foldnorm.sf(np.arange(0,(YEAR_OUT+1)-year), storage_lifetime_interpol.loc[year,tech]/(storage_lifetime_interpol.loc[year,tech]*0.2), 0, scale=storage_lifetime_interpol.loc[year,tech]*0.2)           
                # then apply the survival to the inflow in current cohort, both the inflow & the survival are entered into the stock_cohort dataframe in 1 step
                stock_cohorts_list.append(inflow_by_tech.loc[idx[region,year],tech]  *  survival)
                
            stock_cohorts.loc[idx[region,list(range(year,YEAR_OUT+1))],idx[:,year]] = list(map(list, zip(*stock_cohorts_list)))        
        
    # separate the outflow (by cohort) calculation (separate shift calculation for each region & tech is MUCH more efficient than including it in additional loop over years)
    # calculate the outflow by cohort based on the stock by cohort that was just calculated
    for region in stock.columns:
        for tech in inflow_by_tech.columns:
            outflow_cohorts.loc[idx[region,:],idx[tech,:]] = stock_cohorts.loc[idx[region,:],idx[tech,:]].shift(1,axis=0) - stock_cohorts.loc[idx[region,:],idx[tech,:]]

    return inflow_by_tech, stock_cohorts, outflow_cohorts

# calculate inflow & outflow
# first define a Function in which the stock-driven DSM is applied to return (the moving average of the) inflow & outflow for all regions
def inflow_outflow(stock, lifetime, material_intensity, key):

    YEAR_START = 1971  # start year of the simulation period
    YEAR_END = 2060    # end year of the calculations
    YEAR_OUT = 2060    # year of output generation = last year of reporting

    initial_year = stock.first_valid_index()
    outflow_mat  = pd.DataFrame(index=pd.MultiIndex.from_product([range(startyear,YEAR_OUT+1), material_intensity.columns]), columns=stock.columns)
    inflow_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(startyear,YEAR_OUT+1), material_intensity.columns]), columns=stock.columns)   
    stock_mat    = pd.DataFrame(index=pd.MultiIndex.from_product([range(startyear,YEAR_OUT+1), material_intensity.columns]), columns=stock.columns)
    out_oc_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid,YEAR_OUT+1), material_intensity.columns]), columns=stock.columns)
    out_sc_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid,YEAR_OUT+1), material_intensity.columns]), columns=stock.columns)
    out_in_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid,YEAR_OUT+1), material_intensity.columns]), columns=stock.columns)

    # define mean & standard deviation
    mean_list = list(lifetime)
    stdev_list = [mean_list[i] * stdev_mult for i in range(0,len(stock))]  

    for region in list(stock.columns):
        # define and run the DSM                                                                                            # list with the fixed (=mean) lifetime of grid elements, given for every timestep (1926-2100), needed for the DSM as it allows to change lifetime for different cohort (even though we keep it constant)
        DSMforward = DSM(t = np.arange(0,len(stock[region]),1), s=np.array(stock[region]), lt = {'Type': 'FoldedNormal', 'Mean': np.array(mean_list), 'StdDev': np.array(stdev_list)})  # definition of the DSM based on a folded normal distribution
        out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model(NegativeInflowCorrect = True)                                                                 # run the DSM, to give 3 outputs: stock_by_cohort, outflow_by_cohort & inflow_per_year

        #convert to pandas df before multiplication with material intensity
        index=list(range(first_year_grid, YEAR_OUT+1))
        out_sc_pd = pd.DataFrame(out_sc, index=index,  columns=index)
        out_oc_pd = pd.DataFrame(out_oc, index=index,  columns=index)
        out_in_pd = pd.DataFrame(out_i,  index=index)

        # sum the outflow & stock by cohort (using cohort specific material intensities)
        for material in list(material_intensity.columns):    
           out_oc_mat.loc[idx[:,material],region] = out_oc_pd.mul(material_intensity.loc[:,material], axis=1).sum(axis=1).to_numpy()
           out_sc_mat.loc[idx[:,material],region] = out_sc_pd.mul(material_intensity.loc[:,material], axis=1).sum(axis=1).to_numpy() 
           out_in_mat.loc[idx[:,material],region] = out_in_pd.mul(material_intensity.loc[:,material], axis=0).to_numpy()                
    
           # apply moving average to inflow & outflow & return only 1971-2050 values
           outflow_mat.loc[idx[:,material],region] = pd.Series(out_oc_mat.loc[idx[2:,material],region].astype('float64').values, index=list(range(initial_year, YEAR_OUT + 1))).rolling(window=5).mean().loc[list(range(1971,YEAR_OUT + 1))].to_numpy()    # Apply moving average                                                                                                      # sum the outflow by cohort to get the total outflow per year
           inflow_mat.loc[idx[:,material],region]  = pd.Series(out_in_mat.loc[idx[2:,material],region].astype('float64').values, index=list(range(initial_year, YEAR_OUT + 1))).rolling(window=5).mean().loc[list(range(1971,YEAR_OUT + 1))].to_numpy()
           stock_mat.loc[idx[:,material],region]   = out_sc_mat.loc[idx[:,material],region].loc[list(range(1971,YEAR_OUT + 1))].to_numpy()                                                                                                        # sum the stock by cohort to get the total stock per year
        
    return pd.concat([inflow_mat.stack().unstack(level=1)], keys=[key], axis=1), pd.concat([outflow_mat.stack().unstack(level=1)], keys=[key], axis=1), pd.concat([stock_mat.stack().unstack(level=1)], keys=[key], axis=1)

# -----------------------------------------------------------------------------------------------------

