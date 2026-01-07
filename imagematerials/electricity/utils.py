
import numpy as np
import pandas as pd
import xarray as xr
import math
import scipy.stats
import prism

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
    stock_new  = stock_used.reindex(list(range(YEAR_FIRST_GRID,YEAR_OUT+1))).interpolate() #TODO: why YEAR:OUT (2060) and not YEAR_START (1971)?
    # The explanation above is from Sebastiaan and indicates it should be 1971 (which also makes sense), however in his code it is YEAR_OUT (2060)
    # I think I had indeed some jumps around 1971, maybe this is the explanation
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
    """
    Interpolate an xarray.DataArray over a continuous time range and 
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
    da_interp = data_array.interp({dim: new_range}, method = interp_method)

    # Fill values outside original range
    da_interp.loc[{dim: slice(None, coord_values.min())}] = da_interp.sel({dim: coord_values.min()})
    da_interp.loc[{dim: slice(coord_values.max(), None)}] = da_interp.sel({dim: coord_values.max()})

    # Reattach the unit if it existed
    if unit != prism.Unit('dimensionless'):
        da_interp = prism.Q_(da_interp, unit)

    return da_interp


def MNLogit(df, logitpar):
    '''
    Multinomial Logit function, assumes input of an ordered dataframe with rows as years and columns as technologies, values as prices. 
    
    logitpar: calibrated Logit parameter (usually a nagetive number between 0 and 1)
    - represents the price sensitivity or substitution elasticity between different technologies.
    -> how responsive consumers/markets are to price differences between technologies.
    - More negative values (e.g., -2, -5) = higher price sensitivity -> Small price differences lead to large changes in market share = Technologies are highly substitutable
    - Less negative values (e.g., -0.1, -0.5) = lower price sensitivity -> Price differences have less impact on market share = Technologies are less substitutable (perhaps due to quality differences, switching costs, etc.)

    - mathematically: exp(logitpar * price) means: When logitpar is negative and prices are positive, higher prices get exponentially smaller weights
    
    TODO: check: is this true?
    '''
    new_dataframe = pd.DataFrame(index=df.index, columns=df.columns)
    for year in range(df.index[0],df.index[-1]+1): #from first to last year
        yearsum = 0
        for column in df.columns:
            yearsum += math.exp(logitpar * df.loc[year,column]) # calculate the sum of the prices
        for column in df.columns:
            new_dataframe.loc[year,column] = math.exp(logitpar * df.loc[year,column])/yearsum
    return new_dataframe    # the retuned dataframe contains the market shares


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
    """ Sanitize the attributes of a DataArray and its coordinates for safe serialization.

    This function converts all attribute values that are not of type str, int, or float 
    into strings. It applies this transformation to both the DataArray's `.attrs` and 
    each coordinate's `.attrs`. This is useful when saving xarray objects to formats 
    like NetCDF, which require attribute values to be basic serializable types.

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

def compare_da(path, da_new): # for testing xarray objects
    """ Compare a saved DataArray to a new one, ignoring units.

    Parameters
    ----------
    path : str or Path
        Path to the saved DataArray file.
    da_new : xarray.DataArray
        The new DataArray to compare.

    Returns
    -------
    equal : bool
        True if the DataArrays match numerically after removing units.
    diff_nonzero : xarray.DataArray or None
        Differences where values differ; None if equal.
    """
    # use as:
    # compare_da(path_test / "grid_lines_inflow_v0.nc", model_lines.inflow.to_array())
    
    da_old = xr.open_dataarray(path)

    da_old_clean = da_old.pint.dequantify()
    da_new_clean = da_new.pint.dequantify()

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

