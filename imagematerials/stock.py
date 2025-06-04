import prism
import xarray as xr

from imagematerials.constants import SUBTYPE_SEPARATOR
from imagematerials.survival import SurvivalMatrix


def compute_historic(stock: prism.TimeVariable,
                     survival: SurvivalMatrix,
                     start_simulation: int,
                     stock_by_cohort: prism.TimeVariable,
                     inflow: prism.TimeVariable,
                     outflow_by_cohort: prism.TimeVariable,
                     shares: prism.TimeVariable,
                     stock_function: callable):
    """Compute the historic time series using the stock_function.

    Parameters
    ----------
    stock
        Stock, either used as an input (with the stock driven model), or
        as an output (with the dynamic inflow driven model).
    survival
        Survival matrix (can be dynamic) to compute the survival of stocks.
    start_simulation
        Year for which the simulation is started and thus the historic tail ends.
    stock_by_cohort
        Available stocks for each cohort at each year, mostly an output.
    inflow
        Inflow of stocks at each year, can be input or output.
    outflow_by_cohort
        Outflow of the stocks for each of the cohorts at time t.
    stock_function
        Function to compute the stocks/inflow/outflow for one timestep.
        Currently there are two functions: compute_dynamic_stock_driven and
        compute_dynamic_inflow_driven.

    """
    first_year = stock.coords["Time"][0]
    for t in stock.coords["Time"].loc[first_year+1:start_simulation]:
        # We assume zeros for all variables in the first year
        stock_function(stock, stock_by_cohort, inflow, outflow_by_cohort, survival, t, shares)

def compute_dynamic_stock_driven(stock, stock_by_cohort, inflow, outflow_by_cohort, survival, t,
                                 shares):
    input_stock = stock
    typical_missing = _find_missing(input_stock.loc[t])
    stock_diff = input_stock.loc[t] - stock_by_cohort.loc[t].sum("Cohort")
    # Drop dimension cohort
    stock_diff = xr.where(stock_diff>0, stock_diff/survival[t, t].drop("Cohort"), 0)
    _add_subtype_stock(stock_diff, shares, typical_missing, t)
    inflow[t] = stock_diff

    try:
        t_str = str(t.values)
    except AttributeError:
        t_str = str(t)
    print(f"{t_str}", end="\r")
    stock_by_cohort.loc[t:, t] = inflow[t]*survival[t:, t]
    # for t_future in stock_by_cohort[t].coords["Cohort"].loc[t_str:]:
        # t_future = int(t_future)
        # stock_by_cohort[t_future].loc[{"Cohort": t_str}] = inflow[t]*survival[t_future, t]

    # Prevent out of bounds error, assume first outflow to be 0.
    if t-1 < stock_by_cohort.coords["Time"].min():
        outflow_by_cohort[t] = 0.0
    else:
        outflow_by_cohort[t] = stock_by_cohort.loc[t-1] - stock_by_cohort.loc[t]

# TODO: function below is broken, use the same fix as compute_dynamic_stock_driven.
def compute_dynamic_inflow_driven(stock, stock_by_cohort, inflow, outflow_by_cohort, survival, t):
    input_inflow = inflow
    stock_by_cohort[t] = input_inflow[t] * survival[t, :]
    stock[t] = stock_by_cohort[t].sum("Cohort")
    outflow_by_cohort[t] = stock_by_cohort[t-1]-stock_by_cohort[t]

def _add_subtype_stock(stock_diff, shares, typical_missing, cur_cohort):
    for cur_mode in typical_missing:
        split_mode = cur_mode.split(SUBTYPE_SEPARATOR)
        if len(split_mode) != 2:
            raise ValueError("Can only do basetype -> specific type conversion.")
        base_mode = split_mode[0]
        stock_diff.loc[{"Type": cur_mode}] = (stock_diff.loc[{"Type": base_mode}]
                                              * shares.loc[{"Type": cur_mode, "Cohort": cur_cohort}])

def _find_missing(input_stock):
    all_modes = input_stock.coords["Type"].values
    non_na_modes = input_stock.dropna("Type").coords["Type"].values
    return set(all_modes) - set(non_na_modes)


def material_computation(input_array, material_fractions, weights):
    return input_array*material_fractions*weights

def battery_computation(input_array, battery_shares, battery_materials, battery_weights):
    return input_array*battery_shares*battery_materials*battery_weights
