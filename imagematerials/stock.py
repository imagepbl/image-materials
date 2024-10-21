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
    first_year = stock.coords["time"][0]
    for t in stock.coords["time"].loc[first_year+1:start_simulation]:
        # We assume zeros for all variables in the first year
        stock_function(stock, stock_by_cohort, inflow, outflow_by_cohort, survival, t, shares)

def compute_dynamic_stock_driven(stock, stock_by_cohort, inflow, outflow_by_cohort, survival, t,
                                 shares):
    input_stock = stock
    typical_missing = _find_missing(input_stock.loc[t])
    stock_diff = input_stock.loc[t] - stock_by_cohort[t].sum("cohort")
    # Drop dimension cohort
    stock_diff = xr.where(stock_diff>0, stock_diff/survival[t, t].drop("cohort"), 0)
    _add_subtype_stock(stock_diff, shares, typical_missing, t)
    inflow[t] = stock_diff

    try:
        t_str = str(t.values)
    except AttributeError:
        t_str = str(t)
    print(t_str)
    for t_future in stock_by_cohort[t].coords["cohort"].loc[t_str:]:
        t_future = int(t_future)
        stock_by_cohort[t_future].loc[{"cohort": t_str}] = inflow[t]*survival[t_future, t]
    outflow_by_cohort[t] = stock_by_cohort[t]-stock_by_cohort[t-1]

# TODO: function below is broken, use the same fix as compute_dynamic_stock_driven.
def compute_dynamic_inflow_driven(stock, stock_by_cohort, inflow, outflow_by_cohort, survival, t):
    input_inflow = inflow
    stock_by_cohort[t] = input_inflow[t] * survival[t, :]
    stock[t] = stock_by_cohort[t].sum("cohort")
    outflow_by_cohort[t] = stock_by_cohort[t]-stock_by_cohort[t-1]

def _add_subtype_stock(stock_diff, shares, typical_missing, cur_cohort):
    for cur_mode in typical_missing:
        split_mode = cur_mode.split(SUBTYPE_SEPARATOR)
        if len(split_mode) != 2:
            raise ValueError("Can only do basetype -> specific type conversion.")
        base_mode = split_mode[0]
        stock_diff.loc[{"mode": cur_mode}] = (stock_diff.loc[{"mode": base_mode}]
                                              * shares.loc[{"mode": cur_mode, "cohort": cur_cohort}])

def _find_missing(input_stock):
    all_modes = input_stock.coords["mode"].values
    non_na_modes = input_stock.dropna("mode").coords["mode"].values
    return set(all_modes) - set(non_na_modes)
