import prism
import xarray as xr

from imagematerials.survival import SurvivalMatrix


def compute_historic(stock: prism.TimeVariable,
                     survival: SurvivalMatrix,
                     start_simulation: int,
                     stock_by_cohort: prism.TimeVariable,
                     inflow: prism.TimeVariable,
                     outflow_by_cohort: prism.TimeVariable,
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
        stock_function(stock, stock_by_cohort, inflow, outflow_by_cohort, survival, t)

def compute_dynamic_stock_driven(stock, stock_by_cohort, inflow, outflow_by_cohort, survival, t):
    input_stock = stock
    stock_diff = input_stock.loc[t] - stock_by_cohort[t].sum("cohort")
    # Drop dimension cohort
    stock_diff = xr.where(stock_diff>0, stock_diff/survival[t, t].drop("cohort"), 0)
    inflow[t] = stock_diff

    try:
        t_str = str(t.values)
    except AttributeError:
        t_str = str(t)
    print(t_str)
    for t_future in stock_by_cohort[t].coords["cohort"].loc[t_str:]:  # Should be + dt
        t_future = int(t_future)
        stock_by_cohort[t_future].loc[{"cohort": t_str}] = inflow[t]*survival[t_future, t]
    outflow_by_cohort[t] = stock_by_cohort[t]-stock_by_cohort[t-1]

def compute_dynamic_inflow_driven(stock, stock_by_cohort, inflow, outflow_by_cohort, survival, t):
    input_inflow = inflow
    stock_by_cohort[t] = input_inflow[t] * survival[t, :]
    stock[t] = stock_by_cohort[t].sum("cohort")
    outflow_by_cohort[t] = stock_by_cohort[t]-stock_by_cohort[t-1]
