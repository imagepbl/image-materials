import xarray as xr


def compute_historic(input_stock, survival, start_simulation, stock_by_cohort, inflow,
                     outflow_by_cohort, stock_function):
    first_year = input_stock.coords["time"][0]
    for t in input_stock.coords["time"].loc[first_year+1:start_simulation]:
        # We assume zeros for all variables in the first year 
        stock_function(input_stock, stock_by_cohort, inflow, outflow_by_cohort, survival, t)

def compute_dynamic_stock_driven(stock, stock_by_cohort, inflow, outflow_by_cohort, survival, t):
    input_stock = stock
    stock_diff = input_stock.loc[t] - stock_by_cohort[t].sum("cohort")
    # Drop dimension cohort
    stock_diff = xr.where(stock_diff>0, stock_diff/survival[t, t].drop("cohort"), 0)
    inflow[t] = stock_diff
    stock_by_cohort[t] = inflow[t]*survival[t, :]
    outflow_by_cohort[t] = stock_by_cohort[t]-stock_by_cohort[t-1]

def compute_dynamic_inflow_driven(stock, stock_by_cohort, inflow, outflow_by_cohort, survival, t):
    input_inflow = inflow
    stock_by_cohort[t] = input_inflow[t] * survival[t, :]
    stock[t] = stock_by_cohort[t].sum("cohort")
    outflow_by_cohort[t] = stock_by_cohort[t]-stock_by_cohort[t-1]
