import xarray as xr


def compute_historic(input_stock, survival, start_simulation, stock_by_cohort, inflow,
                     outflow_by_cohort):
    first_year = input_stock.coords["time"][0]
    for t in input_stock.coords["time"].loc[first_year+1:start_simulation]:
        stock_diff = input_stock.loc[t] - stock_by_cohort[t].sum("cohort")
        # print(t)
        print(stock_diff.coords, survival[t, t].coords, t)
        stock_diff = xr.where(stock_diff>0, stock_diff/survival[t, t], 0)
        inflow[t] = stock_diff
        stock_by_cohort[t] = inflow[t]*survival[t, :]
        outflow_by_cohort[t] = stock_by_cohort[t]-stock_by_cohort[t-1]
