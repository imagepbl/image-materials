def compute_historic(input_stock, survival, start_simulation, stock_by_cohort, inflow,
                     outflow_by_cohort):
    first_year = input_stock.coords["time"][0]
    for t in input_stock.coords["time"].loc[first_year+1:start_simulation]:
        stock_diff = input_stock.loc[t] - stock_by_cohort[t].sum("cohort")
        inflow[t] = stock_diff
        stock_by_cohort[t] = inflow[t]*survival[t, :]
        outflow_by_cohort[t] = stock_by_cohort[t]-stock_by_cohort[t-1]
