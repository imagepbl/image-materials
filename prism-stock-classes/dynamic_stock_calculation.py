def dynamic_stock_model(
        self,
        time: prism.Time,
        input_stock: prism.Array['count'],
        survival: Survival):
    t = time.t
    dt = time.dt

    # determine inflow from mass balance
    stockDiff = input_stock - self.stock_cohort[t].sum('cohort') # Sum remainder of previous cohorts left in this timestep
    self.inflow[t] = prism.switch(
        stockDiff > 0,
        stockDiff / self.survival.loc[{Cohort.dim_label: t, Year.dim_label: t}], 
        # if less than all survives increase inflow to fulfill demand in first year
        default = prism.Array['count'](0.0) # TODO should this be a single number or should this have a dimenaion?
    )
    self.stock_cohort[t] = self.inflow[t] * self.survival.loc[{Cohort.dim_label: t}] # depreciation of inflow at this timestep
    # TODO consider calculating survival for past stocks not future
    self.outflow_cohort[t] = -1 * np.diff(self.stock_cohort[t], dt, axis=0, prepend=0) # TODO potential improvement use np.tril to not include what is above the diagonal
    #TODO check if this results in a negative of positive number
