from dataclasses import dataclass

import numpy as np
import xarray as xr

from prism import switch, Time, Timeline

from survival import Survival, RecursiveSurvival


@dataclass
class BaseStock:
    """Base class for stock models"""
    _time: Timeline
    _data: xr.DataArray
    inflow: xr.DataArray
    outflow: xr.DataArray

    def __post_init__(self):
        timeslots = self._time.all_timeslots()
        nt = len(timeslots)
        self._matrix = xr.DataArray(
            dims=["time", "cohort"],
            coords=dict(
                time=timeslots,
                cohort=timeslots,
            ),
            data=np.zeros((nt, nt)),
        )

    def __getitem__(self, key: tuple) -> xr.DataArray:
        """Get stock at time `t`."""
        t, cohort, *dims = key
        time_index = self._time.time_to_index(t)
        cohort_index = self._time.time_to_index(cohort)
        return self._data[time_index, cohort_index]
    
    def __setitem__(self, key: tuple, value: xr.DataArray):
        """Set stock at time `t`."""
        t, cohort, *dims = key
        time_index = self._time.time_to_index(t)
        cohort_index = self._time.time_to_index(cohort)
        self._data[time_index, cohort_index] = value

    # def add(self, value: xr.DataArray):
    #     pass

    # def remove(self, value: xr.DataArray):
    #     pass

    def total(self, time: float) -> xr.DataArray:
        return self._data.sel(time=time).sum(dim="cohort")


@dataclass
class Stock:
    stock: BaseStock

    def compute_from_stock(self, time: Time, required_stock: xr.DataArray, survival: Survival):
        """Compute inflow, outflow, and stock values for a given time step."""
        t, dt = time.t, time.dt
        survival.compute(t)
        # stock[t] = stock[t-dt] + inflow[t] - outflow[t]
        # where outflow[t] = sum(outflow_cohort[t])
        # and required_stock = sum(stock[t]) for all cohorts

        # determine inflow from mass balance
        self.stock.outflow[t] = self.stock[t - dt,:] * survival[t, :]
        stock_to_add = required_stock - self.stock.total(t - dt)  # Sum remainder of previous cohorts left in this timestep
        self.stock.inflow[t] = switch(
            stock_to_add > 0, stock_to_add / survival[t, t], default = 0.0
            )  # TODO check whether we need/want to adjust what prism accepts for default
        self.stock[t, t] = stock_to_add

    def compute_from_inflow(self, time: Time, inflow: xr.DataArray, survival: RecursiveSurvival):
        """Compute outflow and stock values for a given time step."""
        t, dt = time.t, time.dt
        # stock[t] = stock[t-dt] + inflow[t] - outflow[t]
        # where outflow[t] = sum(outflow_cohort[t])

        # determine outflow from mass balance
        self.stock.inflow[t] = inflow
        stock_to_add = inflow * survival[t, t]
        self.stock[t,t] = stock_to_add
        self.stock.outflow[t] = self.stock[t - dt,:] * survival[t, :]
