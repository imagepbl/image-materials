from dataclasses import dataclass

import xarray as xr


@dataclass
class TimeVariable[*Dimensions, Unit]:
    _data: xr.DataArray
