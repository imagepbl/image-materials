from imagematerials.end_of_life.constants import SCENARIO_SELECT, start_year, end_year
from imagematerials.end_of_life.preprocessing import (
    get_eol_rates,
    get_eol_xr,
    compute_eol,

)

def eol_preprocessing (xr_collection, xr_reuse, xr_recycling)
    # inter and extrapolated collection, reuse and recycling xr 
    collection = compute_eol(xr_collection, start_year= start_year, end_year=end_year,min_value = 0,max_value = 1)
    reuse = compute_eol(xr_reuse, start_year= start_year, end_year=end_year,min_value = 0,max_value = 1)
    recyling = compute_eol(xr_recycling, start_year= start_year, end_year=end_year,min_value = 0,max_value = 1)

    return collection, reuse, recycling