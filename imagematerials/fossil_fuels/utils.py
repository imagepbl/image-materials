import numpy as np
import pandas as pd

from imagematerials.fossil_fuels import constants

# TODO: chekc if this can be substituted with add_historic_stock() from electricity/utils

# Generate/prepend historic fossil fuel consumption profile (to avoid inflow shocks in the first year of 'stock-driven' dynamic stock calculations)
# add historic timeseries based on initial/first year of production, with the possibility to include a stabilisation period before the last know year of data
def generate_historic_profile(fuel_use_total, fuel):

    year_list = list(
        range(constants.YEAR_FIRST_PRODUCTION[fuel], constants.START_YEAR)
    )
    prepend = pd.DataFrame(
        np.nan,
        index=year_list,
        columns=list(range(1, constants.NUM_REGIONS + 1)),
    )

    # set first year to 0 & latest unknown year to latest known year
    prepend.loc[year_list[0]] = float(0)
    prepend.loc[year_list[-1]] = fuel_use_total.loc[constants.START_YEAR]

    # if the stabilisation year is within the time series, then extend the last know year first
    if constants.YEAR_STABILISATION[fuel] < year_list[-1]:
        for year in range(constants.YEAR_STABILISATION[fuel], year_list[-1]):
            prepend.loc[year] = fuel_use_total.loc[constants.START_YEAR]

    prepend = prepend.interpolate(method="index")
    combine = pd.concat([prepend, fuel_use_total])

    return combine