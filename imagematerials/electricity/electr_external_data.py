""""External validation data for the electricity model."""
from pathlib import Path
import os
import pint
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ureg = pint.UnitRegistry(force_ndarray_like=True)


#%% IEA ---------------------------------------------------------------------

# "Mineral demand for clean energy"
# APS: Announced Pledges Scenario
# NZS: Net Zero Emissions by 2050 Scenario
years        = np.array([2024, 2030, 2035, 2040, 2045, 2050])

# *1_000: values in kt -> to t
# Cu -------------
values_gen_solar    = np.array([1657, 2803, 2726, 2626, 2448, 2758])        * 1_000 
values_gen_wind     = np.array([534, 1171, 1038, 919, 968, 1314])           * 1_000
values_gen_other    = np.array([76, 166, 155, 135, 137, 121])               * 1_000
values_stor_ev      = np.array([497, 1789, 3018, 3976, 4372, 4743])         * 1_000
values_stor_gridbat = np.array([76, 244, 366, 567, 645, 806])               * 1_000
values_grid         = np.array([4929, 6543, 7266, 8073, 7970, 7479])        * 1_000
values_total_clean  = np.array([7737, 12578, 14579, 16352, 16624, 17176])   * 1_000
values_other_uses   = np.array([18980, 20042, 20005, 20388, 21504, 22465])  * 1_000
values_total_demand = np.array([26717, 32620, 34584, 36740, 38127, 39641])  * 1_000
df_iea_cu_aps = pd.DataFrame({ 
    'Year': years,
    'Gen_solar': values_gen_solar,
    'Gen_wind': values_gen_wind,
    'Gen_other': values_gen_other,
    'Stor_ev': values_stor_ev,
    'Stor_grid-batteries': values_stor_gridbat,
    'Grid': values_grid,
    'Total_clean': values_total_clean,
    'Total_other-uses': values_other_uses,
    'Total_demand': values_total_demand
})
df_iea_cu_aps.set_index('Year', inplace=True)
df_iea_cu_aps.name = 'IEA_APS Cu'

values_gen_solar    = np.array([1657, 2849, 3239, 2570, 2176, 2734])        * 1_000
values_gen_wind     = np.array([534, 1527, 1602, 1126, 1105, 1379])         * 1_000
values_gen_other    = np.array([76, 220, 232, 175, 139, 118])               * 1_000
values_stor_ev      = np.array([497, 2665, 4589, 5219, 5521, 5676])         * 1_000
values_stor_gridbat = np.array([76, 295, 495, 698, 843, 971])               * 1_000
values_grid         = np.array([4929, 7723, 9342, 9972, 9568, 8853])        * 1_000
values_total_clean  = np.array([7737, 15166, 19420, 19846, 19407, 19679])   * 1_000
values_other_uses   = np.array([18980, 19250, 18996, 19634, 20680, 21598])  * 1_000
values_total_demand = np.array([26717, 34416, 38416, 39480, 40087, 41277])  * 1_000
df_iea_cu_nzs = pd.DataFrame({ 
    'Year': years,
    'Gen_solar': values_gen_solar,
    'Gen_wind': values_gen_wind,
    'Gen_other': values_gen_other,
    'Stor_ev': values_stor_ev,
    'Stor_grid-batteries': values_stor_gridbat,
    'Grid': values_grid,
    'Total_clean': values_total_clean,
    'Total_other-uses': values_other_uses,
    'Total_demand': values_total_demand
})
df_iea_cu_nzs.set_index('Year', inplace=True)
df_iea_cu_nzs.name = 'IEA_NZS Cu'

# Co -------------
values_stor_ev      = np.array([66.5, 161.8, 176.5, 183.6, 220.4, 239.8]) *1_000 # battery electric vehicles
values_stor_gridbat = np.array([4.06, 5.74, 4.59, 0, 0, 0])               *1_000 # grid storage technologies
df_iea_co_aps = pd.DataFrame({ 
    'Year': years,
    'Stor_ev': values_stor_ev,
    'Stor_grid-batteries': values_stor_gridbat
})
df_iea_co_aps.set_index('Year', inplace=True)
df_iea_co_aps.name = 'IEA_APS Co'

# Mn -------------
values_stor_ev      = np.array([60.3, 402.9, 1145.8, 2297.3, 3101.1, 3717.8])   *1_000 # battery electric vehicles
values_gen_wind     = np.array([104.8, 208.9, 190.8, 169, 175.5, 238.8])        *1_000 # wind
values_gen_other    = np.array([16.6, 40.7, 69.8, 76.2, 77.3, 67.1])            *1_000 # other low emission power generation
values_stor_gridbat = np.array([4.3, 78, 164.9, 328.3, 432.1, 662.2])           *1_000 # grid battery storage
df_iea_mn_aps = pd.DataFrame({ 
    'Year': years,
    'Stor_ev': values_stor_ev,
    'Gen_wind': values_gen_wind,
    'Gen_other': values_gen_other,
    'Stor_grid-batteries': values_stor_gridbat
})
df_iea_mn_aps.set_index('Year', inplace=True)
df_iea_mn_aps.name = 'IEA_APS Mn'

# Ni -------------
values_gen_solar    = np.array([1, 1, 1, 1, 1, 1])                      * 1_000 # solar PV
values_gen_wind     = np.array([47, 92, 84, 74, 76, 102])               * 1_000 # wind generation
values_gen_other    = np.array([174, 361, 353, 381, 358, 444])          * 1_000 # other low emission power generation
values_stor_ev      = np.array([321, 1083, 1922, 2474, 2803, 2942])     * 1_000 # battery electric vehicles
values_stor_gridbat = np.array([17, 77, 137, 238, 313, 480])            * 1_000 # grid battery storage
values_hydrogen     = np.array([1, 33, 32, 32, 45, 64])                 * 1_000
values_total_clean  = np.array([562, 1647, 2529, 3201, 3596, 4033])     * 1_000 # total demand for clean technologies
values_other_uses   = np.array([2809, 2974, 2998, 3033, 3119, 3197])    * 1_000 # demand for other uses
values_total_demand = np.array([3371, 4620, 5527, 6233, 6715, 7230])    * 1_000 # total demand
df_iea_ni_aps = pd.DataFrame({ 
    'Year': years,
    'Gen_wind': values_gen_wind,
    'Gen_solar': values_gen_solar,
    'Gen_other': values_gen_other,
    'Hydrogen': values_hydrogen,
    'Stor_ev': values_stor_ev,
    'Stor_grid-batteries': values_stor_gridbat,
    'Total_clean': values_total_clean,
    'Total_other-uses': values_other_uses,
    'Total_demand': values_total_demand
})
df_iea_ni_aps.set_index('Year', inplace=True)
df_iea_ni_aps.name = 'IEA_APS Ni'
#------- ---------------------------------------------------------------------
