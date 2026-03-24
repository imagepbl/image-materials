# General constants
NUM_REGIONS = 26  # 26 IMAGE regions
FIRST_YEAR = 1900  # first year for the historically generated dataset
START_YEAR = 1971  # first year available in IMAGE scenarios
END_YEAR = 2100  # last year available in IMAGE scenarios
BASE_YEAR = 2019  # last year before COVID pandemic, so realistic numbers

# Unit conversions
PJ_TO_GJ = 10**6  # Peta-joule to Giga-joule
TJ_TO_PJ = 10**3  # Tera-joule to Peta-joule
MB_TO_M3 = 159000  # million barrels of oil to cubic meters
GJ_TO_M3_GAS = (
    27.77778  # m3/GJ gas (source: BP, Statistical Review of World Energy)
)
GJ_TO_KG_OIL = 23.88  # kg/GJ oil (source: IEA, https://www.iea.org/reports/unit-converter-and-glossary)
GJ_TO_KG_COAL = 34.12  # in kg/GJ of hard coal (source:IEA, https://www.iea.org/reports/unit-converter-and-glossary)
M3_petro_TO_GJ = 36.00648  # https://www.convert-measurement-units.com/convert+m3+Petrol+to+GJ.php
GJ_TO_M3_OIL = 0.027772778677616  # https://www.convert-measurement-units.com/convert+m3+Petrol+to+GJ.php
M3_TO_KG_OIL = 711.22
MB_TO_KG_OIL = 138000000  # http://globalshift.co.uk/conv.html
Bcm_TO_KG_GAS = 735000000  # chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.bp.com/content/dam/bp/business-sites/en/global/corporate/pdfs/energy-economics/statistical-review/bp-stats-review-2022-approximate-conversion-factors.pdf
Bcm_To_GJ = 36000000  # billion cubic meter gas to Gj gas BP
GJ_TO_EJ = 10**-9  # Giga-joule to Exajoule
KG_TO_MT = 10**-9
KG_TO_KT = 10**-6
# Standard deviation
SD = 0.3  # lifetimes are expressed as the mean lifetime. Lifetime (probability density) distribution is then determined using a standard deviation of 10% of the mean.
