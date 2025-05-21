""""Global constants for the VEMA model."""
from pathlib import Path
import os
import pint

ureg = pint.UnitRegistry(force_ndarray_like=True)

base_dir = Path.cwd()

# --- Settings & constants

# start year of historic IMAGE data
START_YEAR = 1971
# start year of the full model period (including stock-development from
# scratch, which needs to be the oldest year of any vehicle, all stock
# calculations are initiated in this year, so this has an effect on
# runtime)
# TODO: set FIRST_YEAR based on minimum value in data-files
FIRST_YEAR = 1807  # first_year_vehicle.values.min()
END_YEAR = 2060
# year of output generation
OUT_YEAR = 2060
REGIONS = 26
# reference loadfactor of cars in TIMER (the trp_trvl_Load.out file is
# relative to this BASE loadfcator (persons/car))
LOAD_FACTOR = 1.6

LIGHT_COMMERCIAL_VEHICLE_SHARE = 0.04 
# 0.04 is the fraction of the tkms driven by light commercial vehicles according to the IEA

MEGA_TO_TERA = 1_000_000  # 1 Tera = 1,000,000 Mega
PKMS_TO_VKMS = 1_000_000_000_000
TONNES_TO_KGS = 1000

SHIPS_YEARS_RANGE = list(range(2005,2018+1))

# --- Labels
bus_label = ["BusOil", "BusBio", "BusGas", "BusElecTrolley", "Bus Hybrid1",
             "Bus Hybrid2", "BusBattElectric", "", "", "", "", "", "", "", "",
             "", "", "", "", "", "", "", "", "", ""]
truck_label = ["Conv. ICE(2000)", "Conv. ICE(2010)", "Adv. ICEOil",
               "Adv. ICEH2", "Turbo-petrol IC", "Diesel ICEOil",
               "Diesel ICEBio", "ICE-HEV-gasoline", "ICE-HEV-diesel oil",
               "ICE-HEV-H2", "ICE-HEV-CNG Gas", "ICE-HEV-diesel bio",
               "FCV Oil", "FCV Bio", "FCV H2", "PEV-10 OilElec.",
               "PEV-30 OilElec.", "PEV-60 OilElec.", "PEV-10 BioElec.",
               "PEV-30 BioElec.", "PEV-60 BioElec.", "BEV Elec.", "", "", ""]
car_label = ["Conv. ICE(2000)", "Conv. ICE(2010)", "Adv. ICEOil",
             "Adv. ICEH2", "Turbo-petrol IC", "Diesel ICEOil",
             "Diesel ICEBio", "ICE-HEV-gasoline", "ICE-HEV-diesel oil",
             "ICE-HEV-H2", "ICE-HEV-CNG Gas", "ICE-HEV-diesel bio", "FCV Oil",
             "FCV Bio", "FCV H2", "PEV-10 OilElec.", "PEV-30 OilElec.",
             "PEV-60 OilElec.", "PEV-10 BioElec.", "PEV-30 BioElec.",
             "PEV-60 BioElec.", "BEV Elec.", "PHEV_BEV", "BEV", "Gas car"]

# lables of original IMAGE input files, Capital Letters are used 1-on-1 lower case modes are either ignored or disaggregated
tkms_label = ["Inland Ships", "Freight Trains", "Medium Freight Trucks",
              "Heavy Freight Trucks", "Freight Planes", "international shipping", "empty",
              "total"]
pkms_label = ["walking", "Bikes", "bus", "Trains", "Cars", "High Speed Trains", 
              "Passenger Planes", "total"]

labels_simple  = ['Passenger Planes', 'Bikes', 'Freight Planes','Freight Trains', 
                  'High Speed Trains', 'Inland Ships', 'Large Ships', 'Medium Ships', 
                  'Small Ships', 'Trains', 'Very Large Ships']
labels_typical = ['Cars', 'Light Commercial Vehicles', 'Medium Freight Trucks',
                  'Heavy Freight Trucks', 'Midi Buses', 'Regular Buses']

# Output labels differ from the main code, because buses & planes & trucks & ships are summed
columns_vehicle_output = ["Buses", "Trains", "High Speed Trains", "Cars", 
                          "Planes", "Bikes", "Trucks", "Freight Trains", 
                          "Ships", "Inland Ships", "Freight Planes"]

# TODO: plot labels should be reconsidered
# Names used to shorten plots
labels_pas = ["bicycle", "rail_reg", "rail_hst", "midi_bus", "reg_bus",
              "air_pas", "ICE", "HEV", "PHEV", "BEV", "FCV"]
# Names used to shorten plots
labels_fre = ["inland_shipping", "rail_freight", "LCV", "MFT", "HFT",
              "air_freight", "sea_shipping_small", "sea_shipping_med",
              "sea_shipping_large", "sea_shipping_vl"]
labels_materials = ["Steel", "Aluminium", "Cu", "Plastics", "Glass", "Ti",
                    "Wood", "Rubber", "Li", "Co", "Ni", "Mn", "Nd", "Pb"]
labels_ev_batt = ["NiMH", "LMO", "NMC", "NCA", "LFP", "Lithium Sulfur",
                  "Lithium Ceramic", "Lithium-air"]


# define #TODO Roel is that fine?
ureg = pint.UnitRegistry(force_ndarray_like=True)
ureg.define('percent = dimensionless * 100 = %')

# Define the units for each dimension
unit_mapping = {
    'time': ureg.year,
    'year': ureg.year,
    'kg': ureg.kilogram,
    'yr': ureg.year,
    '%': ureg.percent,
    't': ureg.tonne,
}

# aggregate car types into 5 car types
BEV_collist  = [22, 24]
PHEV_collist = [23, 21, 20, 19, 18, 17, 16]
ICE_collist  = [1,2,3,4,5,6,7,25]             # Gas car is considered ICE
HEV_collist  = [8,9,10,11,12]
FCV_collist  = [13,14,15]
car_collists = {
    'BEV': BEV_collist,
    'PHEV': PHEV_collist,
    'ICE': ICE_collist,
    'HEV': HEV_collist,
    'FCV': FCV_collist
}

drive_trains = ['ICE', 'HEV', 'PHEV', 'BEV', 'FCV', 'Trolley']
typical_modes = ['Cars', 'Regular Buses', 'Midi Buses', 
                'Heavy Freight Trucks', 'Medium Freight Trucks', 
                'Light Commercial Vehicles']
simple_modes = ["Trains", "High Speed Trains", "Freight Trains",
              "Passenger Planes", "Freight Planes",
              "Bikes",  "Small Ships",
              "Medium Ships", "Large Ships", "Very Large Ships",
              "Inland Ships"]
all_modes = typical_modes + simple_modes

cap_adjustment  = [1, 1, 1, 1]
mile_adjustment = [1, 1, 1, 1]

years_range = list(range(START_YEAR, END_YEAR + 1))

#%% BATTERY VEHICLE CALCULATIONS - Determine the fraction of the fleet that uses batteries, based on vehicle share files
# Batteries are relevant for 1) BUSES 2) TRUCKS
# We use fixed weight & material content assumptions, but we use the development of battery energy density 
# (from the electricity storage calculations) to derive a changing battery capacity (and thus range)
# Battery weight is assumed to be in-addition to the regular vehicle weight

bus_label_ICE    = ['BusOil',	'BusBio',	'BusGas']
bus_label_HEV    = ['BusElecTrolley',	'Bus Hybrid1']
# truck_label_extended = ['Conv. ICE(2000)',	'Conv. ICE(2010)', 'Adv. ICEOil',	'Adv. ICEH2', 'Turbo-petrol IC', \
#                    'Diesel ICEOil', 'Diesel ICEBio', 'ICE-HEV-gasoline', 'ICE-HEV-diesel oil', 'ICE-HEV-H2', \
#                    'ICE-HEV-CNG Gas', 'ICE-HEV-diesel bio', 'FCV Oil', 'FCV Bio', 'FCV H2', 'PEV-10 OilElec.', \
#                    'PEV-30 OilElec.', 'PEV-60 OilElec.', 'PEV-10 BioElec.', 'PEV-30 BioElec.', 'PEV-60 BioElec.', \
#                    'BEV Elec.', 'BEV Elec.', 'BEV Elec.', 'BEV Elec.']
truck_label_ICE  = ['Conv. ICE(2000)',	'Conv. ICE(2010)', 'Adv. ICEOil', 'Adv. ICEH2', 'Turbo-petrol IC',
                    'Diesel ICEOil', 'Diesel ICEBio']
truck_label_HEV  = ['ICE-HEV-gasoline', 'ICE-HEV-diesel oil', 'ICE-HEV-H2', 'ICE-HEV-CNG Gas', 'ICE-HEV-diesel bio']
truck_label_PHEV = ['PEV-10 OilElec.', 'PEV-30 OilElec.', 'PEV-60 OilElec.', 'PEV-10 BioElec.', 'PEV-30 BioElec.',
                    'PEV-60 BioElec.']
truck_label_BEV  = ['BEV Elec.', 'BEV Elec.', 'BEV Elec.']
truck_label_FCV  = ['FCV Oil', 'FCV Bio', 'FCV H2']
vshares_label    = ['ICE', 'HEV', 'PHEV', 'BEV', 'FCV', 'Trolley']

# --- Paths

# Scenario settings
SCEN = "SSP2"
# CP or 2D (Add "_RE" for Resource Efficiency)
VARIANT = "2D_RE"
PROJECT = "mock_project"
FOLDER = SCEN + "_" + VARIANT
OUTPUT_FOLDER = base_dir.joinpath("..", "..", "output", PROJECT, FOLDER)

maintenance_lifetime_per_mode = {
    'Cars': 14,
    'Heavy Freight Trucks': 8,
    'High Speed Trains': 30,
    'Light Commercial Vehicles': 14,
    'Medium Freight Trucks': 8,
    'Midi Buses': 13,
    'Regular Buses': 13,
    'Trains': 35,
    'Buses': 13,
}

all_types = ['Bikes', 'Cars', 'Cars - BEV', 'Cars - FCV', 'Cars - HEV',
       'Cars - ICE', 'Cars - PHEV', 'Cars - Trolley', 'Freight Planes',
       'Freight Trains', 'Heavy Freight Trucks',
       'Heavy Freight Trucks - BEV', 'Heavy Freight Trucks - FCV',
       'Heavy Freight Trucks - HEV', 'Heavy Freight Trucks - ICE',
       'Heavy Freight Trucks - PHEV', 'Heavy Freight Trucks - Trolley',
       'High Speed Trains', 'Inland Ships', 'Large Ships',
       'Light Commercial Vehicles', 'Light Commercial Vehicles - BEV',
       'Light Commercial Vehicles - FCV',
       'Light Commercial Vehicles - HEV',
       'Light Commercial Vehicles - ICE',
       'Light Commercial Vehicles - PHEV',
       'Light Commercial Vehicles - Trolley', 'Medium Freight Trucks',
       'Medium Freight Trucks - BEV', 'Medium Freight Trucks - FCV',
       'Medium Freight Trucks - HEV', 'Medium Freight Trucks - ICE',
       'Medium Freight Trucks - PHEV', 'Medium Freight Trucks - Trolley',
       'Medium Ships', 'Midi Buses', 'Midi Buses - BEV',
       'Midi Buses - FCV', 'Midi Buses - HEV', 'Midi Buses - ICE',
       'Midi Buses - PHEV', 'Midi Buses - Trolley', 'Passenger Planes',
       'Regular Buses', 'Regular Buses - BEV', 'Regular Buses - FCV',
       'Regular Buses - HEV', 'Regular Buses - ICE',
       'Regular Buses - PHEV', 'Regular Buses - Trolley', 'Small Ships',
       'Trains', 'Very Large Ships']