""""Global constants for the VEMA model."""

# --- Settings & constants

# start year of historic IMAGE data
START_YEAR = 1971
# start year of the full model period (including stock-development from
# scratch, which needs to be the oldest year of any vehicle, all stock
# calculations are initiated in this year, so this has an effect on
# runtime)
FIRST_YEAR = START_YEAR  # first_year_vehicle.values.min()
END_YEAR = 2060
# year of output generation
OUT_YEAR = 2060
REGIONS = 26
FIRST_YEAR_BOATS = 1900
# reference loadfactor of cars in TIMER (the trp_trvl_Load.out file is
# relative to this BASE loadfcator (persons/car))
LOAD_FACTOR = 1.6

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
tkms_label = ["inland shipping", "freight train", "medium truck",
              "heavy truck", "air cargo", "international shipping", "empty",
              "total"]
pkms_label = ["walking", "biking", "bus", "train", "car", "hst", "air",
              "total"]
columns_vehicle_output = ["Buses", "Trains", "HST", "Cars", "Planes", "Bikes",
                          "Trucks", "Cargo Trains", "Ships", "Inland ships",
                          "Cargo Planes"]

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

# --- Paths

# Scenario settings
SCEN = "SSP2"
# CP or 2D (Add "_RE" for Resource Efficiency)
VARIANT = "2D_RE"
PROJECT = "mock_project"
FOLDER = SCEN + "_" + VARIANT
# TODO: deprecate this constant
OUTPUT_FOLDER = "../../output/" + PROJECT + "/" + FOLDER
