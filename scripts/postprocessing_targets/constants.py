
target_variables_narrow = {"Total pass. travel demand":"p-km/cap",      #Done
                    "Total freight. travel demand":"t-km/cap",          #Done
                    "passenger car":"Modal split - pkm/capita",
                    "passenger bus":"Modal split - pkm/capita",
                    "passenger aviation":"Modal split - pkm/capita",
                    "passenger walk":"Modal split - %",
                    "passenger bike":"Modal split - %",
                    "freight road heavy":"Modal split - tkm/capita",
                    "freight road light":"Modal split - tkm/capita",
                    "freight rail":"Modal split - tkm/capita",
                    "freight aviation":"Modal split - tkm/capita",
                    "Numb passengers per private vehicle": "Occupancy rate",
                    "Numb passengers per BUS": "Occupancy rate",
                    "Numb passengers per train": "Occupancy rate",
                    "Numb passengers per aviation passengers": "Occupancy rate",
	                "Ton per heavy freight": "Load rate",
                    "Ton per light freight": "Load rate",
                    "Ton per aviation freight": "Load rate",
                    "Ton per navigation freight": "Load rate"
}

# TODO Modal split, occupancy, load factor, weight

target_variables_slow = {
                    "car": "kg/unit",
                    "bus": "kg/unit",
                    "truck": "kg/unit",
                    "airplane": "kg/unit",
                    "rail": "kg/unit",
                    "ships": "kg/unit",
                    "bicycle": "kg/unit",
}

# Mapping from xr_image_output regions to target regions
region_mapping = {
    "IMAGE 3.4|Canada": "Canada",
    "IMAGE 3.4|Central Europe": "Central Europe",
    "IMAGE 3.4|China": "China",
    "IMAGE 3.4|India": "India",
    "IMAGE 3.4|Japan": "Japan",
    "IMAGE 3.4|USA": "USA",
    "IMAGE 3.4|Western Europe": "Western Europe",
    "Latin America (R10)": "Latin America",
    "Latin America (R5)": "Latin America",      # is that correct?
    "IMAGE 3.4|Middle East": "Middle East and Northern Africa",
    "IMAGE 3.4|Northern Africa": "Middle East and Northern Africa",
    "Rest of Asia (R10)": "Other Asia",
    "IMAGE 3.4|Indonesia": "Other Asia",
    "IMAGE 3.4|Korea": "Other OECD",
    "OECD & EU (R5)": "Other OECD",
    "Reforming Economies (R10)": "Reforming Economies",
    "Reforming Economies (R5)": "Reforming Economies",           # is that correct?
    "IMAGE 3.4|Rest of Southern Africa": "Subsaharan Africa",
    "IMAGE 3.4|Western Africa": "Subsaharan Africa",
    "IMAGE 3.4|Eastern Africa": "Subsaharan Africa"
}

# Define the mapping from detailed vehicle types to broad categories
vehicle_type_mapping = {
    "Bikes": "bicycle",
    "Cars": "car", 
    "Cars - BEV": "car", 
    "Cars - FCV": "car",
    "Cars - HEV": "car", 
    "Cars - ICE": "car", 
    "Cars - PHEV": "car", 
    "Cars - Trolley": "car", 
    "Freight Planes": "airplane", 
    "Freight Trains": "rail", 
    "Heavy Freight Trucks": "truck",
    "Heavy Freight Trucks - BEV": "truck", 
    "Heavy Freight Trucks - FCV": "truck",
    "Heavy Freight Trucks - HEV": "truck", 
    "Heavy Freight Trucks - ICE": "truck", 
    "Heavy Freight Trucks - PHEV": "truck", 
    "Heavy Freight Trucks - Trolley": "truck", 
    "High Speed Trains": "rail", 
    "Inland Ships": "ships", 
    "Large Ships": "ships", 
    "Light Commercial Vehicles": "truck",
    "Light Commercial Vehicles - BEV": "truck", 
    "Light Commercial Vehicles - FCV": "truck", 
    "Light Commercial Vehicles - HEV": "truck", 
    "Light Commercial Vehicles - ICE": "truck", 
    "Light Commercial Vehicles - PHEV": "truck", 
    "Light Commercial Vehicles - Trolley": "truck", 
    "Medium Freight Trucks": "truck", 
    "Medium Freight Trucks - BEV": "truck", 
    "Medium Freight Trucks - FCV": "truck", 
    "Medium Freight Trucks - HEV": "truck", 
    "Medium Freight Trucks - ICE": "truck", 
    "Medium Freight Trucks - PHEV": "truck", 
    "Medium Freight Trucks - Trolley": "truck", 
    "Medium Ships": "ships", 
    "Midi Buses": "bus", 
    "Midi Buses - BEV": "bus", 
    "Midi Buses - FCV": "bus", 
    "Midi Buses - HEV": "bus", 
    "Midi Buses - ICE": "bus", 
    "Midi Buses - PHEV": "bus", 
    "Midi Buses - Trolley": "bus", 
    "Passenger Planes": "airplane", 
    "Regular Buses": "bus", 
    "Regular Buses - BEV": "bus", 
    "Regular Buses - FCV": "bus", 
    "Regular Buses - HEV": "bus", 
    "Regular Buses - ICE": "bus", 
    "Regular Buses - PHEV": "bus", 
    "Regular Buses - Trolley": "bus", 
    "Small Ships": "ships", 
    "Trains": "rail", 
    "Very Large Ships": "ships"
}



# Define mapping from aggregated dataset variables to vehicle_targets variables
variable_mapping = {
    "Energy Service|Transportation|Freight": "Total freight. travel demand",
    "Energy Service|Transportation|Passenger": "Total pass. travel demand",
}

target_regions= ["Canada",
"Central Europe",
"China",
"India",
"Japan",
"USA",
"Western Europe",
"Latin America",
"Middle East and Northern Africa",
"Other Asia",
"Other OECD",
"Reforming Economies",
"Subsaharan Africa"]