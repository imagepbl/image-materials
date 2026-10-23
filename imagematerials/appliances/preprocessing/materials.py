# DUMMY DATA!!
# THIS IS MATERIAL INTENSITIES

# Material intensities for appliances are taken from IMAGE, and are in kg/appliance

# DUMMY VALUES: total mass per appliance (kg) times a rough material composition
# share, loosely based on typical WEEE / appliance teardown compositions (steel
# TODO: replace with real IMAGE material intensity data.

import xarray as xr
import prism

def define_material_intensities(count_appliances):

    materials = ["steel", "plastic", "copper", "aluminium", "glass"]

    total_mass_kg = {
        "Fan": 2, "Air Cooler": 15, "Air Conditioner": 45, "Refrigerator": 60,
        "Microwave": 15, "Washing Machine": 65, "Clothes Dryer": 35,
        "Dish Washer": 40, "TV": 15, "VCR/DVD": 2, "PC/Other": 5,
    }

    material_shares = {
        #                  steel plastic copper aluminium glass
        "Fan":             (0.35, 0.45, 0.10, 0.05, 0.05),
        "Air Cooler":      (0.40, 0.40, 0.05, 0.10, 0.05),
        "Air Conditioner": (0.45, 0.20, 0.10, 0.25, 0.00),
        "Refrigerator":    (0.60, 0.25, 0.05, 0.05, 0.05),
        "Microwave":       (0.45, 0.35, 0.10, 0.05, 0.05),
        "Washing Machine": (0.65, 0.20, 0.08, 0.05, 0.02),
        "Clothes Dryer":   (0.65, 0.20, 0.08, 0.05, 0.02),
        "Dish Washer":     (0.55, 0.30, 0.05, 0.05, 0.05),
        "TV":              (0.15, 0.55, 0.05, 0.05, 0.20),
        "VCR/DVD":         (0.15, 0.70, 0.10, 0.05, 0.00),
        "PC/Other":        (0.30, 0.50, 0.10, 0.10, 0.00),
    }

    appliance_types = list(count_appliances.Type.values)
    material_intensity_per_appliance = xr.DataArray(
        [[total_mass_kg[a] * share for share in material_shares[a]] for a in appliance_types],
        dims=["Type", "material"],
        coords={"Type": appliance_types, "material": materials},
    )
    material_intensity_per_appliance = prism.Q_(material_intensity_per_appliance, "kg/count")

    # broadcast the (dummy, currently Region- and Cohort-independent) material
    # intensities over all regions and cohorts present in count_appliances.
    # The time-like dim for material intensities is "Cohort" (build year of the
    # appliance), not "time", matching the convention in e.g.
    # buildings/preprocessing/materials.py and vehicles/preprocessing/materials.py.
    material_intensity_per_appliance = material_intensity_per_appliance.broadcast_like(
        count_appliances
    ).rename({"time": "Cohort"})

    return material_intensity_per_appliance