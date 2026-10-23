import xarray as xr
import prism

# ESTIMATIONS: will need to be replaced by real data!!

# TODO: should these be regionalized and take efficiency improvements into account? 
# For now, we just use a global average (dummy) for all regions and years.
energy_use_one_fan = 0.8
energy_use_one_air_cooler = 1.5  # energy use of one air cooler per year
energy_use_one_ac = 6  # energy use of one AC per year

energy_use_per_appliance = xr.DataArray(
    [energy_use_one_fan, energy_use_one_air_cooler, energy_use_one_ac],
    dims="Appliances",
    coords={"Appliances": ["Fan", "Air Cooler", "Air Conditioner"]},
)
energy_use_per_appliance = prism.Q_(energy_use_per_appliance, "GJ/count")  # energy use per appliance per year