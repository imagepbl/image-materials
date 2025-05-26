# -*- coding: utf-8 -*-
"""
Created on Fri May 23 15:30:34 2025

@author: MvEng
"""

""""Global constants for the TRIPI model."""
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
FIRST_YEAR = 1911  # first_year_vehicle.values.min()
END_YEAR = 2070
# year of output generation
OUT_YEAR = 2070
REGIONS = 26

years_range = list(range(START_YEAR, END_YEAR + 1))


# lables of original IMAGE input files, Capital Letters are used 1-on-1 lower case modes are either ignored or disaggregated
tkms_label  = ['regions','Inland ships', 'Cargo Trains', 'medium truck', 'heavy truck', 'Cargo Planes', 'Ships', 'empty', 'total']
pkms_label  = ['regions','walking', 'Bikes', 'Buses', 'Trains', 'Cars', 'HST', 'Planes', 'total']


labels_simple  = ['Passenger Planes', 'Bikes', 'Freight Planes','Freight Trains', 
                  'High Speed Trains', 'Inland Ships', 'Large Ships', 'Medium Ships', 
                  'Small Ships', 'Trains', 'Very Large Ships']
labels_typical = ['Cars', 'Light Commercial Vehicles', 'Medium Freight Trucks',
                  'Heavy Freight Trucks', 'Midi Buses', 'Regular Buses']


# Define the units for each dimension
unit_mapping = {
    'time': ureg.year,
    'year': ureg.year,
    'kg': ureg.kilogram,
    'yr': ureg.year,
    '%': ureg.percent,
    't': ureg.tonne,}

# --- Paths

# Scenario settings
SCEN = "SSP2"
# CP or 2D (Add "_RE" for Resource Efficiency)
VARIANT = "2D_RE"
PROJECT = "mock_project"
FOLDER = SCEN + "_" + VARIANT
OUTPUT_FOLDER = base_dir.joinpath("..", "..", "output", PROJECT, FOLDER)