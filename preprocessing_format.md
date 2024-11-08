# File format as stock modeling input

This document describes how to create the netCDF4 file that can be used to run the stock modeling.

The netCDF4 file needs two groups: "main" and "lifetimes". 

# main group

## total_nr_vehicles -> item_count

This contains the total number of vehicles/items. Its dimensions are:

- "time"
- "mode" -> "type", for subtypes the numbers will be nan.
- "region".

## vehicle_shares -> item_shares

This dataset is to be able create the totals for all vehicle subtypes. Its dimensions are:

- "cohort"
- "mode" -> "type", for supertypes the numbers will be nan.
- "region"

# lifetimes group

The lifetimes are defined using Scipy distributions. For each mode/type there should be one distribution type associated with it. Scipy defines their distributions by three parameters (generally): `c`, `loc` and `scale`. The parameters therefore should be converted to these three before storing it in the netCDF4 file.

## weibull

This data array is for all modes/types that use the Weibull distribution.

## folded_norm

This data array is for all modes/types that use the folded Normal distribution.

## General format

Each data array should have the following dimensions:

- "time"
- "mode" -> "type", nans when the type doesn't use this distribution. Only the modes/types of the super types are available, subtypes are not included here, but they can be (partially) included if the data is present.
- "scipy_param", for all scipy parameters (out of `c`, `loc` and `scale`) that are not constant

Additionally, for all scipy parameters that are the same always, set the attribute of the data array with the same name to that value.
