Electricity_sector_new: 
Sebastiaans ELMA code

Electricity_sector_new_JT: 
Same, just formatted a bit cleaner and some minor changes to accomodate package changes (not supported functionality)

Electricity_sector_restructured_oldstockmodeling: 
Same, but restructured: separated generation and storage; clearer devision between read in of files, preprocessing, modelling and postprocessing.

preprocessing.py:
Separation of preprocessing and main modelling. Currently, main modelling is still happening within this file, as I first want to make it run and then make a function out of preprocessing that I can call in electricity.py. Like this it is easier to access the variables and understand what is going on.
Generation is running (but not properly checked yet).
Decision need to be made, if first make one preprocessing function within one script (how it was in the vehicle preprocessing until now) or directly use the modular structure that we decided on (Split preprocessing into different files: materials, timer-dependent, utils, interpolate, lifetimes, shares
