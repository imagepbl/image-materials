utils.py
Contains functions used for the preprocessing.

constants.py
contains important constants used

Electricity_sector_new
Sebastiaans ELMA code

Electricity_sector_new_JT
Same, just formatted a bit cleaner and some minor changes to accomodate package changes (not supported functionality)

Electricity_sector_restructured_oldstockmodeling
Same, but restructured: separated generation and storage; clearer devision between read in of files, preprocessing, modelling and postprocessing.

preprocessing.py
Separation of preprocessing and main modelling. Currently, main modelling is still happening within this file, as I first want to make it run and then make a function out of preprocessing that I can call in electricity.py. Like this it is easier to access the variables and understand what is going on.
Generation is running (but not properly checked yet).
Decision need to be made, if first make one preprocessing function within one script (how it was in the vehicle preprocessing until now) or directly use the modular structure that we decided on (Split preprocessing into different files: materials, timer-dependent, utils, interpolate, lifetimes, shares

electricity.py
Script in which the stock modelling should take place. Currently it still contains part of storage preprocessing, as the process of splitting this up is not finished yet nor does it contain the actual stock modelling, as the preprocessing function is not created yet.

grid_materials.py
Sebastiaans code to model the electricity transmission grid infrastructure. I have not looked at this yet.

graphs_elec.py
Sebastiaans code: visualizations of the electricity module, figures used for his publication.

dynamic_stock_model.py
Stock model of Stefan Pauliuk (https://github.com/stefanpauliuk/dynamic_stock_model) which was used in Sebastiaans code. Not needed anymore, once transitioned to new stock modelling approach.


