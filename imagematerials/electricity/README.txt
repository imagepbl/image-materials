# Most relevant ones ----------------------------------------------

utils.py
Contains functions used for the preprocessing.

constants.py
contains important constants used

preprocessing.py
preprocessing functions - one for each electricity sub-sector. Currently implemented: generation

script_test.py
(was previously - before 29.07 - named preprocessing.py)
Separation of preprocessing and main modelling. Currently, main modelling is still happening within this file, as I first want to make it run and then make a function out of preprocessing that I can call in electricity.py. Like this it is easier to access the variables and understand what is going on.
Generation & Grid are running and checked.

electricity.py
Script in which the stock modelling should take place. Currently it still contains part of storage preprocessing, as the process of splitting this up is not finished yet nor does it contain the actual stock modelling, as the preprocessing function is not created yet.


# Old ones -------------------------------------------

Electricity_sector_new.py (in electricity/archive)
Sebastiaans ELMA code

Electricity_sector_new_JT.py (in electricity/archive)
Same, just formatted a bit cleaner and some minor changes to accomodate package changes (not supported functionality)

Electricity_sector_restructured.py
Same, but restructured: separated generation and storage; clearer devision between read in of files, preprocessing, modelling and postprocessing.

grid_materials.py (in electricity/archive)
Sebastiaans code to model the electricity transmission grid infrastructure.

grid_materials_JT.py
Same, but a bit of formatting done.

graphs_elec.py (in electricity/archive)
Sebastiaans code: visualizations of the electricity module, figures used for his publication.

dynamic_stock_model.py
Stock model of Stefan Pauliuk (https://github.com/stefanpauliuk/dynamic_stock_model) which was used in Sebastiaans code. Not needed anymore, once transitioned to new stock modelling approach.


