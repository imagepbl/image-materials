# Most relevant ones ----------------------------------------------

utils.py
Contains functions used for the preprocessing.

constants.py
contains important constants used

preprocessing.py
preprocessing functions - one for each electricity sub-sector. Currently implemented: generation, grid and storage

electricity.py
Script in which the stock modelling should take place. Imports preprocessed data by calling the functions from preprocessing.py

examples/electricity.ipynb
Same as electricity.py, but as a jupyter notebook

script_test.py
(was previously - before 29.07 - named preprocessing.py)
For testing purposes. Preprocessing and stock modelling in one file, preprocessing not inside a function so that variables are accessible
and can be checked more easily.


# Other ----------------------------------------------

electr_external_data.py
Contains data from the IEA and stores them as dictionaries. Data should be moved to an external file and creating the 
dictionary to a function maybe?

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


