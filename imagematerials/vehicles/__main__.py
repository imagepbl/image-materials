from preprocessing import preprocessing
from postprocessing import postprocess
from vemamodeling import run_model
from visualization import visualize_data


prep_data = preprocessing()
raw_data = run_model(prep_data)
processed_data = postprocess(raw_data)
visualize_data(processed_data)
