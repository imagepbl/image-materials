from preprocessing import preprocess
from postprocessing import postprocess
from vemamodeling import run_model
from visualization import visualize_data


if __name__ == "__main":
    prep_data = preprocess()
    raw_data = run_model(prep_data)
    processed_data = postprocess(raw_data)
    visualize_data(processed_data)
