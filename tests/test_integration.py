# from pathlib import Path

# import numpy as np
# import xarray as xr
# from pytest import fixture, mark

# from imagematerials.__main__ import export_summary_netcdf, simulate_stocks
# from imagematerials.util import export_to_netcdf, import_from_netcdf
# import imagematerials.vehicles as vhc
# import imagematerials.buildings as bld

# DATA_DIR = Path("data", "raw")
# COMPARE_SUMMARY_FP = Path("tests", "data", "data_sums.nc")

# @fixture(scope="module")
# def summary_fp(tmpdir_factory):
#     new_summary_fp = tmpdir_factory.mktemp("data").join("data_sums.nc")
#     prep_fp = tmpdir_factory.mktemp("data").join("prep_data.nc")
#     _, orig_prep_data = preprocessing(DATA_DIR)
#     export_to_netcdf(orig_prep_data, prep_fp)
#     prep_data = import_from_netcdf(prep_fp)
#     model = simulate_stocks(prep_data)
#     export_summary_netcdf(model, new_summary_fp)
#     return new_summary_fp

# def all_summary_names():
#     array = xr.open_dataarray(COMPARE_SUMMARY_FP, group="summary").load()
#     return array.attrs["summary_names"]

# @mark.parametrize(
#     "key",
#     all_summary_names()
# )
# def test_integration(summary_fp, key):
#     new_summary = xr.open_dataarray(summary_fp, group=key).load()
#     old_summary = xr.open_dataarray(COMPARE_SUMMARY_FP, group=key).load()
#     assert len(new_summary) >= 1
#     if not new_summary.equals(old_summary):
#         assert new_summary.shape == old_summary.shape, "Dimensions are different or have changed."
#         coor = key.split("-")[-1]
#         wrong_idx = np.where(new_summary.values == old_summary.values)[0]
#         if len(wrong_idx) == len(old_summary):
#             assert False, "All values in this summary array have changed/are wrong."
#         assert False, f"Values are wrong for: {coor.iloc[wrong_idx].values}"
