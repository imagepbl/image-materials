"""Quick test for infrastructure preprocessing."""
from pathlib import Path
import traceback

from imagematerials.infrastructure.preprocessing import get_preprocessing_data_infrastructure

path_base = Path(r"C:\IMAGE\image-materials")
print("Running preprocessing...")
try:
    res = get_preprocessing_data_infrastructure(path_base, "SSP2")
    print("SUCCESS! Keys:", list(res.keys()))
    print()
    stocks = res["stocks"]
    print("Stocks shape:", stocks.shape)
    print("Stocks dims:", stocks.dims)
    print("Stocks units:", stocks.attrs.get("units"))
    print()
    mi = res["material_intensities"]
    print("MI shape:", mi.shape)
    print("MI dims:", mi.dims)
    print("MI units:", mi.attrs.get("units"))
    print("MI material coords:", list(mi.coords["material"].values))
    first_reg = mi.coords["Region"].values[0]
    print(f"Sample MI (urban_paved_motorway, asphalt, {first_reg}, 2020):",
          float(mi.sel(Type="urban_paved_motorway", material="asphalt",
                       Region=first_reg, Cohort=2020)))
    print()
    lt = res["lifetimes"]
    print("Lifetimes shape:", lt.shape)
    print("Lifetimes dims:", lt.dims)
    print("Lifetimes attrs:", lt.attrs)
    for t in lt.coords["Type"].values[:8]:
        c_val = float(lt.sel(ScipyParam="c", Type=t))
        s_val = float(lt.sel(ScipyParam="scale", Type=t))
        print(f"  {t}: shape(c)={c_val}, scale={s_val}")
except Exception as e:
    traceback.print_exc()
