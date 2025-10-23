# sanitize attrs before saving: convert any non-serializable objects (e.g. pint Unit) to strings
def sanitize_attrs_xarray(obj):
    import numbers
    import numpy as np

    # Helper to check simple allowed types
    def is_allowed(v):
        return isinstance(v, (str, bytes, numbers.Number, np.ndarray, list, tuple, bool, type(None)))

    # If Dataset
    if hasattr(obj, 'data_vars'):
        # global attrs
        for k, v in dict(obj.attrs).items():
            try:
                obj.attrs[k] = v if isinstance(v, (str, bytes)) else str(v)
            except Exception:
                obj.attrs[k] = str(v)
        # each variable attrs
        for var in obj.data_vars:
            for ak, av in dict(obj[var].attrs).items():
                try:
                    obj[var].attrs[ak] = av if is_allowed(av) or isinstance(av, (str, bytes)) else str(av)
                except Exception:
                    obj[var].attrs[ak] = str(av)
        return obj

    # If DataArray
    if hasattr(obj, 'attrs'):
        for ak, av in dict(obj.attrs).items():
            try:
                obj.attrs[ak] = av if is_allowed(av) or isinstance(av, (str, bytes)) else str(av)
            except Exception:
                obj.attrs[ak] = str(av)
    return obj



def compute_inflow(dict_materials, material_name):
       inflow_material = dict_materials.get(material_name.lower())
       region_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                 '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
                 '24', '25', '26']
       inflow_material = inflow_material.assign_coords(Region=[str(r) for r in region_list])
       return inflow_material
