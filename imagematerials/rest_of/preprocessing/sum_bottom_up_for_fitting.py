def export_material_inflows(model, output_dir="data/raw/rest-of", include_maintenance=True):
    """Aggregate sector material inflows and export one CSV per material."""
    from pathlib import Path

    from imagematerials.rest_of.const import REGION_TO_CLASS_DICT_IMAGE_MAT

    cement_in_concrete_factor = 0.12
    sand_in_concrete = cement_in_concrete_factor * 0.17
    sand_in_glass_conversion = 0.7
    materials = ["steel", "aluminium", "copper", "cement", "sand_gravel_crushed_rock"]
    # Material categorization for output directory structure
    material_categories = {
        "steel": "metals",
        "aluminium": "metals",
        "copper": "metals",
        "cement": "nmm",
        "sand_gravel_crushed_rock": "nmm",
    }
    sector_names = [
        "buildings", "elc_gen", "elc_grid_lines", "elc_grid_add",
        "elc_stor_phs", "elc_stor_other", "vehicles",
    ]
    image_code_to_region_name = {
        "CAN": "Canada", "USA": "US", "MEX": "Mexico", "RCAM": "Rest C.Am.",
        "BRA": "Brazil", "RSAM": "Rest S.Am.", "NAF": "N.Africa", "WAF": "W.Africa",
        "EAF": "E.Africa", "RSAF": "South Africa", "WEU": "W.Europe", "CEU": "C.Europe",
        "TUR": "Turkey", "UKR": "Ukraine", "STAN": "Stan", "RUS": "Russia",
        "ME": "M.East", "INDIA": "India", "KOR": "Korea", "CHN": "China",
        "SEAS": "SE.Asia", "INDO": "Indonesia", "JAP": "Japan", "OCE": "Oceania",
        "RSAS": "Rest S.Asia", "SAF": "Rest S.Africa",
    }
    class_order = [f"class_ {index}" for index in range(1, 27)]
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    def get_material_total(data, material_name):
        dims_to_sum = [
            dimension for dimension in ["Type", "Quintile", "Quintiles"]
            if dimension in data.dims
        ]
        return data.sum(dims_to_sum).pint.to("Mt").sel(material=material_name)

    def get_derived_sand_total(data):
        derived_total = None
        for source_material, conversion_factor in [
            ("concrete", sand_in_concrete),
            ("glass", sand_in_glass_conversion),
        ]:
            try:
                converted_total = get_material_total(data, source_material) * conversion_factor
                derived_total = converted_total if derived_total is None else derived_total + converted_total
            except (KeyError, ValueError):
                continue
        return derived_total

    sources = []
    for sector_name in sector_names:
        try:
            sector = getattr(model, sector_name)
            sources.append((sector_name, sector.get("inflow_materials").to_array()))
        except (AttributeError, KeyError, ValueError):
            continue

    if include_maintenance:
        try:
            sources.append(("vehicle maintenance", model.vehicles.get("inflow_maintenance").to_array()))
        except (AttributeError, KeyError, ValueError):
            pass

    aggregated = {}
    for material in materials:
        source_material = "concrete" if material == "cement" else material
        totals = []
        for source_name, data in sources:
            try:
                total = (
                    get_derived_sand_total(data)
                    if material == "sand_gravel_crushed_rock"
                    else get_material_total(data, source_material)
                )
                if total is not None:
                    totals.append(total)
            except (KeyError, ValueError):
                continue

        if not totals:
            continue

        total = sum(totals)
        if material == "cement":
            total = (total * cement_in_concrete_factor).assign_coords(material="cement")
        elif material == "sand_gravel_crushed_rock":
            total = total.assign_coords(material=material)
        aggregated[material] = total

        export_total = total.pint.to("t").rename({"Region": "class_"})
        export_total = export_total.assign_coords(
            class_=[
                f"class_ {REGION_TO_CLASS_DICT_IMAGE_MAT[region_name]}"
                if region in image_code_to_region_name
                else str(region)
                for region in export_total.coords["class_"].values
                for region_name in [image_code_to_region_name.get(region, region)]
            ]
        )
        dataframe = export_total.to_dataframe(name=material).reset_index()
        pivot = dataframe.pivot(index="time", columns="class_", values=material)
        pivot = pivot.reindex(columns=[column for column in class_order if column in pivot.columns])
        pivot.index.name = "time"
        
        # Determine output subdirectory based on material category
        material_category = material_categories.get(material, "other")
        output_dir = output_base / material_category
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pivot.to_csv(output_dir / f"image_materials_{material}.csv")

    return aggregated