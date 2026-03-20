import prism


REGION = prism.Dimension("Region")
STOCK_TYPE = prism.Dimension("Type")
COHORT = prism.Dimension("Cohort")
TIME = prism.Dimension("Time")
MATERIAL_TYPE = prism.Dimension("material")


@prism.interface
class TIMERVehicleMaterials(prism.Model):
    """
    """

    # Dimensions - TODO: deal with them differently?
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    Cohort: prism.Coords[COHORT]
    Time: prism.Coords[TIME]

    # Exported variables
    inflow_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()
    outflow_by_cohort_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()
    
    def compute_initial_values(self, timeline: prism.Timeline):
        # Handle historic tail
        # And preprocessing?
        pass

    def compute_values(
        self,
        time: prism.Time,
        passengerkms: prism.Array[Region, Type, "km"],  # TODO: check unit
        tonkms: prism.Array[Region, Type, "Tkm"]  # TODO: check unit
    ):
        pass
        # Interact with Stock model and Materials model from here
