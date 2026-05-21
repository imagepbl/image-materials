import prism

from imagematerials.constants import START_YEAR_HISTORIC
from imagematerials.vehicles.stocks_prism import VehicleStocks


REGION = prism.Dimension("Region")
STOCK_TYPE = prism.Dimension("Type")
COHORT = prism.Dimension("Cohort")
TIME = prism.Dimension("Time")
MATERIAL_TYPE = prism.Dimension("material")


@prism.interface
class TIMERMaterials(prism.Model):
    """
    """

    # Dimensions - TODO: deal with them differently?
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    #Cohort: prism.Coords[COHORT]
    Time: prism.Coords[TIME]

    # Exported variables
    #inflow_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()
    #outflow_by_cohort_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()
    
    def compute_initial_values(self, timeline: prism.Timeline):
        # Handle historic tail
        self.historic_tail_computed = False

        self.complete_timeline = prism.Timeline(START_YEAR_HISTORIC, timeline.end, timeline.stepsize)

    def init_submodels(self, timeline: prism.Timeline):
        vehicles = VehicleStocks(timeline)
        self.submodels = [vehicles]


    def compute_values(
        self,
        time: prism.Time#,
        #passengerkms: prism.Array[Region, Type, "km"],  # TODO: check unit
        #tonkms: prism.Array[Region, Type, "Tkm"]  # TODO: check unit
    ):
        if not self.historic_tail_computed:
        # TODO: make a cache for the historic tail calculation
            for historic_time in self.complete_timeline:
                prism_time = prism.Time(self.complete_timeline.start,
                                        self.complete_timeline.end,
                                        self.complete_timeline.stepsize, historic_time)
                if historic_time >= time.t:
                    break
                self._compute_one_timestep(prism_time)
            self.historic_tail_computed = True
        self._compute_one_timestep(time)
        
    def _compute_one_timestep(self, time: prism.Time):
        print(f"{time.t}", end="\r")
        for model in self.submodels:
            model.compute_values(time, **model.compute_args)

        # Interact with Stock model and Materials model from here
