import scipy


class WeibullDistribution():
    """Weibull Distribution with parameters shape and scale"""
    name = "weibull"
    method = scipy.stats.weibull_min.sf
    params = ["shape", "scale"]
    variable_scipy_param = ["c", "scale"]

    @staticmethod
    def get_param(param_dict):
        """Get the parameters into the right format for Scipy."""
        return {
            "c": param_dict["shape"],
            "scale": param_dict["scale"],
            "loc": 0
        }

    @staticmethod
    def has_param(param):
        """Used to check whether the parameters are compatible with the Weibull dist."""
        if "shape" in param and "scale" in param:
            return True
        return False


class FoldedNormalDistribution():
    """Folded Normal distribution with parameters mean and stdev"""
    name = "folded_norm"
    method = scipy.stats.foldnorm.sf
    params = ["mean", "stdev"]
    variable_scipy_param = ["c", "scale"]

    @staticmethod
    def get_param(param_dict):
        """Get the parameters into the right format for Scipy."""
        mean, stdev = param_dict["mean"], param_dict["stdev"]
        return {"c": mean/stdev, "scale": stdev, "loc": 0}

    @staticmethod
    def has_param(param):
        """Used to check whether the parameters are compatible with the fold_norm dist."""
        if "mean" in param and "stdev" in param:
            return True
        return False


ALL_DISTRIBUTIONS = [WeibullDistribution, FoldedNormalDistribution]
NAME_TO_DIST = {dist.name: dist for dist in ALL_DISTRIBUTIONS}

