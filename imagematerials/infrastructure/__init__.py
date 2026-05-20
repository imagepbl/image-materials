"""Subpackage of image-materials containing the infrastructure sector."""
from imagematerials.infrastructure.preprocessing.main import (
    get_preprocessing_data_infrastructure,
)
from imagematerials.infrastructure.preprocessing.main import (
    get_preprocessing_data_infrastructure as preprocess,
)

__all__ = ["preprocess", "get_preprocessing_data_infrastructure"]
