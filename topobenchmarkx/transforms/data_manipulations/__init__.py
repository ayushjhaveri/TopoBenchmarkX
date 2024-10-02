"""Data manipulations module."""

from .calculate_simplicial_curvature import (
    CalculateSimplicialCurvature,
)
from .equal_gaus_features import EqualGausFeatures
from .identity_transform import IdentityTransform
from .infere_knn_connectivity import InfereKNNConnectivity
from .infere_radius_connectivity import InfereRadiusConnectivity
from .keep_only_connected_component import KeepOnlyConnectedComponent
from .keep_selected_data_fields import KeepSelectedDataFields
from .node_degrees import NodeDegrees
from .node_features_to_float import NodeFeaturesToFloat
from .one_hot_degree_features import OneHotDegreeFeatures
from .attention2graph import Attention2Graph

DATA_MANIPULATIONS = {
    "Identity": IdentityTransform,
    "InfereKNNConnectivity": InfereKNNConnectivity,
    "InfereRadiusConnectivity": InfereRadiusConnectivity,
    "NodeDegrees": NodeDegrees,
    "OneHotDegreeFeatures": OneHotDegreeFeatures,
    "EqualGausFeatures": EqualGausFeatures,
    "NodeFeaturesToFloat": NodeFeaturesToFloat,
    "CalculateSimplicialCurvature": CalculateSimplicialCurvature,
    "KeepOnlyConnectedComponent": KeepOnlyConnectedComponent,
    "KeepSelectedDataFields": KeepSelectedDataFields,
    "Attention2Graph": Attention2Graph,
}

__all__ = [
    "IdentityTransform",
    "InfereKNNConnectivity",
    "InfereRadiusConnectivity",
    "EqualGausFeatures",
    "NodeFeaturesToFloat",
    "NodeDegrees",
    "KeepOnlyConnectedComponent",
    "CalculateSimplicialCurvature",
    "OneHotDegreeFeatures",
    "KeepSelectedDataFields",
    "Attention2Graph",
    "DATA_MANIPULATIONS",
]
