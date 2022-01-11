from typing import Any, Dict, List

JsonDict = Dict[str, Any]
PolygonVertices = List[float]


class LabelizationError(ValueError):
    """[summary]

    Args:
        ValueError ([type]): [description]
    """


class PolygonError(LabelizationError):
    """[summary]

    Args:
        LabelizationError ([type]): [description]
    """


def validate_polygons(
    X_coordinates: PolygonVertices,
    Y_coordinates: PolygonVertices,
    labels: List[int],
):
    """[summary]

    Args:
        X_coordinates (PolygonVertices): [description]
        Y_coordinates (PolygonVertices): [description]
        labels (List[int]): [description]

    Raises:
        PolygonError: [description]
        LabelizationError: [description]
    """

    if len(X_coordinates) - len(Y_coordinates) != 0:
        raise PolygonError(len(X_coordinates), len(Y_coordinates))

    elif len(Y_coordinates) - len(labels) != 0:
        raise LabelizationError(len(X_coordinates), len(Y_coordinates))
