from pathlib import Path
from typing import Any, Dict, List

JsonDict = Dict[str, Any]
PolygonVertices = List[float]


class LabelizationError(ValueError):
    """[summary]

    Args:
        ValueError ([type]): [description]

    Raises:
        PolygonError: [description]
        LabelizationError: [description]
        EmptyLabelizationFilesError: [description]
    """


class PolygonError(LabelizationError):
    """[summary]

    Args:
        LabelizationError ([type]): [description]
    """


class EmptyLabelizationFilesError(LabelizationError):
    """[summary]

    Args:
        LabelizationError ([type]): [description]
    """


class StructureError(ValueError):
    """[summary]

    Args:
        ValueError ([type]): [description]

    Raises:
        PolygonError: [description]
        LabelizationError: [description]
        EmptyLabelizationFilesError: [description]
    """


class VGGStructureError(StructureError):
    """[summary]

    Args:
        StructureError ([type]): [description]
    """


class COCOStructureError(StructureError):
    """[summary]

    Args:
        StructureError ([type]): [description]
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


def validate_non_empty_vgg_files(item_list: List[Path]):

    if item_list:
        pass
    else:
        raise EmptyLabelizationFilesError(item_list)
