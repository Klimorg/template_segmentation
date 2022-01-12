from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class VggDataset(BaseModel):
    """[summary]

    Args:
        BaseModel ([type]): [description]
    """


class VggShapeAttributesSection(VggDataset):
    name: str
    all_points_x: List[float]
    all_points_y: List[float]


class VggRegionAttributesSection(VggDataset):
    label: str


class VggRegionsSection(VggDataset):
    shape_attributes: VggShapeAttributesSection
    region_attributes: VggRegionAttributesSection


class VggAnnotations(VggDataset):
    fileref: str = ""
    size: int
    filename: str
    base64_img_data: str
    file_attributes: Dict[str, Any]
    regions: Dict[str, VggRegionsSection]
