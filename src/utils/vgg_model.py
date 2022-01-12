from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class VggShapeAttributesSection(BaseModel):
    name: str
    all_points_x: List[float]
    all_points_y: List[float]


class VggRegionAttributesSection(BaseModel):
    label: str


class VggRegionsSection(BaseModel):
    shape_attributes: VggShapeAttributesSection
    region_attributes: VggRegionAttributesSection


class VggDataset(BaseModel):
    fileref: str = ""
    size: int
    filename: str
    base64_img_data: str
    file_attributes: Dict[str, Any]
    regions: Dict[str, VggRegionsSection]
