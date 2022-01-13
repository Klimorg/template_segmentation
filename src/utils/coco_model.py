from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class CocoDataset(BaseModel):
    pass


class CocoInfoSection(CocoDataset):
    description: str


class CocoLicencesSection(CocoDataset):
    licenses: Optional[Dict[str, Any]]


class CocoImagesSection(CocoDataset):
    id: int
    width: int
    height: int
    file_name: str


class CocoAnnotationsSection(CocoDataset):
    id: int
    iscrowd: Optional[int]
    image_id: int
    category_id: int
    segmentation: List[List[float]]
    bbox: List[float]
    area: Optional[float]


class CocoCategoriesSection(CocoDataset):
    id: int
    name: str


class CocoSegmentInfoSection(CocoDataset):
    pass


class CocoAnnotations(CocoDataset):
    info: CocoInfoSection
    licenses: Optional[CocoLicencesSection]
    images: List[CocoImagesSection]
    annotations: List[CocoAnnotationsSection]
    categories: List[CocoCategoriesSection]
    segment_info: Optional[List[CocoSegmentInfoSection]]
