from enum import Enum, unique
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


@unique
class JsonFormat(Enum):
    vgg = "vgg"
    coco = "coco"


class CocoDataset(BaseModel):
    """[summary]

    Args:
        BaseModel ([type]): [description]

    Returns:
        [type]: [description]
    """


class CocoInfoSection(CocoDataset):
    description: str


class CocoLicencesSection(CocoDataset):
    licenses: Optional[List[Dict[str, Any]]]
    # licenses: Optional[List[Any]]


class CocoImagesSection(CocoDataset):
    id: int
    width: Optional[int]
    height: Optional[int]
    file_name: str


class CocoAnnotationsSection(CocoDataset):
    id: int
    iscrowd: Optional[int] = 0
    image_id: int
    category_id: int
    segmentation: List[List[float]]
    bbox: Optional[List[float]] = None
    area: Optional[float] = None


class CocoCategoriesSection(CocoDataset):
    id: int
    name: str


class CocoSegmentInfoSection(CocoDataset):
    pass


class CocoAnnotations(CocoDataset):
    info: CocoInfoSection
    licenses: Optional[CocoLicencesSection] = None
    images: List[CocoImagesSection]
    annotations: List[CocoAnnotationsSection]
    categories: List[CocoCategoriesSection]
    segment_info: Optional[List[CocoSegmentInfoSection]] = None


class VggDataset(BaseModel):
    """[summary]

    Args:
        BaseModel ([type]): [description]
    """


class VggShapeAttributesSection(VggDataset):
    name: Optional[str] = "polygon"
    all_points_x: List[float]
    all_points_y: List[float]


class VggRegionAttributesSection(VggDataset):
    label: str


class VggRegionsSection(VggDataset):
    shape_attributes: VggShapeAttributesSection
    region_attributes: VggRegionAttributesSection


class VggFileAttributes(VggDataset):
    height: Optional[int]
    width: Optional[int]


class VggStructure(VggDataset):
    fileref: str = ""
    size: int = 0
    filename: str
    base64_img_data: str = ""
    file_attributes: Optional[VggFileAttributes] = None
    regions: Dict[str, VggRegionsSection]


class VggAnnotations(VggDataset):
    __root__: Dict[str, VggStructure]

    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, item):
        return self.__root__[item]
