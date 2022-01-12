from typing import Any, Dict, List

from pydantic import BaseModel

from src.utils.utils import get_items_list


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


class VggStructure(VggDataset):
    fileref: str = ""
    size: int
    filename: str
    base64_img_data: str
    file_attributes: Dict[str, Any]
    regions: Dict[str, VggRegionsSection]


class VggAnnotations(VggDataset):
    __root__: Dict[str, VggStructure]

    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, item):
        return self.__root__[item]


if __name__ == "__main__":

    json_files = get_items_list(
        directory="datas/raw_datas/ML/labels/",
        extension=".json",
    )[0]

    print(f"{json_files}")

    vgg_dataset = VggAnnotations.parse_file(json_files)

    print(f"{vgg_dataset['m200_31972.jpeg']}")
