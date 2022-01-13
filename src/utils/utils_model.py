import json

import arrow
import orjson
from loguru import logger

from src.utils.data_models import (
    CocoAnnotations,
    CocoAnnotationsSection,
    CocoCategoriesSection,
    CocoImagesSection,
    CocoInfoSection,
    VggAnnotations,
)


def vgg2coco(vgg_json_model_path, coco_json_model_path):

    vgg_dataset = VggAnnotations.parse_file(vgg_json_model_path)
    images = sorted(vgg_dataset)

    info = CocoInfoSection(
        description=f"Converted from vgg format {vgg_json_model_path} to coco format.",
    )

    logger.info("Creating Coco categories section.")
    categories = []
    categories_section = []
    for img in images:
        regions = list(vgg_dataset[img].regions)
        for region in regions:
            categories.append(vgg_dataset[img].regions[region].region_attributes.label)

    categories = sorted(set(categories))

    for idx in range(len(categories)):
        categories_section.append(
            CocoCategoriesSection(id=idx + 1, name=categories[idx]),
        )

    logger.info("Creating Coco images section.")
    logger.info("Creating Coco annotations section.")

    images_section = []
    id_img = 0

    annotations_section = []
    id_annotation = 0

    for img in images:
        images_section.append(
            CocoImagesSection(
                id=id_img + 1,
                width=None,
                height=None,
                file_name=images[id_img],
            ),
        )

        all_regions = list(vgg_dataset[img].regions)
        for region in all_regions:

            all_x = vgg_dataset[img].regions[region].shape_attributes.all_points_x[:-1]
            all_y = vgg_dataset[img].regions[region].shape_attributes.all_points_y[:-1]

            # https://stackoverflow.com/a/6356099
            annotations = [item for pair in zip(all_x, all_y) for item in pair]

            max_x, min_x = max(all_x), min(all_x)
            max_y, min_y = max(all_y), min(all_y)

            width = max_x - min_x
            height = max_y - min_y
            bbox = [min_x, min_y, width, height]
            area = width * height

            label = vgg_dataset[img].regions[region].region_attributes.label

            category_id = [
                category.id for category in categories_section if category.name == label
            ]
            annotations_section.append(
                CocoAnnotationsSection(
                    id=id_annotation,
                    image_id=id_img + 1,
                    category_id=int(category_id[0]),
                    segmentation=[annotations],
                    bbox=bbox,
                    area=area,
                ),
            )
            id_annotation += 1
        id_img += 1

    coco_dataset = CocoAnnotations(
        info=info,
        images=images_section,
        annotations=annotations_section,
        categories=categories_section,
    )

    path = f"report_{arrow.now()}.json"

    logger.info("Conversion done, now dumping.")
    with open(path, "w") as outfile:
        json.dump(orjson.loads(coco_dataset.json()), outfile)


def coco2vgg():
    pass


if __name__ == "__main__":

    from src.utils.utils import get_items_list

    json_files = get_items_list(
        directory="datas/raw_datas/ML/labels/",
        extension=".json",
    )

    vgg2coco(json_files[0], "toto")
