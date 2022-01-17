import json
from pathlib import Path

import arrow
import orjson
import typer
from loguru import logger

from src.utils.pydantic_data_models import (
    CocoAnnotations,
    CocoAnnotationsSection,
    CocoCategoriesSection,
    CocoImagesSection,
    CocoInfoSection,
    VggAnnotations,
    VggFileAttributes,
    VggRegionAttributesSection,
    VggRegionsSection,
    VggShapeAttributesSection,
    VggStructure,
)

app = typer.Typer()


@app.command()
def vgg2coco(
    vgg_json_model_path: Path,
    coco_json_model_path: Path,
    timestamp: bool = False,
) -> None:

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
                width=vgg_dataset[img].file_attributes.width,
                height=vgg_dataset[img].file_attributes.height,
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

    if timestamp:
        path = f"{coco_json_model_path}_{arrow.now()}.json"
    else:
        path = f"{coco_json_model_path}.json"

    logger.info("Conversion done, now dumping.")
    with open(path, "w") as outfile:
        json.dump(orjson.loads(coco_dataset.json()), outfile)


@app.command()
def coco2vgg(
    coco_json_model_path: Path,
    vgg_json_model_path: Path,
    timestamp: bool = False,
) -> None:

    coco_dataset = CocoAnnotations.parse_file(coco_json_model_path)

    categories = {
        idx + 1: coco_dataset.categories[idx].name
        for idx in range(len(coco_dataset.categories))
    }

    logger.info("Retreiving all image names.")
    image_names = {
        idx + 1: coco_dataset.images[idx].file_name
        for idx in range(len(coco_dataset.images))
    }

    logger.info("Creating Vgg annotations section.")
    vgg_annotations = {}
    for image_id, image_name in image_names.items():

        height = coco_dataset.images[image_id - 1].height
        width = coco_dataset.images[image_id - 1].width

        file_attributes = VggFileAttributes(height=height, width=width)

        annotations = [
            coco_dataset.annotations[idx]
            for idx in range(len(coco_dataset.annotations))
            if coco_dataset.annotations[idx].image_id == image_id
        ]

        id_annotation = 0
        regions = {}
        for annotation in annotations:
            even = [
                idx for idx in range(len(annotation.segmentation[0])) if idx % 2 == 0
            ]
            odd = [
                idx for idx in range(len(annotation.segmentation[0])) if idx % 2 == 1
            ]

            all_points_x = [annotation.segmentation[0][idx] for idx in even] + [
                annotation.segmentation[0][even[0]],
            ]
            all_points_y = [annotation.segmentation[0][idx] for idx in odd] + [
                annotation.segmentation[0][odd[0]],
            ]
            label = {"label": categories[annotation.category_id]}

            shape_attribute = VggShapeAttributesSection(
                all_points_x=all_points_x,
                all_points_y=all_points_y,
            )
            region_attribute = VggRegionAttributesSection(**label)

            region_section = VggRegionsSection(
                shape_attributes=shape_attribute,
                region_attributes=region_attribute,
            )
            regions[str(id_annotation)] = region_section
            id_annotation += 1

        vgg_structure = VggStructure(
            filename=image_name,
            file_attributes=file_attributes,
            regions=regions,
        )
        vgg_annotations[image_name] = vgg_structure

    vgg_dataset = VggAnnotations.parse_obj(vgg_annotations)

    if timestamp:
        path = f"{vgg_json_model_path}_{arrow.now()}.json"
    else:
        path = f"{vgg_json_model_path}.json"

    logger.info("Conversion done, now dumping.")
    with open(path, "w") as outfile:
        json.dump(orjson.loads(vgg_dataset.json()), outfile)


if __name__ == "__main__":
    app()
