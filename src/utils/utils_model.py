from src.utils.data_models import (
    CocoAnnotations,
    CocoImagesSection,
    CocoInfoSection,
    VggAnnotations,
)


def vgg2coco(vgg_json_model_path, coco_json_model_path):

    vgg_dataset = VggAnnotations.parse_file(vgg_json_model_path)

    info = CocoInfoSection(
        description=f"Converted from vgg format {vgg_json_model_path} to coco format.",
    )

    images = sorted(vgg_dataset)
    images_section = []
    for idx in range(len(images)):
        images_section.append(
            CocoImagesSection(
                id=idx + 1,
                width=None,
                height=None,
                file_name=images[idx],
            ),
        )


def coco2vgg():
    pass
