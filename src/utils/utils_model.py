from src.utils.data_models import CocoAnnotations, VggAnnotations


def vgg2coco(vgg_json_model_path, coco_json_model_path):

    vgg_dataset = VggAnnotations.parse_file(vgg_json_model_path)


def coco2vgg():
    pass
