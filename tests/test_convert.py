from pathlib import Path

import pytest

from src.utils.convert import coco2vgg, vgg2coco
from src.utils.pydantic_data_models import CocoAnnotations, VggAnnotations


@pytest.fixture
def vgg_json_file():
    return Path("tests/test_datas/vgg_test.json")


@pytest.fixture
def coco_json_file():
    return Path("tests/test_datas/coco_test.json")


def test_coco2vgg(tmp_path: Path, coco_json_file) -> None:

    vgg_json_file_path = tmp_path / Path("vgg_test")

    coco2vgg(
        coco_json_model_path=coco_json_file,
        vgg_json_model_path=vgg_json_file_path,
    )

    dataset = VggAnnotations.parse_file(f"{vgg_json_file_path}.json")
    assert isinstance(dataset, VggAnnotations)


def test_vgg2coco(tmp_path: Path, vgg_json_file) -> None:

    coco_json_file_path = tmp_path / Path("coco_test")

    vgg2coco(
        vgg_json_model_path=vgg_json_file,
        coco_json_model_path=coco_json_file_path,
    )

    dataset = CocoAnnotations.parse_file(f"{coco_json_file_path}.json")
    assert isinstance(dataset, CocoAnnotations)


def test_vgg2coco2vgg(tmp_path: Path, vgg_json_file) -> None:

    test_dir = tmp_path

    coco_json_file_path = test_dir / Path("coco_test")
    vgg_json_file_path = test_dir / Path("vgg_test")

    vgg2coco(
        vgg_json_model_path=vgg_json_file,
        coco_json_model_path=coco_json_file_path,
    )

    coco2vgg(
        coco_json_model_path=Path(f"{coco_json_file_path}.json"),
        vgg_json_model_path=vgg_json_file_path,
    )

    dataset = VggAnnotations.parse_file(f"{vgg_json_file_path}.json")
    assert isinstance(dataset, VggAnnotations)


def test_coco2vgg2coco(tmp_path: Path, coco_json_file) -> None:

    test_dir = tmp_path

    coco_json_file_path = test_dir / Path("coco_test")
    vgg_json_file_path = test_dir / Path("vgg_test")

    coco2vgg(
        coco_json_model_path=coco_json_file,
        vgg_json_model_path=vgg_json_file_path,
    )

    vgg2coco(
        vgg_json_model_path=Path(f"{vgg_json_file_path}.json"),
        coco_json_model_path=coco_json_file_path,
    )

    dataset = CocoAnnotations.parse_file(f"{coco_json_file_path}.json")
    assert isinstance(dataset, CocoAnnotations)
