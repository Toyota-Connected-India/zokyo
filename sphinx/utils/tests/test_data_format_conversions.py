import pytest  # noqa: F401
from pathlib import Path
import os
import json
from sphinx.utils import data_format_conversions

VOC = "tests/annotation"
COCO_STD = "tests/annotation_coco_std/coco_ann.json"
COCO_TOYO = "tests/annotation_coco_toyo"


class TestDataConv:

    def test_coco_std_2_voc(self):

        data_format_conversions.coco_std_2_voc(COCO_STD, save_folder=VOC)

        with open(COCO_STD, 'r') as f:
            ann = json.load(f)
            coco_classes = ann["categories"]

        assert len(ann["images"]) == len(list(Path(VOC).glob('*.xml')))

        with open(str(Path(VOC) / "voc.names"), 'r') as f:
            voc_classes = f.readlines()

        assert len(coco_classes) == len(voc_classes)

        for v_cl, c_cl in zip(voc_classes, coco_classes):
            assert v_cl.rstrip('\n') == c_cl["name"]

    def test_voc_2_coco_std(self):

        data_format_conversions.voc_2_coco_std(
            VOC, save_folder=str(Path(COCO_STD).parent))

        with open(COCO_STD, 'r') as f:
            ann = json.load(f)
            coco_classes = ann["categories"]

        assert len(ann["images"]) == len(list(Path(VOC).glob('*.xml')))

        with open(str(Path(VOC) / "voc.names"), 'r') as f:
            voc_classes = f.readlines()

        assert len(coco_classes) == len(voc_classes)

        for v_cl, c_cl in zip(voc_classes, coco_classes):
            assert v_cl.rstrip('\n') == c_cl["name"]

    def test_coco_std_2_coco_toyo(self):

        data_format_conversions.coco_std_2_coco_toyo(
            COCO_STD, save_folder=COCO_TOYO)

        with open(COCO_STD, 'r') as f:
            ann = json.load(f)
            coco_std_classes = ann["categories"]

        assert len(ann["images"]) == len(os.listdir(COCO_TOYO))

        with open(str(list(Path(COCO_TOYO).glob('*.json'))[0])) as f:
            coco_toyo_classes = json.load(f)["categories"]

        assert coco_std_classes == coco_toyo_classes

    def test_coco_toyo_2_coco_std(self):

        data_format_conversions.coco_toyo_2_coco_std(
            COCO_TOYO, save_folder=str(Path(COCO_STD).parent))

        with open(COCO_STD, 'r') as f:
            ann = json.load(f)
            coco_std_classes = ann["categories"]

        assert len(ann["images"]) == len(os.listdir(COCO_TOYO))

        with open(str(list(Path(COCO_TOYO).glob('*.json'))[0])) as f:
            coco_toyo_classes = json.load(f)["categories"]

        assert coco_std_classes == coco_toyo_classes

    def test_coco_toyo_2_voc(self):

        data_format_conversions.coco_toyo_2_voc(
            COCO_TOYO, save_folder=VOC)

        assert len(list(Path(VOC).glob('*.xml'))
                   ) == len(list(Path(COCO_TOYO).glob('*.json')))

        with open(str(list(Path(COCO_TOYO).glob('*.json'))[0])) as f:
            coco_classes = json.load(f)["categories"]

        with open(str(Path(VOC) / "voc.names"), 'r') as f:
            voc_classes = f.readlines()

        assert len(coco_classes) == len(voc_classes)

        for v_cl, c_cl in zip(voc_classes, coco_classes):
            assert v_cl.rstrip('\n') == c_cl["name"]

    def test_voc_2_coco_toyo(self):

        data_format_conversions.voc_2_coco_toyo(
            VOC, save_folder=COCO_TOYO)

        assert len(list(Path(VOC).glob('*.xml'))
                   ) == len(list(Path(COCO_TOYO).glob('*.json')))

        with open(str(list(Path(COCO_TOYO).glob('*.json'))[0])) as f:
            coco_classes = json.load(f)["categories"]

        with open(str(Path(VOC) / "voc.names"), 'r') as f:
            voc_classes = f.readlines()

        assert len(coco_classes) == len(voc_classes)

        for v_cl, c_cl in zip(voc_classes, coco_classes):
            assert v_cl.rstrip('\n') == c_cl["name"]
