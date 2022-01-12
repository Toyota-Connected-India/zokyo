import pytest  # noqa: F401
from pathlib import Path
import os
import json
import shutil
import cv2
import numpy as np
from zokyo.utils import semantic_seg

out_dir = "tests/semantic_seg/semantic_map"


@pytest.mark.parametrize(
    "out_dir,gt,color_map,fill_poly",
    [
        pytest.param(out_dir, True, [i for i in range(20, 255, 10)], True),
        pytest.param(out_dir, False, np.random.randint(
            0, 255, size=(16, 3)), False),
        pytest.param(None, True, None, True)
    ],
    ids=[
        "output dir-gt-grey color-fill poly",
        "output dir-no gt-RGB-no fill poly",
        "No output dir-gt-No color-fill poly"
    ]
)
class TestSemantic:

    @pytest.fixture(autouse=True)
    def builder_teardown(self, request):

        os.makedirs(out_dir, exist_ok=True)

        def teardown():
            shutil.rmtree(out_dir, ignore_errors=True)

        request.addfinalizer(teardown)

    def test_semantic_map(self, out_dir, gt, color_map, fill_poly):

        img_path = "tests/semantic_seg/frame/Negley_Black_1_1.jpg"
        ann_path = "tests/semantic_seg/annotation/negley_black_1st_1.json"

        res = semantic_seg.generate_semantic_map(
            img_path, ann_path, out_dir=out_dir, gt=gt, color_map=color_map, fill_poly=fill_poly, verbose=False)

        img = cv2.imread(img_path)

        img_path = Path(img_path)

        if gt:
            sem_seg, mask = res
            assert img.shape == sem_seg.shape
            assert img.shape[:2] == mask.shape

            with open(ann_path, 'r') as anns_json:
                anns = json.load(anns_json)
            n_classes = len(anns["categories"])

            assert n_classes == np.max(mask) + 1

            if out_dir:
                out_dir = Path(out_dir)
                assert Path(str(out_dir / img_path.stem) + '.jpg').is_file()
                assert Path(str(out_dir / img_path.stem) + '_gt.npy').is_file()

        else:
            assert img.shape == res.shape

            if out_dir:
                out_dir = Path(out_dir)
                assert Path(str(out_dir / img_path.stem) + '.jpg').is_file()
