# -*- coding: utf-8 -*-
import os
import pytest
from sphinx.augmentation import Builder
import shutil
import cv2


class TestBuilder:
    @pytest.fixture(autouse=True)
    def builder_teardown(self, request):
        def teardown():
            shutil.rmtree('tests/output/images')
            shutil.rmtree('tests/output/masks')
            shutil.rmtree('tests/output/annotations')

        request.addfinalizer(teardown)

    def test_build_run(self):
        builder = Builder("tests/builder_test_config.json")
        builder.process_and_save()
        assert len(os.listdir('tests/output/images/')) == builder.sample
        assert len(os.listdir('tests/output/masks')) == builder.sample

    def test_save_annotation_run(self):
        builder = Builder("tests/save_annotation_test_config.json")
        builder.process_and_save()
        assert len(os.listdir('tests/output/images/')) == builder.sample
        assert len(os.listdir('tests/output/masks')) == builder.sample
        assert len(os.listdir('tests/output/annotation_mask/')
                   ) == builder.sample
        shutil.rmtree('tests/output/annotation_mask')

    def test_sequential_sampling_run(self):
        builder = Builder(
            config_json="tests/sequential_sampling_test_config.json")
        builder.calculate_and_set_generator_params(batch_size=2)
        gen = builder.process_and_generate()

        imgs = sorted(os.listdir("tests/images/"))
        i = 0

        while True:
            try:
                res = next(gen)
                for item in res:
                    assert item.name == imgs[i % len(imgs)][:-4]
                    i += 1

            except StopIteration:
                break

    @pytest.mark.parametrize(
        "task",
        ["classification", "detection", "segmentation"],
        ids=["classification", "detection", "segmentation"]
    )
    @pytest.mark.parametrize(
        "batch_size",
        [2, 3, 4],
        ids=lambda batch_size: f'batch_size: {batch_size}'
    )
    def test_keras_generator(self, task, batch_size):
        builder = Builder(
            config_json="tests/keras_generator_test_config.json")
        builder.calculate_and_set_generator_params(
            batch_size=batch_size, internal_batch=batch_size)
        gen = builder.get_keras_generator(task=task)

        idx = 0
        n_batch = len(gen)
        i = 0
        imgs = sorted(os.listdir("tests/images/"))

        if task == "segmentation":
            masks = sorted(os.listdir("tests/masks/"))

        while idx < n_batch:
            # try:
            x_batch, y_batch = gen[idx]
            assert len(x_batch) == len(y_batch) == min(
                builder.sample - i, batch_size)

            if task == "segmentation":
                for x, y in zip(x_batch, y_batch):
                    assert x.shape == cv2.imread(os.path.join(
                        "tests", "images", imgs[i % len(imgs)])).shape
                    assert y.shape == cv2.imread(os.path.join(
                        "tests", "masks", masks[i % len(imgs)]), 0).shape
                    i += 1
            else:
                for x in x_batch:
                    assert x.shape == cv2.imread(os.path.join(
                        "tests", "images", imgs[i % len(imgs)])).shape
                    i += 1
            idx += 1
            # except StopIteration:
            #     break

        assert idx == n_batch
