from sphinx.augmentation import Builder
pipeline = Builder(config_json="tests/radial_lens_distortion_test_config.json")
pipeline.process_and_save()