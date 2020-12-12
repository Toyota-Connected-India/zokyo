from sphinx.augmentation import Builder
pipeline = Builder(config_json="tests/sun_flare_test_config.json")
pipeline.process_and_save()