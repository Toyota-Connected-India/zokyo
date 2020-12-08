from sphinx.augmentation import Builder
pipeline = Builder(config_json="tests/brighten_scene_test_config.json")
pipeline.process_and_save()