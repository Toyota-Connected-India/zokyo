from sphinx.augmentation import Builder
import shutil

try:
    shutil.rmtree('tests/output/images')
    shutil.rmtree('tests/output/masks')
    shutil.rmtree('tests/output/annotations')
except:
    pass

pipeline = Builder(config_json="tests/sun_flare_test_config.json")
pipeline.process_and_save()