from os import PRIO_PROCESS
from sphinx.augmentation import Builder
import shutil

try:
    shutil.rmtree('tests/output/images')
    shutil.rmtree('tests/output/masks')
    shutil.rmtree('tests/output/annotations')
except:
    pass

pipeline = Builder(config_json="tests/test_config.json")
pipeline.calculate_and_set_generator_params(batch_size=2)
gen = pipeline.process_and_generate()

i = 0
total_entities = 0

while True:
    try:
        res = next(gen)
        total_entities += len(res)
    except StopIteration:
        break

print(total_entities)

