from sphinx.augmentation import Builder
from PIL import Image

pipeline = Builder(config_json="tests/test_config.json")
pipeline.calculate_and_set_generator_params(batch_size=1)
gen = pipeline.process_and_generate()

print(pipeline.class_dictionary)
print(pipeline.classes)

i = 0

while True:
    try:
        res = next(gen)
        for item in res:
            item.image.save("samples/{}.png".format(i))
            i+=1
    except StopIteration:
        break

