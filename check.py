from sphinx.augmentation import Builder
from PIL import Image
import os
import shutil
try:
    shutil.rmtree('tests/output/images')
    shutil.rmtree('tests/output/masks')
    shutil.rmtree('tests/output/annotations')
    shutil.rmtree('tests/output/annotation_mask')
    shutil.rmtree('samples')
except:
    pass

pipeline = Builder(config_json="tests/test_config.json")
# pipeline.process_and_save()
# pipeline.calculate_and_set_generator_params(batch_size=2)
# gen = pipeline.process_and_generate()

gen = pipeline.get_keras_generator(batch_size=2)

print(pipeline.class_dictionary)
print(pipeline.classes)

i = 0
imgs = sorted(os.listdir("tests/images/"))
os.makedirs("samples", exist_ok=True)

while True:
    try:
        # res = next(gen)
        res = gen.__getitem__(i)
        for item in res:
            # print(imgs[i % len(imgs)][:-4], item.name)
            # item.image.save("samples/{}.png".format(i))
            print(res[0].shape)
        i+=1
    except StopIteration:
        break

# from os import PRIO_PROCESS
# from sphinx.augmentation import Builder
# import shutil

# try:
#     shutil.rmtree('tests/output/images')
#     shutil.rmtree('tests/output/masks')
#     shutil.rmtree('tests/output/annotations')
# except:
#     pass

# pipeline = Builder(config_json="tests/test_config.json")
# pipeline.calculate_and_set_generator_params(batch_size=2)
# gen = pipeline.process_and_generate()

# i = 0
# total_entities = 0

# while True:
#     try:
#         res = next(gen)
#         for r in res:
#             r.image.save("samples/{}.png".format(i))
#             i += 1
#     except StopIteration:
#         break

# from os import PRIO_PROCESS
# from sphinx.augmentation import Builder
# import shutil
# try:
#     shutil.rmtree('tests/output/images')
#     shutil.rmtree('tests/output/masks')
#     shutil.rmtree('tests/output/annotations')
#     shutil.rmtree('tests/output/annotation_mask')
# except:
#     pass
# pipeline = Builder(config_json="tests/sun_flare_test_config.json")
# pipeline.process_and_save()
