from zokyo.augmentation import Builder
from PIL import Image
import os
import xml.etree.ElementTree as ET
import shutil
import cv2
try:
    shutil.rmtree('tests/output/images')
    shutil.rmtree('tests/output/masks')
    shutil.rmtree('tests/output/annotations')
    shutil.rmtree('tests/output/annotation_mask')
    shutil.rmtree('samples')
except:
    pass

pipeline = Builder(config_json="tests/test_config.json")

# root = ET.parse("tests/annotation/0.xml")
# ann_mask = pipeline._generate_mask_for_annotation(root)
# cv2.imshow("0", 255. *ann_mask[:, :, 0])
# cv2.imshow("1", 255. *ann_mask[:, :, 1])
# cv2.imshow("2", 255. *ann_mask[:, :, 2])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# pipeline.process_and_save()

pipeline.calculate_and_set_generator_params(batch_size=2, internal_batch=1)
gen = pipeline.get_keras_generator(batch_size=2, task="segmentation")

print(pipeline.class_dictionary)
print(pipeline.classes)

i = 0
imgs = sorted(os.listdir("tests/images/"))
# os.makedirs("samples", exist_ok=True)

while i<len(gen):
    x_batch, y_batch = gen[i]
    print(len(x_batch), len(y_batch))
    for x, y in zip(x_batch, y_batch):
        # print(imgs[i % len(imgs)][:-4], item.name)
        # item.image.save("samples/{}.png".format(i))
        print(x.shape, y.shape)
    i+=1

# from os import PRIO_PROCESS
# from zokyo.augmentation import Builder
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
# from zokyo.augmentation import Builder
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
