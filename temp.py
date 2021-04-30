from sphinx.augmentation import Builder
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def equalizeHist(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_rgb

def apply_augmentation(image, mask, label, function):
    image = np.array(image, dtype=np.uint8)
    mask = np.array(mask, dtype=np.uint8)
    augmented_segment = function(image)
    image[mask == label] = augmented_segment[mask == label]
    return image

pipeline = Builder(config_json="tests/test_config.json")
xmlobject = ET.parse("tests/annotation/0.xml")
image = cv2.imread("tests/images/0.png")
annotation_mask = pipeline._generate_mask_for_annotation(
                    xmlobject)

aug_image = apply_augmentation(
    image,
    annotation_mask,
    1,
    equalizeHist
)

print(pipeline.class_dictionary)
print(pipeline.classes)
aug_image = cv2.resize(aug_image, (1280,720))
annotation_mask = cv2.resize(annotation_mask, (1280,720))
annotation_mask = annotation_mask * 255
annotation_mask[annotation_mask == 255*2] = 100
cv2.imwrite("demo_aug.png", aug_image)
cv2.imwrite("demo_mask.png", annotation_mask)