from sphinx.augmentation.utils import annotations_to_mask
import xml.etree.ElementTree as ET
import numpy as np
import cv2

tree = ET.parse("tests/annotation/0.xml")
root = tree.getroot()
class_dictionary = {}
classes = 1
class_dictionary["background"] = 0
class_bnd_box = {}
class_bnd_box["classes"] = {}
class_bnd_box["size"] = {}
for child in root:
    if child.tag == "size":
        for size in child:
            class_bnd_box["size"][size.tag] = int(size.text)
    if child.tag == "object":
        current_tag = ""
        for elem in child:
            bnd_dict = {}
            if elem.tag == "name":
                if elem.text not in class_dictionary.keys():
                    class_dictionary[elem.text] = classes
                    classes += 1
                if elem.text not in class_bnd_box["classes"].keys():
                    class_bnd_box["classes"][elem.text] = []
                current_tag = elem.text
            if elem.tag == "bndbox":
                for coord in elem:
                    bnd_dict[coord.tag] = int(coord.text)
                class_bnd_box["classes"][current_tag].append(bnd_dict)

print(class_dictionary)
print(class_bnd_box)

annotation_mask = np.zeros((class_bnd_box["size"]["height"], class_bnd_box["size"]["width"], class_bnd_box["size"]["depth"]), dtype=np.uint8)
for cat in class_bnd_box["classes"].keys():
    for bnd in class_bnd_box["classes"][cat]:
        color = (class_dictionary[cat], class_dictionary[cat], class_dictionary[cat])
        annotation_mask = cv2.rectangle(annotation_mask, (bnd["xmin"],bnd["ymin"]), (bnd["xmax"],bnd["ymax"]), color, -1)

print(annotation_mask.shape)
cv2.imwrite("annotation_mask.jpg",annotation_mask)
cv2.waitKey(0)



# class DataStructure:
#     num : int
#     string : str

# v = DataStructure()
# v.num = 10
# v.string = "asdf"


# print(v.num)