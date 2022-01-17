# Contributors : [srinivas.v@toyotaconnected.co.in, ]
import numpy as np
import os
import cv2


def apply_augmentation(image, mask, label, function):
    """
        Function to apply augmentation operation to a certain labels only
    """

    image = np.array(image, dtype=np.uint8)
    mask = np.array(mask, dtype=np.uint8)
    augmented_segment = function(image)
    if len(np.squeeze(mask).shape) == 2:
        image[mask == label] = augmented_segment[mask == label]
    else:
        image[mask[:, :, label] == 1,
              :] = augmented_segment[mask[:, :, label] == 1, :]
    return image


def change_pascal_annotation(annotation, image_dir, filename):
    """
        Function add path infos to pascal annotation
    """

    root = annotation.getroot()
    for child in root:
        if child.tag == "folder":
            child.text = image_dir
        if child.tag == "filename":
            child.text = filename
        if child.tag == "path":
            child.text = os.path.join(image_dir, filename)
    return annotation


def get_annotation_dictionary(annotation):
    """
        Method to parse XML annotation and return a dict with class names as keys
        and their corresponding bounding boxes as values.
    """

    root = annotation.getroot()
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
                    if elem.text not in class_bnd_box["classes"].keys():
                        class_bnd_box["classes"][elem.text] = []
                    current_tag = elem.text
                if elem.tag == "bndbox":
                    for coord in elem:
                        bnd_dict[coord.tag] = int(coord.text)
                    class_bnd_box["classes"][current_tag].append(bnd_dict)
    return class_bnd_box


def generate_mask_for_annotation_for_xml(annotation, num_classes, label_id):
    """
        Method to generate class-wise binary mask from the bounding boxes of each class  (including BG)
    """

    current_image_class_data_dict = get_annotation_dictionary(annotation)
    annotation_mask = np.zeros(
        (current_image_class_data_dict["size"]["height"],
            current_image_class_data_dict["size"]["width"],
            num_classes),
        dtype=np.uint8)
    ann_bg = np.ones(
        (current_image_class_data_dict["size"]["height"],
            current_image_class_data_dict["size"]["width"]),
        dtype=np.uint8)
    for cat in current_image_class_data_dict["classes"].keys():
        ann_cl = np.zeros(
            (current_image_class_data_dict["size"]["height"],
                current_image_class_data_dict["size"]["width"]),
            dtype=np.uint8)
        for bnd in current_image_class_data_dict["classes"][cat]:
            if cat != "background":
                ann_cl = cv2.rectangle(
                    ann_cl,
                    (bnd["xmin"], bnd["ymin"]),
                    (bnd["xmax"], bnd["ymax"]), 1, -1)
                ann_bg = cv2.rectangle(
                    ann_bg,
                    (bnd["xmin"], bnd["ymin"]),
                    (bnd["xmax"], bnd["ymax"]), 0, -1)
        annotation_mask[:, :, label_id] = ann_cl
    annotation_mask[:, :, 0] = ann_bg
    return annotation_mask
