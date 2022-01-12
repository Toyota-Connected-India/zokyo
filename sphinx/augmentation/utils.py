# Contributors : [srinivas.v@toyotaconnected.co.in, ]
import numpy as np
import os


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
