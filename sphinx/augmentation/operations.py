# -*- coding: utf-8 -*-
# Contributors : [srinivas.v@toyotaconnected.co.in, ]

import Augmentor
import cv2
from Augmentor.Operations import Operation
import os
from ..utils.CustomExceptions import CoefficientNotinRangeError, InvalidImageArrayError, CrucialValueNotFoundError
from PIL import Image, ImageOps
import numpy as np
import warnings


class ArgsClass(object):
    def __init__(self, **kwargs):
        if "probability" not in kwargs.keys():
            kwargs["probability"] = 1
        self.__dict__.update((key, kwargs[key]) for key in kwargs)


class ColorEqualize(Operation):
    def __init__(self, **kwargs):
        args = ArgsClass(**kwargs)
        Operation.__init__(self, args.probability)

    def perform_operation(self, images):
        def do(image):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ImageOps.equalize(image)

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class DarkenScene(Operation):
    def __init__(self, **kwargs):
        args = ArgsClass(**kwargs)
        Operation.__init__(self, args.probability)

        if args.coefficient is None:
            raise CrucialValueNotFoundError(
                "DarkenScene", sample_type="coefficient")

        if (args.coefficient != -1):
            if (args.coefficient < 0.0 or args.coefficient > 1.0):
                raise CoefficientNotinRangeError(
                    args.coefficient, "DarknessCoefficient", 0, 1)
        self.darkness_coeff = args.coefficient

    def perform_operation(self, images):

        def do(image):
            image = np.array(image, dtype=np.uint8)
            image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            image_HLS[:, :, 1] = image_HLS[:, :, 1] * self.darkness_coeff
            image_HLS[:, :, 1][image_HLS[:, :, 1] < 0] = 0
            image_HLS = np.array(image_HLS, dtype=np.uint8)
            image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
            return Image.fromarray(image_RGB)

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        return augmented_images
