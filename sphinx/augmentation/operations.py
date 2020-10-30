# -*- coding: utf-8 -*-
# Contributors : [srinivas.v@toyotaconnected.co.in, ]

import Augmentor
import cv2
from Augmentor.Operations import Operation
import os
from ..utils.CustomExceptions import CoefficientNotinRangeError, InvalidImageArrayError
from PIL import Image, ImageOps
import numpy as np
import warnings

class ColorEqualize(Operation):
    def __init__(self, probability):
        Operation.__init__(self, probability)

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
    def __init__(self, probability, darkness_coeff=-1):
        Operation.__init__(self, probability)
        if (darkness_coeff != -1):
            if (darkness_coeff < 0.0 or darkness_coeff > 1.0):
                raise CoefficientNotinRangeError(darkness_coeff, "DarknessCoefficient", 0, 1)
        self.darkness_coeff = darkness_coeff

    def perform_operation(self,images):

        def do(image):
            image_array = np.array(image).astype('uint8')
            #TODO : To implement Darken Road scene
            return PIL.Image.fromarray(image_array)

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        return augmented_images