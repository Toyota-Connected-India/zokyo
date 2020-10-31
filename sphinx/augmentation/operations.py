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
        Operation.__init__(self, args)
        
        if args.coefficient is None:
            raise CrucialValueNotFoundError("DarkenScene", sample_type="Coefficient")
        
        if (args.darkness_coeff != -1):
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