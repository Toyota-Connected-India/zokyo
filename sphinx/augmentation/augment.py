# -*- coding: utf-8 -*-
# Contributors : [ashok.ramadass@toyotaconnected.com, srinivas.v@toyotaconnected.co.in, ]

import Augmentor
import cv2
from Augmentor.Operations import Operation
import os
from ..utils.CustomExceptions import CoefficientNotinRangeError
from PIL import Image

class DarkenScene(Operation):
    def __init__(self, probability, darkness_coeff=-1):
        Operation.__init__(self, probability)
        if(darkness_coeff!=-1):
            if(darkness_coeff<0.0 or darkness_coeff>1.0):
                raise CoefficientNotinRangeError(darkness_coeff, "DarknessCoefficient", 0, 1)
        self.darkness_coeff = darkness_coeff
    
    def __str__(self):
        return self.__class__.__name__

    def perform_operation(self,images):
        def do(image):
            pass
        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        return augmented_images
            

