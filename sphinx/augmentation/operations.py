# -*- coding: utf-8 -*-
# Contributors : [srinivas.v@toyotaconnected.co.in, ashok.ramadass@toyotaconnected.com, ]

import Augmentor
import cv2
from Augmentor.Operations import Operation
import os

from numpy.lib import histograms
from ..utils.CustomExceptions import CoefficientNotinRangeError, InvalidImageArrayError, CrucialValueNotFoundError
from ..utils.misc import from_float, to_float
from PIL import Image, ImageOps
import numpy as np
import random
import warnings

class ArgsClass(object):
    def __init__(self, **kwargs):
        if "probability" not in kwargs.keys():
            kwargs["probability"] = 1
        
        if "is_mask" not in kwargs.keys():
            kwargs["is_mask"] = False

        if kwargs["is_mask"] == True and "label" not in kwargs:
            kwargs["label"] = None

        self.__dict__.update((key, kwargs[key]) for key in kwargs)


class EqualizeScene(Operation):
    def __init__(self, **kwargs):
        self.args = ArgsClass(**kwargs)
        Operation.__init__(self, self.args.probability)

    def perform_operation(self, images):
        def do(images):
            if self.args.is_mask == True:
                if self.args.label == None:
                    return [ImageOps.equalize(image) for image in images]
                else:
                    image = np.array(images[0], dtype=np.uint8)
                    image_mask = np.array(images[1], dtype=np.uint8)
                    augmented_segment = cv2.equalizeHist(image)
                    image[image_mask == self.args.label] = augmented_segment[image_mask == self.args.label]
                    return [Image.fromarray(image), Image.fromarray(image_mask)]
            else:
                if len(images) == 2:
                    return [ImageOps.equalize(images[0]), images[1]]
                else:
                    return [ImageOps.equalize(images[0])]
        
        return do(images)
            


class DarkenScene(Operation):
    def __init__(self, **kwargs):
        self.args = ArgsClass(**kwargs)
        Operation.__init__(self, self.args.probability)

        if 'darkness' not in self.args.__dict__.keys():
            raise CrucialValueNotFoundError(
                "DarkenScene", value_type="darkness")

        if (self.args.darkness != -1):
            if (self.args.darkness < 0.0 or self.args.darkness > 1.0):
                raise CoefficientNotinRangeError(
                    self.args.darkness, "darkness", 0, 1)

        self.darkness_coeff = 1 - self.args.darkness

    def perform_operation(self, images):
        def darken(image):
            image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            image_HLS[:, :, 1] = image_HLS[:, :, 1] * self.darkness_coeff
            image_HLS[:, :, 1][image_HLS[:, :, 1] < 0] = 0
            image_HLS = np.array(image_HLS, dtype=np.uint8)
            image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
            return image_RGB

        def do(images):
            if self.args.is_mask == True:
                if self.args.label == None:
                    return [Image.fromarray(darken(image)) for image in images]
                else:
                    image = np.array(images[0], dtype=np.uint8)
                    image_mask = np.array(images[1], dtype=np.uint8)
                    augmented_segment = darken(image)
                    image[image_mask == self.args.label] = augmented_segment[image_mask == self.args.label]
                    return [Image.fromarray(image), Image.fromarray(image_mask)]
            else:
                if len(images) == 2:
                    return [Image.fromarray(darken(images[0])), images[1]]
                else:
                    return [Image.fromarray(darken(images[0]))]

        return do(images)


class BrightenScene(Operation):
    def __init__(self, **kwargs):
        self.args = ArgsClass(**kwargs)
        Operation.__init__(self, self.args.probability)

        if 'brightness' not in self.args.__dict__.keys():
            raise CrucialValueNotFoundError(
                "BrightenScene", value_type="brightness")

        if (self.args.brightness != -1):
            if (self.args.brightness < 0.0 or self.args.brightness > 1.0):
                raise CoefficientNotinRangeError(
                    self.args.brightness, "BrightnessCoefficient", 0, 1)

        self.brightness_coeff = 1 + self.args.brightness

    def perform_operation(self, images):

        def brighten(image):
            image = np.array(image, dtype=np.uint8)
            image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            image_HLS[:, :, 1] = image_HLS[:, :, 1] * self.brightness_coeff
            image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
            image_HLS[:, :, 1][image_HLS[:, :, 1] < 0] = 0
            image_HLS = np.array(image_HLS, dtype=np.uint8)
            image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
            return image_RGB

        def do(images):
            if self.args.is_mask == True:
                if self.args.label == None:
                    return [Image.fromarray(brighten(image)) for image in images]
                else:
                    image = np.array(images[0], dtype=np.uint8)
                    image_mask = np.array(images[1], dtype=np.uint8)
                    augmented_segment = brighten(image)
                    image[image_mask == self.args.label] = augmented_segment[image_mask == self.args.label]
                    return [Image.fromarray(image), Image.fromarray(image_mask)]
            else:
                if len(images) == 2:
                    return [Image.fromarray(brighten(images[0])), images[1]]
                else:
                    return [Image.fromarray(brighten(images[0]))]

        return do(images)


class RandomBrightness(Operation):
    def __init__(self, **kwargs):
        self.args = ArgsClass(**kwargs)
        Operation.__init__(self, self.args.probability)

        if 'distribution' not in self.args.__dict__.keys():
            raise CrucialValueNotFoundError(
                "RandomBrightness", value_type="distribution")

        if self.args.distribution == "normal":
            self.coeff = 2 * np.random.normal(0, 1)
        elif self.args.distribution == "uniform":
            self.coeff = 2 * np.random.uniform(0, 1)

    def perform_operation(self, images):

        def random_brighten(image):
            image = np.array(image, dtype=np.uint8)
            image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            image_HLS[:, :, 1] = image_HLS[:, :, 1] * self.coeff
            image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
            image_HLS[:, :, 1][image_HLS[:, :, 1] < 0] = 0
            image_HLS = np.array(image_HLS, dtype=np.uint8)
            image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
            return image_RGB

        def do(images):
            if self.args.is_mask == True:
                if self.args.label == None:
                    return [Image.fromarray(random_brighten(image)) for image in images]
                else:
                    image = np.array(images[0], dtype=np.uint8)
                    image_mask = np.array(images[1], dtype=np.uint8)
                    augmented_segment = random_brighten(image)
                    image[image_mask == self.args.label] = augmented_segment[image_mask == self.args.label]
                    return [Image.fromarray(image), Image.fromarray(image_mask)]
            else:
                if len(images) == 2:
                    return [Image.fromarray(random_brighten(images[0])), images[1]]
                else:
                    return [Image.fromarray(random_brighten(images[0]))]

        return do(images)


class SnowScene(Operation):
    def __init__(self, **kwargs):
        args = ArgsClass(**kwargs)
        Operation.__init__(self, args.probability)

    def perform_operation(self, images):

        def snow(image):
            coefficient = random.gauss(0.3, 0.1)
            coefficient *= 255 / 2
            coefficient += 255 / 3
            coefficient = 255 - coefficient
            image = np.array(image, dtype=np.uint8)
            image_HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL)
            rand = np.random.randint(225,255,(image_HLS[:,:,1].shape[0],image_HLS[:,:,1].shape[1]))
            image_HLS[:,:,1][image_HLS[:,:,1] > coefficient] = rand[image_HLS[:,:,1] > coefficient]
            image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2BGR_FULL)
            return image_RGB

        def do(images):
            if self.args.is_mask == True:
                if self.args.label == None:
                    return [Image.fromarray(snow(image)) for image in images]
                else:
                    image = np.array(images[0], dtype=np.uint8)
                    image_mask = np.array(images[1], dtype=np.uint8)
                    augmented_segment = snow(image)
                    image[image_mask == self.args.label] = augmented_segment[image_mask == self.args.label]
                    return [Image.fromarray(image), Image.fromarray(image_mask)]
            else:
                if len(images) == 2:
                    return [Image.fromarray(snow(images[0])), images[1]]
                else:
                    return [Image.fromarray(snow(images[0]))]

        return do(images)


class RainScene(Operation):
    def __init__(self, **kwargs):
        self.args = ArgsClass(**kwargs)
        Operation.__init__(self, self.args.probability)

        if self.args.rain_type not in ["drizzle", "heavy", "torrential", None]:
            raise ValueError(
                "raint_type must be one of ({}). Got: {}".format(
                    ["drizzle", "heavy", "torrential", None], self.args.rain_type)
            )
        if not -20 <= self.args.slant_lower <= self.args.slant_upper <= 20:
            raise ValueError(
                "Invalid combination of slant_lower and slant_upper. Got: {}".format(
                    (self.args.slant_lower, self.args.slant_upper))
            )
        if not 1 <= self.args.drop_width <= 5:
            raise ValueError(
                "drop_width must be in range [1, 5]. Got: {}".format(
                    self.args.drop_width))
        if not 0 <= self.args.drop_length <= 100:
            raise ValueError(
                "drop_length must be in range [0, 100]. Got: {}".format(
                    self.args.drop_length))
        if not 0 <= self.args.brightness_coefficient <= 1:
            raise ValueError(
                "brightness_coefficient must be in range [0, 1]. Got: {}".format(
                    self.args.brightness_coefficient))

        if not self.args.drop_color and isinstance(self.args.drop_color, list):
            raise ValueError(
                "drop_color must be a list of length 3 and each value must be in range [0, 255] . Got: {}".format(
                    self.args.drop_color))
        self.slant_lower = self.args.slant_lower
        self.slant_upper = self.args.slant_upper

        self.drop_width = self.args.drop_width
        self.drop_color = tuple(self.args.drop_color)
        self.blur_value = self.args.blur_value
        self.brightness_coefficient = self.args.brightness_coefficient
        self.rain_type = self.args.rain_type

    def perform_operation(self, images):

        def rain(image):
            image = np.array(image, dtype=np.uint8)
            drop_length, rain_drops, slant = get_params(image)
            for (rain_drop_x0, rain_drop_y0) in rain_drops:
                rain_drop_x1 = rain_drop_x0 + slant
                rain_drop_y1 = rain_drop_y0 + drop_length

                cv2.line(
                    image,
                    (rain_drop_x0,
                     rain_drop_y0),
                    (rain_drop_x1,
                     rain_drop_y1),
                    self.drop_color,
                    self.drop_width)

            # rainy view are blurry
            image = cv2.blur(image, (self.blur_value, self.blur_value))
            image_hls = cv2.cvtColor(
                image, cv2.COLOR_RGB2HLS).astype(
                np.float32)
            image_hls[:, :, 1] *= self.brightness_coefficient

            image_rgb = cv2.cvtColor(
                image_hls.astype(
                    np.uint8), cv2.COLOR_HLS2RGB)

            return image_rgb

        def get_params(img):
            slant = int(random.uniform(self.slant_lower, self.slant_upper))

            height, width = img.shape[:2]
            area = height * width

            if self.rain_type == "drizzle":
                num_drops = area // 770
                drop_length = 10
            elif self.rain_type == "heavy":
                num_drops = width * height // 600
                drop_length = 30
            elif self.rain_type == "torrential":
                num_drops = area // 500
                drop_length = 60
            else:
                drop_length = self.drop_length
                num_drops = area // 600

            rain_drops = []

            for _i in range(
                    num_drops):  # If You want heavy rain, try increasing this
                if slant < 0:
                    x = random.randint(slant, width)
                else:
                    x = random.randint(0, width - slant)

                y = random.randint(0, height - drop_length)
                rain_drops.append((x, y))

            return drop_length, rain_drops, slant

        def do(images):
            if len(images) == 2:
                return [Image.fromarray(rain(images[0])), images[1]]
            else:
                return [Image.fromarray(rain(images[0]))]

        return do(images)

class MotionBlur(Operation):
    def __init__(self, **kwargs):
        args = ArgsClass(**kwargs)
        if 'blurness' not in args.__dict__.keys():
            raise CrucialValueNotFoundError("MotionBlur", "blurness coefficient")
        self.blurness = args.blurness

    def perform_operation(self, images):
        pass

class FogScene(Operation):
    def __init__(self, **kwargs):
        args = ArgsClass(**kwargs)
        if 'fogness' not in args.__dict__.keys():
            raise CrucialValueNotFoundError("FogScene", "Fogness coefficient")
        self.fogness = args.fogness

    def perform_operation(self, images):
        pass



