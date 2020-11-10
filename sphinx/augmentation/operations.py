# -*- coding: utf-8 -*-
# Contributors : [srinivas.v@toyotaconnected.co.in, ]

import Augmentor
import cv2
from Augmentor.Operations import Operation
import os
from ..utils.CustomExceptions import CoefficientNotinRangeError, InvalidImageArrayError, CrucialValueNotFoundError
from ..utils.misc import from_float, to_float
from PIL import Image, ImageOps
import numpy as np
import warnings
import random


class ArgsClass(object):
    def __init__(self, **kwargs):
        if "probability" not in kwargs.keys():
            kwargs["probability"] = 1
        self.__dict__.update((key, kwargs[key]) for key in kwargs)


class EqualizeScene(Operation):
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

# TODO : Add Sky augmentation operation

# TODO : Add a common util to change lighting instead of repeating the
# implementation


class BrightenScene(Operation):
    def __init__(self, **kwargs):
        args = ArgsClass(**kwargs)
        Operation.__init__(self, args.probability)

        if args.coefficient is None:
            raise CrucialValueNotFoundError(
                "BrightenScene", sample_type="coefficient")

        if (args.coefficient != -1):
            if (args.coefficient < 0.0 or args.coefficient > 1.0):
                raise CoefficientNotinRangeError(
                    args.coefficient, "BrightnessCoefficient", 0, 1)

        self.brightness_coeff = 1 + args.coefficient

    def perform_operation(self, images):

        def do(image):
            image = np.array(image, dtype=np.uint8)
            image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            image_HLS[:, :, 1] = image_HLS[:, :, 1] * self.brightness_coeff
            image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
            image_HLS[:, :, 1][image_HLS[:, :, 1] < 0] = 0
            image_HLS = np.array(image_HLS, dtype=np.uint8)
            image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
            return Image.fromarray(image_RGB)

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        return augmented_images


class RandomBrightness(Operation):
    def __init__(self, **kwargs):
        args = ArgsClass(**kwargs)
        Operation.__init__(self, args.probability)

        if args.distribution is None:
            raise CrucialValueNotFoundError(
                "RandomBrightness", sample_type="distribution")

        if args.distribution == "normal":
            self.coeff = 2 * np.random.normal(0, 1)
        elif args.distribution == "uniform":
            self.coeff = 2 * np.random.uniform(0, 1)

    def perform_operation(self, images):

        def do(image):
            image = np.array(image, dtype=np.uint8)
            image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            image_HLS[:, :, 1] = image_HLS[:, :, 1] * self.coeff
            image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
            image_HLS[:, :, 1][image_HLS[:, :, 1] < 0] = 0
            image_HLS = np.array(image_HLS, dtype=np.uint8)
            image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
            return Image.fromarray(image_RGB)

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        return augmented_images


class SnowScene(Operation):
    def __init__(self, **kwargs):
        args = ArgsClass(**kwargs)
        Operation.__init__(self, args.probability)

        if args.distribution is None:
            raise CrucialValueNotFoundError(
                "SnowScene", sample_type="coefficient")

        if (args.coefficient != -1):
            if (args.coefficient < 0.0 or args.coefficient > 1.0):
                raise CoefficientNotinRangeError(
                    args.coefficient, "SnownessCoefficient", 0, 1)

        args.coefficient *= 255 / 2
        args.coefficient += 255 / 3
        self.snowness = args.coefficient

    def perform_operation(self, images):

        def do(image):
            image = np.array(image, dtype=np.uint8)
            image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            brightness_coefficient = np.random.uniform(1, 3)
            image_HLS[:, :, 1][image_HLS[:, :, 1] < self.snowness] = image_HLS[:,
                                                                               :, 1][image_HLS[:, :, 1] < self.snowness] * brightness_coefficient
            image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
            image_HLS = np.array(image_HLS, dtype=np.uint8)
            image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
            return Image.fromarray(image_RGB)

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        return augmented_images


class RainScene(Operation):
    def __init__(self, **kwargs):
        args = ArgsClass(**kwargs)
        Operation.__init__(self, args.probability)

        if args.rain_type not in ["drizzle", "heavy", "torrential", None]:
            raise ValueError(
                "raint_type must be one of ({}). Got: {}".format(["drizzle", "heavy", "torrential", None], args.rain_type)
            )
        if not -20 <= args.slant_lower <= args.slant_upper <= 20:
            raise ValueError(
                "Invalid combination of slant_lower and slant_upper. Got: {}".format((args.slant_lower, args.slant_upper))
            )
        if not 1 <= args.drop_width <= 5:
            raise ValueError("drop_width must be in range [1, 5]. Got: {}".format(args.drop_width))
        if not 0 <= args.drop_length <= 100:
            raise ValueError("drop_length must be in range [0, 100]. Got: {}".format(args.drop_length))
        if not 0 <= args.brightness_coefficient <= 1:
            raise ValueError("brightness_coefficient must be in range [0, 1]. Got: {}".format(args.brightness_coefficient))

        if not args.drop_color and isinstance(args.drop_color, list):
            raise ValueError("drop_color must be a list of length 3 and each value must be in range [0, 255] . Got: {}".format(args.drop_color))
        self.slant_lower = args.slant_lower
        self.slant_upper = args.slant_upper

        self.drop_width = args.drop_width
        self.drop_color = tuple(args.drop_color)
        self.blur_value = args.blur_value
        self.brightness_coefficient = args.brightness_coefficient
        self.rain_type = args.rain_type

    def perform_operation(self, images):

        def do(image):
            image = np.array(image, dtype=np.uint8)
            drop_length, rain_drops, slant = get_params(image)
            for (rain_drop_x0, rain_drop_y0) in rain_drops:
                rain_drop_x1 = rain_drop_x0 + slant
                rain_drop_y1 = rain_drop_y0 + drop_length

                cv2.line(image, (rain_drop_x0, rain_drop_y0), (rain_drop_x1, rain_drop_y1), self.drop_color, self.drop_width)

            image = cv2.blur(image, (self.blur_value, self.blur_value))  # rainy view are blurry
            image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float32)
            image_hls[:, :, 1] *= self.brightness_coefficient

            image_rgb = cv2.cvtColor(image_hls.astype(np.uint8), cv2.COLOR_HLS2RGB)

            return Image.fromarray(image_rgb)

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

            for _i in range(num_drops):  # If You want heavy rain, try increasing this
                if slant < 0:
                    x = random.randint(slant, width)
                else:
                    x = random.randint(0, width - slant)

                y = random.randint(0, height - drop_length)

                rain_drops.append((x, y))

            return drop_length, rain_drops, slant

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        return augmented_images