# -*- coding: utf-8 -*-
# Contributors : [srinivas.v@toyotaconnected.co.in,
# ashok.ramadass@toyotaconnected.com, ]

import cv2
from Augmentor.Operations import Operation

from numpy.lib import histograms
from ..utils.CustomExceptions import CoefficientNotinRangeError, InvalidImageArrayError, CrucialValueNotFoundError
from ..utils.misc import from_float, to_float
from PIL import Image, ImageOps
import numpy as np
import random
from random import randint
import warnings
import math

class ArgsClass(object):
    def __init__(self, **kwargs):
        if "probability" not in kwargs.keys():
            kwargs["probability"] = 1

        if "is_mask" not in kwargs.keys():
            kwargs["is_mask"] = False

        if kwargs["is_mask"] is True and "label" not in kwargs:
            kwargs["label"] = None

        self.__dict__.update((key, kwargs[key]) for key in kwargs)


class EqualizeScene(Operation):
    def __init__(self, **kwargs):
        self.args = ArgsClass(**kwargs)
        Operation.__init__(self, self.args.probability)

    def perform_operation(self, images):
        def do(images):
            if self.args.is_mask is True:
                if self.args.label is None:
                    return [ImageOps.equalize(image) for image in images]
                else:
                    image = images[0]
                    image_mask = images[1]
                    augmented_segment = ImageOps.equalize(image)
                    image = np.array(image, dtype=np.uint8)
                    augmented_segment = np.array(
                        augmented_segment, dtype=np.uint8)
                    image[image_mask ==
                          self.args.label] = augmented_segment[image_mask == self.args.label]
                    return [Image.fromarray(image), image_mask]
            else:
                if len(images) > 1:
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
            if self.args.is_mask is True:
                if self.args.label is None:
                    return [Image.fromarray(darken(image)) for image in images]
                else:
                    image = np.array(images[0], dtype=np.uint8)
                    image_mask = np.array(images[1], dtype=np.uint8)
                    augmented_segment = darken(image)
                    image[image_mask ==
                          self.args.label] = augmented_segment[image_mask == self.args.label]
                    return [Image.fromarray(
                        image), Image.fromarray(image_mask)]
            else:
                if len(images) > 1:
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
            if self.args.is_mask is True:
                if self.args.label is None:
                    return [Image.fromarray(brighten(image))
                            for image in images]
                else:
                    image = np.array(images[0], dtype=np.uint8)
                    image_mask = np.array(images[1], dtype=np.uint8)
                    augmented_segment = brighten(image)
                    image[image_mask ==
                          self.args.label] = augmented_segment[image_mask == self.args.label]
                    return [Image.fromarray(
                        image), Image.fromarray(image_mask)]
            else:
                if len(images) > 1:
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
            if self.args.is_mask is True:
                if self.args.label is None:
                    return [Image.fromarray(random_brighten(image))
                            for image in images]
                else:
                    image = np.array(images[0], dtype=np.uint8)
                    image_mask = np.array(images[1], dtype=np.uint8)
                    augmented_segment = random_brighten(image)
                    image[image_mask ==
                          self.args.label] = augmented_segment[image_mask == self.args.label]
                    return [Image.fromarray(
                        image), Image.fromarray(image_mask)]
            else:
                if len(images) > 1:
                    return [Image.fromarray(
                        random_brighten(images[0])), images[1]]
                else:
                    return [Image.fromarray(random_brighten(images[0]))]

        return do(images)

class SnowScene(Operation):
    def __init__(self, **kwargs):
        args = ArgsClass(**kwargs)
        Operation.__init__(self, args.probability)

    def perform_operation(self, images):

        def snow(image):
            coefficient = random.gauss(0.35, 0.15)
            coefficient *= 255 / 2
            coefficient += 255 / 3
            coefficient = 255 - coefficient
            image = np.array(image, dtype=np.uint8)
            image_HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL)
            rand = np.random.randint(
                225, 255, (image_HLS[:, :, 1].shape[0], image_HLS[:, :, 1].shape[1]))
            image_HLS[:, :, 1][image_HLS[:, :, 1] >
                               coefficient] = rand[image_HLS[:, :, 1] > coefficient]
            image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2BGR_FULL)
            return image_RGB

        def do(images):
            if self.args.is_mask is True:
                if self.args.label is None:
                    return [Image.fromarray(snow(image)) for image in images]
                else:
                    image = np.array(images[0], dtype=np.uint8)
                    image_mask = np.array(images[1], dtype=np.uint8)
                    augmented_segment = snow(image)
                    image[image_mask ==
                          self.args.label] = augmented_segment[image_mask == self.args.label]
                    return [Image.fromarray(
                        image), Image.fromarray(image_mask)]
            else:
                if len(images) > 1:
                    return [Image.fromarray(snow(images[0])), images[1]]
                else:
                    return [Image.fromarray(snow(images[0]))]

        return do(images)

class RadialLensDistortion(Operation):
    def __init__(self, **kwargs):
        args = ArgsClass(**kwargs)
        Operation.__init__(self, args.probability)

        if args.distortiontype not in ["NegativeBarrel", "PinCushion"]:
            raise ValueError(
                "distortiontype must be one of ({}). Got: {}".format(["NegativeBarrel", "PinCushion"], args.rain_type)
            )
        
        if (args.distortiontype == "NegativeBarrel"):
            self.radialk1 = -1 * randint(0, 10) / 10
        elif (args.distortiontype != "PinCushion"):
            self.radialk1 = randint(0, 10) / 10

    def perform_operation(self, images):
        def do(image):
            image = np.array(image, dtype=np.uint8)
            d_coef = (self.radialk1, 0, 0, 0, 0)
            # get the height and the width of the image
            h, w = image.shape[:2]
            # compute its diagonal
            f = (h ** 2 + w ** 2) ** 0.5
            # set the image projective to carrtesian dimension
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
            # Generate new camera matrix from parameters
            M, _ = cv2.getOptimalNewCameraMatrix(K, d_coef, (w, h), 0)
            # Generate look-up tables for remapping the camera image
            remap = cv2.initUndistortRectifyMap(K, d_coef, None, M, (w, h), 5)
            # Remap the original image to a new image
            image = cv2.remap(image, *remap, cv2.INTER_LINEAR)
            return Image.fromarray(image)
        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        return augmented_images

class TangentialLensDistortion(Operation):
    def __init__(self, **kwargs):
        args = ArgsClass(**kwargs)
        Operation.__init__(self, args.probability)
        self.tangentialP1 = randint(-10, 10) / 100
        self.tangentialP2 = randint(-10, 10) / 100

    def perform_operation(self, images):
        def do(image):
            image = np.array(image, dtype=np.uint8)
            d_coef = (0, 0, self.tangentialP1, self.tangentialP2, 0)
            # get the height and the width of the image
            h, w = image.shape[:2]
            # compute its diagonal
            f = (h ** 2 + w ** 2) ** 0.5
            # set the image projective to carrtesian dimension
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
            # Generate new camera matrix from parameters
            M, _ = cv2.getOptimalNewCameraMatrix(K, d_coef, (w, h), 0)
            # Generate look-up tables for remapping the camera image
            remap = cv2.initUndistortRectifyMap(K, d_coef, None, M, (w, h), 5)
            # Remap the original image to a new image
            image = cv2.remap(image, *remap, cv2.INTER_LINEAR)
            return Image.fromarray(image)
        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        return augmented_images

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
            if len(images) > 1:
                return [Image.fromarray(rain(images[0])), images[1]]
            else:
                return [Image.fromarray(rain(images[0]))]

        return do(images)


class MotionBlur(Operation):
    def __init__(self, **kwargs):
        args = ArgsClass(**kwargs)
        if 'blurness' not in args.__dict__.keys():
            raise CrucialValueNotFoundError(
                "MotionBlur", "blurness coefficient")
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


class SunFlare(Operation):
    def __init__(self, **kwargs):
        args = ArgsClass(**kwargs)
        Operation.__init__(self, args.probability)

    def perform_operation(self, images):
        def flare_source(image, point, radius, src_color):
            overlay = image.copy()
            output = image.copy()
            num_times = radius//10
            alpha = np.linspace(0.0, 1, num = num_times)
            rad = np.linspace(1, radius, num = num_times)
            for i in range(num_times):
                cv2.circle(overlay, point, int(rad[i]), src_color, -1)
                alp = alpha[num_times-i-1] * alpha[num_times-i-1] * alpha[num_times-i-1]
                cv2.addWeighted(overlay, alp, output, 1 -alp , 0, output)
            return output

        def add_sun_flare_line(flare_center, angle, imshape):
            x=[]
            y=[]
            i=0
            for rand_x in range(0, imshape[1], 10):
                rand_y= math.tan(angle) * (rand_x-flare_center[0]) + flare_center[1]
                x.append(rand_x)
                y.append(2 * flare_center[1]-rand_y)
            return x, y

        def add_sun_process(image, no_of_flare_circles, flare_center, src_radius, x, y, src_color):
            overlay = image.copy()
            outpu = image.copy()
            imshape =image.shape
            for i in range(no_of_flare_circles):
                alpha = random.uniform(0.05, 0.2)
                r = random.randint(0, len(x)-1)
                rad = random.randint(1, imshape[0] // 100 - 2)
                cv2.circle(overlay,(int(x[r]), int(y[r])), rad * rad * rad, (random.randint(max(src_color[0]-50, 0), src_color[0]), random.randint(max(src_color[1]-50, 0), src_color[1]), random.randint(max(src_color[2]-50, 0), src_color[2])), -1)
                cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)                      
            output = flare_source(output, (int(flare_center[0]), int(flare_center[1])), src_radius, src_color)
            return output

        def add_sun_flare(image, flare_center=-1, angle=-1, no_of_flare_circles=3,src_radius=100, src_color=(255,255,255)):
            image = np.array(image, dtype=np.uint8)
            imshape = image.shape
            if(angle == -1):
                angle_t = random.uniform(0, 2 * math.pi)
                if angle_t == math.pi / 2:
                    angle_t = 0
            else:
                angle_t = angle
            if flare_center == -1:
                flare_center_t = (random.randint(0,imshape[1]),random.randint(0,imshape[0]//2))
            else:
                flare_center_t = flare_center
            x,y = add_sun_flare_line(flare_center_t, angle_t, imshape)
            output = add_sun_process(image, no_of_flare_circles, flare_center_t, src_radius, x, y, src_color)
            image_RGB = output
            return image_RGB

        def do(images):
            if len(images) > 1:
                return [Image.fromarray(add_sun_flare(images[0])), images[1]]
            else:
                return [Image.fromarray(add_sun_flare(images[0]))]

        return do(images)

        
    