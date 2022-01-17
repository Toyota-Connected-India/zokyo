# -*- coding: utf-8 -*-
# Contributors : [srinivas.v@toyotaconnected.co.in,srivathsan.govindarajan@toyotaconnected.co.in,
# harshavardhan.thirupathi@toyotaconnected.co.in,
# ashok.ramadass@toyotaconnected.com ]

import cv2
from ..utils.CustomExceptions import (CoefficientNotinRangeError,
                                      CrucialValueNotFoundError)
from PIL import Image, ImageOps
import numpy as np
import random
from random import randint
import math
from .utils import apply_augmentation


class Operation(object):
    """
        Base class for operations
    """

    def __init__(self, probability):
        self.probability = probability

    def __str__(self):
        return self.__class__.__name__

    def perform_operation(self, images):
        raise RuntimeError("Illegal call to base class.")


class ArgsClass(object):
    """
        Class used to handle arguments passed to operations
    """

    def __init__(self, **kwargs):
        if "probability" not in kwargs.keys():
            kwargs["probability"] = 1

        if "is_mask" not in kwargs.keys():
            kwargs["is_mask"] = False

        if kwargs["is_mask"] is True and "mask_label" not in kwargs:
            kwargs["mask_label"] = None

        if "is_annotation" not in kwargs.keys():
            kwargs["is_annotation"] = False

        if (kwargs["is_annotation"] is True and
                "annotation_label" not in kwargs):
            kwargs["annotation_label"] = None

        self.__dict__.update((key, kwargs[key]) for key in kwargs)


class EqualizeScene(Operation):
    """
        Class to equalize an image or specific classes of an image
    """

    def __init__(self, **kwargs):
        self.args = ArgsClass(**kwargs)
        Operation.__init__(self, self.args.probability)

    def perform_operation(self, entities):
        def equalizeHist(image):
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            return img_rgb

        def do(entities):
            if self.args.is_mask is True:
                if self.args.mask_label is None:
                    entities.image = ImageOps.equalize(entities.image)
                    return entities
                else:
                    image = entities.image
                    image_mask = entities.mask
                    image = apply_augmentation(
                        image, image_mask, self.args.mask_label, equalizeHist)
                    entities.image = Image.fromarray(image)
                    return entities

            if self.args.is_annotation is True:
                if self.args.annotation_label is None:
                    entities.image = ImageOps.equalize(entities.image)
                    return entities
                else:
                    image = entities.image
                    image_mask = entities.annotation_mask
                    image = apply_augmentation(
                        image, image_mask, self.args.annotation_label,
                        equalizeHist)
                    entities.image = Image.fromarray(image)
                    return entities

            if not self.args.is_mask and not self.args.is_annotation:
                entities.image = ImageOps.equalize(entities.image)
                return entities

        return do(entities)


class DarkenScene(Operation):
    """
        Class to darken an image or specific classes of an image based on the darkness parameter[0, 1]
    """

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

    def perform_operation(self, entities):
        def darken(image):
            image = np.array(image)
            image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            image_HLS[:, :, 1] = image_HLS[:, :, 1] * self.darkness_coeff
            image_HLS[:, :, 1][image_HLS[:, :, 1] < 0] = 0
            image_HLS = np.array(image_HLS, dtype=np.uint8)
            image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
            return image_RGB

        def do(entities):
            if self.args.is_mask is True:
                if self.args.mask_label is None:
                    entities.image = Image.fromarray(darken(entities.image))
                    return entities
                else:
                    image = entities.image
                    image_mask = entities.mask
                    image = apply_augmentation(
                        image, image_mask, self.args.mask_label, darken)
                    entities.image = Image.fromarray(image)
                    return entities

            if self.args.is_annotation is True:
                if self.args.annotation_label is None:
                    entities.image = Image.fromarray(darken(entities.image))
                    return entities
                else:
                    image = entities.image
                    image_mask = entities.annotation_mask
                    image = apply_augmentation(
                        image, image_mask, self.args.annotation_label, darken)
                    entities.image = Image.fromarray(image)
                    return entities

            if not self.args.is_mask and not self.args.is_annotation:
                entities.image = Image.fromarray(darken(entities.image))
                return entities

        return do(entities)


class BrightenScene(Operation):
    """
        Class to brighten an image or specific classes of an image based on the brightness parameter[0, 1]
    """

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

    def perform_operation(self, entities):
        def brighten(image):
            image = np.array(image, dtype=np.uint8)
            image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            image_HLS[:, :, 1] = image_HLS[:, :, 1] * self.brightness_coeff
            image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
            image_HLS[:, :, 1][image_HLS[:, :, 1] < 0] = 0
            image_HLS = np.array(image_HLS, dtype=np.uint8)
            image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
            return image_RGB

        def do(entities):
            if self.args.is_mask is True:
                if self.args.mask_label is None:
                    entities.image = Image.fromarray(brighten(entities.image))
                    return entities
                else:
                    image = entities.image
                    image_mask = entities.mask
                    image = apply_augmentation(
                        image, image_mask, self.args.mask_label, brighten)
                    entities.image = Image.fromarray(image)
                    return entities

            if self.args.is_annotation is True:
                if self.args.annotation_label is None:
                    entities.image = Image.fromarray(brighten(entities.image))
                    return entities
                else:
                    image = entities.image
                    image_mask = entities.annotation_mask
                    image = apply_augmentation(
                        image, image_mask, self.args.annotation_label,
                        brighten)
                    entities.image = Image.fromarray(image)
                    return entities

            if not self.args.is_mask and not self.args.is_annotation:
                entities.image = Image.fromarray(brighten(entities.image))
                return entities

        return do(entities)


class RandomBrightness(Operation):
    """
        Class to randomly brighten an image or specific classes of an image based on the random distribution
        parameter (normal or uniform)
    """

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

    def perform_operation(self, entities):
        def random_brighten(image):
            image = np.array(image, dtype=np.uint8)
            image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            image_HLS[:, :, 1] = image_HLS[:, :, 1] * self.coeff
            image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
            image_HLS[:, :, 1][image_HLS[:, :, 1] < 0] = 0
            image_HLS = np.array(image_HLS, dtype=np.uint8)
            image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
            return image_RGB

        def do(entities):
            if self.args.is_mask is True:
                if self.args.mask_label is None:
                    entities.image = Image.fromarray(
                        random_brighten(entities.image))
                    return entities
                else:
                    image = entities.image
                    image_mask = entities.mask
                    image = apply_augmentation(
                        image, image_mask, self.args.mask_label,
                        random_brighten)
                    entities.image = Image.fromarray(Image.fromarray(image))
                    return entities

            if self.args.is_annotation is True:
                if self.args.annotation_label is None:
                    entities.image = Image.fromarray(
                        random_brighten(entities.image))
                    return entities
                else:
                    image = entities.image
                    image_mask = entities.annotation_mask
                    image = apply_augmentation(
                        image, image_mask, self.args.annotation_label,
                        random_brighten)
                    entities.image = Image.fromarray(image)
                    return entities

            if not self.args.is_mask and not self.args.is_annotation:
                entities.image = Image.fromarray(
                    random_brighten(entities.image))
                return entities

        return do(entities)


class SnowScene(Operation):
    """
        Class to add snow effect to an image or specific classes of an image
    """

    def __init__(self, **kwargs):
        self.args = ArgsClass(**kwargs)
        Operation.__init__(self, self.args.probability)

    def perform_operation(self, entities):
        def snow(image):
            coefficient = random.gauss(0.35, 0.15)
            coefficient *= 255 / 2
            coefficient += 255 / 3
            coefficient = 255 - coefficient
            image = np.array(image, dtype=np.uint8)
            image_HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL)
            rand = np.random.randint(
                225, 255, (image_HLS[:, :, 1].shape[0],
                           image_HLS[:, :, 1].shape[1]))
            image_HLS[:, :, 1][image_HLS[:, :, 1] > coefficient] = rand[image_HLS[:, :, 1] > coefficient]  # noqa
            image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2BGR_FULL)
            return image_RGB

        def do(entities):
            if self.args.is_mask is True:
                if self.args.mask_label is None:
                    entities.image = Image.fromarray(snow(entities.image))
                    return entities
                else:
                    image = entities.image
                    image_mask = entities.mask
                    image = apply_augmentation(
                        image, image_mask, self.args.mask_label, snow)
                    entities.image = Image.fromarray(image)
                    return entities

            if self.args.is_annotation is True:
                if self.args.annotation_label is None:
                    entities.image = Image.fromarray(snow(entities.image))
                    return entities
                else:
                    image = entities.image
                    image_mask = entities.annotation_mask
                    image = apply_augmentation(
                        image, image_mask, self.args.annotation_label, snow)
                    entities.image = Image.fromarray(image)
                    return entities

            if not self.args.is_mask and not self.args.is_annotation:
                entities.image = Image.fromarray(snow(entities.image))
                return entities

        return do(entities)


class RadialLensDistortion(Operation):
    """
        Class to apply radial distortion to an image or specific classes of an image using the
        distortion type parameter (NegativeBarrel, PinCushion). It is applied to masks/ annotation masks too
    """

    def __init__(self, **kwargs):
        self.args = ArgsClass(**kwargs)
        Operation.__init__(self, self.args.probability)

        if self.args.distortiontype not in ["NegativeBarrel", "PinCushion"]:
            raise ValueError(
                "distortiontype must be one of ({}). Got: {}".format(
                    ["NegativeBarrel", "PinCushion"], self.args.rain_type)
            )

        if (self.args.distortiontype == "NegativeBarrel"):
            self.radialk1 = -1 * randint(0, 10) / 10
        elif (self.args.distortiontype != "PinCushion"):
            self.radialk1 = randint(0, 10) / 10

    def perform_operation(self, entities):
        def radial_distort(image):
            image = np.array(image, dtype=np.uint8)
            d_coef = (self.radialk1, 0, 0, 0, 0)
            h, w = image.shape[:2]
            f = (h ** 2 + w ** 2) ** 0.5
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
            M, _ = cv2.getOptimalNewCameraMatrix(K, d_coef, (w, h), 0)
            remap = cv2.initUndistortRectifyMap(K, d_coef, None, M, (w, h), 5)
            image = cv2.remap(image, *remap, cv2.INTER_LINEAR)
            return image

        def do(entities):
            if self.args.is_mask is True:
                if self.args.mask_label is None:
                    entities.image = Image.fromarray(
                        radial_distort(entities.image))
                    entities.mask = Image.fromarray(
                        radial_distort(entities.mask))
                    return entities
                else:
                    image = entities.image
                    image_mask = entities.mask
                    image = apply_augmentation(
                        image, image_mask, self.args.mask_label,
                        radial_distort)
                    entities.image = Image.fromarray(image)
                    return entities

            if self.args.is_annotation is True:
                if self.args.annotation_label is None:
                    entities.image = Image.fromarray(
                        radial_distort(entities.image))
                    entities.annotation_mask = Image.fromarray(
                        radial_distort(entities.annotation_mask))
                    return entities
                else:
                    image = entities.image
                    image_mask = entities.annotation_mask
                    image = apply_augmentation(
                        image, image_mask, self.args.annotation_label,
                        radial_distort)
                    entities.image = Image.fromarray(image)
                    return entities

            if not self.args.is_mask and not self.args.is_annotation:
                entities.image = Image.fromarray(
                    radial_distort(entities.image))
                return entities

        return do(entities)


class TangentialLensDistortion(Operation):
    """
        Class to apply tangential distortion to an image or specific classes of an image.
        It is applied to masks/ annotation masks too
    """

    def __init__(self, **kwargs):
        self.args = ArgsClass(**kwargs)
        Operation.__init__(self, self.args.probability)
        self.tangentialP1 = randint(-10, 10) / 100
        self.tangentialP2 = randint(-10, 10) / 100

    def perform_operation(self, entities):
        def tangential_distort(image):
            image = np.array(image, dtype=np.uint8)
            d_coef = (0, 0, self.tangentialP1, self.tangentialP2, 0)
            h, w = image.shape[:2]
            f = (h ** 2 + w ** 2) ** 0.5
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
            M, _ = cv2.getOptimalNewCameraMatrix(K, d_coef, (w, h), 0)
            remap = cv2.initUndistortRectifyMap(K, d_coef, None, M, (w, h), 5)
            image = cv2.remap(image, *remap, cv2.INTER_LINEAR)
            return image

        def do(entities):
            if self.args.is_mask is True:
                if self.args.mask_label is None:
                    entities.image = Image.fromarray(
                        tangential_distort(entities.image))
                    entities.mask = Image.fromarray(
                        tangential_distort(entities.mask))
                    return entities
                else:
                    image = entities.image
                    image_mask = entities.mask
                    image = apply_augmentation(
                        image, image_mask, self.args.mask_label,
                        tangential_distort)
                    entities.image = Image.fromarray(image)
                    return entities

            if self.args.is_annotation is True:
                if self.args.annotation_label is None:
                    entities.image = Image.fromarray(
                        tangential_distort(entities.image))
                    entities.annotation_mask = Image.fromarray(
                        tangential_distort(entities.annotation_mask))
                    return entities
                else:
                    image = entities.image
                    image_mask = entities.annotation_mask
                    image = apply_augmentation(
                        image, image_mask, self.args.annotation_label,
                        tangential_distort)
                    entities.image = Image.fromarray(image)
                    return entities

            if not self.args.is_mask and not self.args.is_annotation:
                entities.image = Image.fromarray(
                    tangential_distort(entities.image))
                return entities

        return do(entities)


class RainScene(Operation):
    """
        Class to apply rain effect to an image based on the paramters rain type (drizzle, heavy, torrential),
        drop width, length and color, slant of the rain and a brightness coefficient
    """

    def __init__(self, **kwargs):
        self.args = ArgsClass(**kwargs)
        Operation.__init__(self, self.args.probability)

        if self.args.rain_type not in ["drizzle", "heavy", "torrential", None]:
            raise ValueError(
                "raint_type must be one of ({}). Got: {}".format(
                    ["drizzle", "heavy", "torrential", None],
                    self.args.rain_type)
            )
        if not -20 <= self.args.slant_lower <= self.args.slant_upper <= 20:
            raise ValueError(
                '''Invalid combination of slant_lower and slant_upper. Got:
                {}'''.format(
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
                '''brightness_coefficient must be in range [0, 1]. Got:
                {}'''.format(
                    self.args.brightness_coefficient))

        if not self.args.drop_color and isinstance(self.args.drop_color,
                                                   list):
            raise ValueError(
                '''drop_color must be a list of length 3 and each value
                must be in range [0, 255] . Got: {}'''.format(
                    self.args.drop_color))
        self.slant_lower = self.args.slant_lower
        self.slant_upper = self.args.slant_upper

        self.drop_width = self.args.drop_width
        self.drop_color = tuple(self.args.drop_color)
        self.blur_value = self.args.blur_value
        self.brightness_coefficient = self.args.brightness_coefficient
        self.rain_type = self.args.rain_type

    def perform_operation(self, entities):

        def rain(image):
            def get_params(img):
                slant = int(random.uniform(self.slant_lower,
                                           self.slant_upper))

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
                        num_drops):
                    # If You want heavy rain, try increasing this
                    if slant < 0:
                        x = random.randint(slant, width)
                    else:
                        x = random.randint(0, width - slant)
                    y = random.randint(0, height - drop_length)
                    rain_drops.append((x, y))
                return drop_length, rain_drops, slant

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
            # image_hls[:, :, 1] *= self.brightness_coefficient

            image_rgb = cv2.cvtColor(
                image_hls.astype(
                    np.uint8), cv2.COLOR_HLS2RGB)

            return image_rgb

        def do(entities):
            entities.image = Image.fromarray(rain(entities.image))
            return entities

        return do(entities)


class SunFlare(Operation):
    """
        Class to apply sun flare effect to an image
    """

    def __init__(self, **kwargs):
        self.args = ArgsClass(**kwargs)
        Operation.__init__(self, self.args.probability)

    def perform_operation(self, entities):
        def sun_flare(image, flare_center=-1, angle=-1, no_of_flare_circles=3,
                      src_radius=100, src_color=(255, 255, 255)):
            def flare_source(image, point, radius, src_color):
                overlay = image.copy()
                output = image.copy()
                num_times = radius // 10
                alpha = np.linspace(0.0, 1, num=num_times)
                rad = np.linspace(1, radius, num=num_times)
                for i in range(num_times):
                    cv2.circle(overlay, point, int(rad[i]), src_color, -1)
                    alp = alpha[num_times - i - 1] * \
                        alpha[num_times - i - 1] * alpha[num_times - i - 1]
                    cv2.addWeighted(overlay, alp, output, 1 - alp, 0, output)
                return output

            def add_sun_flare_line(flare_center, angle, imshape):
                x = []
                y = []
                for rand_x in range(0, imshape[1], 10):
                    rand_y = math.tan(angle) * ((rand_x -
                                                flare_center[0]) +
                                                flare_center[1])
                    x.append(rand_x)
                    y.append(2 * flare_center[1] - rand_y)
                return x, y

            def add_sun_process(image, no_of_flare_circles,
                                flare_center, src_radius, x, y, src_color):
                overlay = image.copy()
                output = image.copy()
                imshape = image.shape
                for _ in range(no_of_flare_circles):
                    alpha = random.uniform(0.05, 0.2)
                    r = random.randint(0, len(x) - 1)
                    rad = random.randint(1, imshape[0] // 100 - 2)
                    cv2.circle(
                        overlay,
                        (int(x[r]), int(y[r])),
                        rad**3,
                        (random.randint(max(src_color[0] - 50, 0),
                                        src_color[0]),
                         random.randint(
                             max(src_color[1] - 50, 0), src_color[1]),
                         random.randint(max(src_color[2] - 50, 0),
                                        src_color[2])),
                        - 1
                    )
                    cv2.addWeighted(
                        overlay, alpha, output, 1 - alpha, 0, output)
                output = flare_source(
                    output, (int(
                        flare_center[0]), int(
                        flare_center[1])), src_radius, src_color)
                return output

            image = np.array(image, dtype=np.uint8)
            imshape = image.shape
            if(angle == -1):
                angle_t = random.uniform(0, 2 * math.pi)
                if angle_t == math.pi / 2:
                    angle_t = 0
            else:
                angle_t = angle
            if flare_center == -1:
                flare_center_t = (
                    random.randint(
                        0, imshape[1]), random.randint(
                        0, imshape[0] // 2))
            else:
                flare_center_t = flare_center
            x, y = add_sun_flare_line(flare_center_t, angle_t, imshape)
            output = add_sun_process(
                image,
                no_of_flare_circles,
                flare_center_t,
                src_radius,
                x,
                y,
                src_color)
            image_RGB = output
            return image_RGB

        def do(entities):
            entities.image = Image.fromarray(sun_flare(entities.image))
            return entities

        return do(entities)


class MotionBlur(Operation):
    """
        TODO: Class to apply motion blur to an image
    """

    def __init__(self, **kwargs):
        self.args = ArgsClass(**kwargs)
        if 'blurness' not in self.args.__dict__.keys():
            raise CrucialValueNotFoundError(
                "MotionBlur", "blurness coefficient")
        self.blurness = self.args.blurness

    def perform_operation(self, images):
        raise NotImplementedError("Motionblur not implemented")


class FogScene(Operation):
    """
        TODO: Class to apply fog effect to an image
    """

    def __init__(self, **kwargs):
        self.args = ArgsClass(**kwargs)
        if 'fogness' not in self.args.__dict__.keys():
            raise CrucialValueNotFoundError("FogScene", "Fogness coefficient")
        self.fogness = self.args.fogness

    def perform_operation(self, images):
        raise NotImplementedError("FogScene not implemented")
