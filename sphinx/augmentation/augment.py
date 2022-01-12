# -*- coding: utf-8 -*-
# Contributors : [srinivas.v@toyotaconnected.co.in, srivathsan.govindarajan@toyotaconnected.co.in
# harshavardhan.thirupathi@toyotaconnected.co.in,
# ashok.ramadass@toyotaconnected.com]

import os
from os.path import join
import json
import importlib
from pathlib import Path
from .pipeline import _DataPipeline
from .generators import _KerasGenerator
from PIL import Image
from ..utils.CustomExceptions import (CrucialValueNotFoundError,
                                      OperationNotFoundOrImplemented)
import numpy as np
from tqdm import tqdm
import random
import uuid
from abc import ABC, abstractclassmethod
import xml.etree.ElementTree as ET
from .data import SphinxData
import cv2
from .utils import change_pascal_annotation
from ..utils.logging import get_logger
import logging
import math
from itertools import cycle


class AbstractBuilder(ABC):

    @abstractclassmethod
    def _add_operation(self, pipeline):
        raise NotImplementedError("This method is not implemented")

    @abstractclassmethod
    def _load_entities(self, data_path_list):
        raise NotImplementedError("This is method is not implemented")

    @abstractclassmethod
    def process_and_generate(self):
        raise NotImplementedError("This method is not implemented")

    @abstractclassmethod
    def process_and_save(self):
        raise NotImplementedError("This method is not implemented")

    @abstractclassmethod
    def get_keras_generator(self):
        raise NotImplementedError("This method is not implemented")


class Builder(AbstractBuilder):
    """
        TODO : Parallelize saving images to disk
        TODO : Parallelize operations from data pipeline

        Builder class to create augmentor Pipeline object
        Config file is a json map used to define Operations and their
        properties
        {
            "input_dir" : "tests/images",
            "mask_dir" : "tests/masks",
            "output_dir": "tests/output",
            "annotation_dir": "tests/annotation",
            "annotation_format": "pascal_voc",
            "sample" : 50,
            "multi_threaded" : false,
            "run_all" : false,
            "batch_ingestion": true,
            "shuffle": false,
            "debug": false,
            "internal_batch": 5,
            "save_annotation_mask" : false,
            "operations":[
                {
                    "operation": "DarkenScene",
                    "operation_module" : "sphinx.augmentation",
                    "args": {
                        "probability": 0.7,
                        "darkness" : 0.5,
                        "is_mask" : true,
                        "mask_label" : 2,
                        "is_annotation" : true,
                        "annotation_label" : 1
                    }
                },
                {
                    "operation": "EqualizeScene",
                    "operation_module" : "sphinx.augmentation",
                    "args": {
                        "probability": 0.5,
                        "is_mask" : true,
                        "label" : 2
                    }
                },
                {
                    "operation": "RadialLensDistortion",
                    "operation_module" : "sphinx.augmentation",
                    "args": {
                        "probability": 0.5,
                        "is_annotation" : true,
                        "distortiontype" : "NegativeBarrel",
                        "is_mask" : true
                    }
                }
            ]
        }
    """

    def __init__(self, config_json="config.json"):
        if not os.path.exists(config_json):
            raise FileNotFoundError("{} not found".format(config_json))

        self.batch_size = None
        self._image_extension_list = ["png", "jpg", "jpeg", "bmp"]
        self._annotation_extension_list = ["xml"]
        self.logger = get_logger("Sphinx Builder", level=logging.INFO)

        with open(config_json) as config_file:
            self.config = json.load(config_file)

        if "operations" not in self.config.keys():
            raise CrucialValueNotFoundError(
                operation="augmentation configurations",
                value_type="operations")

        if "input_dir" not in self.config.keys():
            raise CrucialValueNotFoundError(
                operation="configuration json file",
                value_type="input_dir")

        if "input_dir" in self.config.keys() and len(
                os.listdir(self.config["input_dir"])) == 0:
            raise FileNotFoundError("No files found in input directory")

        if "output_dir" not in self.config.keys():
            self.output_dir = join(os.getcwd(), "output")
            try:
                os.mkdir(self.output_dir)
                os.mkdir(join(self.output_dir, "images"))
                os.mkdir(join(self.output_dir, "masks"))
                os.mkdir(join(self.output_dir, "annotations"))
            except FileExistsError:
                pass

        if "output_dir" in self.config.keys():
            self.output_dir = self.config["output_dir"]
            if os.path.exists(self.output_dir):
                try:
                    os.mkdir(join(self.output_dir, "images"))
                    os.mkdir(join(self.output_dir, "masks"))
                    os.mkdir(join(self.output_dir, "annotations"))
                except FileExistsError:
                    pass
            else:
                os.mkdir(self.output_dir)
                os.mkdir(join(self.output_dir, "images"))
                os.mkdir(join(self.output_dir, "masks"))
                os.mkdir(join(self.output_dir, "annotations"))

        if not os.path.exists(self.config["input_dir"]):
            raise FileNotFoundError("{} not found", self.config["input_dir"])

        self.data_len = len(os.listdir(self.config["input_dir"]))

        if self.data_len == 0:
            raise FileExistsError("No files found in the input directory")

        if "mask_dir" in self.config.keys() and len(
                os.listdir(self.config["mask_dir"])) == 0:
            raise FileNotFoundError("No files found in mask directory")

        if "annotation_dir" in self.config.keys() and len(
                os.listdir(self.config["annotation_dir"])) == 0:
            raise FileNotFoundError("No files found in annotation directory")

        if "output_dir" not in self.config.keys():
            self.output_dir = "output"

        if "sample" not in self.config.keys():
            self.sample = self.data_len

        if "run_all" not in self.config.keys():
            self.run_all = False

        if "multi_threaded" not in self.config.keys():
            self.multi_threaded = False

        if "shuffle" not in self.config.keys():
            self.shuffle = False

        if "batch_ingestion" not in self.config.keys():
            self.batch_ingestion = False

        if "debug" not in self.config.keys():
            self.debug = False

        if "annotation_dir" in self.config.keys():
            if "annotation_format" not in self.config.keys():
                raise CrucialValueNotFoundError(
                    operation="annotation data",
                    value_type="annotation_format")
            else:
                if self.config["annotation_format"] != "pascal_voc":
                    raise NotImplementedError(
                        '''Annotation format not supported, pascal_voc is the
                         only supported format''')

        if "save_annotation_mask" not in self.config.keys():
            self.save_annotation_mask = False

        if "save_annotation_mask" in self.config.keys(
        ) and self.config["save_annotation_mask"] is True:
            try:
                os.mkdir(join(self.output_dir, "annotation_mask"))
            except BaseException:
                pass

        self.setting_generator_params = False
        self.class_dictionary = {}
        self.class_dictionary["background"] = 0
        self.classes = 1

        self.__dict__.update(
            (key,
             self.config[key]) for key in (
                'input_dir',
                'sample',
                'output_dir',
                'run_all',
                'multi_threaded',
                'operations',
                'mask_dir',
                'annotation_dir',
                'annotation_format',
                'save_annotation_mask',
                'batch_ingestion',
                'shuffle',
                'debug',
                'internal_batch') if key in self.config.keys())

        if "annotation_dir" in self.config.keys(
        ) and self.config["annotation_format"] == "pascal_voc":
            voc_names_path = list(
                Path(self.config['annotation_dir']).glob('*.names'))[0]

            with open(voc_names_path, 'r') as f:
                voc_names = [n.rstrip('\n') for n in f.readlines()]

            for i, cat in enumerate(voc_names, 1):
                if cat not in self.class_dictionary.keys():
                    self.class_dictionary[cat] = i
                    self.classes += 1

    def get_builder_logger(self):
        """
            Method to return builder's logger.
        """

        return self.logger

    def _get_annotations(self, annotation):
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
                        # if elem.text not in self.class_dictionary.keys():
                        #    self.class_dictionary[elem.text] = self.classes
                        #    self.classes += 1
                        if elem.text not in class_bnd_box["classes"].keys():
                            class_bnd_box["classes"][elem.text] = []
                        current_tag = elem.text
                    if elem.tag == "bndbox":
                        for coord in elem:
                            bnd_dict[coord.tag] = int(coord.text)
                        class_bnd_box["classes"][current_tag].append(bnd_dict)
        return class_bnd_box

    def _generate_mask_for_annotation(self, annotation):
        """
            Method to generate class-wise binary mask from the bounding boxes of each class  (including BG)
        """

        current_image_class_data_dict = self._get_annotations(annotation)
        annotation_mask = np.zeros(
            (current_image_class_data_dict["size"]["height"],
             current_image_class_data_dict["size"]["width"],
             self.classes),
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
            annotation_mask[:, :, self.class_dictionary[cat]] = ann_cl
        annotation_mask[:, :, 0] = ann_bg
        return annotation_mask

    def _add_operation(self, pipeline):
        """
            Adds operation to sphinx pipeline. Dynamic module loading.
        """

        for operation in self.operations:
            if "operation_module" not in operation:
                operation_module = "sphinx.augmentation"
            else:
                operation_module = operation["operation_module"]

            try:
                module = importlib.import_module(operation_module)
            except BaseException:
                raise ModuleNotFoundError(
                    "\"{0}\" module not found".format(operation_module))
            try:
                operation_class = getattr(module, operation["operation"])
            except BaseException:
                raise OperationNotFoundOrImplemented(
                    operation_module, operation["operation"])

            OperationInstance = operation_class(**operation["args"])
            pipeline.add_operation(OperationInstance)
        return pipeline

    def _image_mask_pair_list_factory(self):
        """
            Method that creates a list of pair of image files with its
            respective masks
            TODO: Implement name checks and verification
        """
        _image_mask_pair = []

        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(
                "Input folder not found in the directory {}".format(
                    self.input_dir))

        image_list = sorted(os.listdir(self.input_dir))
        image_list = list(filter(lambda x: x.split(
            '.')[-1] in self._image_extension_list, image_list))

        if ("mask_dir" in self.__dict__.keys() and
                "annotation_dir" in self.__dict__.keys()):

            mask_list = os.listdir(self.mask_dir)
            mask_list = list(filter(lambda x: x.split(
                '.')[-1] in self._image_extension_list, mask_list))
            annotation_list = os.listdir(self.annotation_dir)
            annotation_list = list(filter(lambda x: x.split(
                '.')[-1] in self._annotation_extension_list, annotation_list))

            mask_list.sort()
            annotation_list.sort()
            for image, mask, annotation in zip(
                    image_list, mask_list, annotation_list):
                if image.split('.')[-1] in self._image_extension_list and \
                    mask.split('.')[-1] in self._image_extension_list and \
                        annotation.split('.')[-1] in ["xml"]:
                    image_mask_annotation_dict = {
                        "image": join(self.input_dir, image),
                        "mask": join(self.mask_dir, mask),
                        "annotation": join(self.annotation_dir, annotation)
                    }
                    _image_mask_pair.append(image_mask_annotation_dict)
            return _image_mask_pair

        if "mask_dir" in self.__dict__.keys():
            mask_list = sorted(os.listdir(self.mask_dir))
            mask_list = list(filter(lambda x: x.split(
                '.')[-1] in self._image_extension_list, mask_list))
            for image, mask in zip(image_list, mask_list):
                if image.split('.')[-1] in self._image_extension_list and \
                        mask.split('.')[-1] in self._image_extension_list:
                    image_mask_dict = {
                        "image": join(self.input_dir, image),
                        "mask": join(self.mask_dir, mask),
                        "annotation": None
                    }
                    _image_mask_pair.append(image_mask_dict)
            return _image_mask_pair

        if "annotation_dir" in self.__dict__.keys():
            annotation_list = sorted(os.listdir(self.annotation_dir))
            annotation_list = list(filter(lambda x: x.split(
                '.')[-1] in self._annotation_extension_list, annotation_list))
            for image, annotation in zip(image_list, annotation_list):
                if image.split('.')[-1] in self._image_extension_list and \
                        annotation.split('.')[-1] in ["xml"]:
                    image_annotation_dict = {
                        "image": join(self.input_dir, image),
                        "mask": None,
                        "annotation": join(self.annotation_dir, annotation)
                    }
                    _image_mask_pair.append(image_annotation_dict)
            return _image_mask_pair

    def _image_list_factory(self):
        """
            Method that creates a list of image files
        """
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(
                "Input folder not found in the directory {}".format(
                    self.input_dir))
        input_data_list = [
            {
                "image": join(self.input_dir, filename),
                "mask": None,
                "annotation": None
            }
            for filename in os.listdir(self.input_dir)]
        return input_data_list

    def _check_and_populate_path(self):
        """
            Method to return list of all image paths, annotation paths (if given), and mask paths (if given)
        """

        if ("mask_dir" in self.config.keys() or
                "annotation_dir" in self.config.keys()):
            data_path_list = self._image_mask_pair_list_factory()
        else:
            data_path_list = self._image_list_factory()
        return data_path_list

    def _load_entities(self, data_sample_list):
        """
            Method to return images, annotations and masks as SphinxData objects for the given paths
        """

        image_mask_list = []
        for data_dict in data_sample_list:
            sd = SphinxData()
            sd.name = Path(data_dict["image"]).stem
            sd.image = Image.open(data_dict["image"])
            if data_dict["mask"] is not None:
                sd.mask = Image.open(data_dict["mask"])
            if data_dict["annotation"] is not None:
                xmlobject = ET.parse(data_dict["annotation"])
                sd.annotation_mask = self._generate_mask_for_annotation(
                    xmlobject)
                sd.annotation = xmlobject
            image_mask_list.append(sd)
            del sd
        return image_mask_list

    def _save_entities_to_disk(self, entities):
        """
            Method to save the SphinxData objects to the output directory
            TODO: Parallelize saving the images to disk
        """

        for ets in entities:
            filename = str(uuid.uuid4())
            ets.image.save(
                join(
                    self.output_dir,
                    "images") +
                "/{}.png".format(filename))
            if ets.mask is not None:
                ets.mask.save(
                    join(
                        self.output_dir,
                        "masks") +
                    "/{}.png".format(filename))
            if ets.annotation is not None:
                image_dir = join(self.output_dir, "images")
                ets.annotation = change_pascal_annotation(
                    ets.annotation, image_dir, filename=filename + ".png")
                ets.annotation.write(
                    open(
                        join(
                            self.output_dir,
                            "annotations") +
                        "/{}.xml".format(filename),
                        'a'),
                    encoding='unicode')
            if self.save_annotation_mask:
                np.save(
                    join(
                        self.output_dir,
                        "annotation_mask") +
                    "/{}.npy".format(filename), ets.annotation_mask)

    def calculate_and_set_generator_params(
            self, batch_size=None, internal_batch=None):
        """
            Method to set the batch size, internal batch size (for batch ingestion) and
            calculate sample factor count. Should be called before calling process_and_generate
            or get_keras_generator.
        """

        self.setting_generator_params = True

        if self.batch_ingestion:
            if batch_size is None and internal_batch is None:
                raise ValueError(
                    '''Provide batch size or internal batch as batch_ingestion
                    mode is set to True''')

            elif batch_size is None:
                self.sample_factor = math.ceil(
                    self.data_len / internal_batch)
                self.batch_size = math.ceil(self.sample / self.sample_factor)
                self.internal_batch = internal_batch

            elif internal_batch is None:
                self.sample_factor = math.ceil(self.sample / batch_size)
                self.internal_batch = math.ceil(
                    self.data_len / self.sample_factor)
                self.batch_size = batch_size

            else:

                self.sample_factor = math.ceil(
                    self.data_len / internal_batch)
                self.batch_size = batch_size
                self.internal_batch = internal_batch

        else:
            if batch_size is None:
                raise ValueError("Batch size cannot be none !!")
            self.batch_size = batch_size
            self.sample_factor = math.ceil(self.sample / self.batch_size)

    def _augment_data_batch(self, data_list, batch_size):

        entities = self._load_entities(data_list)
        if self.debug:
            self.logger.info(
                "Entities num: {}".format(
                    len(entities)))

        pipeline = _DataPipeline(
            entities=entities, shuffle=self.shuffle)
        pipeline = self._add_operation(pipeline=pipeline)
        result_entities = pipeline.sample_for_generator(
            batch_size=batch_size)
        del pipeline
        return result_entities

    def process_and_generate(self, infinite_generator=False):
        """
           Process the images and yields the results in batches.
           NOTE : On batch ingestion mode if both internal batch and output
           batch size is given, sample number cannot be achieved.
        """

        if not self.setting_generator_params:
            raise Exception(
                "Did you call calculate_and_set_generator_params method ?")

        data_path_list = self._check_and_populate_path()
        if self.debug:
            self.logger.info("sample factor : {}".format(self.sample_factor))
            self.logger.info("batch size : {}".format(self.batch_size))

        if self.batch_ingestion:
            if self.debug:
                self.logger.info(
                    "internal batch : {}".format(
                        self.internal_batch))

            if not infinite_generator:
                sample_count = 0
                for i in cycle(range(0, self.data_len, self.internal_batch)):
                    if sample_count == self.sample:
                        break
                    if self.debug:
                        self.logger.info("val : {}".format(i))

                    data_list = data_path_list[i:(i + self.internal_batch)]
                    out_batch = min(
                        self.sample - sample_count, self.batch_size)
                    result_entities = self._augment_data_batch(
                        data_list, out_batch)

                    sample_count += out_batch
                    yield result_entities
            else:
                if self.batch_size is None:
                    raise ValueError("Batch Size not found")
                while True:
                    indexlist = random.sample(
                        range(0, self.data_len), self.internal_batch)
                    data_list = [data_path_list[i] for i in indexlist]
                    result_entities = self._augment_data_batch(
                        data_list, self.batch_size)
                    yield result_entities

        else:
            result_entities = self._augment_data_batch(
                data_path_list, self.sample)
            if not infinite_generator:
                for i in range(0, self.sample, self.batch_size):
                    yield result_entities[i:(i + self.batch_size)]
            else:
                while True:
                    indexlist = random.sample(
                        range(0, self.sample), self.batch_size)
                    results = [result_entities[i] for i in indexlist]
                    yield results

    def process_and_save(self, batch_save_size=None, internal_batch_size=None):
        """
            Process the files and save to disk
        """

        if not self.batch_ingestion:
            data_path_list = self._check_and_populate_path()
            entities = self._load_entities(data_path_list)
            pipeline = _DataPipeline(entities=entities, shuffle=self.shuffle)
            pipeline = self._add_operation(pipeline)
            result_entities = pipeline.sample(self.sample)
            self._save_entities_to_disk(result_entities)
        else:
            self.calculate_and_set_generator_params(
                batch_size=batch_save_size, internal_batch=internal_batch_size)
            image_generator = self.process_and_generate(
                infinite_generator=False)
            pbar = tqdm(total=self.sample_factor)
            while True:
                try:
                    result_entities = next(image_generator)
                    self._save_entities_to_disk(result_entities)
                    pbar.update(1)
                except StopIteration:
                    break

    def get_keras_generator(self, batch_size=None, internal_batch=None, input_func=None,
                            output_func=None, task="classification"):
        """
            Method to return a Keras generator. If internal batch is not set, batch size value is used
        """

        if not self.setting_generator_params:
            raise Exception(
                "Did you call calculate_and_set_generator_params method ?")

        if not batch_size:
            batch_size = self.batch_size

        if not internal_batch:
            if self.batch_ingestion:
                internal_batch = self.internal_batch
            else:
                internal_batch = batch_size

        return _KerasGenerator(builder=self, internal_batch=internal_batch, batch_size=batch_size,
                               input_func=input_func, output_func=output_func, task=task)
