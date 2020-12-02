# -*- coding: utf-8 -*-
# Contributors : [ashok.ramadass@toyotaconnected.com, srinivas.v@toyotaconnected.co.in, ]

import os
from os.path import join
import json
import importlib
import re
from .pipeline import DataPipeline
from PIL import Image
from ..utils.CustomExceptions import CrucialValueNotFoundError, OperationNotFoundOrImplemented, ConfigurationError
import numpy as np
from tqdm import tqdm
import random
import uuid

class Builder(object):
    '''
        TODO : Image checks and exception handling, filter all the files that are not of acceptable format ( png, jpg, bmp, jpeg )
        TODO : Parallelize saving images to disk

        Builder class to create augmentor Pipeline object
        Config file is a json map used to define Operations and their properties
        {
            "input_dir" : "images",
            "output_dir" : "output",
            "mask_dir" : "mask"
            "sample" : 5000,
            "multi_threaded" : true,
            "run_all" : false,
            "batch_ingestion": true,
            "internal_batch": 20,
            "operations":[
                {
                    "operation": "DarkenScene",
                    "operation_module" : "sphinx.augmentation",
                    "args": {
                        "probability": 0.7,
                        "darkness" : 0.5,
                        "is_mask" : true,
                        "label" : 2,
                    }
                },
                {
                    "operation": "Equalize",
                    "operation_module" : "sphinx.augmentation",
                    "args": {
                        "probability": 0.5,
                    }
                },
            ]
        }
    '''

    def __init__(self, config_json="config.json"):
        if not os.path.exists(config_json):
            raise FileNotFoundError("{} not found".format(config_json))

        with open(config_json) as config_file:
            self.config = json.load(config_file)

        if "operations" not in self.config.keys():
            raise CrucialValueNotFoundError(
                operation="augmentation configurations",
                value_type="operations")

        if "input_dir" not in self.config.keys():
            raise CrucialValueNotFoundError(
                operation="augmentation configurations",
                value_type="input_dir")

        if "output_dir" not in self.config.keys():
            self.output_dir = join(self.config["input_dir"], "output")
            try:
                os.mkdir(self.output_dir)
                os.mkdir(join(self.output_dir, "images"))
                os.mkdir(join(self.output_dir, "masks"))
            except FileExistsError:
                pass
        
        if "output_dir" in self.config.keys():
            self.output_dir = self.config["output_dir"]
            try:
                os.mkdir(join(self.output_dir, "images"))
                os.mkdir(join(self.output_dir, "masks"))
            except FileExistsError:
                pass

        if not os.path.exists(self.config["input_dir"]):
            raise FileNotFoundError("{} not found", self.config["input_dir"])

        self.data_len = len(os.listdir(self.config["input_dir"]))

        if self.data_len == 0:
            raise FileExistsError("No files found in the input directory")

        if "mask_dir" in self.config.keys() and len(os.listdir(self.config["mask_dir"])) == 0:
            raise FileNotFoundError("No files found in mask directory")

        if "output_dir" not in self.config.keys():
            self.output_dir = "output"

        if "sample" not in self.config.keys():
            self.sample = self.data_len

        if "run_all" not in self.config.keys():
            self.run_all = False

        if "multi_threaded" not in self.config.keys():
            self.multi_threaded = False

        if "batch_ingestion" not in self.config.keys():
            self.batch_ingestion = False
        
        if "internal_batch" not in self.__dict__.keys():
            self.internal_batch = None

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
                'batch_ingestion',
                'internal_batch') if key in self.config.keys())


    def _add_operation(self,pipeline):
        '''
            Adds operation to sphinx pipeline. Dynamic module loading.
        '''
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
        '''
            Function that create a list of pair of image files with its respective masks
        '''
        _image_mask_pair = []
    
        if not os.path.exists(self.mask_dir):
            raise FileExistsError("Mask folder not found in the directory {}".format(self.mask_dir))
        for filename in os.listdir(self.input_dir):
            r = re.compile(filename.split('.')[0])
            filematch = [mask for mask in list(filter(r.match, os.listdir(self.mask_dir)))]
            if len(filematch) == 1: 
                _image_mask_pair.append([join(self.input_dir,filename),join(join(self.mask_dir, filematch[0]))])
            elif len(filematch > 1):
                raise ConfigurationError("More than 1 mask image found for the image " + filename)
        return _image_mask_pair

    def _image_list_factory(self):
        '''
            Function that create a list of pair of image files with its respective masks 
        '''
        if not os.path.exists(self.input_dir):
            raise FileExistsError("Input folder not found in the directory {}".format(self.mask_dir))
        
        input_data_list = [join(self.input_dir, filename) for filename in os.listdir(self.input_dir)]
        return [input_data_list]

    def _check_and_populate_path(self):
        if "mask_dir" in self.config.keys():
            data_path_list = self._image_mask_pair_list_factory()
        else:
            data_path_list = self._image_list_factory()
        return data_path_list

    def _load_images(self):
        data_path_list = self._check_and_populate_path()
        images = [[np.array(Image.open(y)) for y in x] for x in data_path_list]
        return images

    def _save_images_to_disk(self, images):
        '''
            TODO: Parallelize saving the images to disk
        ''' 
        print(len(images))
        for ims in images:
            filename = str(uuid.uuid4())
            image = Image.fromarray(ims[0])
            image.save(join(self.output_dir, "images")+"/{}.png".format(filename))
            if len(ims) == 2:
                mask = Image.fromarray(ims[1])
                mask.save(join(self.output_dir, "masks")+"/{}.png".format(filename))

    def calculate_and_set_generator_params(self, batch_size=None):
        if batch_size is None and self.internal_batch is None:
                raise ValueError("Provide batch size or internal batch as batch_ingestion mode is set to \"True\"")

        elif batch_size is None:
            self.sample_factor = self.data_len // self.internal_batch
            self.batch_size = self.sample // self.sample_factor
            self.internal_batch_split = self.internal_batch

        elif self.internal_batch is None:
            self.sample_factor = self.sample // self.batch_size
            self.internal_batch_split = self.data_len // self.sample_factor
            self.batch_size = batch_size

        else:
            if batch_size < self.internal_batch:
                raise ValueError("Batch size cannot be greater than internal batch split")
            self.sample_factor = self.data_len // self.internal_batch
            self.internal_batch_split = self.internal_batch
            self.batch_size = self.batch_size

            
    def process_and_generate(self, batch_size=None, infinite_generator=False):
        '''
           Process the images and yields the results in batches. 
           NOTE : If both internal batch and output batch size is given, sample number cannot be achieved.
        '''

        data_path_list = self._check_and_populate_path()
        
        if not infinite_generator:
            for i in range(self.sample_factor):
                images = [[np.array(Image.open(y)) for y in x] for x in data_path_list[i:(i+1)*(self.internal_batch_split+1) - 1]]
                pipeline = DataPipeline(images=images)
                pipeline = self._add_operation(pipeline=pipeline)
                images = pipeline.generator(batch_size=self.batch_size)
                del pipeline # clear pipeline memory
                yield images
        
        else:
            if batch_size is None:
                raise ValueError("Batch Size not found")
            while True:
                indexlist = random.sample(range(0, self.data_len), batch_size)
                sample_data_path = [data_path_list[i] for i in indexlist]
                images = [[np.array(Image.open(y)) for y in x] for x in sample_data_path]
                pipeline = DataPipeline(images=images)
                pipeline = self._add_operation(pipeline=pipeline)
                images = pipeline.generator(batch_size=batch_size)
                del pipeline #clear pipeline memory
                yield images
        
            
    def process_and_save(self, batch_size=None):
        '''
            Process the files and save to disk
        '''
        if not self.batch_ingestion:
            images = self._load_images()
            pipeline = DataPipeline(images=images)
            pipeline = self._add_operation(pipeline)
            images = pipeline.sample(self.sample)
            self._save_images_to_disk(images)
            
        else:
            self.calculate_and_set_generator_params(batch_size=batch_size)
            image_generator = self.process_and_generate(infinite_generator=False)
            pbar = tqdm(total = self.sample_factor)
            while True:
                try:
                    images = next(image_generator)
                    self._save_images_to_disk(images)
                    pbar.update(1)
                except StopIteration:
                    break