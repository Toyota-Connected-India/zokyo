# -*- coding: utf-8 -*-
# Contributors : [ashok.ramadass@toyotaconnected.com, srinivas.v@toyotaconnected.co.in, ]

from os import pipe
from os.path import join
import Augmentor
import os
import json
import importlib
import re
from Augmentor.Pipeline import Pipeline, DataPipeline
from PIL import Image
from numpy.lib.function_base import percentile
from ..utils.CustomExceptions import CrucialValueNotFoundError, OperationNotFoundOrImplemented, ConfigurationError
import numpy as np

class Builder(object):
    '''
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

        if not os.path.exists(self.config["input_dir"]):
            raise FileNotFoundError("{} not found", self.config["input_dir"])

        if "output_dir" not in self.config.keys():
            self.output_dir = "output"

        if "sample" not in self.config.keys():
            self.sample = len(os.listdir(self.input_dir))

        if "run_all" not in self.config.keys():
            self.run_all = False

        if "multi_threaded" not in self.config.keys():
            self.multi_threaded = False

        if "batch_ingestion" not in self.config.keys():
            self.batch_ingestion = False
        
        self.pipeline = None

        self.__dict__.update(
            (key,
             self.config[key]) for key in (
                'input_dir',
                'sample',
                'output_dir',
                'run_all',
                'multi_threaded',
                'operations',
                'mask_dir'
                'batch_ingestion',
                'internal_batch',
                'operation_module') if key in self.config.keys())

        if self.batch_ingestion and "internal_batch" not in self.__dict__.keys():
            self.internal_batch = 1
        

    def _add_operation(self,pipeline):
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
                _image_mask_pair.append([os.path.join(self.input_dir,filename),os.path.join(os.path.join(self.mask_dir, filematch[0]))])
            elif len(filematch > 1):
                raise ConfigurationError("More than 1 mask image found for the image " + filename)
        return _image_mask_pair

    def _image_list_factory(self):
        '''
            Function that create a list of pair of image files with its respective masks 
        '''
        if not os.path.exists(self.input_dir):
            raise FileExistsError("Input folder not found in the directory {}".format(self.mask_dir))
        
        input_data_list = [os.path.join(self.input_dir, filename) for filename in os.listdir(self.input_dir)]
        return [input_data_list]
        

    def _generator_pipeline(self, batch_size):
        if "mask_dir" in self.config.keys():
            data_path_list = self._image_mask_pair_list_factory()
        else:
            data_path_list = self._image_list_factory()

        self.sample_factor = self.sample // batch_size
        internal_batch_split = len(data_path_list) // self.sample_factor
        for i in range(self.sample_factor):
            images = [[Image.open(y) for y in x] for x in data_path_list[i:(i+1)*(internal_batch_split+1) - 1]]
            pipeline = Augmentor.DataPipeline(images=images)
            pipeline = self._add_operation(pipeline=pipeline)
            yield pipeline.sample(batch_size)
    
    def _load_images(self):
        if "mask_dir" in self.config.keys():
            data_path_list = self._image_mask_pair_list_factory()
        else:
            data_path_list = self._image_list_factory()
        images = [[Image.open(y) for y in x] for x in data_path_list]
        return images





    def process_and_save(self):
        '''
            Build pipeline object
        '''
        if not self.batch_ingestion:
            pipeline = Augmentor.DataPipeline(
                source_directory=self.input_dir, output_directory=self.output_dir)
            pipeline, mask_operations = self._add_operation(pipeline)

            if self.run_all:
                pipeline.process()
            else:
                pipeline.sample(self.sample, multi_threaded=self.multi_threaded)

        else:
            pipeline = Augmentor.DataPipeline(

            )
        
        

