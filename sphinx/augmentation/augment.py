# -*- coding: utf-8 -*-
# Contributors : [ashok.ramadass@toyotaconnected.com,
# srinivas.v@toyotaconnected.co.in, ]

import Augmentor
import cv2
from Augmentor.Operations import Operation
import os
import numpy as np
import warnings
import json
import importlib
from .operations import ColorEqualize
from ..utils.CustomExceptions import CrucialValueNotFoundError, OperationNotFoundOrImplemented


class Builder(object):
    '''
        Builder class to create augmentor Pipeline object
        Config file is a json map used to define Operations and their properties
        {
            "input_dir" : "images",
            "output_dir" : "output",
            "sample" : 5000,
            "multi_threaded" : true,
            "run_all" : false,
            "type" : "sphinx.augmentation | Augmentor.Operations"
            "operations":[
                {
                    "operation": "DarkenScene",
                    "args": {
                        "probability": 0.7,
                        "coefficient" : 0.5
                    }
                },
                {
                    "operation": "Equalize",
                    "args": {
                        "probability": 0.5
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

        self.__dict__.update(
            (key,
             self.config[key]) for key in (
                'input_dir',
                'sample',
                'output_dir',
                'run_all',
                'multi_threaded',
                'operations',
                'operation_module') if key in self.config.keys())

    def build_and_run(self):
        pipeline = Augmentor.Pipeline(
            source_directory=self.input_dir, output_directory=self.output_dir)

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

        if self.run_all:
            pipeline.process()
        else:
            pipeline.sample(self.sample, multi_threaded=self.multi_threaded)
