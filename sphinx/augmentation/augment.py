# -*- coding: utf-8 -*-
# Contributors : [ashok.ramadass@toyotaconnected.com, srinivas.v@toyotaconnected.co.in, ]

import Augmentor
import cv2
from Augmentor.Operations import Operation
import os
import numpy as np
import warnings
import json
import importlib
from .operations import ColorEqualize
from ..utils.CustomExceptions import CrucialValueNotFoundError

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

        if self.config["operations"] is None:
            raise CrucialValueNotFoundError(operation="augmentation configurations", value_type="operations")

        if self.config["input_dir"] is None:
            raise CrucialValueNotFoundError(operation="augmentation configurations", value_type="input_dir")

        if not os.path.exits(self.config["input_dir"]):
            raise FileNotFoundError("{} not found", self.config["input_dir"])
        
        if self.config["output_dir"] is None:
            self.output_dir = input_dir + "/output"
        
        if self.config["sample"] is None:
            self.sample = len(os.listdir(self.input_dir))

        if self.config["run_all"] is None:
            self.run_all = False

        if self.config["multi_threaded"] is None:
            self.multi_threaded = False
 
        self.__dict__.update((key, self.config[key]) for key in ('sample','output_dir', 'run_all', 'multi_threaded', 'operations') if key in self.config)

    def build_and_run(self):
        pipeline = Augmentor.Pipeline(self.input_dir, output_directory=self.output_dir)
        module = importlib.import_module('augmentation')
       
        for operation in self.operations:
            Operation = getattr(module, operation["operation"])
            OperationInstance = Operation(operation["args"])
            pipeline.add_operation(OperationInstance)
        
        if self.run_all:
            pipeline.process()
        else:
            pipeline.sample(self.sample, multi_threaded=self.multi_threaded)
        



            



        


        

        



