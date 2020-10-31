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

class Builder(object):
    '''
        Builder class to create augmentor Pipeline object
    '''
    def __init__(self, input_dir, **kwargs):
        self.input_dir = input_dir
        if kwargs["output_dir"] == None:
            self.output_dir = input_dir + "/output"
        if kwargs is not None:
            self.__dict__.update((key, kwargs[key]) for key in ('sample','output_dir', 'run_all', 'multi_threaded') if key in kwargs)
    
    def build_and_run(self, config_json):
        '''
        Config file is a json map used to define Operations and their properties
        {
            "operations":[
                {
                    "operation": "DarkenScene",
                    "args": {
                        "probability": 0.7,
                        "coefficient" : 0.5
                    }
                }
            ]
        }
        '''

        if not os.path.exists(config_json):
            raise FileNotFoundError("{} not found".format(config_json))
        
        with open(config_json) as config_file:
            self.config = json.load(config_file)
       
        pipeline = Augmentor.Pipeline(self.input_dir, output_directory=self.output_dir)
        module = importlib.import_module('augmentation')
       
        for operation in self.config["operations"]:
            Operation = getattr(module, operation["operation"])
            OperationInstance = Operation(operation["args"])
            pipeline.add_operation(OperationInstance)
        
        if self.run_all:
            pipeline.process()
        else:
            pipeline.sample(self.sample, multi_threaded=self.multi_threaded)
        



            



        


        

        



