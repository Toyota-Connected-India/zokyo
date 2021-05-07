from os import pipe
import cv2
from tensorflow.keras.utils import Sequence, to_categorical
import math
import numpy as np
import random
import warnings


class KerasGenerator(Sequence):
    def __init__(self, builder, internal_batch, batch_size, input_func=None, output_func=None, shuffle=False, task="classification"):
        'Initialization'
        self.batch_size = batch_size
        self.internal_batch = internal_batch
        self.builder = builder
        self.builder.shuffle = shuffle
        self.input_func = input_func
        self.output_func = output_func

        if task not in ["classification", "detection", "segmentation"]:
            raise ValueError(
                f'Expected task to be one of classification, detection, segmentation but got {task}')
        else:
            self.task = task
        
        self.builder.batch_ingestion = True
        self.builder.calculate_and_set_generator_params(
            batch_size=self.batch_size, internal_batch=self.internal_batch)
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.builder.sample) // self.batch_size

    def __getitem__(self, index):

        X, y = self.__data_generation()

        return X, y

    def on_epoch_end(self):
        self.__data_generator = self.builder.process_and_generate()

    def __data_generation(self):
        data_batch = next(self.__data_generator)

        if not self.input_func:
            self.input_func = lambda x: x
        if not self.output_func:
            self.output_func = lambda x: x
        
        X = []
        y = []
        for data in data_batch:
            X.append(self.input_func(np.array(data.image)))
            
            if self.task == "classification":
                ann = self.builder._get_annotations(data.annotation)
                y.append(self.output_func(ann))

            elif self.task == "detection":
                ann = self.builder._get_annotations(data.annotation)
                y.append(self.output_func(ann))

            elif self.task == "segmentation":
                mask = np.array(data.mask)
                y.append(self.output_func(mask))

        return np.array(X), np.array(y)
