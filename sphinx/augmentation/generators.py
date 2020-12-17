from os import pipe
import cv2
from tensorflow.keras.utils import Sequence
import math
import numpy as np
import random
import warnings

class SegmentationGenerator(Sequence):
    def __init__(self, pipeline, batch_size, internal_batch) -> None:
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.internal_batch = internal_batch
        super(SegmentationGenerator, self).__init__()

        
    
    