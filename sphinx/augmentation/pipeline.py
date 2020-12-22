from PIL import Image
import numpy as np
from Augmentor.Operations import Operation
import random
from tqdm import tqdm


class DataPipeline():
    def __init__(self, data_dictionary):
        self.augmentor_images = data_dictionary
        self.operations = []

    def add_operation(self, operation):
        if isinstance(operation, Operation):
            self.operations.append(operation)
        else:
            raise TypeError("Must be of type Augmentor Operation or a Sphinx Operation to be added to the pipeline.")

    def remove_operation(self, operation_index=-1):
        self.operations.pop(operation_index)

    def sample_for_generator(self, batch_size=1):
        batch_size = 1 if (batch_size < 1) else batch_size
        batch = []
        for _ in range(0, batch_size):
            index = random.randint(0, len(self.augmentor_images) - 1)
            entities_to_yield = [x for x in self.augmentor_images[index]]
            for operation in self.operations:
                r = round(random.uniform(0, 1), 1)
                if r <= operation.probability:
                    entities_to_yield = operation.perform_operation(entities_to_yield)

            entities_to_yield = [x for x in entities_to_yield]
            batch.append(entities_to_yield)
            return batch

    def sample(self, n):
        batch = []
        for _ in tqdm(range(0, n)):
            index = random.randint(0, len(self.augmentor_images) - 1)
            entities_to_return = [x for x in self.augmentor_images[index]]
            for operation in self.operations:
                r = round(random.uniform(0, 1), 1)
                if r <= operation.probability:
                    entities_to_return = operation.perform_operation(entities_to_return)

            entities_to_return = [x for x in entities_to_return]
            batch.append(entities_to_return)
            return batch
