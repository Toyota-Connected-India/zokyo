from .operations import Operation
import random
from tqdm import tqdm
from itertools import cycle


class _DataPipeline(object):
    """
        Class which accepts SphinxData objects to pass through different operations to perform augmentation
    """

    def __init__(self, entities, shuffle=False, **kwargs):
        self.augmentable_entities = entities
        self.operations = []
        self.shuffle = shuffle

    def add_operation(self, operation):
        """
            Method to add augmentation operations to the pipeline
        """

        if isinstance(operation, Operation):
            self.operations.append(operation)
        else:
            raise TypeError(
                '''Must be of type Augmentor Operation or a Sphinx
                Operation to be added to the pipeline.''')

    def remove_operation(self, operation_index=-1):
        """
            Method to remove an operation from the pipeline. If index is not given, the last operation is removed
        """

        self.operations.pop(operation_index)

    def sample_for_generator(self, batch_size=1):
        """
            Method to sample entities in batches and augment them to pass to the generator
        """

        batch_size = 1 if (batch_size < 1) else batch_size
        batch = []
        for index in cycle(range(0, len(self.augmentable_entities))):
            if len(batch) == batch_size:
                break
            if self.shuffle:
                index = random.randint(0, len(self.augmentable_entities) - 1)
            entities_to_yield = self.augmentable_entities[index].copy()
            for operation in self.operations:
                r = round(random.uniform(0, 1), 1)
                if r <= operation.probability:
                    entities_to_yield = operation.perform_operation(
                        entities_to_yield)
            batch.append(entities_to_yield)
        return batch

    def sample(self, batch_size):
        """
            Method to sample entities in batches and augment them. Used by proccess_and_save and displays progress bar
        """

        batch = []
        for index in tqdm(
                cycle(range(0, len(self.augmentable_entities))), total=batch_size):
            if len(batch) == batch_size:
                break
            if self.shuffle:
                index = random.randint(0, len(self.augmentable_entities) - 1)
            entities_to_return = self.augmentable_entities[index].copy()
            for operation in self.operations:
                r = round(random.uniform(0, 1), 1)
                if r <= operation.probability:
                    entities_to_return = operation.perform_operation(
                        entities_to_return)
            batch.append(entities_to_return)
        return batch
