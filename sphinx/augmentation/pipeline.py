from PIL import Image
import numpy as np
from Augmentor.Pipeline import Pipeline
import random
from tqdm import tqdm


class DataPipeline(Pipeline):
    def __init__(self, images, labels=None):

        # We will not use this member variable for now.
        # if output_directory:
        #    self.output_directory = output_directory
        # else:
        #    self.output_directory = None

        self.augmentor_images = images
        self.labels = labels

        self.operations = []

    ##########################################################################
    # Properties
    ##########################################################################

    # @property
    # def output_directory(self):
    #     return self._output_directory

    # @output_directory.setter
    # def output_directory(self, value):
    #     if os.path.isdir(value):
    #         self._output_directory = value
    #     else:
    #         raise IOError("The provided argument, %s, is not a directory." % value)

    @property
    def augmentor_images(self):
        return self._augmentor_images

    @augmentor_images.setter
    def augmentor_images(self, value):
        self._augmentor_images = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    ##########################################################################
    # End Properties
    ##########################################################################

    def __call__(self, augmentor_image):
        """
        Multi-threading support to be enabled in a future release of sphinx.
        """
        return self._execute(augmentor_image)

    def sample_for_generator(self, batch_size=1):

        batch_size = 1 if (batch_size < 1) else batch_size

        batch = []
        y = []

        for _ in range(0, batch_size):
            index = random.randint(0, len(self.augmentor_images) - 1)
            images_to_yield = [
                Image.fromarray(x) for x in self.augmentor_images[index]]

            for operation in self.operations:
                r = round(random.uniform(0, 1), 1)
                if r <= operation.probability:
                    images_to_yield = operation.perform_operation(
                        images_to_yield)

            images_to_yield = [np.asarray(x) for x in images_to_yield]

            if self.labels:
                batch.append(images_to_yield)
                y.append(self.labels[index])
            else:
                batch.append(images_to_yield)

        if self.labels:
            return batch, y
        else:
            return batch

    def sample(self, n):
        batch = []
        y = []

        for _ in tqdm(range(0, n)):
            index = random.randint(0, len(self.augmentor_images) - 1)
            images_to_return = [
                Image.fromarray(x) for x in self.augmentor_images[index]]

            for operation in self.operations:
                r = round(random.uniform(0, 1), 1)
                if r <= operation.probability:
                    images_to_return = operation.perform_operation(
                        images_to_return)

            images_to_return = [np.asarray(x) for x in images_to_return]

            if self.labels:
                batch.append(images_to_return)
                y.append(self.labels[index])
            else:
                batch.append(images_to_return)

        if self.labels:
            return batch, y
        else:
            return batch
