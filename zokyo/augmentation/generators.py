from tensorflow.keras.utils import Sequence
import numpy as np


class _KerasGenerator(Sequence):
    """
        Keras generator class
    """

    def __init__(self, builder, internal_batch, batch_size, input_func=None,
                 output_func=None, task="classification"):

        self.batch_size = batch_size
        self.internal_batch = internal_batch
        self.builder = builder
        self.input_func = input_func
        self.output_func = output_func
        self.data_list = self.builder._check_and_populate_path()

        if task not in ["classification", "detection", "segmentation"]:
            raise ValueError(
                f'Expected task to be one of classification, detection, segmentation but got {task}')
        else:
            self.task = task

    def __len__(self):
        return self.builder.sample // self.batch_size

    def __getitem__(self, index):

        X, y = self.__data_generation(index)

        return X, y

    def on_epoch_end(self):
        """
        TODO: Generic implementation
        """
        pass

    def __data_generation(self, i):
        """
            Method to load and augment data batches for training. Returns bounding box
            annotation when task is "classification" or "detection" and mask for "segmentation".
            Uses input and output functions provided to preprocess inputs and targets respectively.
        """

        data_list = self.data_list[i *
                                   self.internal_batch:(i +
                                                        1) *
                                   self.internal_batch]
        data_batch = self.builder._augment_data_batch(
            data_list, self.batch_size)

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

        return X, y
