from tensorflow.keras.utils import Sequence


class SegmentationGenerator(Sequence):
    def __init__(self, pipeline, batch_size, internal_batch) -> None:
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.internal_batch = internal_batch
        super(SegmentationGenerator, self).__init__()
