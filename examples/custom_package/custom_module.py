# -*- coding: utf-8 -*-

from zokyo.augmentation.operations import Operation, ArgsClass

class CustomOp(Operation):
    """
    Custom operation class which should inherit base Operation class and use ArgsClass to parse arguments from Zokyo
    """

    def __init__(self, **kwargs):
        self.args = ArgsClass(**kwargs)
        Operation.__init__(self, self.args.probability)
    
    def perform_operation(self, entities):
        # dummy custom operation
        return entities
