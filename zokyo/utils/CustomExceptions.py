# -*- coding: utf-8 -*-
# Contributors : [srinivas.v@toyotaconnected.co.in,srivathsan.govindarajan@toyotaconnected.co.in,
# harshavardhan.thirupathi@toyotaconnected.co.in,
# ashok.ramadass@toyotaconnected.com ]

class CoefficientNotinRangeError(Exception):
    """
        Class to throw exception when a coefficient is not in the specified range
    """

    def __init__(self, coefficient, coeff_type="Default",
                 range_min=0, range_max=1):
        self.range_max = range_max
        self.range_min = range_min
        self.coeff_type = coeff_type
        self.coefficient = coefficient
        super().__init__()

    def __str__(self):
        return '''\"{0}\" coefficient of value {1} is not in range {2} and {3}'''.format(
            self.coeff_type, self.coefficient, self.range_min, self.range_max)


class InvalidImageArrayError(Exception):
    """
        Class to throw exception when an image is not valid
    """

    def __init__(self, image_type="PIL"):
        self.image_type = image_type
        super().__init__()

    def __str__(self):
        return "Image is not a {} Image".format(self.image_type)


class CrucialValueNotFoundError(Exception):
    """
        Class to throw exception when an expected value is not found
    """

    def __init__(self, operation, value_type="sample"):
        self.value_type = value_type
        self.operation = operation
        super().__init__()

    def __str__(self):
        return "\"{0}\" value not found for the \"{1}\" mentioned".format(
            self.value_type, self.operation)


class OperationNotFoundOrImplemented(Exception):
    """
        Class to throw exception when an operation is not found
    """

    def __init__(self, module, class_name):
        self.module = module
        self.class_name = class_name
        super().__init__()

    def __str__(self):
        return "\"{0}\" not found or implemented in the module \"{1}\"".format(
            self.class_name, self.module)


class ConfigurationError(Exception):
    """
        Class to throw exception when a configuration is not right
    """

    def __init__(self, exception_string) -> None:
        self.exception_string = exception_string
        super().__init__()

    def __str__(self) -> str:
        return self.exception_string
