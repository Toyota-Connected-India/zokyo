class CoefficientNotinRangeError(Exception):
    def __init__(self, coefficient, coeff_type="Default", range_min = 0, range_max = 1):
        self.range_max = range_max
        self.range_min = range_min
        self.coeff_type = coeff_type
        self.coefficient = coefficient
        super().__init__()

    def __str__(self):
        return "{0} coefficient of value {1} is not in range {2} and {3}".format(self.coeff_type,self.coefficient, self.range_min, self.range_max)

class InvalidImageArrayError(Exception):
    def __init__(self, image_type="PIL"):
        self.image_type = image_type
        super().__init__()

    def __str__(self):
        return "Image is not a {} Image".format(self.image_type)

class CrucialValueNotFoundError(Exception):
    def __init__(self, operation, value_type="sample"):
        self.value_type = value_type
        self.operation = operation
        super.__init__()

    def __str__(self):
        return "{0} value not found for operation {1}".format(self.value_type, self.operation)