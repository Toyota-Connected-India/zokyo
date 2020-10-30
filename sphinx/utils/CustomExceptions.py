class CoefficientNotinRangeError(Exception):
    def __init__(self, coefficient, coefftype="Default", rangeMin = 0, rangeMax = 1):
        self.rangeMax = rangeMax
        self.rangeMin = rangeMin
        self.coefftype = coefftype
        self.coefficient = coefficient
        super().__init__()

    def __str__(self):
        return "{0} coefficient of value {1} is not in range {2} and {3}".format(self.coefftype,self.coefficient, self.rangeMin, self.rangeMax)

class InvalidImageArrayError(Exception):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Image is not a numpy ndarray"
