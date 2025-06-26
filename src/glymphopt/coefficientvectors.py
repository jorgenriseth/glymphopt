import numpy as np


class CoefficientVector:
    def __init__(self, default_coefficients, variable_coefficients):
        self.coefficients = default_coefficients
        self.vars = variable_coefficients

    def to_vector(self, **kwargs):
        coefficients = self.coefficients | kwargs
        return np.array([coefficients[key] for key in self.vars])

    def from_vector(self, x):
        return self.coefficients | {key: x[i] for i, key in enumerate(self.vars)}
