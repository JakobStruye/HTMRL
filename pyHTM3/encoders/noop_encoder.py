import numpy as np

class NoopEncoder:
    def __init__(self, size):
        self.size = size


    def encode(self, x):
        return np.ones(self.size, dtype=np.bool)

