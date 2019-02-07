import numpy as np

class SanityEncoder:
    def __init__(self, size):
        self.size = size


    def encode(self, x):
        # Represent by encoder of size+4, with on bits in 5x5 square around coordinate
        indices = np.zeros((self.size,))
        indices[x:x+1] = 1
        flat= indices.flatten()

        #todo temp hack
        flat = np.repeat(flat,20)
        return flat

