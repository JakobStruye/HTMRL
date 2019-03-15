import numpy as np

class MazeEncoder:
    def __init__(self, maze_size):
        self.maze_size = maze_size


    def encode(self, x, y):
        # Represent by encoder of size+4, with on bits in 5x5 square around coordinate
        indices = np.zeros((self.maze_size, self.maze_size))
        indices[x:x+1,y:y+1] = 1
        flat= indices.flatten()

        #todo temp hack
        flat = np.repeat(flat,20)
        return flat

