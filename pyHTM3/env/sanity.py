from enum import Enum
import time
import numpy as np
import pyglet



class Sanity():
    def __init__(self, env_config):
        size = env_config["size"]
        self.size = size
        self.current = np.array([0,])
        self.optimals = np.random.choice(4,self.size)



    def get_state(self):
        return self.current

    def do_action(self, action):
        reward = 1. if self.optimals[self.current] == action else -1.
        self.current = np.random.randint(self.size, size=1)
        return (self.current, reward)

    def is_done(self):
        return False
    def get_action_count(self):
        return 4

    def get_debug_info(self):
        return None
