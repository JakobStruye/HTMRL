from enum import Enum
import time
import numpy as np
import pyglet



class Sanity():
    def __init__(self, env_config):
        size = env_config["size"]
        self.size = size
        self.arms = env_config["arms"]
        self.current = np.array([0,])
        self.optimals = np.random.choice(self.arms,self.size)
        #self.optimals = np.array([330,184,71,179,260,212,170,341,91,77])



    def get_state(self):
        return self.current

    def do_action(self, action):
        reward = 1. if self.optimals[self.current] == action else -1.0 #-0.
        self.current = np.random.randint(self.size, size=1)
        return (self.current, reward)

    def is_done(self):
        return False
    def get_action_count(self):
        return self.arms

    def get_debug_info(self):
        return None

    def get_all_states(self):
        return[np.array([loc,]) for loc in range(self.size)]
