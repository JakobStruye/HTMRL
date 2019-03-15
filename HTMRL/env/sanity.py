import numpy as np


class Sanity:
    def __init__(self, env_config):
        size = env_config["size"]
        self.size = size
        self.arms = env_config["arms"]
        self.current = 0
        self.optimals = np.random.choice(self.arms,self.size)


    def get_state(self):
        return self.current

    def do_action(self, action):
        reward = 1. if self.optimals[self.current] == action else -1.0
        self.current = np.random.randint(self.size, size=1)[0]
        return self.current, reward

    def is_done(self):
        return False

    def get_action_count(self):
        return self.arms

    def get_debug_info(self):
        return None

    def get_all_states(self):
        return[np.array([loc,]) for loc in range(self.size)]
