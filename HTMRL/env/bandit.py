import numpy as np


class Bandit:
    def __init__(self, config):
        self.k = config["k"]
        self.arms = np.random.normal(0,1,self.k)
        self.best = self.arms.argmax()
        self.counter = 0
        self.shuffle = False if "shuffle" not in config else False if config["shuffle"] == 0 else True

    def do_action(self, i):

        self.counter += 1
        if self.counter % 2000 == 0:
            if not self.shuffle:
                self.arms = np.random.normal(0,1,self.k)
            else:
                np.random.shuffle(self.arms)
            self.best = self.arms.argmax()
        return (None, np.random.normal(self.arms[i], 1))

    def is_best(self, k):
        return k == self.best

    def step(self, action):
        rew = self.get_reward(action)
        return [0], rew, np.array([True]), {}

    def reset(self):
        return [0]

    def get_action_count(self):
        return self.k

    def get_debug_info(self):
        return self.arms

    def get_state(self):
        return None
