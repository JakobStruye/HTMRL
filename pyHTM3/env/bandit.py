import numpy as np

class Bandit():#gym.Env):
    def __init__(self, config):
        self.k = config["k"]
        self.arms = np.random.normal(0,1,self.k)
        #self.arms = [-3,1,2.,3]
        self.best = self.arms.argmax()
        self.counter = 0

        #self.num_envs = 1
        #self.observation_space = spaces.Discrete(1)
        #self.action_space = spaces.Discrete(k)
    def do_action(self, i):

        self.counter += 1
        if self.counter % 2000 == 0:
            np.random.shuffle(self.arms)
            self.best = self.arms.argmax()
        return (None, np.random.normal(self.arms[i], 1))
        #return 1 if i == self.best else -1
        #if  np.random.randn(1) > self.arms[i]:
        #    return 1
        #else:
        #    return -1
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