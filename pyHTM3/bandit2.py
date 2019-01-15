import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pyHTM3.spatial_pooler as spatial_pooler
#import gym
#from gym import spaces
#import os
#np.random.seed(int(os.environ["MYRANDSEED"]))

class Bandit():#gym.Env):
    def __init__(self, k):
        self.k = k
        self.arms = np.random.normal(0,1,k)
        #self.arms = [-3,1,2.,3]
        self.best = self.arms.argmax()
        self.counter = 0

        #self.num_envs = 1
        #self.observation_space = spaces.Discrete(1)
        #self.action_space = spaces.Discrete(k)
    def get_reward(self, i):

        self.counter += 1
        if self.counter % 2000 == 0:
            np.random.shuffle(self.arms)
            self.best = self.arms.argmax()
        return np.random.normal(self.arms[i], 1)
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



def run_greedy(k, eps, steps):
    b = Bandit(k)
    rews = []
    sample_avgs = np.zeros((k,))
    sample_counts = np.zeros((k,))
    for step in range(steps):
        if np.random.uniform(0.,1.) <= eps:
            selection = np.random.randint(0,k)
        else:
            selection = np.argmax(sample_avgs)
        rew = b.get_reward(selection)
        sample_avg = sample_avgs[selection]
        sample_count = sample_counts[selection]
        sample_avgs[selection] = (sample_count * sample_avg + rew) / (sample_count + 1)
        sample_counts[selection] += 1
        rews.append(rew)
    return np.array(rews)

def repeat_greedy(k, eps, steps, repeats):
    avg_rews = np.zeros((steps,))
    for i in range(repeats):
        new_rews = run_greedy(k,eps,steps)
        avg_rews = (i * avg_rews + new_rews) / (i+1)
    return avg_rews

#for eps in [0.1,0.01,0.0]:
#    results = repeat_greedy(10, eps, 1000, 2000)
#    print(results.shape)
#    plt.plot(range(1000), results)
#plt.show()


def encoding_to_action(encoding, actions, i=1):
    buckets = np.floor(encoding / (2050. / actions))
    buckets = buckets.astype(np.int32)
    counts = np.bincount(buckets)
    #print(counts)
    #if i%200 == 0:
    #    print(counts)
    return counts.argmax()

if __name__ == "__main__":
    input_size = (60,)
    input_sparsity = 0.1
    k = 10

    fixed_input_indices = np.random.choice(input_size[0], round(input_size[0] * input_sparsity))
    fixed_input = np.zeros(input_size)
    fixed_input[fixed_input_indices] = 1

    total_episodes = 100000 #Set total number of episodes to train agent on.
    e = 0.0 #Set the chance of taking a random action.


    repeats = 2000
    steps = 6000
    tot_rews = np.zeros((steps,))

    for repeat in range(repeats):
        print(repeat)
        b = Bandit(k)

        sp = spatial_pooler.SpatialPooler(input_size, k)
        rews = []
        best_count = 0
        total_reward = np.zeros(k)  # Set scoreboard for bandits to 0.
        total_selections = np.zeros(k)

        for step in range(steps):
            #if step % 1000 == 0:
            #    print(step)
            #Choose either a random action or one from our network.
            if np.random.rand(1) < e:
                action = np.random.randint(k)
            else:
                encoding = sp.step(fixed_input)
                action = encoding_to_action(encoding, k, step)
                net_weight = action

            reward = b.get_reward(action) #Get our reward from picking one of the bandits.

            best_count += 1 if b.is_best(action) else 0
            #if reward >= 0:
            #sp.perm_inc_step = np.tanh(reward) * 0.01
            #sp.perm_dec_step = np.tanh(reward) * 0.005
            #else:
            #    sp.perm_inc_step = reward * 0.01
            #    sp.perm_dec_step = reward * 0.005
            sp.reinforce(action, reward)
            #Update our running tally of scores.
            total_reward[action] += reward
            total_selections[action] += 1
            #if i % 50 == 0:
            #    #print("Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward))
            #    #print(total_selections)
            #    #print("CUR " + str(net_weight) + " BEST " + str(np.argmax(-np.array(bandits))))
            #if i % 1000 == 0:
            #    shuffle(bandits)
            rews.append(reward)
            if (step == 199 and best_count <100):
                print(action, b.arms, total_selections)
        print("BEST:", best_count)
        tot_rews = (repeat * tot_rews + np.array(rews)) / (repeat + 1)
    plt.plot(range(steps), tot_rews, alpha=0.5)
    best_count = 0
    repeat_rews = np.zeros((steps,))
    repeats = 20

    for rep in range(repeats):
        rews = []
        b = Bandit(k)
        for step in range(steps):

            action = np.random.randint(k)
            reward = b.get_reward(action)
            best_count += 1 if b.is_best(action) else 0
            rews.append(reward)
        repeat_rews = (rep * repeat_rews + np.array(rews)) / (rep + 1)
    print("BEST:", best_count)
    plt.plot(range(steps), repeat_rews, alpha=0.5)
    for eps in [0.1]:
        results = repeat_greedy(10, eps, steps, repeats)
        print(results.shape)
        plt.plot(range(steps), results, alpha=0.5)
    plt.savefig("plot.png")
