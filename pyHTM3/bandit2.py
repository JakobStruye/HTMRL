import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pyHTM3.spatial_pooler as spatial_pooler
import yaml
import os
import datetime
import pickle

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

def run_htmrl(bandit_config, steps, htmrl_config):
    k = bandit_config["k"]
    b = Bandit(k)

    input_size = (htmrl_config["input_size"],)
    input_sparsity = htmrl_config["input_sparsity"]

    fixed_input_indices = np.random.choice(input_size[0], round(input_size[0] * input_sparsity))
    fixed_input = np.zeros(input_size)
    fixed_input[fixed_input_indices] = 1

    boost_strength = float(htmrl_config["boost_strength"])
    only_reinforce_selected = bool(htmrl_config["only_reinforce_selected"])
    reward_scaled_reinf = bool(htmrl_config["reward_scaled_reinf"])
    normalized_rewards =  bool(htmrl_config["normalized_rewards"])
    boost_scaled_reinf = bool(htmrl_config["boost_scaled_reinf"])

    sp = spatial_pooler.SpatialPooler(input_size, k, boost_strength=boost_strength,
                                      only_reinforce_selected=only_reinforce_selected,
                                      reward_scaled_reinf=reward_scaled_reinf, normalize_rewards=normalized_rewards,
                                      boost_scaled_reinf=boost_scaled_reinf)
    rews = []
    actions = []
    best_count = 0
    total_reward = np.zeros(k)  # Set scoreboard for bandits to 0.
    total_selections = np.zeros(k)

    for step in range(steps):
        encoding = sp.step(fixed_input)
        action = encoding_to_action(encoding, k, step)
        net_weight = action

        reward = b.get_reward(action)  # Get our reward from picking one of the bandits.

        best_count += 1 if b.is_best(action) else 0

        sp.reinforce(action, reward)
        # Update our running tally of scores.
        total_reward[action] += reward
        total_selections[action] += 1
        rews.append(reward)
        actions.append(action)
        # if (step == 199 and best_count <100):
        #    print(action, b.arms, total_selections)
    print("BEST:", best_count)
    return (rews, actions, b.arms)


def run_greedy(bandit_config, steps, eps):
    k = bandit_config["k"]
    b = Bandit(k)
    rews = []
    actions = []
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
        actions.append(selection)
    return (np.array(rews), actions, b.arms)

def run_random(bandit_config, steps):
    rews = []
    actions = []
    k = bandit_config["k"]
    b = Bandit(k)
    for step in range(steps):
        action = np.random.randint(k)
        reward = b.get_reward(action)
        rews.append(reward)
        actions.append(action)
    return (np.array(rews), actions, b.arms)


def repeat_algo(bandit_config, steps, repeats, algo, outfile, **kwargs):
    avg_rews = np.zeros((steps,))
    all_rews = []
    all_acts = []
    all_arms = []
    for i in range(repeats):
        (new_rews, new_actions, new_b) = algo(bandit_config, steps, **kwargs)
        outfile.write(str(new_rews))
        outfile.write(str(new_actions))
        outfile.write(str(new_b))
        #all_rews.append(new_rews)
        #all_acts.append(new_actions)
        #all_arms.append(new_b)
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
    with open("config/bandit.yml", 'r') as stream:
        try:
            yml = yaml.load(stream)
            config_main = yml["general"]
            bandit_main = yml["bandit"]
            algorithms_main = yml["algorithms"]
            experiments = yml["experiments"]
            print(yml)
        except yaml.YAMLError as exc:
            print(exc)


    outdir = "output/" + datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S") + "/"

    try:
        os.makedirs(outdir)
    except:
        pass



    for exp_dict in experiments:
        exp_name = list(exp_dict.keys())[0]
        os.makedirs(outdir + exp_name)
        exp = exp_dict[exp_name]
        if exp is None:
            exp = {} #ease of use
        if "general" in exp:
            config = {**config_main, **exp["general"]}
        else:
            config = config_main
        if "bandit" in exp:
            print(exp["bandit"])
            bandit = {**bandit_main, **exp["bandit"]}
        else:
            bandit = bandit_main
        k = bandit["k"]
        repeats = config["repeats"]
        steps = config["steps"]


        #HTMRL
        print(algorithms_main)
        if "algorithms" in exp and "htmrl" in exp['algorithms']:
            htmrl = {**algorithms_main["htmrl"], **exp["algorithms"]["htmrl"]}
        else:
            htmrl = algorithms_main["htmrl"]

        with open(outdir + exp_name + "/htmrl", "w") as rawfile:
            results = repeat_algo(bandit, steps, repeats, run_htmrl, rawfile, htmrl_config=htmrl)

        plt.plot(range(steps), results, alpha=0.5)

        #eps-greedy
        if "algorithms" in exp and "eps" in exp['algorithms']:
            eps = {**algorithms_main["eps"], **exp["algorithms"]["eps"]}
        else:
            eps = algorithms_main["eps"]
        with open(outdir + exp_name + "/eps", "w") as rawfile:
            results = repeat_algo(bandit, steps, repeats, run_greedy, rawfile, eps=eps["e"])
        print(results.shape)
        plt.plot(range(steps), results, alpha=0.5)


        #Random
        with open(outdir + exp_name + "/random", "w") as rawfile:
            results = repeat_algo(bandit, steps, repeats, run_random, rawfile)

        plt.plot(range(steps), results, alpha=0.5)

        with open(outdir + exp_name + "/config", "w") as writefile:
            writefile.write("\n".join([str(config), str(bandit), str(htmrl), str(eps)]))
        plt.savefig(outdir + exp_name + ".png")
        plt.gcf().clear()
