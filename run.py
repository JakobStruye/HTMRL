import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import HTMRL.spatial_pooler as spatial_pooler
import sys
import yaml
import os
import datetime
import time

from HTMRL.env.bandit import Bandit
from HTMRL.env.sanity import Sanity

from HTMRL.encoders.sanity_encoder import SanityEncoder
from HTMRL.encoders import encoder_for_env

from multiprocessing import Pool
import psutil

matplotlib.use("Agg")

np.random.seed(0)

outdir = "output/" + datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S") + "/"


def run_htmrl(env, steps, htmrl_config):
    input_size = (htmrl_config["input_size"],)

    boost_strength = float(htmrl_config["boost_strength"])
    only_reinforce_selected = bool(htmrl_config["only_reinforce_selected"])
    reward_scaled_reinf = bool(htmrl_config["reward_scaled_reinf"])
    normalized_rewards = bool(htmrl_config["normalized_rewards"])
    boost_scaled_reinf = bool(htmrl_config["boost_scaled_reinf"])
    cell_count = int(htmrl_config["cell_count"])
    active_count = int(htmrl_config["active_count"])
    boost_until = int(htmrl_config["boost_until"])
    reward_window = int(htmrl_config["reward_window"])

    k = env.get_action_count()
    sp = spatial_pooler.SpatialPooler(input_size, k, boost_strength=boost_strength,
                                      only_reinforce_selected=only_reinforce_selected,
                                      reward_scaled_reinf=reward_scaled_reinf, normalize_rewards=normalized_rewards,
                                      boost_scaled_reinf=boost_scaled_reinf, cell_count=cell_count,
                                      active_count=active_count, boost_until=boost_until, reward_window=reward_window)
    rews = []
    actions = []

    encoder = encoder_for_env(env, htmrl_config)

    state = env.get_state()
    latest_bad = 0
    start = time.time()
    for step in range(steps):
        if step % 1000 == 0:
            print(step)
        input_enc = encoder.encode(state)
        encoding = sp.step(input_enc)
        action = encoding_to_action(encoding, k, sp.size, step)

        state, reward = env.do_action(action)  # Get our reward from picking one of the bandits.
        if reward == 1.0:
            if step - latest_bad > 100:
                break
        else:
            latest_bad = step

        sp.reinforce(action, reward)
        # Update our running tally of scores.
        rews.append(reward)
        actions.append(action)


    stop = time.time()
    print("TIMER", str(stop - start), step)

    if len(rews) < steps:
        print("Skipped", steps - len(rews))
        rews.extend((steps - len(rews)) * [1.0])

    return (rews, actions, env.get_debug_info())


def run_greedy(env, steps, eps):
    k = env.get_action_count()
    rews = []
    actions = []
    sample_avgs = np.zeros((k,))
    sample_counts = np.zeros((k,))
    for step in range(steps):
        if np.random.uniform(0., 1.) <= eps:
            selection = np.random.randint(0, k)
        else:
            selection = np.argmax(sample_avgs)
        state, rew = env.do_action(selection)
        sample_avg = sample_avgs[selection]
        sample_count = sample_counts[selection]
        sample_avgs[selection] = (sample_count * sample_avg + rew) / (sample_count + 1)
        sample_counts[selection] += 1
        rews.append(rew)
        actions.append(selection)
    return (np.array(rews), actions, env.get_debug_info())


def run_random(env, steps):
    k = env.get_action_count()
    rews = []
    actions = []
    for step in range(steps):
        action = np.random.randint(k)
        state, reward = env.do_action(action)
        rews.append(reward)
        actions.append(action)
    return np.array(rews), actions, env.get_debug_info()


def repeat_algo(env_init, env_config, steps, repeats, algo, outfile, **kwargs):
    avg_rews = np.zeros((steps,))

    p = Pool(psutil.cpu_count(logical=False))
    all_retvals = []
    for i in range(repeats):
        print(env_init)
        env = env_init(env_config)
        retval = p.apply_async(algo, [env, steps], kwargs)
        all_retvals.append(retval)

    p.close()
    p.join()

    for retval in all_retvals:
        new_rews = retval.get()[0]
        for line in new_rews:
            outfile.write(str(line) + '\n')

    return avg_rews


def encoding_to_action(encoding, actions, sp_size, i=1):
    buckets = np.floor(encoding / (float(sp_size) / actions))
    buckets = buckets.astype(np.int32)
    counts = np.bincount(buckets)

    return counts.argmax()


def rate_predictions(states, actions, env, sp):
    sp.boost_strength = 0.0
    encoder = SanityEncoder(states)
    sp.boost_anneal_until = 0
    for state in range(states):
        input_enc = encoder.encode(state)
        encoding = sp.step(input_enc)
        action = encoding_to_action(encoding, actions, sp.size)
        best_action = env.optimals[state]
        print(state, action, best_action)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide config filename as argument")
        exit(1)
    if len(sys.argv) > 2:
        enabled_experiments = sys.argv[2:]
    else:
        enabled_experiments = []
    with open(sys.argv[1], 'r') as stream:
        try:
            yml = yaml.load(stream)
            config_main = yml["general"]
            env_main = yml["env"]
            if env_main["name"] == "Bandit":
                env_init = Bandit
            elif env_main["name"] == "Sanity":
                env_init = Sanity
            else:
                raise Exception("Unknown env type: " + env_main["name"])
            algorithms_main = yml["algorithms"]
            experiments = yml["experiments"]
            print(yml)
        except yaml.YAMLError as exc:
            print(exc)

    try:
        os.makedirs(outdir)
    except:
        pass

    plt.figure(1)
    for exp_dict in experiments:
        exp_name = list(exp_dict.keys())[0]
        if enabled_experiments and exp_name not in enabled_experiments:
            continue
        os.makedirs(outdir + exp_name)
        exp = exp_dict[exp_name]
        if exp is None:
            exp = {}  # ease of use
        if "algorithms" not in exp:
            exp["algorithms"] = {}  # ease of use
        if "general" in exp:
            config = {**config_main, **exp["general"]}
        else:
            config = config_main
        if "env" in exp:
            env_config = {**env_main, **exp["env"]}
        else:
            env_config = env_main
        repeats = config["repeats"]
        steps = config["steps"]


        # HTMRL
        if "algorithms" in exp and "htmrl" in exp['algorithms']:
            htmrl = {**algorithms_main["htmrl"], **exp["algorithms"]["htmrl"]}
        elif "htmrl" in algorithms_main:
            htmrl = algorithms_main["htmrl"]
        else:
            htmrl = None
        print(htmrl)
        if htmrl is not None and not ("enabled" in htmrl and htmrl["enabled"] == 0):
            with open(outdir + exp_name + "/htmrl", "w") as rawfile:
                results = repeat_algo(env_init, env_config, steps, repeats, run_htmrl, rawfile, htmrl_config=htmrl)
            plt.figure(1)
            plt.plot(range(steps), results, alpha=0.5, label="HTM")

        # eps-greedy
        if "algorithms" in exp and "eps" in exp['algorithms']:
            eps = {**algorithms_main["eps"], **exp["algorithms"]["eps"]}
        elif "eps" in algorithms_main:
            eps = algorithms_main["eps"]
        else:
            eps = None
        if eps is not None and not ("enabled" in eps and eps["enabled"] == 0):
            with open(outdir + exp_name + "/eps", "w") as rawfile:
                results = repeat_algo(env_init, env_config, steps, repeats, run_greedy, rawfile, eps=eps["e"])
            print(results.shape)
            plt.plot(range(steps), results, alpha=0.5, label="eps-greedy")

        # Random
        if "random" in algorithms_main:
            with open(outdir + exp_name + "/random", "w") as rawfile:
                results = repeat_algo(env_init, env_config, steps, repeats, run_random, rawfile)

            plt.plot(range(steps), results, alpha=0.5, label="random")

        with open(outdir + exp_name + "/config", "w") as writefile:
            writefile.write("\n".join([str(config), str(env_config), str(htmrl), str(eps)]))

        plt.legend()
        plt.savefig(outdir + exp_name + ".png")
        plt.gcf().clear()
