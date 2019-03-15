import numpy as np
#np.random.seed(0)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import HTMRL.spatial_pooler as spatial_pooler
import yaml
import os
import datetime
import time

from HTMRL.env.bandit import Bandit
from HTMRL.old.maze import Maze
from HTMRL.old.qlearn import QLearn

from HTMRL.encoders.maze_encoder import MazeEncoder


outdir = "output/" + datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S") + "/"


def run_htmrl(env, steps, htmrl_config):

    input_size = (htmrl_config["input_size"],)
    input_sparsity = htmrl_config["input_sparsity"]

    #fixed_input_indices = np.random.choice(input_size[0], round(input_size[0] * input_sparsity))
    #fixed_input = np.zeros(input_size)
    #fixed_input[fixed_input_indices] = 1

    boost_strength = float(htmrl_config["boost_strength"])
    only_reinforce_selected = bool(htmrl_config["only_reinforce_selected"])
    reward_scaled_reinf = bool(htmrl_config["reward_scaled_reinf"])
    normalized_rewards =  bool(htmrl_config["normalized_rewards"])
    boost_scaled_reinf = bool(htmrl_config["boost_scaled_reinf"])

    k = env.get_action_count()
    sp = spatial_pooler.SpatialPooler(input_size, k, boost_strength=boost_strength,
                                      only_reinforce_selected=only_reinforce_selected,
                                      reward_scaled_reinf=reward_scaled_reinf, normalize_rewards=normalized_rewards,
                                      boost_scaled_reinf=boost_scaled_reinf)
    rews = []
    actions = []
    best_count = 0
    total_reward = np.zeros(k)  # Set scoreboard for bandits to 0.
    total_selections = np.zeros(k)

    encoder = MazeEncoder(env_config["size"])

    state = env.get_state()
    latest_bad = 0
    latest_good = 0
    start = time.time()
    for step in range(steps):
        if step % 1000 == 0:
            print(step)
        input_enc = encoder.encode(state[0], state[1])
        encoding = sp.step(input_enc)
        action = encoding_to_action(encoding, k, sp.size, step, state)
        net_weight = action

        state, reward = env.do_action(action)  # Get our reward from picking one of the bandits.
        if reward > 0.0:
            print(step - latest_good, sp.boost_strength)
            latest_good = step
        #if reward == 1.0:
        #    if step - latest_bad > 100:
        #        break
        #else:
        #    latest_bad = step

        #best_count += 1 if env.is_best(action) else 0
        #if step > 5000:
        env.visualize()

        #if step < 5000:
        sp.reinforce(action, reward)
        # Update our running tally of scores.
        total_reward[action] += reward
        total_selections[action] += 1
        rews.append(reward)
        actions.append(action)
        # if (step == 199 and best_count <100):
        #    print(action, b.arms, total_selections)
    #print("BEST:", best_count)

    # debugging
    #all_states = env.get_all_states()
    #all_encs = [encoder.encode(state[0]) for state in all_states]
    #sp.visualize_cell_usage(all_encs, outdir)

    #rate_predictions(env.size, k, env, sp)

    stop = time.time()
    print("TIMER", str(stop-start), step)

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
        if np.random.uniform(0.,1.) <= eps:
            selection = np.random.randint(0,k)
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

def run_q(env, steps):
    rews = []
    actions = []
    ql = QLearn((25,),200,0.1)
    state = env.get_state()
    for step in range(steps):
        action = ql.get_action(state)
        next_state, rew = env.do_action(action)
        ql.learn(state, next_state, action, rew)
        prev_state = state
        state = next_state

        rews.append(rew)
        actions.append(action)
        if prev_state == 50:
            print(prev_state, action, rew, ql.eps)
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
    return (np.array(rews), actions, env.get_debug_info())


def repeat_algo(env_init, env_config, steps, repeats, algo, outfile, **kwargs):
    avg_rews = np.zeros((steps,))
    all_rews = []
    all_acts = []
    all_arms = []
    for i in range(repeats):
        print(env_init)
        env = env_init(env_config)
        (new_rews, new_actions, new_b) = algo(env, steps, **kwargs)
        #outfile.write(str(new_rews))
        #outfile.write(str(new_actions))
        #outfile.write(str(new_b))
        #all_rews.append(new_rews)
        #all_acts.append(new_actions)
        #all_arms.append(new_b)
        new_rews = np.cumsum(new_rews)
        new_rews[100:] = new_rews[100:] - new_rews[:-100]
        new_rews /= 100.
        for line in new_rews:
            outfile.write(str(line) + '\n')
        avg_rews = (i * avg_rews + new_rews) / (i+1)
    #outfile.write("###FINAL RESULTS###")
    for line in avg_rews:
        outfile.write(str(line) + '\n')
    return avg_rews

#for eps in [0.1,0.01,0.0]:
#    results = repeat_greedy(10, eps, 1000, 2000)
#    print(results.shape)
#    plt.plot(range(1000), results)
#plt.show()


def encoding_to_action(encoding, actions, sp_size, i=1, state=(0,0)):
    buckets = np.floor(encoding / (float(sp_size) / actions))
    buckets = buckets.astype(np.int32)
    counts = np.bincount(buckets)
    #print(counts)
    #if state[0] == 0 and state[1] == 0:
    #    print(counts)
    # if i>5000:
    #     print(counts)
    return counts.argmax()

def rate_predictions(states, actions, env, sp):
    sp.boost_strength = 0.0
    encoder = MazeEncoder(states)
    sp.boost_anneal_until = 0
    for state in range(states):
        input_enc = encoder.encode(state)
        encoding = sp.step(input_enc)
        action = encoding_to_action(encoding, actions, sp.size)
        best_action = env.optimals[state]
        print(state, action, best_action)

if __name__ == "__main__":
    with open("config/maze.yml", 'r') as stream:
        try:
            yml = yaml.load(stream)
            config_main = yml["general"]
            env_main = yml["env"]
            if env_main["name"] == "Bandit":
                env_init = Bandit
            elif env_main["name"] == "Maze":
                env_init = Maze
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
        os.makedirs(outdir + exp_name)
        exp = exp_dict[exp_name]
        if exp is None:
            exp = {} #ease of use
        if "algorithms" not in exp:
            exp["algorithms"] = {} #ease of use
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

        #with open(outdir + exp_name + "/q", "w") as rawfile:

        #    results = repeat_algo(env_init, env_config, steps, repeats, run_q, rawfile)
        #plt.plot(range(steps), results, alpha=0.5, label="Q-learn")

        #HTMRL
        if "algorithms" in exp and "htmrl" in exp['algorithms']:
            htmrl = {**algorithms_main["htmrl"], **exp["algorithms"]["htmrl"]}
        elif "htmrl" in algorithms_main:
            htmrl = algorithms_main["htmrl"]
        else:
            htmrl = None
        print(htmrl)
        if htmrl is not None:
            with open(outdir + exp_name + "/htmrl", "w") as rawfile:
                results = repeat_algo(env_init, env_config, steps, repeats, run_htmrl, rawfile, htmrl_config=htmrl)
            plt.figure(1)
            plt.plot(range(steps), results, alpha=0.5, label="HTM")

        #eps-greedy
        if "algorithms" in exp and "eps" in exp['algorithms']:
            eps = {**algorithms_main["eps"], **exp["algorithms"]["eps"]}
        elif "eps" in algorithms_main:
            eps = algorithms_main["eps"]
        else:
            eps = None
        if eps is not None:
            with open(outdir + exp_name + "/eps", "w") as rawfile:
                results = repeat_algo(env_init, env_config, steps, repeats, run_greedy, rawfile, eps=eps["e"])
            print(results.shape)
            plt.plot(range(steps), results, alpha=0.5, label="eps-greedy")


        #Random
        if "random" in algorithms_main:
            with open(outdir + exp_name + "/random", "w") as rawfile:
                results = repeat_algo(env_init, env_config, steps, repeats, run_random, rawfile)

            plt.plot(range(steps), results, alpha=0.5, label="random")

        with open(outdir + exp_name + "/config", "w") as writefile:
            writefile.write("\n".join([str(config), str(env_config), str(htmrl), str(eps)]))
        plt.legend()
        plt.savefig(outdir + exp_name + ".png")
        plt.gcf().clear()
