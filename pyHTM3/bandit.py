
import numpy as np
import spatial_pooler
from random import shuffle

#List out our bandits. Currently bandit 4 (index#3) is set to most often provide a positive reward.
bandits = [-3,1,2.,3]
num_bandits = len(bandits)
def pullBandit(bandit):
    #Get a random number.
    result = np.random.randn(1)
    if result > bandit:
        #return a positive reward.
        return 1
    else:
        #return a negative reward.
        return -1

def encoding_to_action(encoding, i=1):
    buckets = np.floor(encoding / 512.)
    buckets = buckets.astype(np.int32)
    counts = np.bincount(buckets)
    if i%200 == 0:
        print(counts)
    return counts.argmax()

input_size = (60,)
input_sparsity = 0.1
sp = spatial_pooler.SpatialPooler(input_size)

fixed_input_indices = np.random.choice(input_size[0], round(input_size[0] * input_sparsity))
fixed_input = np.zeros(input_size)
fixed_input[fixed_input_indices] = 1

total_episodes = 10000000 #Set total number of episodes to train agent on.
total_reward = np.zeros(num_bandits) #Set scoreboard for bandits to 0.
total_selections = np.zeros(num_bandits)
e = 0.0 #Set the chance of taking a random action.


i = 0
while i < total_episodes:

    #Choose either a random action or one from our network.
    if np.random.rand(1) < e:
        action = np.random.randint(num_bandits)
    else:
        encoding = sp.step(fixed_input, False)
        action = encoding_to_action(encoding,i)
        net_weight = action

    reward = pullBandit(bandits[action]) #Get our reward from picking one of the bandits.
    if reward >= 0:
        sp.perm_inc_step = 0.01
        sp.perm_dec_step = 0.005
    else:
        sp.perm_inc_step = -0.01
        sp.perm_dec_step = -0.005
    sp.step(fixed_input, True, action)
    #Update our running tally of scores.
    total_reward[action] += reward
    total_selections[action] += 1
    if i % 50 == 0:
        #print("Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward))
        #print(total_selections)
        print("CUR " + str(net_weight) + " BEST " + str(np.argmax(-np.array(bandits))))
    i+=1
    #if i % 1000 == 0:
    #    shuffle(bandits)
print("The agent thinks bandit " + net_weight + " is the most promising....")
if net_weight == np.argmax(-np.array(bandits)):
    print("...and it was right!")
else:
    print("...and it was wrong!")
