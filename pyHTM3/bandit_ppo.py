from baselines.ppo2.ppo2 import learn
from pyHTM3.bandit2 import Bandit
import numpy as np
p = dict()
p[0] = dict()
p[0][0] = [(1.0, 0, 1., 1)]


e = Bandit(10)
model = learn(network="mlp", env=e, total_timesteps=50000)

#e.reset()
goods, bads = (0,0)
for i in range(1000):
    a, _, _, _ = model.step([0])
    _, r, _, _ = e.step(a)
    vals = (a, r, e.is_best(a))
    print(vals)
    if vals[2]:
        goods += 1
    else:
        bads += 1
print(goods, bads)
