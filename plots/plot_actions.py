import matplotlib.pyplot as plt
import os

def act_count(path):
    name = path.split("/")[-1]
    digits = int(''.join([a for a in name if a.isdigit()]))
    return digits

datadir = "./data/actions_floor"
dirs = [x[0] for x in os.walk(datadir)][1:]
#print(dirs)
dirs.sort(key=act_count)
for d in dirs:
    with open(d + "/htmrl", "r") as f:
        points = [float(val) for val in f.readlines()]
        name = d.split("/")[-1]
        points = points[100:]
        try:
            done = points.index(1)
            print(done)
            points = points[:done]
        except:
            pass
        plt.plot(points, label=name)       
ax = plt.gca()
ax.set_xscale("log")
ax.legend()
plt.savefig("performance_actions.eps", format="eps", dpi=1000)
#plt.show(block=True)
