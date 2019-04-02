import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import os
import numpy as np
from scipy.signal import savgol_filter

types = ["longrew", "noboost", "reorder-h", "reorder-e"]
graphtype = types[3]

plt.rcParams.update({'font.size': 13})

learner_labels = {"default/htmrl": "Default",
                  "noboost/htmrl": "No boosting",
                  "longwindow/htmrl": "Reward window: 2000",
                  "verylongwindow/htmrl": "Reward window: unbounded",
                  "shuffle/htmrl": "Shuffled",
                  "shuffle/eps": "Shuffled",
                  "eps001/eps": "Default (0.01)"}


#colors = ["#d7191c", "#fdae61", "#ffffbf", "#abdda4", "#2b83ba"]
if graphtype == "longrew":
    colors = ["#1d577c", "#4da83f", "#34702a"]
elif graphtype == "noboost":
    colors = ["#1d577c", "#70672a"]
elif graphtype == "reorder-h":
    colors = ["#1d577c", "#ac41f4"]
elif graphtype == "reorder-e":
    colors = ["#d7191c", "#f2741c"]


styles = ["-", "-", "-", "-", "-"]

def act_count(path):
    name = path.split("/")[-1]
    digits = name_to_digits(name)
    return digits

datadir = "./data/bandit"
#dirs = [x[0] for x in os.walk(datadir)][1:]

repeats = 1000
length = 10000
#print(dirs)
ax = plt.gca()

ctr = 0
d = datadir# + "/boosted"
if graphtype == "longrew":
    learners = ["default/htmrl", "longwindow/htmrl", "verylongwindow/htmrl"]
elif graphtype == "noboost":
    learners = ["default/htmrl", "noboost/htmrl"]
elif graphtype == "reorder-h":
    learners = ["default/htmrl", "shuffle/htmrl"]
elif graphtype == "reorder-e":
    learners = ["eps001/eps", "shuffle/eps"]

mask = list(range(10000))
for a in range(0,10000,2000):
    for b in range(1900,2000):
        mask.remove(a+b)
        print(a+b)

for learner in learners: 
    if d == "raw":
        continue
    with open(d + "/" + learner, "r") as f:
        points = [float(val) for val in f.readlines()]
        #assert len(points ) == (repeats+1) * length
        points = np.array(points[:(repeats*length)])
        name = d.split("/")[-1]
        points = np.reshape(points, (repeats, length))

        #undo moving window cumsum
        #points *= 100

        #for i in range(length - 100):
        #    points[:,i + 100] += points[:,i]

        #points[:,1:] -= points[:,:-1].copy()

        #redo with larger window
        #points = np.cumsum(points, axis=1)
        #points[:,10:] -= points[:,:-10]
        #points[:,:10] /= (np.array(np.arange(10.)) + 1.0)
        #points[:,10:] /= 10.

        #points = points[:,10:]
        meanpoints = np.mean(points, axis=0)
        stdpoints = np.std(points,axis=0)
        print(stdpoints.shape)
        #print(points[:,1000])

        try:
            #done = np.where(meanpoints == 1.0)[0][0]
            done = meanpoints.shape[0] - 1
            print(done)
            meanpoints = meanpoints[:done]
            stdpoints = stdpoints[:done]
        except:
            pass
        meanpoints = savgol_filter(meanpoints, 201,3)
        alpha = 0.4 if learner == learners[0] else 0.9
        plt.plot(mask, meanpoints[mask], label=learner_labels[learner], alpha=alpha, ls=styles[ctr], color=colors[ctr])
        print(stdpoints)
        #ax.fill_between(range(len(meanpoints)), meanpoints - stdpoints, meanpoints + stdpoints, alpha=0.5)
        ctr += 1


#ax.set_xscale("log")
handles, labels = ax.get_legend_handles_labels()

if graphtype == "longrew":
    order = [0,1,2]
elif graphtype == "noboost":
    order = [0,1]
elif graphtype == "reorder-h":
    order = [0,1]
elif graphtype == "reorder-e":
    order = [0,1]
print(len(order), len(handles), len(labels))
handles = [handles[i] for i in order]
labels = [labels[i] for i in order]
iseps = types.index(graphtype) == 3
ax.legend(handles, labels, loc=4 if not iseps else 1)
#ax.legend(loc=4)
plt.xlim(right=11000)
plt.ylim(bottom=0.8 if not iseps else 0.0, top=1.6)
#plt.xticks(range(0,11000,1000))
#plt.ylim(bottom=-1.1743108423442274, top=1.1705477696310274)
plt.xlabel("Training steps")
plt.ylabel("Reward (smoothed)")
plt.grid()
if graphtype == "longrew":
    name = "nonstatic-longrewhtmrl"
elif graphtype == "noboost":
    name = "nonstatic-noboosthtmrl"
elif graphtype == "reorder-h":
    name = "nonstatic-reorderhtmrl"
elif graphtype == "reorder-e":
    name = "nonstatic-reordereps"
plt.savefig("performance_{}.pdf".format(name), format="pdf", dpi=1200, bbox_inches='tight', pad_inches = 0)
#plt.show(block=True)
