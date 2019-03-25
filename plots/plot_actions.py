import matplotlib.pyplot as plt
import os
import numpy as np
from math import log
plt.rcParams.update({'font.size': 11})

#CONFIGURE THIS TO GENERATE PLOT FOR EITHER STATES OR ACTIONS
isstates = True

colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
styles = [[8,0],[8,2,8,2],[8,2,2,2],[2,2,2,2], [8,2,2,2,2,2], [8,2,8,2,2,2]]
#styles = ["--", "-", "-.", ":"]
styles_legend = [[8,0],[8,1,8,100],[8,1,2,1],[2,1,2,1],[8,1,2,1,2,1],[8,1,8,1,2,100]]
def logfilter(vals, totvals, remaining):
    minlog = 0#log(al)/log(10)
    maxlog = log(totvals)/log(10)
    indslog = np.arange(minlog, maxlog, (maxlog-minlog) / float(remaining))
    base = 10 * np.ones(indslog.shape[0])
    inds = base ** indslog
    inds = np.round(inds).astype(np.int)
    inds = np.unique(inds)
    final =np.searchsorted(inds, vals.shape[0]-1)
    inds = inds[:final]
    inds = inds.tolist()
    if inds[-1] < vals.shape[0]-1:
        inds.append(vals.shape[0]-1)
    return vals[inds]

def name_to_digits(name):
    digits = int(''.join([a for a in name if a.isdigit()]))
    return digits

def act_count(path):
    name = path.split("/")[-1]
    digits = name_to_digits(name)
    return digits
if isstates:
    datadir = "./data/states"
else:
    datadir = "./data/actions"
dirs = [x[0] for x in os.walk(datadir)][1:]

repeats = 20
length = 500000
#print(dirs)
try:
    dirs.sort(key=act_count)
except:
    print("Sorting failed")
ax = plt.gca()

labels = []
ctr = 0
for d in dirs:
    if d == "raw":
        continue
    with open(d + "/htmrl", "r") as f:
        print(d)
        points = [float(val) for val in f.readlines()]
        #assert len(points ) == (repeats+1) * length
        points = np.array(points[:(repeats*length)])
        name = d.split("/")[-1]
        points = np.reshape(points, (repeats, length))

        #undo moving window cumsum
        points *= 100

        for i in range(length - 100):
            points[:,i + 100] += points[:,i]

        points[:,1:] -= points[:,:-1].copy()

        #redo with larger window
        points = np.cumsum(points, axis=1)
        points[:,1000:] -= points[:,:-1000]
        points[:,:1000] /= (np.array(np.arange(1000.)) + 1.0)
        points[:,1000:] /= 1000.

        points = points[:,10:]
        meanpoints = np.mean(points, axis=0)
        stdpoints = np.std(points,axis=0)

        try:
            done = np.where(meanpoints == 1.0)[0][0]
            print(done)
            meanpoints = meanpoints[:done]
            stdpoints = stdpoints[:done]
        except:
            pass
        xvals = np.arange(0,meanpoints.shape[0],1, dtype=np.int)
        xvals = logfilter(xvals,350000,1000)
        yvals = logfilter(meanpoints,350000,1000)
        stdpoints = logfilter(stdpoints,350000,1000)
        plt.plot(xvals, yvals, label=name_to_digits(name), color=colors[ctr], ls="-", dashes=styles[ctr % len(styles)], alpha=0.8)
        ax.fill_between(xvals, yvals - stdpoints, yvals + stdpoints, alpha=0.15, facecolor=colors[ctr])

        labels.append(name_to_digits(name))
        ctr += 1


ax.set_xscale("log")

typename = "states" if isstates else "actions"
typename_cap = typename[0:1].upper() + typename[1:]

from matplotlib.lines import Line2D

custom_lines = []
for i in range(ctr):
    custom_lines.append(Line2D([0], [0], color=colors[i], ls='-', dashes=styles_legend[i], alpha=0.8))
#ax.legend(title=typename_cap)
ax.legend(custom_lines, labels, handlelength=3)
#plt.xlim(left=100)
plt.ylim(bottom=-1.1743108423442274, top=1.1705477696310274)
plt.xlabel("Training steps")
plt.ylabel("Mean reward (latest 1000 steps)")
plt.grid()
plt.savefig("performance_{}.pdf".format(typename), format="pdf", dpi=1200, bbox_inches='tight', pad_inches=0)
#plt.show(block=True)
