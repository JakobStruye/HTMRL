import matplotlib.pyplot as plt
import os
import numpy as np

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

datadir = "./data/states/"
datafile = "speedvals"

fig,ax1 = plt.subplots()
with open(datadir + datafile, "r") as f:
    lines = f.readlines()
    incr_states = []
    incr_steptime = []
    incr_stepcount = []
    for line in lines:
        [inc1, inc2, inc3] = line.split()
        incr_states.append(int(inc1))
        incr_steptime.append(float(inc2))
        incr_stepcount.append(int(inc3))
    lns1 = ax1.plot(incr_states, incr_steptime, "-o", color="orange",  label="steptime")
    ax2 = ax1.twinx()
    lns2 = ax2.plot(incr_states, incr_stepcount, "-o", color="blue", label="stepcount")

#make sure increases are properly scaled
print(ax1.get_ylim())
print(ax2.get_ylim())
ymax = ax2.get_ylim()[1]
yscale = ymax / float(incr_stepcount[0])
ymax_time = incr_steptime[0] * yscale
ax1.set_ylim(top=ymax_time)
align_yaxis(ax2,0,ax1,0)

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left')

ax1.set_xscale("log")
#ax1.set_yscale("log")
ax2.set_xscale("log")
#ax2.set_yscale("log")
#ax1.legend()
#plt.xlim(left=100)
#plt.savefig("performance_actions.eps", format="eps", dpi=1000)
plt.show(block=True)
