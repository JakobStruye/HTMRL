from scipy.sparse import csr_matrix
import numpy as np
import pyHTM3.spatial_pooler as spatial_pooler
import pyHTM3.temporal_memory as temporal_memory
import time
import random
random.seed(666)

def calc_overlap(l1, l2):
    overlap = l1.multiply(l2).sum()
    # overlap = 0
    # for item in l1:
    #     overlap += item in l2
    return overlap

n_inputs = 60
input_size = (n_inputs,)
input_size_flat = np.prod(input_size)
tm_size = (2048*32,1)

sp = spatial_pooler.SpatialPooler(input_size)
tm = temporal_memory.TemporalMemory()
step = 0
act_cells1 = csr_matrix(tm_size, dtype=np.bool)
act_cells2 = csr_matrix(tm_size, dtype=np.bool)
act_cells3 = csr_matrix(tm_size, dtype=np.bool)
act_cells4 = csr_matrix(tm_size, dtype=np.bool)
steptotal = 0

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput


#with PyCallGraph(output=GraphvizOutput()):
init_sp_out = None
first = None
start = time.time()
while steptotal < 100:
        steptotal += 1
        step = (step + 1) % 8
        inputs = np.zeros(input_size, dtype=bool)

        # if step == 0:
        #     inputs[0:10] = True
        # elif step == 4:
        #     inputs[10:20] = True
        # elif step== 1 or step == 5:
        #     inputs[20:30] = True
        # elif step == 2 or step == 6:
        #     inputs[30:40] = True
        # elif step == 3:
        #     inputs[40:50] = True
        # elif step == 7:
        #     inputs[50:60] = True
        inputs[:10] = True
        activated = sp.step(inputs)
        if first is None:
            first = activated
        activated = first
        #print(activated)
        active_cells = tm.step(activated)
        #active_cells = csr_matrix(activated, dtype=np.bool)
        #print(steptotal)

        if step == 2:
            print("Same input:")
            print("Good: {} out of {}".format(calc_overlap(active_cells, act_cells1), active_cells.getnnz()))
            print("Bad: {} out of {}".format(calc_overlap(active_cells, act_cells2), active_cells.getnnz()))
            act_cells1 = active_cells
        elif step  == 6:
            print("Same input:")
            print("Good: {} out of {}".format(calc_overlap(active_cells, act_cells2), active_cells.getnnz()))
            print("Bad: {} out of {}".format(calc_overlap(active_cells, act_cells1), active_cells.getnnz()))
            act_cells2 = active_cells
        elif step == 3:
            print("Different input:")
            print("Good: {} out of {}".format(calc_overlap(active_cells, act_cells3), active_cells.getnnz()))
            print("Bad: {} out of {}".format(calc_overlap(active_cells, act_cells4), active_cells.getnnz()))
            act_cells3 = active_cells
        elif step  == 7:
            print("Different input:")
            print("Good: {} out of {}".format(calc_overlap(active_cells, act_cells4), active_cells.getnnz()))
            print("Bad: {} out of {}".format(calc_overlap(active_cells, act_cells3), active_cells.getnnz()))
            act_cells4 = active_cells

end = time.time()
print(end-start)
with PyCallGraph(output=GraphvizOutput()):
    while steptotal < 10:
        steptotal += 1
        step = (step + 1) % 8
        inputs = np.zeros(input_size, dtype=bool)

        # if step == 0:
        #     inputs[0:10] = True
        # elif step == 4:
        #     inputs[10:20] = True
        # elif step== 1 or step == 5:
        #     inputs[20:30] = True
        # elif step == 2 or step == 6:
        #     inputs[30:40] = True
        # elif step == 3:
        #     inputs[40:50] = True
        # elif step == 7:
        #     inputs[50:60] = True
        inputs[:10] = True
        activated = sp.step(inputs)

        active_cells = tm.step(activated)
        #print(steptotal)
        if step == 3:
            print("Good: {} out of {}".format(calc_overlap(active_cells, act_cells1), active_cells.getnnz()))
            print("Bad: {} out of {}".format(calc_overlap(active_cells, act_cells2), active_cells.getnnz()))
            act_cells1 = active_cells
        elif step  == 7:
            print("Good: {} out of {}".format(calc_overlap(active_cells, act_cells2), active_cells.getnnz()))
            print("Bad: {} out of {}".format(calc_overlap(active_cells, act_cells1), active_cells.getnnz()))
            act_cells2 = active_cells
