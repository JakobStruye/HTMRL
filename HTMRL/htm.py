from scipy.sparse import csr_matrix
import numpy as np
import HTMRL.spatial_pooler as spatial_pooler
import HTMRL.temporal_memory as temporal_memory
import time

from sys import argv
import HTMRL.log as log

def calc_overlap(l1, l2):
    overlap = l1.multiply(l2).sum()

    return overlap

def test():
    n_inputs = 60
    input_size = (n_inputs,)
    tm_size = (2048*32,1)

    sp = spatial_pooler.SpatialPooler(input_size)
    tm = temporal_memory.TemporalMemory()
    step = 0
    act_cells1 = csr_matrix(tm_size, dtype=np.bool)
    act_cells2 = csr_matrix(tm_size, dtype=np.bool)
    act_cells3 = csr_matrix(tm_size, dtype=np.bool)
    act_cells4 = csr_matrix(tm_size, dtype=np.bool)
    steptotal = 0


    start = time.time()
    while steptotal < 1000:
            steptotal += 1
            step = (step + 1) % 8
            inputs = np.zeros(input_size, dtype=bool)
            #inputs_fake = np.zeros(40)

            if step == 0:
                inputs[0:10] = True
                #inputs_fake[:] = list(range(0,40))
            elif step == 4:
                inputs[10:20] = True
                #inputs_fake[:] = list(range(40, 80))
            elif step== 1 or step == 5:
                inputs[20:30] = True
                #inputs_fake[:] = list(range(80, 120))
            elif step == 2 or step == 6:
                inputs[30:40] = True
                #inputs_fake[:] = list(range(120, 160))
            elif step == 3:
                inputs[40:50] = True
                #inputs_fake[:] = list(range(160, 200))
            elif step == 7:
                inputs[50:60] = True
                #inputs_fake[:] = list(range(200, 240))
            activated = sp.step(inputs)

            active_cells = tm.step(activated)

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
                tm.reset()
            elif step  == 7:
                print("Different input:")
                print("Good: {} out of {}".format(calc_overlap(active_cells, act_cells4), active_cells.getnnz()))
                print("Bad: {} out of {}".format(calc_overlap(active_cells, act_cells3), active_cells.getnnz()))
                act_cells4 = active_cells
                tm.reset()

    end = time.time()
    print(end-start)

if __name__ == '__main__':
    if "--fixed-seed" in argv:
        np.random.seed(666)
    if "--trace" in argv:
        log.set_trace()
    elif "--log" in argv:
        log.set_debug()

    test()