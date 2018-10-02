import numpy as np
import pyHTM3.sp as sp
import pyHTM3.tm as tm

def calc_overlap(l1, l2):
    overlap = 0
    for item in l1:
        overlap += item in l2
    return overlap

n_inputs = 60
input_size = (n_inputs,)
input_size_flat = np.prod(input_size)


inputs = np.zeros(input_size, dtype=bool)
perms = sp.get_sp(input_size_flat)
inputs[:20] = True
act1 = np.sort(sp.get_active_cols(inputs, perms))

tempmem = tm.TemporalMemory()
step = 0
act_cells1 = []
act_cells2 = []
while (True):
    step += 1
    #inputs = np.random.choice(a=[False, True], size=input_size, p=[0.9, 0.1])
    inputs = np.zeros(input_size, dtype=bool)
    if step % 4 == 0:
        inputs[-5:] = True
    elif step % 4 == 1 or step % 4 == 3:
        inputs[:5] = True
    else:
        inputs[-10:-5] = True

    activated = sp.get_active_cols(inputs, perms)
    perms = sp.reinforce(inputs, perms, activated)

    active_cells = tempmem.step(activated)
    if step % 4 == 1:
        print("Good: {} out of {}".format(calc_overlap(active_cells, act_cells1), len(active_cells)))
        print("Bad: {} out of {}".format(calc_overlap(active_cells, act_cells2), len(active_cells)))
        act_cells1 = active_cells[:]
    elif step % 4 == 3:
        print("Good: {} out of {}".format(calc_overlap(active_cells, act_cells2), len(active_cells)))
        print("Bad: {} out of {}".format(calc_overlap(active_cells, act_cells1), len(active_cells)))
        act_cells2 = active_cells[:]

    #print(np.sort(activated))
    # if step % 100 == 0:
    #     #print(step)
    #     inputs = np.zeros(input_size, dtype=bool)
    #     inputs[0:20] = True
    #     act2 = np.sort(sp.get_active_cols(inputs, perms))
    # 
    #     overlaps = 0
    #     for val in act1:
    #         overlaps += 1 if val in act2 else 0
    #     # print(act1, act2)
    #     print(overlaps)

#print(permanences)