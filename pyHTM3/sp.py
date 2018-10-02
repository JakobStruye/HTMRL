import numpy as np
n_cols = 2048
stimulus_thresh = 0
boost_strength = 0.
init_synapse_count = 20
connected_perm = 0.2
active_columns = 40

perm_inc_step = 0.01
perm_dec_step = 0.005
perm_min = 0.0
perm_max = 1.0

sp_size = (n_cols,)


sp_size_flat = np.prod(sp_size)



def get_rand_init_perm(n):
    vals = np.zeros((n,), dtype=float)
    rands = np.random.random(n)
    for i in range(n):
        if rands[i] > 0.5:
            vals[i] = connected_perm + (perm_max - connected_perm) * np.random.random()
        else:
            vals[i] = connected_perm * np.random.random()
    return(vals)

def get_active_cols(inputs, permanences):
    #second part needed because nan is cast to True
    connecteds = np.array((permanences - connected_perm).clip(min=0), dtype=bool) * ( ~ np.isnan(permanences))
    conn_counts = np.dot(np.expand_dims(inputs, 0), np.array(connecteds, dtype=int))
    conn_counts = np.squeeze(conn_counts)
    #print(np.sort(conn_counts))
    activated = np.argpartition(- conn_counts, active_columns)[:active_columns,]
    return activated

def reinforce(inputs, permanences, activated):
    inputs_pos = inputs * perm_inc_step
    inputs_neg = (inputs - 1) * perm_dec_step
    inputs_shift = inputs_pos + inputs_neg
    inputs_shift = np.expand_dims(inputs_shift, 1)
    permanences[:,activated] = permanences[:,activated] + inputs_shift
    return permanences


def get_sp(input_size_flat):
    #Initialize permanences
    permanences = np.empty((input_size_flat, sp_size_flat), dtype=np.float)
    permanences[:,:] = np.nan
    for col in range(sp_size_flat):
        rand_selection = np.random.choice(input_size_flat,init_synapse_count, replace=False)
        permanences[rand_selection, col] = get_rand_init_perm(init_synapse_count)
    #print(permanences[:,1000].tolist())
    #print(permanences[:,1001].tolist())
    return permanences

