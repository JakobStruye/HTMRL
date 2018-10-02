import numpy as np
import random
from collections import defaultdict


cells_per_col = 32

activation_thresh = 16
initial_perm = 0.2
connected_perm = 0.25 #0.5
learning_thresh = 12
learning_enabled = True

perm_inc_step = 0.01
perm_dec_step = 0.001
perm_dec_predict_step = 0.0005
max_segments_per_cell = 128
max_synapses_per_segment = 32

sp_size = (2048,)
sp_size_flat = np.prod(sp_size)

tm_size = (sp_size_flat,cells_per_col)

"""
Uniquely identifying objects:
col: its index
cell: its cell's index and its index
segment: its from and to cell UIDs


"""

def assert_cell(cell):
    assert(type(cell) == tuple)
    assert(len(cell) == 2)
    assert(type(cell[0]) == type(cell[1]) == int)



class Synapse():
    def __init__(self, from_cell, to_cell):
        assert_cell(from_cell)
        assert_cell(to_cell)
        self.from_cell = from_cell
        self.to_cell = to_cell
        self.perm = initial_perm

    def __eq__(self, other):
        try:
            return self.to_cell == other.to_cell
        except:
            return False



class Segment():
    def __init__(self, cell):
        assert_cell(cell)
        self.cell = cell
        self.synapses = []

    def grow_synapses(self, count, potentials):
        potentials = potentials[:] #copy
        random.shuffle(potentials)
        grown = 0
        for potential in potentials:
            if grown >= count:
                break
            if potential in self.synapses:
                continue
            self.synapses.append(Synapse(self.cell, potential))
            grown += 1




class SegmentDict():
    def __init__(self, sp_size_flat, cells_per_col):
        #self.segments[col][cell] is list of segments
        self.segments =  [[[] for _ in range(cells_per_col)] for _ in range(sp_size_flat)]

    def get_segments(self, col, cell):
        return self.segments[col][cell]

    def get_seg_count(self, col, cell):
        return len(self.segments[col][cell])

    def add_segment(self, col_id, cell_id):
        seg = Segment((col_id, cell_id))
        self.segments[col_id][cell_id].append(seg)
        return seg

    def get_least_used_cell(self, col):
        least_segs = min([len(segs) for segs in self.segments[col]])
        least_cells = [i for i, x in enumerate(self.segments[col]) if len(x) == least_segs]
        return random.choice(least_cells)

    def get_best_matching_seg(self, col, matching_segs, active_pot_counts):
        best_match = None
        best_score = -1
        for cell in self.segments[col]:
            for seg in cell:
                if seg not in matching_segs:
                    continue
                if active_pot_counts[seg.cell] > best_score:
                    best_match = seg
                    best_score = active_pot_counts[seg.cell]
        return best_match



class TemporalMemory():
    def __init__(self):
        self.cells = np.zeros(tm_size)
        self.segments = np.empty_like(self.cells).tolist()
        self.actives_old = []
        self.actives = []
        self.winners_old = []
        self.winners = []
        self.active_segs_old = []
        self.active_segs = []
        self.matching_segs_old = []
        self.matching_segs = []

        self.active_pot_counts = defaultdict(int)
        self.active_pot_counts_old = defaultdict(int)

        self.sd = SegmentDict(sp_size_flat, cells_per_col)


    def burst(self, col):
        self.cells[col, :] = True
        for cell in range(cells_per_col):
            self.actives.append((col, cell))
        if len(self.get_matching_segs_for_col(col)) > 0:
            learning_seg = self.sd.get_best_matching_seg(col, self.matching_segs_old, self.active_pot_counts_old)
            winner_id = learning_seg.cell[1]
        else:
            # no matching
            winner_id = self.sd.get_least_used_cell(col)
            if learning_enabled:
                learning_seg = self.sd.add_segment(col, winner_id)
        self.winners.append((col, winner_id))
        if learning_enabled:
            for synapse in learning_seg.synapses:
                if synapse.to_cell in self.actives_old:
                    synapse.perm += perm_inc_step
                else:
                    synapse.perm -= perm_dec_step
            new_syn_count = max_synapses_per_segment - self.active_pot_counts_old[learning_seg.cell]
            learning_seg.grow_synapses(new_syn_count, self.winners_old)


    def activate_predicted_col(self, col):
        for seg in self.get_activated_segs_for_col(col):
            self.actives.append(seg.cell)
            self.winners.append(seg.cell)
            if learning_enabled:
                for syn in seg.synapses:
                    if syn.to_cell in self.actives_old:
                        syn.perm += perm_inc_step
                    else:
                        syn.perm -= perm_dec_step
                new_syn_count = max_synapses_per_segment - self.active_pot_counts_old[seg.cell]
                seg.grow_synapses(new_syn_count, self.winners_old)
    def get_activated_segs_for_col(self, col):
        segs = []
        for seg in self.active_segs_old:
            if seg.cell[0] == col:
                segs.append(seg)
        return segs

    def get_activated_segs_for_col_count(self, col):

        return len(self.get_activated_segs_for_col(col))

    def get_matching_segs_for_col(self, col):
        segs = []
        for seg in self.matching_segs_old:
            if seg.cell[0] == col:
                segs.append(seg)
        return segs

    def get_matching_segs_for_col_count(self, col):

        return len(self.get_matching_segs_for_col(col))

    def punish_predicted(self, col):
        if learning_enabled:
            for seg in self.get_matching_segs_for_col(col):
                for syn in seg.synapses:
                    if syn.to_cell in self.actives_old:
                        syn.perm -= perm_dec_predict_step

    def activate(self):
        for (col_id, col) in enumerate(self.sd.segments):
            for (cell_id, cell) in enumerate(col):
                for segment in cell:
                    active_conn_count = 0
                    active_pot_count = 0
                    for syn in segment.synapses:
                        if syn.to_cell in self.actives:

                            if syn.perm >= connected_perm:
                                active_conn_count += 1
                            if syn.perm > 0:
                                active_pot_count += 1
                    if active_conn_count >= activation_thresh:
                        self.active_segs.append(segment)
                    if active_pot_count >= learning_thresh:
                        self.matching_segs.append(segment)
                    self.active_pot_counts[(col_id, cell_id)] = active_pot_count

    def step(self, activated_cols):

        for col in range(sp_size_flat):
            if col in activated_cols:
                if self.get_activated_segs_for_col_count(col) > 0:
                    self.activate_predicted_col(col)
                else:
                    self.burst(col)


            else:
                if self.get_matching_segs_for_col(col):
                    self.punish_predicted(col)
        self.activate()
        self.actives_old = self.actives
        self.actives = []
        self.winners_old = self.winners
        self.winners = []
        self.active_segs_old = self.active_segs
        self.active_segs = []
        self.matching_segs_old = self.matching_segs
        self.matching_segs = []


        self.active_pot_counts = defaultdict(int)
        self.active_pot_counts_old = defaultdict(int)

        return self.actives_old