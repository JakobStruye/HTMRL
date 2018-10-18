from scipy.sparse import csc_matrix, csr_matrix, dok_matrix, coo_matrix, find, isspmatrix_csr
import numpy as np
import random
from collections import defaultdict
import time

np.random.seed(666)


cells_per_col = 32

activation_thresh = 16
initial_perm = 0.3
connected_perm = 0.25 #0.5
learning_thresh = 12
learning_enabled = True

perm_inc_step = 0.01
perm_dec_step = 0.001
perm_dec_predict_step = 0.0005
max_segments_per_cell = 2
max_synapses_per_segment = 32

sp_size = (2048,)
sp_size_flat = np.prod(sp_size)

tm_size = (sp_size_flat,cells_per_col)
tm_size_flat = (np.prod(tm_size),1)


def to_flat_tm(col, cell):
    return col * cells_per_col + cell

def to_flat_segments(col, cell, seg=0):
    result = col * cells_per_col * max_segments_per_cell + (cell * max_segments_per_cell) + seg
    #print(col, cell, seg)
    assert (col, cell, seg) == unflatten_segments(result)
    return result

def unflatten_segments(flat):
    col = flat // (cells_per_col * max_segments_per_cell)
    remain = flat % (cells_per_col * max_segments_per_cell)
    cell = remain // max_segments_per_cell
    seg = remain % max_segments_per_cell
    return (col, cell, seg)


"""
Uniquely identifying objects:
col: its index
cell: its cell's index and its index
segment: its from and to cell UIDs


"""

grow_ctr = 0
perm_ctr = 0


class Cell(object):
    def __init__(self, col_idx, cell_idx):
        self.col_idx = col_idx
        self.cell_idx = cell_idx
        self.hash = None

    def __eq__(self, other):
        return self.col_idx == other.col_idx and self.cell_idx == other.cell_idx

    def __cmp__(self, other):
        print("cmpcell")
        return super.__cmp__(other)

    def __hash__(self):
        if not self.hash:
            #temp = (self.col_idx, self.cell_idx)
            self.hash =(self.col_idx, self.cell_idx).__hash__()  # precompute it
        return self.hash



class Synapse():
    def __init__(self, from_cell, to_cell):
        self.from_cell = from_cell
        self.to_cell = to_cell
        self.perm = initial_perm


        #This hacks the __eq__ for Cell == Synapse, for speed
        self.col_idx = to_cell.col_idx
        self.cell_idx = to_cell.cell_idx

    def __eq__(self, other):
        #impl: you should never have to compare synapses with different from_cells so don't bother checking
        return self.to_cell == other.to_cell
    def __cmp__(self, other):
        print("cmp")
        return super.__cmp__(other)

    def __hash__(self):
        return self.to_cell.__hash__() + 1


class Segment(object):
    def __init__(self, cell):
        self.cell = cell
        self.synapses = dok_matrix(tm_size_flat, dtype=np.float32)
        self.synapses_compiled = self.synapses.tocsc()

    def copy_and_prep(self, potentials):
        potentials = potentials[:] #copy
        random.shuffle(potentials)
        return potentials

    def grow_synapses(self, count, potentials):

        #potentials = self.copy_and_prep(potentials)
        overlap = potentials - self.synapses#potentials.multiply(self.synapses).astype(np.bool)
        overlap.data = np.clip(overlap.data, a_min=0, a_max=None)
        values = find(overlap)
        inds = list(zip(values[0], values[1]))
        random.shuffle(inds)
        grown = 0
        for ind in inds:
            if grown >= count:
                break
            #if self.synapses[potential.col_idx, potential.cell_idx]:
            #    continue
            #new_syn = Synapse(self.cell, potential)
            self.synapses[ind[0], ind[1]] = initial_perm
            grown += 1
        if grown:
            self.synapses_compiled = self.synapses.tocsc()




class SegmentDict(object):
    def __init__(self, sp_size_flat, cells_per_col):

        #self.segments =  [[[] for _ in range(cells_per_col)] for _ in range(sp_size_flat)]

        self.seg_matrix = dok_matrix((tm_size_flat[0], tm_size_flat[0] * max_segments_per_cell))
        self.seg_matrix_compiled = self.seg_matrix.tocsc()
        self.seg_counts = defaultdict(int)

    def get_segments(self, col, cell):
        return self.segments[col][cell]

    def get_seg_count(self, col, cell):
        #return len(self.segments[col][cell])
        return self.seg_counts[to_flat_tm(col, cell)]

    def add_segment(self, col_id, cell_id):
        #seg = Segment(Cell(col_id, cell_id))
        #self.segments[col_id][cell_id].append(seg)

        index = to_flat_tm(col_id, cell_id)
        self.seg_counts[index] += 1
        return self.seg_counts[index] - 1

    #def get_min(self, col):
    #    minimum = max_segments_per_cell
    #    start_idx = self.get_index(col, 0)
    #    for idx in range(start_idx, start_idx + cells_per_col ):
    #        minimum = min(minimum, self.seg_counts[idx])
    #    return minimum


    def get_least_used_cell(self, col):
        #least_segs = min([len(segs) for segs in self.segments[col]])
        #least_cells = [i for i, x in enumerate(self.segments[col]) if len(x) == least_segs]
        minimum = max_segments_per_cell
        start_idx = to_flat_tm(col, 0)
        mins = []
        for i in range(cells_per_col ):
            this_len = self.seg_counts[to_flat_tm(col, i)]
            #print(this_len)
            if this_len == minimum:
                mins.append(i)
            elif this_len < minimum:
                minimum = this_len
                mins = [i]

        return random.choice(mins)

    def get_best_matching_seg(self, col, matching_segs, active_pot_counts):
        #print(type(active_pot_counts))
        #print("min best", np.min(matching_segs))
        #print("", col, "from", to_flat(col,0), "to", to_flat(col+1, 0))
        best_score = -1
        start = to_flat_segments(col,0)
        best_cell = None
        best_seg = None
        #print(matching_segs.shape, start)
        for idx in find(matching_segs[:,start: to_flat_segments(col+1, 0)])[1]:
            (c, cell, seg_idx) = unflatten_segments(idx + start)
            #print(idx, col, c, cell, seg_idx)
            assert(c == col)

            #print("continue", idx, active_pot_counts[idx])
            if active_pot_counts[0, idx] > best_score:
                best_cell = cell
                best_seg = seg_idx
                best_score = active_pot_counts[0,idx]

            # for seg in cell:
            #     if seg not in matching_segs:
            #         continue
            #     if active_pot_counts[seg.cell] > best_score:
            #         best_match = seg
            #         best_score = active_pot_counts[seg.cell]
        #print(cell_id, best_score)
        return (best_cell, best_seg)

    def getrands(self, idx, potentials, slice):
        overlap = potentials - slice
        overlap.data = np.clip(overlap.data, a_min=0, a_max=None)
        values = find(overlap)
        inds = list(zip(values[0], values[1]))
        random.shuffle(inds)
        return inds

    def grow_synapses(self, col, cell, seg_idx, count, potentials, permlists):
        if count <= 0:
            return
        #print(type(potentials))
        #print(potentials.shape)
        #print("Growing on", col, cell, seg_idx)
        #potentials = self.copy_and_prep(potentials)
        idx = to_flat_segments(col, cell, seg_idx)
        #print(type(potentials))
        #print(type(self.seg_matrix_compiled))
        #temp = np.zeros((2048*32, 128))

        #slice = self.seg_matrix_compiled[:,idx]
        #start = time.time()
        #temp = self.seg_matrix_compiled.indices[self.seg_matrix_compiled.indptr[idx]:self.seg_matrix_compiled.indptr[idx+1]]
        # self.seg_matrix_compiled.indptr[idx]
        # self.seg_matrix_compiled.indptr[idx+1]
        # self.seg_matrix_compiled.indices[0:0]
        # self.seg_matrix_compiled.indices[self.seg_matrix_compiled.indptr[idx]:self.seg_matrix_compiled.indptr[idx+1]].tolist()

        #temp = np.ndarray((0,0))
        #print(len(temp))
        #end = time.time()
        #print("first", end-start)
        #start = time.time()
        #temp = self.seg_matrix_compiled[:,idx]
        #print(len(temp))

        #end = time.time()
        #print("second", end-start)
         #potentials.multiply(self.synapses).astype(np.bool)

        #inds = self.getrands(idx, potentials, slice)
        grown = 0
        # print(potentials)
        # print(type(potentials.todense().ravel()[0]))
        a = potentials.indices
        b = self.seg_matrix_compiled.indices[
                                                  self.seg_matrix_compiled.indptr[idx]:self.seg_matrix_compiled.indptr[
                                                      idx + 1]].ravel()
        #print(a.shape, b.shape)
        # print(np.setdiff1d(a,b))
        # print(self.seg_matrix_compiled.indices[
        #                                           self.seg_matrix_compiled.indptr[idx]:self.seg_matrix_compiled.indptr[
        #                                               idx + 1]].ravel().shape)
        # print(np.setdiff1d(potentials.todense().ravel(), self.seg_matrix_compiled.indices[
        #                                           self.seg_matrix_compiled.indptr[idx]:self.seg_matrix_compiled.indptr[
        #                                               idx + 1]].ravel()).shape)

        # print(np.setdiff1d(potentials.indices.ravel(), self.seg_matrix_compiled.indices[
        #                                           self.seg_matrix_compiled.indptr[idx]:self.seg_matrix_compiled.indptr[
        #                                               idx + 1]].ravel()).shape)
        # print(np.random.choice(np.setdiff1d(potentials.indices, self.seg_matrix_compiled.indices[
        #                                           self.seg_matrix_compiled.indptr[idx]:self.seg_matrix_compiled.indptr[
        #                                               idx + 1]]), count))

        diff = np.setdiff1d(potentials.indices, self.seg_matrix_compiled.indices[
                                                  self.seg_matrix_compiled.indptr[idx]:self.seg_matrix_compiled.indptr[
                                                      idx + 1]])
        if not diff.size:
            return
        count = min(diff.size, count)
        #print(count)

        for ind in np.random.choice(diff, count):
            if grown >= count:
                break
            #if self.synapses[potential.col_idx, potential.cell_idx]:
            #    continue
            #new_syn = Synapse(self.cell, potential)
            #print(ind[0], ind[1], to_flat(ind[0], ind[1]), idx, self.seg_matrix.shape)
            #print(ind)
            #print(idx)
            #print(self.seg_matrix)
            # self.seg_matrix[ind[0], idx] = initial_perm
            #print(ind)
            permlists[0].append(initial_perm)
            permlists[1].append(ind)
            permlists[2].append(idx)
            grown += 1
        #if grown:
        #    self.synapses_compiled = self.synapses.tocsc()


class TemporalMemory(object):
    def __init__(self):
        self.cells = np.zeros(tm_size)
        self.segments = np.empty_like(self.cells).tolist()
        self.actives_old = csc_matrix(tm_size_flat, dtype=bool)
        self.actives_old_dok = set() #dok_matrix(tm_size_flat, dtype=np.bool)
        self.actives = dok_matrix(tm_size_flat, dtype=bool)
        self.actives_compiled = None
        self.winners_old = csc_matrix(tm_size_flat, dtype=np.bool)
        self.winners = dok_matrix(tm_size_flat, dtype=bool)
        self.active_segs_old = csc_matrix((1,tm_size_flat[0]*max_segments_per_cell))
        self.active_segs = csc_matrix((1,tm_size_flat[0]*max_segments_per_cell))
        self.matching_segs_old = csc_matrix((1,tm_size_flat[0]*max_segments_per_cell))
        self.matching_segs = csc_matrix((1,tm_size_flat[0]*max_segments_per_cell))

        self.match2 = np.zeros((sp_size_flat,))#csc_matrix((1,sp_size_flat))
        self.act2 = np.zeros((1,sp_size_flat))

        self.permlists = [[],[],[]]

        self.active_pot_counts = csc_matrix((1,tm_size_flat[0]*max_segments_per_cell), dtype=np.int)
        self.active_pot_counts_old = csc_matrix((1,tm_size_flat[0]*max_segments_per_cell), dtype=np.int)

        self.sd = SegmentDict(sp_size_flat, cells_per_col)

    def burst_help(self, col):
        _from = to_flat_tm(col, 0)
        _to = to_flat_tm(col+1, 0)
        self.actives[_from:_to] = True

    def burst(self, col):
        #self.cells[col, :] = True
        #for cell in range(cells_per_col):
            #new_cell = Cell(col, cell)
        self.burst_help(col)

        if self.get_matching_segs_for_col_count(col):
            (winner_id, learning_seg) = self.sd.get_best_matching_seg(col, self.matching_segs_old, self.active_pot_counts_old)
            #print("WINNER", winner_id)
            self.matches += 1
        else:
            self.nomatches += 1
            # no matching
            winner_id = self.sd.get_least_used_cell(col)
            if learning_enabled:
                learning_seg = self.sd.add_segment(col, winner_id)
                learning_idx = (col, winner_id)
        self.winners[to_flat_tm(col, winner_id)] = True

        #self.actives_compiled = self.actives.tocsc()


        if learning_enabled:
            # import time
            # start = time.time()
            # learning_seg.synapses.multiply(self.actives_old).astype(np.bool)
            # start2 = time.time()
            # dur1 = start2 - start
            # print("d1", dur1)
            # start2 = time.time()
            #
            # learning_seg.synapses -= (learning_seg.synapses.astype(np.bool) * perm_dec_step)
            # start3 = time.time()
            # dur2 = start3 - start2
            # print("d2", dur2)
            # start3 = time.time()
            #
            # learning_seg.synapses += (self.actives_old * perm_inc_step * 2.0)
            # start4 = time.time()
            # dur3 = start4 - start3
            # print("d3", dur3)
            #
            seg_idx = to_flat_segments(col, winner_id, learning_seg)
            #print(self.sd.seg_matrix_compiled.shape)
            seg = self.sd.seg_matrix_compiled[:, seg_idx]
            #print(seg.shape, self.actives_old.shape)
            conns = seg.multiply(self.actives_old)
            active_idxs = set(find(conns)[0])
            #for synapse in learning_seg.synapses.keys():
            for syn_col in find(seg)[0]:
                #if synapse.to_cell in self.actives_old:

                if syn_col in active_idxs:
                    self.sd.seg_matrix[syn_col, seg_idx]  += perm_inc_step
                else:
                    self.sd.seg_matrix[syn_col, seg_idx] -= perm_dec_step
            # start2 = time.time()
            # dur1 = start2 - start
            # print(dur1)

            #print("VAL", self.active_pot_counts_old[12], learning_idx, self.active_pot_counts_old[to_flat(learning_idx[0], learning_idx[1])])
            #print(self.active_pot_counts_old.shape)
            new_syn_count = max_synapses_per_segment - self.active_pot_counts_old[0, seg_idx]
            if new_syn_count:
                self.sd.grow_synapses(col, winner_id, learning_seg, new_syn_count, self.winners_old, self.permlists)
                #learning_seg.grow_synapses(new_syn_count, self.winners_old)
            #learning_seg.synapses_compiled = learning_seg.synapses.tocsc()

    def activate_pred_help(self, col, cell, seg_idx):
        start = time.time()
        #global perm_ctr
        #for syn in seg.synapses.keys():
        #existing_synapses = find(self.sd.seg_matrix_compiled[:,to_flat_segments(col, cell, seg_idx)])[0]
        idx = to_flat_segments(col, cell, seg_idx)
        existing_synapses = self.sd.seg_matrix_compiled.indices[self.sd.seg_matrix_compiled.indptr[idx]:self.sd.seg_matrix_compiled.indptr[idx]+1]
        if len(existing_synapses) == 0:
            return
        #print(existing_synapses)
        #print(type(self.actives_old_perms))
        data = [self.actives_old_perms[idx] for idx in existing_synapses]
        self.permlists[0].extend(data)
        self.permlists[1].extend(existing_synapses)
        self.permlists[2].extend(len(existing_synapses) * [to_flat_segments(col, cell, seg_idx)])
        #modder = coo_matrix((data, (existing_synapses, len(existing_synapses) * [to_flat_segments(col, cell, seg_idx)])), shape=(tm_size_flat[0], tm_size_flat[0]*max_segments_per_cell))
        #self.sd.seg_matrix = (self.sd.seg_matrix + modder)
        end = time.time()
        #print(end-start)
        return
        for idx in find(self.sd.seg_matrix_compiled[:,to_flat_segments(col, cell, seg_idx)])[0]:
            #print(type(self.actives_old))
            # if syn.to_cell in self.actives_old:
            if idx in self.actives_old_dok:
                #perm_ctr += 1
                #continue
                self.sd.seg_matrix[idx, to_flat_segments(col, cell, seg_idx)] += perm_inc_step
                #seg.synapses[syn[0], syn[1]] += perm_inc_step
            else:
                #continue

                self.sd.seg_matrix[idx, to_flat_segments(col, cell, seg_idx)] -= perm_dec_step
                #seg.synapses[syn[0], syn[1]] -= perm_dec_step
        #print(perm_ctr)

    def activate_predicted_col(self, col):
        #print("predicted")

        #for idx in find(self.get_activated_segs_for_col(col))[0]:
        for idx in self.get_activated_segs_for_col(col):
            cell = idx // max_segments_per_cell
            seg_idx = idx % max_segments_per_cell
            self.actives[to_flat_tm(col, cell)] = True
            self.winners[to_flat_tm(col, cell)] = True
            if learning_enabled:
                self.activate_pred_help(col, cell, seg_idx)
                new_syn_count = max_synapses_per_segment - self.active_pot_counts_old[0, to_flat_segments(col, cell, seg_idx)] #TODO
                if new_syn_count:
                    self.sd.grow_synapses(col, cell, seg_idx, new_syn_count, self.winners_old, self.permlists)
            #seg.synapses_compiled = seg.synapses.tocsc()


    def get_activated_segs_for_col(self, col):
        start = to_flat_segments(col,0)
        end = to_flat_segments(col+1,0)
        return self.active_segs_old.indices[self.active_segs_old.indptr[start]:self.active_segs_old.indptr[end]]
        return self.active_segs_old[:,to_flat_segments(col,0):]#.reshape(max_segments_per_cell,cells_per_col, order='F')
        # segs = []
        #
        # for seg in self.active_segs_old:
        #     #if seg.cell.col_idx == col:
        #     if seg >= col * cells_per_col and seg < (col + 1) * cells_per_col:
        #         segs.append(seg)
        # return segs

    def get_activated_segs_for_col_count(self, col):
        val = self.act2[0,col]
        #if val > 0:
        #    print("JIPLA", val, self.get_activated_segs_for_col(col).getnnz())
        return val
        #return self.get_activated_segs_for_col(col).getnnz()

    def get_matching_segs_for_col(self, col):

        return self.matching_segs_old[:,to_flat_segments(col,0):to_flat_segments(col+1,0)]#.reshape(max_segments_per_cell,cells_per_col, order='F')
        # segs = []
        # #print("shape", self.matching_segs_old.shape)
        # for seg in self.matching_segs_old:
        #     if seg >= col*cells_per_col and seg < (col+1) * cells_per_col:
        #         segs.append(seg)
        # return segs

    def get_matching_segs_for_col_count(self, col):
        #print(self.match2.shape, col)
        #print(self.match2.shape)
        val2 = self.match2[col]

        # if val2 > 0:
        #     print("Jipla", val2)
        #     val1 = self.get_matching_segs_for_col(col).getnnz()
        #     assert (val1 == val2)
        return val2

    def punish_predicted(self, col):
        #TODO
        if learning_enabled:
            for match_idx in find(self.get_matching_segs_for_col(col))[1]:
                (c, cell, seg) = unflatten_segments(match_idx)
                for idx in find(self.sd.seg_matrix[:,to_flat_segments(col, cell, seg)])[0]:
                    #print("is {} in {}?".format(idx, find(self.actives_old)[0]))
                    if idx in find(self.actives_old)[0]:
                        self.sd.seg_matrix[idx,to_flat_segments(col, cell, seg)] -= perm_dec_predict_step
                        #print("PERMDECD")
        #     for seg in self.get_matching_segs_for_col(col):
        #         for syn in seg.synapses.keys():
        #             #if syn.to_cell in self.actives_old:
        #             if self.actives_old[syn[0], syn[1]]:
        #                 seg.synapses[syn[0],syn[1]] -= perm_dec_predict_step


    def activate_inner2(self, segment):
        #(s,a) = self.activate_helper(segment.synapses, self.actives)
        return segment.synapses_compiled.multiply(self.actives_compiled)

    def activate_inner(self, segment):
        # active_conn_count = 0
        # active_pot_count = 0
        active_synapse_perms = self.activate_inner2(segment).data
        active_conn_count = len(active_synapse_perms)
        active_pot_count = len(np.where(active_synapse_perms > connected_perm))


        # for (syn_col, syn_cell, perm) in zip(*find(segment.synapses)):
        #     #if syn.to_cell in self.actives:  # syn.to_cell in self.actives:
        #     if self.actives[syn_col, syn_cell]:
        #         if perm >= connected_perm:
        #             active_conn_count += 1
        #         if perm > 0:
        #             active_pot_count += 1


        return (active_conn_count, active_pot_count)


    def activate(self):
        #TEMP get cols with segments
        # temptest = self.sd.seg_matrix_compiled.sum(axis=0) #number of synapses in each segment
        # nonz = temptest.nonzero()
        # print("nonz", nonz)
        # for n in nonz[1]:
        #     unflattened = unflatten_segments(n)
        #     print("col {} cell {} seg {}".format(unflattened[0], unflattened[1], unflattened[2]))

        conns = self.sd.seg_matrix_compiled.multiply(self.actives_compiled)

        #print("connsize", conns.shape)
        #sum converts to int


        temp_copy = conns.copy()

        temp_copy.data[temp_copy.data < connected_perm] = 0.0 # = np.clip(temp_copy.data - connected_perm, a_min=0.0, a_max=None)
        temp_copy.eliminate_zeros()
        #print(type(temp_copy))
        #print(type(temp_copy.astype(np.bool)))
        temp_copy = csc_matrix(temp_copy.astype(bool).sum(axis=0))

        #print(type(temp_copy))
        temp_copy.data[temp_copy.data < activation_thresh] = 0 # = np.argwhere(active_conns > activation_thresh)[:,1]
        temp_copy.eliminate_zeros()
        self.active_segs = temp_copy.astype(bool)
        #print("actives:", self.active_segs.sum())
        #print(conns, type(conns))

        conns = csc_matrix(conns.astype(bool).sum(axis=0))
        self.active_pot_counts = conns.todok()
        conns.data[conns.data < learning_thresh] = 0# = np.argwhere(self.active_pot_counts > learning_thresh)[:,1]
        conns.eliminate_zeros()


        self.matching_segs = conns.astype(bool)
        #print("sizes", self.active_segs.shape, self.matching_segs.shape)
        #print(self.matching_segs.shape, sp_size_flat, self.matching_segs.getnnz())
        #print(find(self.matching_segs))
        #print(np.sum(self.matching_segs.data))
        #if len(find(self.matching_segs)[1]):
        #    print("HASMIN", min(find(self.matching_segs)[1]))
        #print(self.matching_segs.shape, cells_per_col * max_segments_per_cell * sp_size_flat)
        temp1 = self.matching_segs.reshape((cells_per_col*max_segments_per_cell, sp_size_flat), order='F')

        #print("temp1", temp1.shape)
        temp2 = temp1.sum(axis=0)
        #print("temp2", temp2.shape)
        self.match2 = np.asarray(temp2).ravel()
        #print(self.active_segs.getnnz(), find(self.active_segs))
        self.act2 = self.active_segs.reshape((cells_per_col*max_segments_per_cell, sp_size_flat), order='F').sum(axis=0)
        #print(self.act2.getnnz(), find(self.act2))
        #print("MATCH",self.match2.shape)

        #TODO COUNTS
        # for (col_id, col) in enumerate(self.sd.segments):
        #     for (cell_id, cell) in enumerate(col):
        #         for segment in cell:
        #             active_conn_count, active_pot_count = self.activate_inner(segment)
        #             if active_conn_count >= activation_thresh:
        #                 self.active_segs.append(segment)
        #             if active_pot_count >= learning_thresh:
        #                 self.matching_segs[segment] = segment
        #             self.active_pot_counts[(col_id, cell_id)] = active_pot_count



    def step(self, activated_cols):
        #print(sorted(activated_cols))
        actz = []
        for i in range(2048):
            if self.act2[0,i] > 0:
                actz.append(i)
        #print(actz)
        # acts = 0
        # for the_col in range(sp_size_flat):
        #
        #     if self.get_activated_segs_for_col_count(the_col) > 0:
        #         acts += 1
        # print("Acts", acts)
        #print(activated_cols)
        bursts = 0
        acts = 0
        self.matches = 0
        self.nomatches = 0
        for col in range(sp_size_flat):

            if col in activated_cols:

                if self.get_activated_segs_for_col_count(col) > 0:
                    self.activate_predicted_col(col)
                    acts += 1
                else:
                    bursts += 1
                    self.burst(col)


            else:
                if self.get_matching_segs_for_col_count(col):

                    self.punish_predicted(col)
        if len(self.permlists[0]):
            modder = coo_matrix((self.permlists[0], (self.permlists[1], self.permlists[2])), shape=(tm_size_flat[0], tm_size_flat[0]*max_segments_per_cell))
            print("ADDED MODDER")
            self.sd.seg_matrix = (self.sd.seg_matrix + modder).todok()
        #print("{} bursts {} acts {} matches {} no matches".format(bursts, acts, self.matches, self.nomatches))
        # acts = 0
        # for the_col in range(sp_size_flat):
        #
        #     if self.get_activated_segs_for_col_count(the_col) > 0:
        #         acts += 1
        # print("Acts", acts)
        self.actives_compiled = self.actives.tocsc()
        self.sd.seg_matrix_compiled = self.sd.seg_matrix.tocsc()

        self.activate()

        self.actives_old = self.actives_compiled
        self.actives_old_dok = set(find(self.actives)[0])
        dense_acts = self.actives_compiled.todense()
        self.actives_old_perms = np.squeeze(dense_acts * perm_inc_step - np.invert(dense_acts) * perm_dec_step).tolist()[0]
        self.actives = dok_matrix(tm_size_flat, dtype=bool)
        self.actives_compiled = None
        self.winners_old = self.winners.tocsc()
        self.winners = dok_matrix(tm_size_flat, dtype=bool)

        self.active_segs_old = self.active_segs
        self.active_segs = csc_matrix((1,tm_size_flat[0]*max_segments_per_cell))
        # acts = 0
        # for the_col in range(sp_size_flat):
        #
        #     if self.get_activated_segs_for_col_count(the_col) > 0:
        #         acts += 1
        # print("Acts", acts, len(self.active_segs), len(self.active_segs_old))
        self.matching_segs_old = self.matching_segs
        self.matching_segs = csc_matrix((1,tm_size_flat[0]*max_segments_per_cell))

        self.sd.seg_matrix = self.sd.seg_matrix_compiled.todok()


        self.active_pot_counts_old = self.active_pot_counts
        self.active_pot_counts = csc_matrix((1,tm_size_flat[0] * max_segments_per_cell))
        self.permlists = [[],[],[]]
        # acts = 0
        # for the_col in range(sp_size_flat):
        #
        #     if self.get_activated_segs_for_col_count(the_col) > 0:
        #         acts += 1
        # print("Acts", acts)
        return self.actives_old