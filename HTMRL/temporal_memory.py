from scipy.sparse import csr_matrix, dok_matrix, coo_matrix, find
import numpy as np
import random
from collections import defaultdict
import HTMRL.log as log
import time
cells_per_col = 32

activation_thresh = 16
initial_perm = 0.31
connected_perm = 0.4  # 0.5
learning_thresh = 12
learning_enabled = True

perm_inc_step = 0.01
perm_dec_step = 0.001
perm_dec_predict_step = 0.000#5
# max_seg set very low right now: serious impact on performance; should autoscale internally
max_segments_per_cell = 4
max_synapses_per_segment = 32

sp_size = (2048,)
sp_size_flat = np.prod(sp_size)

tm_size = (sp_size_flat, cells_per_col)
tm_size_flat = np.prod(tm_size)
max_segs_total = tm_size_flat * max_segments_per_cell

timer = 0
timera = 0
timerb = 0

ctr = 0
def to_flat_tm(col, cell):
    """
    Col and cell index to flattened representation, assuming no segments
    """
    return col * cells_per_col + cell


def to_flat_segments(col, cell, seg=0):
    """
    Col, cell and segment index to flattened representation
    """
    result = col * cells_per_col * max_segments_per_cell + (cell * max_segments_per_cell) + seg
    return result


def unflatten_segments(flat):
    """
    Given a flattened index with segments, convert to col, cell and segment indices
    :param flat:
    :return:
    """
    col = flat // (cells_per_col * max_segments_per_cell)
    remain = flat % (cells_per_col * max_segments_per_cell)
    cell = remain // max_segments_per_cell
    seg = remain % max_segments_per_cell
    return (col, cell, seg)

def csr_double(a):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one.
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""

    #a.data = np.hstack((a.data,b.data))
    #a.indices = np.hstack((a.indices,b.indices))
    extra_rows = a.shape[0] if a.shape[0] else 1
    a.indptr = np.append(a.indptr, extra_rows * [a.nnz])#np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+extra_rows,a.shape[1])
    return a

def arr_double(a):
    extra_rows = a.shape[0] if a.shape[0] else 1
    return np.append(a, extra_rows * [0])


class TemporalMemory(object):
    def __init__(self):
        self.actives = csr_matrix((1, tm_size_flat), dtype=bool)
        self.winners = csr_matrix((1, tm_size_flat), dtype=bool)
        self.active_segs = csr_matrix((sp_size_flat, cells_per_col * max_segments_per_cell), dtype=bool)
        self.matching_segs = csr_matrix((sp_size_flat, cells_per_col * max_segments_per_cell), dtype=bool)

        self.matches_per_col = np.zeros((sp_size_flat,))  # csc_matrix((1,sp_size_flat))
        self.actives_per_col = np.zeros((sp_size_flat,))

        # These buffers contain a list of values followed by a list of row [and col] indices
        # At the end of each step, each buffer must be added to the appropriate matrix/array
        self.permanence_updates_buffer = [[], [], []]
        self.active_updates_buffer = [[], []]
        self.winner_updates_buffer = [[], []]

        #self.active_pot_counts = [0] * l#csc_matrix((1, tm_size_flat[0] * max_segments_per_cell), dtype=np.int)
        self.active_pot_counts = [0] * max_segs_total #np.array(max_segs_total, dtype=np.int)#csc_matrix((1, tm_size_flat[0] * max_segments_per_cell), dtype=np.int)

        self.seg_matrix = csr_matrix((0,tm_size_flat))#csc_matrix((tm_size_flat[0], tm_size_flat[0] * max_segments_per_cell))
        self.seg_linkings = dict()
        self.seg_linkings_reverse = np.empty((0,), dtype=int)
        self.seg_counts = defaultdict(int)

    def add_segment(self, col_id, cell_id):
        """
        Create a new segment on a specific cell.
        """
        # The synapse matrix is large enough to accomodate for any segment,
        # just remember how many are created on the current cell.
        index = to_flat_tm(col_id, cell_id)
        index_seg = to_flat_segments(col_id, cell_id, self.seg_counts[index])
        if self.seg_matrix.shape[0] == len(self.seg_linkings):
            self.seg_matrix = csr_double(self.seg_matrix)
        self.seg_counts[index] += 1
        if self.seg_linkings_reverse.shape[0] == len(self.seg_linkings):
            self.seg_linkings_reverse = arr_double(self.seg_linkings_reverse)
        self.seg_linkings_reverse[len(self.seg_linkings)] =index_seg
        self.seg_linkings[index_seg] = len(self.seg_linkings)

        return self.seg_counts[index] - 1

    def get_least_used_cell(self, col):
        """
        Of all cells for the given col, pick one so that no other cell has fewer segments.
        """
        minimum = max_segments_per_cell
        mins = []
        # Get all cells for which no cell with fewer segments exists for this col
        for i in range(cells_per_col):
            this_len = self.seg_counts[to_flat_tm(col, i)]
            if this_len == minimum:
                mins.append(i)
            elif this_len < minimum:
                minimum = this_len
                mins = [i]
        # From those, pick one at random
        return random.choice(mins) if len(mins) else None

    def get_best_matching_seg(self, col):
        """
        Of all segments for any cell in a specific column, get the one with most matching synapses.
        In case of ties, get the first one (??)
        """
        best_score = -1
        best_cell = None
        best_seg = None
        #for idx in find(self.matching_segs_old[:, _from:_to])[1]:
        for idx in self.matching_segs.indices[self.matching_segs.indptr[col]:self.matching_segs.indptr[col + 1]]:
            (c, cell, seg_idx) = unflatten_segments(idx + (col * cells_per_col * max_segments_per_cell))
            #assert c == col
            full_idx = to_flat_segments(c, cell, seg_idx)
            #assert full_idx == idx + (col * cells_per_col * max_segments_per_cell)
            seg_linked = self.seg_linkings[full_idx]
            this_score = self.active_pot_counts[seg_linked]
            if this_score > best_score:
                best_cell = cell
                best_seg = seg_idx
                best_score = this_score

        return (best_cell, best_seg)

    def grow_synapses(self, col, cell, seg_idx, count):
        """
        For a given segment, grow up to a number of synapses to winner cells of the previous step.
        """
        if count <= 0:
            return

        idx = to_flat_segments(col, cell, seg_idx)

        idx_toseg = self.seg_linkings[idx]
        # Check which aren't grown yet
        #unconnected = np.setdiff1d(self.winners.indices, self.seg_matrix.indices[
        #                                                     self.seg_matrix.indptr[idx_toseg]:self.seg_matrix.indptr[
        #                                                         idx_toseg + 1]])
        seta = set(self.winners.indices)
        setb = set(self.seg_matrix.indices[
                                                             self.seg_matrix.indptr[idx_toseg]:self.seg_matrix.indptr[
                                                                 idx_toseg + 1]])
        unconnected = seta.difference(setb)
        #assert unconnected.size == len(temp)
        #if not unconnected.size:
        if not len(unconnected):
            # Nothing left to grow to
            return
        #count = min(unconnected.size, count)
        count = min(len(unconnected), count)
        # Pick targets at random
        #for ind in np.random.choice(unconnected, count):
        for ind in random.sample(unconnected, count):
            # Store where to grow to, actually grow them all together later on for efficiency
            self.permanence_updates_buffer[0].append(initial_perm)
            self.permanence_updates_buffer[1].append(idx_toseg)
            self.permanence_updates_buffer[2].append(ind)

    def burst(self, col):
        """
        Burst all cells in an unpredicted column.
        """
        # Activate all cells in the col
        _from = to_flat_tm(col, 0)
        _to = to_flat_tm(col + 1, 0)
        # Actually apply these later on, for efficiency
        self.active_updates_buffer[0].extend(cells_per_col * [True])
        self.active_updates_buffer[1].extend(list(range(_from, _to)))
        is_new_seg = False
        if self.get_matching_segs_for_col_count(col):
            # Winner cell is the one with the best matching segment...
            (winner_cell, learning_seg) = self.get_best_matching_seg(col)
        else:
            # ...or if there are not with the least segments
            winner_cell = self.get_least_used_cell(col)
            if learning_enabled and winner_cell is not None:
                # Grow a new segment because none matched this sequence
                learning_seg = self.add_segment(col, winner_cell)
                is_new_seg = True
            elif learning_enabled and winner_cell is None:
                pass
                #print("Ran out of segment space!")


        # Actually set the winner cell later on
        if winner_cell is not None: #hotfix for low seg count
            self.winner_updates_buffer[0].append(True)
            self.winner_updates_buffer[1].append(to_flat_tm(col, winner_cell))

        if learning_enabled and winner_cell is not None:

            seg_idx = to_flat_segments(col, winner_cell, learning_seg)
            seg_linked = self.seg_linkings[seg_idx]
            seg = self.seg_matrix.indices[self.seg_matrix.indptr[seg_linked]:self.seg_matrix.indptr[seg_linked + 1]]
            # Find which synapses are connected to previously active cells (i.e., contributed to seg being matching)
            #active_idxs = set(np.intersect1d(seg, self.actives.indices))
            active_idxs = set(self.actives.indices)

            for syn_col in seg:
                # Reward contributing synapses, punish others
                if syn_col in active_idxs:
                    # Actually apply later
                    self.permanence_updates_buffer[0].append(perm_inc_step)
                else:
                    self.permanence_updates_buffer[0].append(-perm_dec_step)
                self.permanence_updates_buffer[1].append(seg_linked)
                self.permanence_updates_buffer[2].append(syn_col)

            # Aim for specific number of potential synapses for winner segment
            new_syn_count = max_synapses_per_segment - (self.active_pot_counts[seg_linked] if not is_new_seg else 0)
            if new_syn_count:
                self.grow_synapses(col, winner_cell, learning_seg, new_syn_count)

    def activate_predicted_col(self, col):
        """
        At least one activate segment in this activated column, activate all cells with active segment
        """
        if log.has_trace():
            log.trace("Active col has {} active segs".format(self.get_activated_segs_for_col_count(col)))
        for idx in self.get_activated_segs_for_col(col):
            #idx = idx + (col * cells_per_col * max_segments_per_cell)
            cell = idx // max_segments_per_cell
            seg_idx = idx % max_segments_per_cell
            cell_idx = to_flat_tm(col, cell)
            # Actually apply later
            self.active_updates_buffer[0].append(True)
            self.active_updates_buffer[1].append(cell_idx)
            self.winner_updates_buffer[0].append(True)
            self.winner_updates_buffer[1].append(cell_idx)

            if learning_enabled:
                idx = to_flat_segments(col, cell, seg_idx)
                seg_linked = self.seg_linkings[idx]
                existing_synapses = self.seg_matrix.indices[
                                    self.seg_matrix.indptr[seg_linked]:self.seg_matrix.indptr[seg_linked + 1]]

                # Actives_old_perms contains permanence increase value for previously active cells
                # and the decrease value for previously inactive
                data = [self.actives_old_perms[idx] for idx in existing_synapses]
                if log.has_trace():

                    data_arr = np.array(data)

                    log.trace("active seg has {} inc {} dec for {} synapses".format(len(data_arr[data_arr > 0]), len(data_arr[data_arr < 0]), len(existing_synapses)))
                self.permanence_updates_buffer[0].extend(data)
                self.permanence_updates_buffer[1].extend(len(existing_synapses) * [seg_linked])
                self.permanence_updates_buffer[2].extend(existing_synapses)
                new_syn_count = max_synapses_per_segment - self.active_pot_counts[seg_linked]
                if new_syn_count:
                    self.grow_synapses(col, cell, seg_idx, new_syn_count)

    def get_activated_segs_for_col(self, col):
        """
        Gets the previous step's activated segment indices (flattened) for one column
        """
        return self.active_segs.indices[self.active_segs.indptr[col]:self.active_segs.indptr[col + 1]]
        #return self.active_segs_old[:, to_flat_segments(col, 0):to_flat_segments(col + 1, 0)]

    def get_activated_segs_for_col_count(self, col):
        """
        Counts the number of activated segments in the previous step for one column
        """
        val = self.actives_per_col[col]
        return val

    def get_matching_segs_for_col(self, col):
        """
        Gets the previous step's matching segment indices (flattened) for one column

        """
        return self.matching_segs.indices[self.matching_segs.indptr[col]:self.matching_segs.indptr[col + 1]]
        #return self.matching_segs_old[:, to_flat_segments(col, 0):to_flat_segments(col + 1, 0)]

    def get_matching_segs_for_col_count(self, col):
        """
        Counts the number of matching segments in the previous step for one column
        """
        return self.matches_per_col[col]

    def punish_predicted(self, col):
        """
        Column was predicted to become active but didn't. Punish all synapses contributing to this prediction
        """
        if learning_enabled:
            for match_idx in self.get_matching_segs_for_col(col):

                (c, cell, seg) = unflatten_segments(match_idx + (col * cells_per_col * max_segments_per_cell))
                #assert c == col
                seg_linked = self.seg_linkings[to_flat_segments(col, cell, seg)]
                for idx in self.seg_matrix.indices[
                           self.seg_matrix.indptr[seg_linked]: self.seg_matrix.indptr[
                               seg_linked+1]]:
                    # Actives_old_t is CSC matrix of previously active cells (all in 1 row).
                    # If this and the next cell have the same indptr, there are no nonzero values in that column
                    # so it wasn't active. There are probably cleaner ways of doing this as efficiently.
                    if self.actives_old_t.indptr[idx] != self.actives_old_t.indptr[idx + 1]:
                        self.permanence_updates_buffer[0].append(-perm_dec_predict_step)
                        self.permanence_updates_buffer[1].append(seg_linked)
                        self.permanence_updates_buffer[2].append(idx)

    def activate(self):
        """
        Calculate the matching and active segments, for use in the next step
        """

        # Broadcasting pointwise multiplication, contains permanences of synapses to active cells
        global timer
        t = time.time()
        active_synapses = self.seg_matrix.multiply(self.actives)[0:len(self.seg_linkings),:]
        tt = time.time()
        timer += tt-t
        connected_synapses = active_synapses.copy()  # Copy because we still need this version for potentials
        # Only considered connected if permanence is high enough
        connected_synapses.data[connected_synapses.data < connected_perm] = 0.0
        connected_synapses.eliminate_zeros()

        connected_synapses.has_canonical_format = True # Avoids sum_duplicates; not necessary here and slow
        connected_synapses = connected_synapses.astype(bool)
        conn_syns_counts = connected_synapses.sum(axis=1)
        # Segment is only active if there are enough connected synapses
        conn_syns_counts[conn_syns_counts < activation_thresh] = 0
        #self.active_segs = csc_matrix(conn_syns_counts, dtype=bool)
        conn_syns_counts_full = coo_matrix((np.squeeze(np.asarray(conn_syns_counts)), (self.seg_linkings_reverse[:len(self.seg_linkings)], [0] * len(self.seg_linkings))),
                                           dtype=bool, shape=(sp_size_flat*cells_per_col*max_segments_per_cell,1))
        self.active_segs = conn_syns_counts_full.reshape((sp_size_flat, cells_per_col * max_segments_per_cell),
                                                                    order='C').tocsr()

        #TEMP DEBUG CHECK
        self.active_segs.eliminate_zeros() #TODO apparently the Falses are still in there. More efficient solution possible?
        #print("Active seg count:", self.active_segs.data.shape)
        # finds = find(self.active_segs)
        # for i in range(finds[0].shape[0]):
        #     (row, col, val) = (finds[0][i], finds[1][i], finds[2][i])
        #     cell = col // max_segments_per_cell
        #     seg_idx = col % max_segments_per_cell
        #     this_idx = to_flat_segments(row, cell, seg_idx)
        #     assert this_idx in self.seg_linkings

        # Any permanence is enough to be potential
        active_synapses.has_canonical_format = True # Avoids sum_duplicates; not necessary here and slow
        pot_syns_counts = active_synapses.astype(bool).sum(axis=1)
        self.active_pot_counts = np.squeeze(np.asarray(pot_syns_counts)).tolist()#dok_matrix(pot_syns_counts)

        # Segment is only matching if there are enough potential connections
        pot_syns_counts[pot_syns_counts < learning_thresh] = 0
        pot_syns_counts_full = coo_matrix((np.squeeze(np.asarray(pot_syns_counts)), (self.seg_linkings_reverse[:len(self.seg_linkings)], [0] * len(self.seg_linkings))),
                                          dtype=bool, shape=(sp_size_flat*cells_per_col*max_segments_per_cell,1))

        #self.matching_segs = csr_matrix(pot_syns_counts_full, dtype=bool) #TODO CONFIRM ORDER

        self.matching_segs = pot_syns_counts_full.reshape((sp_size_flat, cells_per_col * max_segments_per_cell), order='C').tocsr()
        self.matching_segs.eliminate_zeros()
        #TEMP DEBUG CHECK
        #print("Matching seg count:", self.matching_segs.data.shape)
        # finds = find(self.matching_segs)
        # for i in range(finds[0].shape[0]):
        #     (row, col, val) = (finds[0][i], finds[1][i], finds[2][i])
        #     cell = col // max_segments_per_cell
        #     seg_idx = col % max_segments_per_cell
        #     this_idx = to_flat_segments(row, cell, seg_idx)
        #     assert this_idx in self.seg_linkings

        # Reshape matrix to have 1 col per TM column instead of per cell, for easy counting
        self.matches_per_col = np.asarray(self.matching_segs
            .sum(
                axis=1)).ravel()
        #print("Should contain ints:", self.matches_per_col.dtype)
        self.actives_per_col = np.asarray(self.active_segs.sum(axis=1)).ravel()
        #print("cols with active:", np.count_nonzero(self.actives_per_col))
        #print("cols with matching:", np.count_nonzero(self.matches_per_col))
        # assert self.actives_per_col.sum() == self.active_segs.data.shape[0]
        # assert self.matches_per_col.sum() == self.matching_segs.data.shape[0]

    def update_synapses(self):
        """
        Performs the actual changes to the permanence matrix for one step.
        Grouped into one addition for efficiency
        """
        if len(self.permanence_updates_buffer[0]):
            # COO for easy creation, to CSC for efficient addition
            modder = csr_matrix((self.permanence_updates_buffer[0],
                                 (self.permanence_updates_buffer[1], self.permanence_updates_buffer[2])),
                                shape=self.seg_matrix.shape)

            self.seg_matrix = (self.seg_matrix + modder)


    def update_actives_and_winners(self):
        """
        Performs the actual changes to the winner and active matrices for one step.
        Grouped into one addition each for efficiency
        """
        if len(self.active_updates_buffer[0]):
            #(data,indices,indptr)
            self.actives = csr_matrix((self.active_updates_buffer[0],
                                       self.active_updates_buffer[1], [0,len(self.active_updates_buffer[0])]),
                                      shape=(1,tm_size_flat), dtype=bool)
        if len(self.winner_updates_buffer[0]):
            self.winners = csr_matrix((self.winner_updates_buffer[0],
                                       self.winner_updates_buffer[1], [0,len(self.winner_updates_buffer[0])]),
                                      shape=(1,tm_size_flat), dtype=bool)


    def step_end(self):
        """
        End-of-step bookkeeping. Move current matrices to their _old versions and reset them
        :return:
        """
        global timera, timerb

        self.actives_old_t = self.actives.transpose().tocsr()

        dense_acts = self.actives.todense()  # Not too big, can densify safely

        # List of cells: active ones replaced by increase in permanence, inactives by decrease
        self.actives_old_perms = \
        dense_acts * perm_inc_step - np.invert(dense_acts) * perm_dec_step
        t = time.time()

        self.actives_old_perms = self.actives_old_perms.tolist()[0]
        tt = time.time()
        timera += tt - t
        #timerb += ttt - tt
        #self.active_pot_counts_old = self.active_pot_counts
        #self.active_pot_counts = csc_matrix((1, tm_size_flat[0] * max_segments_per_cell))

        self.permanence_updates_buffer = [[], [], []]
        self.winner_updates_buffer = [[], []]
        self.active_updates_buffer = [[], []]

    def reset(self):
        self.actives = csr_matrix((1, tm_size_flat), dtype=bool)
        self.actives_old_t = self.actives.transpose().tocsr()
        self.winners = csr_matrix((1, tm_size_flat), dtype=bool)
        self.active_segs = csr_matrix((sp_size_flat, cells_per_col * max_segments_per_cell), dtype=bool)
        self.matching_segs = csr_matrix((sp_size_flat, cells_per_col * max_segments_per_cell), dtype=bool)
        self.active_pot_counts = [0] * max_segs_total #dok_matrix((1, tm_size_flat[0] * max_segments_per_cell))


    def step(self, activated_cols):
        """
        The full step: from SP active columns to TM active cells
        """
        for col in range(sp_size_flat):
            if col in activated_cols:
                # For each active SP column, either...
                if self.get_activated_segs_for_col_count(col) > 0:
                    # ...activate the cell(s) predicted for that column
                    self.activate_predicted_col(col)
                else:
                    # ...or burst all cells
                    self.burst(col)


            else:
                # For each inactive SP column, punish anything that predicted it to be active
                if self.get_matching_segs_for_col_count(col):
                    self.punish_predicted(col)

        # The previous steps generated lists of updates to perform to the synapses and active/winner cells,
        # but they still have to be actually applied to the sparse matrices
        self.update_synapses()
        self.update_actives_and_winners()

        # Given the current state, generate the active/matching segments for the next step
        self.activate()

        # Bookkeeping towards next step
        self.step_end()
        global timera, timerb
        print(timera, timerb)
        # self.actives is already reset to return the "old" actives
        return self.actives
