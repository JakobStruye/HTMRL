from scipy.sparse import csc_matrix, dok_matrix, find, coo_matrix
import numpy as np
import random
from collections import defaultdict

np.random.seed(1234)

cells_per_col = 32

activation_thresh = 16
initial_perm = 0.3
connected_perm = 0.25  # 0.5
learning_thresh = 12
learning_enabled = True

perm_inc_step = 0.01
perm_dec_step = 0.001
perm_dec_predict_step = 0.0005
# max_seg set very low right now: serious impact on performance; should autoscale internally
max_segments_per_cell = 2
max_synapses_per_segment = 32

sp_size = (2048,)
sp_size_flat = np.prod(sp_size)

tm_size = (sp_size_flat, cells_per_col)
tm_size_flat = (np.prod(tm_size), 1)


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


class TemporalMemory(object):
    def __init__(self):
        self.cells = np.zeros(tm_size)
        self.segments = np.empty_like(self.cells).tolist()
        self.actives_old = csc_matrix(tm_size_flat, dtype=bool)
        self.actives_old_dok = set()
        self.actives = dok_matrix(tm_size_flat, dtype=bool)
        self.actives_compiled = None
        self.winners_old = csc_matrix(tm_size_flat, dtype=np.bool)
        self.winners = dok_matrix(tm_size_flat, dtype=bool)
        self.active_segs_old = csc_matrix((1, tm_size_flat[0] * max_segments_per_cell))
        self.active_segs = csc_matrix((1, tm_size_flat[0] * max_segments_per_cell))
        self.matching_segs_old = csc_matrix((1, tm_size_flat[0] * max_segments_per_cell))
        self.matching_segs = csc_matrix((1, tm_size_flat[0] * max_segments_per_cell))

        self.matches_per_col = np.zeros((sp_size_flat,))  # csc_matrix((1,sp_size_flat))
        self.actives_per_col = np.zeros((1, sp_size_flat))

        # These buffers contain a list of values followed by a list of row [and col] indices
        # At the end of each step, each buffer must be added to the appropriate matrix/array
        self.permanence_updates_buffer = [[], [], []]
        self.active_updates_buffer = [[], []]
        self.winner_updates_buffer = [[], []]

        self.active_pot_counts = csc_matrix((1, tm_size_flat[0] * max_segments_per_cell), dtype=np.int)
        self.active_pot_counts_old = csc_matrix((1, tm_size_flat[0] * max_segments_per_cell), dtype=np.int)

        self.seg_matrix = csc_matrix((tm_size_flat[0], tm_size_flat[0] * max_segments_per_cell))
        self.seg_counts = defaultdict(int)

    def add_segment(self, col_id, cell_id):
        """
        Create a new segment on a specific cell.
        """

        # The synapse matrix is large enough to accomodate for any segment,
        # just remember how many are created on the current cell.
        index = to_flat_tm(col_id, cell_id)
        self.seg_counts[index] += 1
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
        return random.choice(mins)

    def get_best_matching_seg(self, col):
        """
        Of all segments for any cell in a specific column, get the one with most matching synapses.
        In case of ties, get the first one (??)
        """
        best_score = -1
        _from = to_flat_segments(col, 0)
        _to = to_flat_segments(col + 1, 0)
        best_cell = None
        best_seg = None
        for idx in find(self.matching_segs_old[:, _from:_to])[1]:
            (c, cell, seg_idx) = unflatten_segments(idx + _from)

            if self.active_pot_counts_old[0, idx] > best_score:
                best_cell = cell
                best_seg = seg_idx
                best_score = self.active_pot_counts_old[0, idx]

        return (best_cell, best_seg)

    def grow_synapses(self, col, cell, seg_idx, count):
        """
        For a given segment, grow up to a number of synapses to winner cells of the previous step.
        """
        if count <= 0:
            return

        idx = to_flat_segments(col, cell, seg_idx)

        # Check which aren't grown yet
        unconnected = np.setdiff1d(self.winners_old.indices, self.seg_matrix.indices[
                                                             self.seg_matrix.indptr[idx]:self.seg_matrix.indptr[
                                                                 idx + 1]])
        if not unconnected.size:
            # Nothing left to grow to
            return
        count = min(unconnected.size, count)

        # Pick targets at random
        for ind in np.random.choice(unconnected, count):
            # Store where to grow to, actually grow them all together later on for efficiency
            self.permanence_updates_buffer[0].append(initial_perm)
            self.permanence_updates_buffer[1].append(ind)
            self.permanence_updates_buffer[2].append(idx)

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

        if self.get_matching_segs_for_col_count(col):
            # Winner cell is the one with the best matching segment...
            (winner_cell, learning_seg) = self.get_best_matching_seg(col)
        else:
            # ...or if there are not with the least segments
            winner_cell = self.get_least_used_cell(col)
            if learning_enabled:
                # Grow a new segment because none matched this sequence
                learning_seg = self.add_segment(col, winner_cell)

        # Actually set the winner cell later on
        self.winner_updates_buffer[0].append(True)
        self.winner_updates_buffer[1].append(to_flat_tm(col, winner_cell))

        if learning_enabled:

            seg_idx = to_flat_segments(col, winner_cell, learning_seg)
            seg = self.seg_matrix.indices[self.seg_matrix.indptr[seg_idx]:self.seg_matrix.indptr[seg_idx + 1]]
            # Find which synapses are connected to previously active cells (i.e., contributed to seg being matching)
            active_idxs = set(np.intersect1d(seg, self.actives_old.indices))

            for syn_col in seg:
                # Reward contributing synapses, punish others
                if syn_col in active_idxs:
                    # Actually apply later
                    self.permanence_updates_buffer[0].append(perm_inc_step)
                else:
                    self.permanence_updates_buffer[0].append(-perm_dec_step)
                self.permanence_updates_buffer[1].append(syn_col)
                self.permanence_updates_buffer[2].append(seg_idx)

            # Aim for specific number of potential synapses for winner segment
            new_syn_count = max_synapses_per_segment - self.active_pot_counts_old[0, seg_idx]
            if new_syn_count:
                self.grow_synapses(col, winner_cell, learning_seg, new_syn_count)

    def activate_predicted_col(self, col):
        """
        At least one activate segment in this activated column, activate all cells with active segment
        """

        for idx in self.get_activated_segs_for_col(col):
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
                existing_synapses = self.seg_matrix.indices[
                                    self.seg_matrix.indptr[idx]:self.seg_matrix.indptr[idx + 1]]

                # Actives_old_perms contains permanence increase value for previously active cells
                # and the decrease value for previously inactive
                data = [self.actives_old_perms[idx] for idx in existing_synapses]
                self.permanence_updates_buffer[0].extend(data)
                self.permanence_updates_buffer[1].extend(existing_synapses)
                self.permanence_updates_buffer[2].extend(
                    len(existing_synapses) * [to_flat_segments(col, cell, seg_idx)])

                new_syn_count = max_synapses_per_segment - self.active_pot_counts_old[
                    0, to_flat_segments(col, cell, seg_idx)]
                if new_syn_count:
                    self.grow_synapses(col, cell, seg_idx, new_syn_count)

    def get_activated_segs_for_col(self, col):
        """
        Gets the previous step's activated segment indices (flattened) for one column
        """
        start = to_flat_segments(col, 0)
        end = to_flat_segments(col + 1, 0)
        return self.active_segs_old.indices[self.active_segs_old.indptr[start]:self.active_segs_old.indptr[end]]

    def get_activated_segs_for_col_count(self, col):
        """
        Counts the number of activated segments in the previous step for one column
        """
        val = self.actives_per_col[0, col]
        return val

    def get_matching_segs_for_col(self, col):
        """
        Gets the previous step's matching segment indices (flattened) for one column

        """
        return self.matching_segs_old[:, to_flat_segments(col, 0):to_flat_segments(col + 1, 0)]

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
            for match_idx in find(self.get_matching_segs_for_col(col))[1]:

                (c, cell, seg) = unflatten_segments(match_idx)
                for idx in self.seg_matrix.indices[
                           self.seg_matrix.indptr[to_flat_segments(col, cell, seg)]: self.seg_matrix.indptr[
                               to_flat_segments(col, cell, seg + 1)]]:
                    # Actives_old_t is CSC matrix of previously active cells (all in 1 row).
                    # If this and the next cell have the same indptr, there are no nonzero values in that column
                    # so it wasn't active. There are probably cleaner ways of doing this as efficiently.
                    if self.actives_old_t.indptr[idx] != self.actives_old_t.indptr[idx + 1]:
                        self.permanence_updates_buffer[0].append(-perm_dec_predict_step)
                        self.permanence_updates_buffer[1].append(idx)
                        self.permanence_updates_buffer[2].append(to_flat_segments(col, cell, seg))

    def activate(self):
        """
        Calculate the matching and active segments, for use in the next step
        """

        # Broadcasting pointwise multiplication, contains permanences of synapses to active cells
        active_synapses = self.seg_matrix.multiply(self.actives_compiled)

        connected_synapses = active_synapses.copy()  # Copy because we still need this version for potentials
        # Only considered connected if permanence is high enough
        connected_synapses.data[connected_synapses.data < connected_perm] = 0.0
        connected_synapses.eliminate_zeros()

        connected_synapses.has_canonical_format = True # Avoids sum_duplicates; not necessary here and slow
        connected_synapses = connected_synapses.astype(bool)

        conn_syns_counts = connected_synapses.sum(axis=0)
        # Segment is only active if there are enough connected synapses
        conn_syns_counts[conn_syns_counts < activation_thresh] = 0
        self.active_segs = csc_matrix(conn_syns_counts, dtype=bool)

        # Any permanence is enough to be potential
        active_synapses.has_canonical_format = True # Avoids sum_duplicates; not necessary here and slow
        pot_syns_counts = active_synapses.astype(bool).sum(axis=0)
        self.active_pot_counts = dok_matrix(pot_syns_counts)

        # Segment is only matching if there are enough potential connections
        pot_syns_counts[pot_syns_counts < learning_thresh] = 0
        self.matching_segs = csc_matrix(pot_syns_counts, dtype=bool)

        # Reshape matrix to have 1 col per TM column instead of per cell, for easy counting
        self.matches_per_col = np.asarray(
            self.matching_segs.reshape((cells_per_col * max_segments_per_cell, sp_size_flat), order='F').sum(
                axis=0)).ravel()
        self.actives_per_col = self.active_segs.reshape((cells_per_col * max_segments_per_cell, sp_size_flat),
                                                        order='F').sum(axis=0)

    def update_synapses(self):
        """
        Performs the actual changes to the permanence matrix for one step.
        Grouped into one addition for efficiency
        """
        if len(self.permanence_updates_buffer[0]):
            # COO for easy creation, to CSC for efficient addition
            modder = coo_matrix((self.permanence_updates_buffer[0],
                                 (self.permanence_updates_buffer[1], self.permanence_updates_buffer[2])),
                                shape=(tm_size_flat[0], tm_size_flat[0] * max_segments_per_cell))
            modder.has_canonical_format = True # Avoids sum_duplicates; not necessary here and slow

            self.seg_matrix = (self.seg_matrix + modder.tocsc())

    def update_actives_and_winners(self):
        """
        Performs the actual changes to the winner and active matrices for one step.
        Grouped into one addition each for efficiency
        """
        if len(self.active_updates_buffer):
            modder = coo_matrix((self.active_updates_buffer[0],
                                 (self.active_updates_buffer[1], len(self.active_updates_buffer[1]) * [0])),
                                shape=(tm_size_flat))
            modder.has_canonical_format = True  # Avoids sum_duplicates; not necessary here and slow
            self.actives_compiled = modder.tocsc()
        if len(self.winner_updates_buffer):
            modder = coo_matrix((self.winner_updates_buffer[0],
                                 (self.winner_updates_buffer[1], len(self.winner_updates_buffer[1]) * [0])),
                                shape=(tm_size_flat))
            modder.has_canonical_format = True  # Avoids sum_duplicates; not necessary here and slow
            self.winners = modder.tocsc()

    def step_end(self):
        """
        End-of-step bookkeeping. Move current matrices to their _old versions and reset them
        :return:
        """
        self.actives_old = self.actives_compiled
        self.actives_old_t = self.actives_old.transpose().tocsc()
        self.actives_old_dok = set(find(self.actives)[0])  # As set for efficient contains

        dense_acts = self.actives_compiled.todense()  # Not too big, can densify safely
        # List of cells: active ones replaced by increase in permanence, inactives by decrease
        self.actives_old_perms = \
        np.squeeze(dense_acts * perm_inc_step - np.invert(dense_acts) * perm_dec_step).tolist()[0]

        self.actives = dok_matrix(tm_size_flat, dtype=bool)
        self.actives_compiled = None

        self.winners_old = self.winners.tocsc()
        self.winners = dok_matrix(tm_size_flat, dtype=bool)

        self.active_segs_old = self.active_segs
        self.active_segs = csc_matrix((1, tm_size_flat[0] * max_segments_per_cell))

        self.matching_segs_old = self.matching_segs
        self.matching_segs = csc_matrix((1, tm_size_flat[0] * max_segments_per_cell))

        self.active_pot_counts_old = self.active_pot_counts
        self.active_pot_counts = csc_matrix((1, tm_size_flat[0] * max_segments_per_cell))

        self.permanence_updates_buffer = [[], [], []]
        self.winner_updates_buffer = [[], []]
        self.active_updates_buffer = [[], []]

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
        # self.actives is already reset to return the "old" actives
        return self.actives_old
