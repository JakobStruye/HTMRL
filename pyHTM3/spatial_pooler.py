import numpy as np

import pyHTM3.log as log


class SpatialPooler:
    def __init__(self, input_size):

        self.size = 2048
        self.stimulus_thresh = 0 #not implemented
        self.boost_strength = 0. #not implemented
        self.init_synapse_count = 20
        self.connected_perm_thresh = 0.2
        self.active_columns_count = 40

        self.perm_inc_step = 0.01
        self.perm_dec_step = 0.005
        self.perm_min = 0.0
        self.perm_max = 1.0

        self.input_size = input_size
        self.input_size_flat = np.prod(input_size)

        self.permanences = self._get_initialized_permanences()

    def _get_initialized_permanences(self):
        # Permanences are represented by a 2D matrix (input size X SP size)
        # If there is a synapse between an input cell and SP cell, the matrix gives its permanence
        # otherwise the matrix value is NaN
        permanences = np.empty((self.input_size_flat, self.size), dtype=np.float)
        permanences[:, :] = np.nan # Init all to NaN first
        for col in range(self.size):
            #First choose which synapses to grow
            rand_selection = np.random.choice(self.input_size_flat, self.init_synapse_count, replace=False)
            #Then choose their initial permanences
            permanences[rand_selection, col] = self._get_initialized_segment()
        if log.has_debug():
            log.debug("SP has {} initialized synapses".format(np.count_nonzero(~np.isnan(permanences))))
        return permanences


    def _get_initialized_segment(self):
        vals = np.zeros((self.init_synapse_count,), dtype=float)
        # ~50% of all synapses should be active (permanence high enough); first decide which
        is_actives = [randval > 0.5 for randval in np.random.random(self.init_synapse_count)]
        # Then determine each permanence uniformly randomly in [min, thresh] or [thresh,max]
        for i in range(self.init_synapse_count):
            if is_actives[i]:
                vals[i] = self.connected_perm_thresh + (self.perm_max - self.connected_perm_thresh) * np.random.random()
            else:
                vals[i] = self.connected_perm_thresh * np.random.random()
        if (log.has_trace()):
            log.debug("Median:", np.median(vals))
            log.debug("Active: {}, average {}".format(len(vals[vals > self.connected_perm_thresh]), np.mean(vals[vals > self.connected_perm_thresh])))
            log.debug("Inactive: {}, average {}".format(len(vals[vals < self.connected_perm_thresh]),
                                                        np.mean(vals[vals < self.connected_perm_thresh])))
        return vals

    def _get_activated_cols(self, inputs):
        """
        Gets the indices of the activated columns
        """

        # Get all the active synapses.
        # impl: (because bool(nan) == True, filter those out manually
        connecteds = np.array((self.permanences - self.connected_perm_thresh).clip(min=0), dtype=bool) * (~ np.isnan(self.permanences))

        # Count the number of connected active input cells for each column
        conn_counts = np.dot(np.expand_dims(inputs, 0), np.array(connecteds, dtype=int))
        conn_counts = np.squeeze(conn_counts)

        # Get the top-k columns in terms of connected active input cells
        # impl: argpartition calculates the indices that sort arg0 such that
        # at least the first arg1 entries are the arg1 smallest values, in order.
        # Makes no guarantees about the rest of the values (but we don't need those)
        activated = np.argpartition(- conn_counts, self.active_columns_count)[:self.active_columns_count, ]
        return activated

    def _reinforce(self, inputs, activated):
        # Synapses to active inputs may be positively reinforced, the others negatively
        inputs_pos = inputs * self.perm_inc_step
        inputs_neg = (inputs - 1) * self.perm_dec_step
        inputs_shift = inputs_pos + inputs_neg
        if log.has_trace():
            log.trace("Reinforcing with {} pos {} neg".format(len(inputs_shift[inputs_shift > 0]), len(inputs_shift[inputs_shift < 0])))
        inputs_shift = np.expand_dims(inputs_shift, 1)
        # Reinforce only the synapses of the activated columns
        # impl: NaN + 1 == NaN, so all non-existing synapses don't get touched here
        self.permanences[:,activated] = self.permanences[:,activated] + inputs_shift
        self.permanences = self.permanences.clip(min=self.perm_min, max=self.perm_max)

    def step(self, inputs, learn=True):
        activated_cols = self._get_activated_cols(inputs)
        if learn:
            self._reinforce(inputs, activated_cols)
        return activated_cols




