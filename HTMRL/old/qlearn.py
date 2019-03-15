import numpy as np
import time

class QLearn:
    def __init__(self, dims, n_acts, eps):
        self.eps = eps
        self.eps_orig = eps
        self.eps_anneal_length = 1000000
        self.i = 0
        self.n_acts = n_acts
        self.qtable = np.zeros(dims + (n_acts,))
        self._tie_break_scale = 0.00001
        self.lr = 0.001
        self.discount = 0.#99

    def get_action(self, state):
        if self.eps_anneal_length > 0:
            self.eps = self.eps_orig * ((self.eps_anneal_length - self.i) / float(self.eps_anneal_length))
        else:
            self.eps = self.eps_orig
        self.i += 1
        #print(self.eps)
        if np.random.rand() <= self.eps:
            return np.random.randint(0,self.n_acts)
        else:
            action_qs = self.qtable[tuple(state)]
            return np.argmax(action_qs + (np.random.rand(self.n_acts) * self._tie_break_scale))

    def learn(self, prev_state, next_state, action, reward):
        cur_q = self.qtable[tuple(prev_state) + (action,)]
        best_next_q = np.max(self.qtable[tuple(next_state)])
        new_q = cur_q + self.lr * (reward + self.discount * best_next_q - cur_q)
        self.qtable[tuple(prev_state) + (action,)] = new_q


if __name__ == "__main__":
    ql = QLearn((10,10),4,0.00)
    from HTMRL.old.maze import Maze
    env = Maze({"size": 10, "visualize": True, "realtime": True})
    # action = ql.get_action(np.array([4,4]))
    # print(action)
    # ql.learn(np.array([4,4]), np.array([4,5]), 0, 0.1)
    # print(ql.qtable[4,4,0])
    state = env.get_state()
    founds = 0
    prev_goal = 0
    prev_dur = -1

    time.sleep(2) #Allow window to initialize
    for i in range(100000):
        #print("STEP", i)
        action = ql.get_action(state)
        next_state, rew = env.do_action(action)
        ql.learn(state, next_state, action, rew)
        state = next_state
        if rew == 1.0:
            if prev_dur != (i - prev_goal):
                prev_dur = i - prev_goal
                #print(prev_dur)
            prev_goal = i
            # founds += 1
            # print("FOUND "+ str(i))
        env.visualize()

    #pyglet.clock.schedule_interval(upd, 1 / 120.0, ql, env, state, prev_dur, prev_goal, i)
    #for i in range(1000000) :
    #    upd(0, ql, env, state, prev_dur, prev_goal, i)
    #pyglet.clock.schedule_once(upd, 1.0, ql, env, state, prev_dur, prev_goal)
    #pyglet.app.run()
