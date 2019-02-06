from enum import Enum
import time
import numpy as np
import pyglet


def action_to_direction(action):
    #0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
    action = (action + 2) % 4
    if action == 0:
        return np.array([0,1])
    elif action == 1:
        return np.array([1,0])
    elif action == 2:
        return np.array([0,-1])
    elif action == 3:
        return np.array([-1,0])

class MazeActions(Enum):
    UP=0
    RIGHT=1
    DOWN=2
    LEFT=3



class Maze():
    def __init__(self, env_config):
        size = env_config["size"]
        self.reward_shape_scale = env_config["reward_shape_scale"] if "reward_shape_scale" in env_config else 0.0
        # (0,0) is bottom left, (size-1,0) is bottom right
        assert(size >= 4)
        self.size = size
        self.goal = np.array([size-2,size-2])
        self.init = np.array([1,1])
        self.current = np.copy(self.init)
        self.sparse = False
        self.remaining = self._get_manhattan_distance(self.goal, self.current)

        self.realtime = False
        self.warned = None
        if "visualize" in env_config and env_config["visualize"] == True:
            self.visualize_enabled = True
            self.window = pyglet.window.Window(620,620)
            if "realtime" in env_config and env_config["realtime"] == True:
                self.realtime = True
                self._prev_time = 0
                self._interval = 0.02
                print("Running at reduced speed for visualization")
            else:
                print("Running at full speed despite visualization. Reduce speed with 'realtime'")

        else:
            self.visualize_enabled = False
            self.window = None


    def get_state(self):
        return self.current


    def _is_valid(self, loc):
        x,y = loc
        return 0 <= x < self.size and 0 <= y < self.size

    def _get_manhattan_distance(self, loc1, loc2):
        return np.sum(np.abs(loc1 - loc2))

    def do_action(self, action):
        now = time.time()
        if self.realtime and self._prev_time + self._interval > now:
            time.sleep((self._prev_time + self._interval) - now)
        self._prev_time = now
        new_loc = self.current + action_to_direction(action)
        if not self._is_valid(new_loc):
            return (self.current, -1.0) #Can't move
        else:
            if np.array_equal(new_loc, self.goal):
                reward = 1
                self.current = np.copy(self.init)
                self.remaining = self._get_manhattan_distance(self.goal, self.current)
            else:
                new_remaining = self._get_manhattan_distance(self.goal, new_loc)
                reward = (self.remaining - new_remaining) * self.reward_shape_scale
                self.current = new_loc
                self.remaining = new_remaining

            #temp hack
            #if reward >0:
            #    reward = 1
            #elif reward <= 0:
            #    reward = -1

            return (self.current, reward)

    def is_done(self):
        return np.array_equal(self.current, self.goal)
    def get_action_count(self):
        return 4

    def get_debug_info(self):
        return None

    def is_best(self, action):
        return action == MazeActions.RIGHT




    def visualize(self):
        if not self.visualize_enabled:
            if self.warned is None:
                print("Trying to visualize, but not enabled through the 'visualize' parameter")
                self.warned = True
            return
        self.window.clear()
        b = pyglet.graphics.Batch()

        self.draw_to_batch(b)
        b.draw()

        # Magic to avoid black window
        self.window.flip()
        self.window.dispatch_events()

    ################
    # PYGLET HELPERS
    ################
    def _index_to_pixel(self, index):
        available_size = 600
        margin = 10
        return int(available_size / self.size) * index + margin

    def _draw_square(self, x, y, color):
        _from_goal = (self._index_to_pixel(x), self._index_to_pixel(y))
        _to_goal = (self._index_to_pixel(x + 1), self._index_to_pixel(y + 1))
        pyglet.graphics.draw(5, pyglet.gl.GL_POLYGON, ('v2i', [_from_goal[0], _from_goal[1], _from_goal[0], _to_goal[1], _to_goal[0], _to_goal[1], _to_goal[0], _from_goal[1], _from_goal[0], _from_goal[1]]), ('c4B', color * 5))

    def draw_to_batch(self, b):

        for i in range(self.size+1):
            _from = self._index_to_pixel(i)
            b.add(2, pyglet.gl.GL_LINES, None,
                  ('v2i', (_from, 10, _from, 610))
                  )
            b.add(2, pyglet.gl.GL_LINES, None,
                  ('v2i', (10, _from, 610, _from))
                  )
        self._draw_square(self.goal[0], self.goal[1], [255,0,0,0])
        self._draw_square(self.init[0], self.init[1], [0,255,0,0])
        self._draw_square(self.current[0], self.current[1], [0,0,255,0])



if __name__ == '__main__':
    window = pyglet.window.Window(620,620)

    #label = pyglet.text.Label('Hello, world',
    #                          font_name='Times New Roman',
    #                          font_size=36,
    #                          x=window.width//2, y=window.height//2,
    #                          anchor_x='center', anchor_y='center')



    #@window.event
    #def on_draw():
    #    window.clear()
    #    label.draw()


    #window.clear()

    #label.draw()
    m = Maze({"size": 10})



    @window.event
    def on_draw():
        b = pyglet.graphics.Batch()

        m.draw_to_batch(b)
        b.draw()



    pyglet.app.run()
