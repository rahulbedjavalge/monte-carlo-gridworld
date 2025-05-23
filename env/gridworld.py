import numpy as np

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.start_state = (0, 0)
        self.goal_state = (size-1, size-1)
        self.state = self.start_state
        self.actions = ['up', 'down', 'left', 'right']

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 'up':
            x = max(x - 1, 0)
        elif action == 'down':
            x = min(x + 1, self.size - 1)
        elif action == 'left':
            y = max(y - 1, 0)
        elif action == 'right':
            y = min(y + 1, self.size - 1)
        next_state = (x, y)
        reward = 0 if next_state == self.goal_state else -1
        done = next_state == self.goal_state
        self.state = next_state
        return next_state, reward, done

    def get_all_states(self):
        return [(i, j) for i in range(self.size) for j in range(self.size)]

    def get_actions(self):
        return self.actions
