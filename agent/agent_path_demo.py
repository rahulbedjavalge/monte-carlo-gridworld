import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.gridworld import GridWorld
from visualize import make_agent_path_video
import random

def generate_random_episode(env):
    state = env.reset()
    path = [state]
    done = False
    while not done:
        action = random.choice(env.get_actions())
        next_state, reward, done = env.step(action)
        path.append(next_state)
        state = next_state
    return path

def main():
    env = GridWorld(size=4)
    path = generate_random_episode(env)
    make_agent_path_video(path, 'results/agent_path.gif', size=4)
    print('Agent path video saved as results/agent_path.gif')

if __name__ == "__main__":
    main()
