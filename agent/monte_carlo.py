import numpy as np
from env.gridworld import GridWorld
import random
import os

class MonteCarloAgent:
    def __init__(self, env, episodes=10000, gamma=1.0, save_every=1000, results_dir='results'):
        self.env = env
        self.episodes = episodes
        self.gamma = gamma
        self.save_every = save_every
        self.results_dir = results_dir
        self.states = env.get_all_states()
        self.actions = env.get_actions()
        self.V = {s: 0.0 for s in self.states}
        self.returns = {s: [] for s in self.states}
        os.makedirs(results_dir, exist_ok=True)
        self.episode_rewards = []

    def random_policy(self, state):
        return random.choice(self.actions)

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        done = False
        while not done:
            action = self.random_policy(state)
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def first_visit_mc(self):
        for ep in range(1, self.episodes + 1):
            episode = self.generate_episode()
            G = 0
            visited = set()
            rewards = [x[2] for x in episode]
            self.episode_rewards.append(sum(rewards))
            for t, (state, action, reward) in enumerate(episode):
                if state not in visited:
                    visited.add(state)
                    G = sum([r * (self.gamma ** i) for i, r in enumerate([x[2] for x in episode[t:]])])
                    self.returns[state].append(G)
                    self.V[state] = np.mean(self.returns[state])
            if ep % self.save_every == 0:
                np.save(os.path.join(self.results_dir, f'V_{ep}.npy'), self.V)
                np.save(os.path.join(self.results_dir, f'rewards_{ep}.npy'), np.array(self.episode_rewards))
