from env.gridworld import GridWorld
from agent.monte_carlo import MonteCarloAgent
import numpy as np
import os
from visualize import plot_heatmap, make_video, plot_rewards

def main():
    env = GridWorld(size=4)
    agent = MonteCarloAgent(env, episodes=10000, save_every=1000, results_dir='results')
    agent.first_visit_mc()

    # Generate heatmaps and video
    image_files = []
    for ep in range(1000, 10001, 1000):
        V = np.load(os.path.join('results', f'V_{ep}.npy'), allow_pickle=True).item()
        img_file = os.path.join('results', f'V_{ep}.png')
        plot_heatmap(V, img_file, size=4)
        image_files.append(img_file)
    make_video(image_files, os.path.join('results', 'V_evolution.gif'))

    # Plot rewards
    rewards = np.load(os.path.join('results', 'rewards_10000.npy'))
    plot_rewards(rewards, os.path.join('results', 'rewards.png'))

if __name__ == "__main__":
    main()
