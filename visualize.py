import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2

def plot_heatmap(V, filename, size=4):
    grid = np.zeros((size, size))
    for (i, j), v in V.items():
        grid[i, j] = v
    plt.figure(figsize=(5, 5))
    plt.title(f"State-Value Function: {filename}")
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='V(s)')
    for i in range(size):
        for j in range(size):
            plt.text(j, i, f"{grid[i, j]:.1f}", ha='center', va='center', color='w')
    plt.savefig(filename)
    plt.close()

def make_video(image_files, output_file):
    images = [imageio.imread(img) for img in image_files]
    imageio.mimsave(output_file, images, duration=0.8)

def plot_rewards(rewards, filename):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards Over Time')
    plt.savefig(filename)
    plt.close()

def plot_agent_path(path, filename, size=4):
    import matplotlib.pyplot as plt
    import numpy as np
    grid = np.zeros((size, size))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap='Greys', vmin=0, vmax=1)
    for (i, j) in path:
        ax.add_patch(plt.Circle((j, i), 0.3, color='blue', alpha=0.3))
    # Mark start and goal
    ax.add_patch(plt.Circle((0, 0), 0.3, color='green', alpha=0.7, label='Start'))
    ax.add_patch(plt.Circle((size-1, size-1), 0.3, color='red', alpha=0.7, label='Goal'))
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(filename)
    plt.close()

def make_agent_path_video(path, out_gif, size=4):
    import imageio
    frames = []
    for t in range(1, len(path)+1):
        fname = f"results/agent_path_{t}.png"
        plot_agent_path(path[:t], fname, size)
        frames.append(imageio.imread(fname))
    imageio.mimsave(out_gif, frames, duration=0.5)
    # Optionally, clean up pngs
    for t in range(1, len(path)+1):
        os.remove(f"results/agent_path_{t}.png")

def plot_process_summary(rewards_file, value_files, output_file):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # Plot rewards
    rewards = np.load(rewards_file)
    axs[0].plot(rewards)
    axs[0].set_title('Episode Rewards Over Time')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')
    # Plot value function heatmaps at different stages
    for vf in value_files:
        V = np.load(vf, allow_pickle=True).item()
        grid = np.zeros((4, 4))
        for (i, j), v in V.items():
            grid[i, j] = v
        axs[1].imshow(grid, cmap='viridis', interpolation='nearest')
        axs[1].set_title(f'V(s) at {os.path.basename(vf)}')
        plt.pause(0.5)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
