from visualize import plot_process_summary
import glob

if __name__ == "__main__":
    rewards_file = 'results/rewards_10000.npy'
    value_files = sorted(glob.glob('results/V_*.npy'))
    output_file = 'results/process_summary.png'
    plot_process_summary(rewards_file, value_files, output_file)
    print(f"Process summary plot saved as {output_file}")
