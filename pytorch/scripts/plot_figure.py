import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import seaborn as sns

# Read the merged CSV file
input_file = 'pytorch-train-throughput-v2-fp16_merged'

df = pd.read_csv(input_file + '.csv', index_col='brand')

columns_of_interest = ['resnet50', 'ssd', 'gnmt', 'bert_base_squad', 'bert_large_squad', 'tacotron2', 'waveglow', 'ncf']


# Set the aesthetics for the plots
sns.set(style="whitegrid")

# Create a figure with subplots
fig, axes = plt.subplots(len(columns_of_interest), 1, figsize=(12, 28))


# Define colors for std shading with higher contrast
dell_colors = ['#AEC6CF', '#B0E0E6', '#ADD8E6']  # Lighter blue shades
smc_colors = ['#FFDAB9', '#FFB6C1', '#FFA07A']  # Lighter orange shades

# Text properties
text_props = {
    'fontsize': 10,
    'fontweight': 'bold',
    'alpha': 0.8,
    'bbox': dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5')
}

# Iterate over columns of interest and corresponding axes
for ax, col in zip(axes, columns_of_interest):
    # Fit a Gaussian distribution for each brand
    dell_data = df.loc['DELL', col]
    smc_data = df.loc['SMC', col]

    dell_mean, dell_std = norm.fit(dell_data)
    smc_mean, smc_std = norm.fit(smc_data)

    # Define the x-axis range to cover the full data range with some padding
    x_min = min(dell_data.min(), smc_data.min()) - 1
    x_max = max(dell_data.max(), smc_data.max()) + 1
    x = np.linspace(x_min, x_max, 1000)

    # Plot the Gaussian distribution for DELL
    p = norm.pdf(x, dell_mean, dell_std)
    ax.plot(x, p, 'b-', label='DELL', linewidth=2, alpha=0.6)

    # Fill the standard deviation areas for DELL with higher contrast
    for i in range(1, 4):
        ax.fill_between(x, norm.pdf(x, dell_mean, dell_std), where=((x >= dell_mean - i * dell_std) & (x <= dell_mean + i * dell_std)), color=dell_colors[i-1], alpha=0.3)

    # Plot the Gaussian distribution for SMC
    p = norm.pdf(x, smc_mean, smc_std)
    ax.plot(x, p, 'orange', linestyle='--', label='SMC', linewidth=2, alpha=0.6)

    # Fill the standard deviation areas for SMC with higher contrast
    for i in range(1, 4):
        ax.fill_between(x, norm.pdf(x, smc_mean, smc_std), where=((x >= smc_mean - i * smc_std) & (x <= smc_mean + i * smc_std)), color=smc_colors[i-1], alpha=0.3)

    # Highlight the mean and std for DELL
    ax.axvline(dell_mean, color='blue', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(dell_mean, ax.get_ylim()[1] * 0.9, f'Mean: {dell_mean:.2f}\nStd: {dell_std:.2f}', color='blue', ha='center', **text_props)

    # Highlight the mean and std for SMC
    ax.axvline(smc_mean, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(smc_mean, ax.get_ylim()[1] * 0.7, f'Mean: {smc_mean:.2f}\nStd: {smc_std:.2f}', color='orange', ha='center', **text_props)

    # Set plot labels and title
    ax.set_xlabel(f'Performance in {col}', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{col}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig('performance_distribution.png', dpi=300)

# Clear the figure to free memory
plt.clf()