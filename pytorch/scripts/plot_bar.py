import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_performance_charts(csv_path, output_dir="charts"):
    # Load the CSV data into a DataFrame, setting the first column as the index
    df = pd.read_csv(csv_path, index_col=0)

    # Extract the full configuration type (which combines type + GPU config) as a new column
    df['full_config'] = df.index.to_series().apply(lambda x: '_'.join(x.split('_')[:6]))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the order for y-axis items
    y_axis_order = [
        "LambdaOD_1x_Texas_H100_80GB_SXM5",
        "LambdaOD_2x_Texas_H100_80GB_SXM5",
        "LambdaOD_4x_Texas_H100_80GB_SXM5",
        "LambdaOD_8x_Texas_H100_80GB_SXM5",
        "LambdaOD_2x_Texas_2xH100_80GB_SXM5",
        "LambdaOD_4x_Texas_2xH100_80GB_SXM5",
        "LambdaOD_8x_Texas_2xH100_80GB_SXM5",
        "LambdaOD_4x_Texas_4xH100_80GB_SXM5",
        "LambdaOD_8x_Texas_4xH100_80GB_SXM5",
        "LambdaOD_8x_Texas_8xH100_80GB_SXM5"
    ]

    models = ['ssd', 'bert_base_squad', 'bert_large_squad', 'gnmt', 'resnet50', 'tacotron2', 'waveglow']

    # Generate plots for each model
    for model in models:
        plt.figure(figsize=(10, 6))

        means = []
        stds = []

        # Gather mean and std for each item in y_axis_order
        for config in y_axis_order:
            config_data = df[df['full_config'] == config][model]
            means.append(config_data.mean())
            stds.append(config_data.std())

        # Plot the horizontal bar chart
        plt.barh(y_axis_order, means, xerr=stds, capsize=5)
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.text(mean + (std + 0.03 * max(means) if std else 0.05 * max(means)), i, f'{mean:.2f}±{std:.2f}', va='center')

        plt.xlim(0, max(means) * 1.25)
        plt.title(f'{model} Performance Across Configurations')
        plt.xlabel('Performance (mean ± std)')
        plt.tight_layout()

        # Save the figure as PNG
        output_path = os.path.join(output_dir, f'{model}_performance_across_configs.png')
        plt.savefig(output_path)
        plt.close()

    print(f"Charts saved in {output_dir}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate performance comparison charts for models across all GPU configurations")
    parser.add_argument('--csv_path', type=str, help='Path to the CSV file containing performance data')
    parser.add_argument('--output_dir', type=str, default='charts', help='Directory to save the output charts')

    # Parse arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    plot_performance_charts(args.csv_path, args.output_dir)
