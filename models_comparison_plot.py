import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Flag to check if the dataset should be mostly neutral
isNeutral = True # Set this to True if you want "neutral", False for "biased"

# File paths for the data
files = {
    "Random": "survey_results_random.csv",
    "OpenAI GPT-4o-Mini": "survey_results_openai_(gpt-4o-mini).csv",
    "Google Gemini-1.5-Flash": "survey_results_google_(gemini-1.5-flash).csv"
}

# Function to update file paths based on the isNeutral flag
def update_file_paths(isNeutral, files):
    updated_files = {}
    prefix = "neutral" if isNeutral else "biased"

    for key, filepath in files.items():
        updated_files[key] = f"{prefix}/{filepath}"

    return updated_files

# Update the file paths with the correct prefix
files = update_file_paths(isNeutral, files)

# Weighted keys for the 6 different weights
weighted_keys = ["W1", "W2", "W3", "W4", "W5", "W6"]

# Placeholder for data
average_percentages = {'Non-Weighted': [], 'Weighted W1': [], 'Weighted W2': [], 'Weighted W3': [], 'Weighted W4': [], 'Weighted W5': [], 'Weighted W6': []}
min_percentages = {'Non-Weighted': [], 'Weighted W1': [], 'Weighted W2': [], 'Weighted W3': [], 'Weighted W4': [], 'Weighted W5': [], 'Weighted W6': []}
max_percentages = {'Non-Weighted': [], 'Weighted W1': [], 'Weighted W2': [], 'Weighted W3': [], 'Weighted W4': [], 'Weighted W5': [], 'Weighted W6': []}

# Read and process each file for the given modes
for mode, file in files.items():
    try:
        # Load data from the CSV file
        df = pd.read_csv(file)

        # Calculate average non-weighted percentage
        avg_non_weighted = df['non_weighted_percentage'].mean()
        # Calculate min and max non-weighted percentages
        min_non_weighted = df['non_weighted_percentage'].min()
        max_non_weighted = df['non_weighted_percentage'].max()

        # Append values for the non-weighted percentages
        average_percentages['Non-Weighted'].append(avg_non_weighted)
        min_percentages['Non-Weighted'].append(min_non_weighted)
        max_percentages['Non-Weighted'].append(max_non_weighted)

        # Calculate average, min, and max for each weighted percentage (W1 to W6)
        for i, key in enumerate(weighted_keys):
            avg_weighted = df[f'weighted_percentages_{key}'].mean()
            min_weighted = df[f'weighted_percentages_{key}'].min()
            max_weighted = df[f'weighted_percentages_{key}'].max()

            average_percentages[f'Weighted W{i+1}'].append(avg_weighted)
            min_percentages[f'Weighted W{i+1}'].append(min_weighted)
            max_percentages[f'Weighted W{i+1}'].append(max_weighted)

    except Exception as e:
        print(f"Error processing file {file} for {mode}: {e}")
        continue

# Define custom colors for each mode
mode_colors = {
    "Random": "#629FCA",  # Blue
    "OpenAI GPT-4o-Mini": "#FFA556",  # Orange
    "Google Gemini-1.5-Flash": "#6BBC6B"  # Green
}

# Create bars for each percentage type (non-weighted and weighted)
for key in average_percentages:
    plt.figure(figsize=(10, 6))  # New plot for each percentage type

    # Mark the minimum and maximum values for each type of percentage
    for i, mode in enumerate(files.keys()):
        plt.bar(i, average_percentages[key][i], 0.4, color=mode_colors[mode])
        # Mark the maximum value with a green triangle and label (below the marker)
        plt.scatter(i, max_percentages[key][i], color="green", s=100, marker="^", label=f"Max Percentage" if i == 0 else "")
        plt.text(i, max_percentages[key][i] - 6, f"{max_percentages[key][i]:.2f}%", ha='center', fontsize=12, color="green")

        # Mark the minimum value with a red triangle and label (below the marker)
        plt.scatter(i, min_percentages[key][i], color="red", s=100, marker="v", label=f"Min Percentage" if i == 0 else "")
        plt.text(i, min_percentages[key][i] - 6, f"{min_percentages[key][i]:.2f}%", ha='center', fontsize=12, color="red")

        # Mark the average value with a black circle and label (on the bar)
        plt.scatter(i, average_percentages[key][i], color="black", s=200, marker=".", label=f"Average Percentage" if i == 0 else "")
        plt.text(i, average_percentages[key][i] - 6, f"{average_percentages[key][i]:.2f}%", ha='center', fontsize=12, color="black")

    # Adding threshold lines to the Y-axis
    plt.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% Threshold')
    plt.axhline(y=75, color='gold', linestyle='--', linewidth=2, label='75% Threshold')
    plt.axhline(y=90, color='green', linestyle='--', linewidth=2, label='90% Threshold')

    # Labeling axes and adding title
    plt.title(f'Average Percentage Comparison across Different Modes ({key})', fontsize=16)
    plt.xlabel('Mode', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(np.arange(len(files)), files.keys())
    plt.yticks(np.arange(0, 101, 10))  # Y-axis with values from 0 to 100

    # Add subticks for percentage thresholds on Y-axis
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(5))
    plt.gca().tick_params(axis="y", which="minor", length=5, color="#cccccc")

    # Customize grid: Disable vertical grid lines, enable horizontal ones
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    # Add legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()
