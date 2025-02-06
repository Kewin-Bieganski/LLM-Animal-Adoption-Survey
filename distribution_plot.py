import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# Define the weight keys
weighted_keys = ["W1", "W2", "W3", "W4", "W5", "W6"]

# Updated function to label the actual bin counts for lowest and most occurring bins
def plot_distribution_with_actual_bin_counts(data, column, title, thresholds=None, bins=20, xlabel=None):
    """
    Plots the distribution of a column with enhanced styling and marks statistical metrics (mean, median, mode).
    Optionally calculates and displays the percentage of data above thresholds.
    Labels the bins with the lowest and most occurring counts.

    Args:
        data (pd.DataFrame): The data containing the column.
        column (str): The column to plot.
        title (str): The title of the plot.
        thresholds (list, optional): Thresholds to mark on the plot.
        bins (int, optional): Number of bins for the histogram.
        xlabel (str, optional): Label for the x-axis.
    """
    plt.figure(figsize=(12, 8))

    # Plot histogram and kernel density estimate (KDE)
    sns.histplot(data[column], bins=bins, kde=True, color="blue", alpha=0.2, line_kws={'lw': 2})

    # Calculate statistics
    mean_value = data[column].mean()
    median_value = data[column].median()
    mode_value = data[column].mode()[0]

    # Plot statistics
    plt.axvline(mean_value, color="#ff0000", linestyle="--", linewidth=2, label=f"Mean: {mean_value:.2f}")
    plt.axvline(median_value, color="#ff9900", linestyle="-.", linewidth=2, label=f"Median: {median_value:.2f}")
    plt.axvline(mode_value, color="#00cc00", linestyle="-.", linewidth=2, label=f"Mode: {mode_value:.2f}")

    # Calculate and display percentages above thresholds
    if thresholds:
        for threshold in thresholds:
            percentage_above = (data[column] > threshold).mean() * 100
            plt.axvline(
                threshold, color="#0000ff", linestyle=":", linewidth=2.5,
                label=f"Threshold: {threshold} ({percentage_above:.1f}%)"
            )

    # Find the lowest and most occurring values
    bin_counts, bin_edges = np.histogram(data[column], bins=bins)

    # Mark the highest occurring bin, including the actual bin count
    highest_value_index = np.argmax(bin_counts)
    highest_value = (bin_edges[highest_value_index], bin_edges[highest_value_index + 1])
    plt.scatter(
        [sum(highest_value) / 2], [bin_counts[highest_value_index]],
        color="green", s=180, marker='^', label=f"Highest Bin: {highest_value[0]:.2f}-{highest_value[1]:.2f} (Count: {bin_counts[highest_value_index]})"
    )

    # Mark lowest bin, including the actual bin count
    lowest_value_index = np.argmin(bin_counts)
    lowest_value = (bin_edges[lowest_value_index], bin_edges[lowest_value_index + 1])
    plt.scatter(
        [sum(lowest_value) / 2], [bin_counts[lowest_value_index]],
        color="red", s=180, marker='v', label=f"Lowest Bin: {lowest_value[0]:.2f}-{lowest_value[1]:.2f} (Count: {bin_counts[lowest_value_index]})"
    )

    # Add title, labels, and legend
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel if xlabel else column, fontsize=14)
    plt.ylabel("Frequency", fontsize=14)

    # Customize ticks and grid
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(color="#cccccc", linestyle="--", linewidth=0.7, alpha=0.7)

    # Add subticks for percentage thresholds
    if "percentage" in column.lower():
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(5))
        plt.gca().tick_params(axis="x", which="minor", length=5, color="#cccccc")

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


# Apply updated function to relevant distributions
for mode, file in files.items():
    try:
        # Load data
        df = pd.read_csv(file)

        # Non-weighted percentage distribution
        plot_distribution_with_actual_bin_counts(
            df, "non_weighted_percentage",
            title=f"Distribution of Non-Weighted Percentage ({mode})",
            xlabel="Non-Weighted Percentage (%)",
            thresholds=[50, 75, 90]
        )

        # Weighted percentages distribution
        for key in weighted_keys:
            column_name = f"weighted_percentages_{key}"
            if column_name in df.columns:
                plot_distribution_with_actual_bin_counts(
                    df, column_name,
                    title=f"Distribution of Weighted Percentages ({key}) - {mode}",
                    xlabel=f"Weighted Percentages ({key})",
                    thresholds=[50, 75, 90]
                )

    except Exception as e:
        print(f"Error processing file {file} for {mode}: {e}")
