import json, os
import pandas as pd
import matplotlib.pyplot as plt

# Flag to check if the dataset should be mostly neutral
isNeutral = True # Set this to True if you want "neutral", False for "biased"

# Append "neutral" or "biased" based on the isNeutral flag
questions_file_path = f"questions_{'neutral' if isNeutral else 'biased'}.json"

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

# Questions for preference analysis (columns 13 to 20)
preference_question_ids = [13, 14, 15, 16, 17, 18, 19, 20]
preference_labels = {
    13: "Preferred Animal",
    14: "Dog or Cat Preference",
    15: "Breed Preference",
    16: "Health Status Preference",
    17: "Animal Size Preference",
    18: "Animal Age Preference",
    19: "Animal Gender Preference",
    20: "Activity Level Preference"
}

# Load and process data
data = {}
for mode, file in files.items():
    try:
        df = pd.read_csv(file)
        # Extract and preprocess preference answers
        df["preference_answers"] = df["preference_answers"].apply(eval)  # Convert string representation to list
        data[mode] = df
    except Exception as e:
        print(f"Error loading file {file}: {e}")

# Check if the number of records is the same across all files
num_records = {mode: len(df) for mode, df in data.items()}
if len(set(num_records.values())) > 1:
    print("Warning: Not all files have the same number of records.")
else:
    print(f"All files have the same number of records: {num_records}")

# Calculate frequency of each answer for each question
preference_frequencies = {qid: {} for qid in preference_question_ids}
for mode, df in data.items():
    for qid in preference_question_ids:
        # Extract all answers for the given question id
        answers = df["preference_answers"].apply(lambda x: x[qid - 13] if qid - 13 < len(x) else None).dropna()
        frequencies = answers.value_counts()

        # Convert frequencies to percentages
        total_answers = len(df)
        percentages = (frequencies / total_answers) * 100
        preference_frequencies[qid][mode] = percentages

import numpy as np

def load_json_object_from_file(file_path):
    """
    Loads a JSON object from a file and returns it.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict or list: The loaded JSON object.

    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If the file content is not valid JSON.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' was not found.")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json_object = json.load(file)
            return json_object
    except json.JSONDecodeError as e:
        raise ValueError(f"The file '{file_path}' contains invalid JSON.") from e


# Create grouped bar plots for preference questions
# Update preference_labels with formatted labels for grouped bar plot
# Load questions to retrieve answer labels
questions = load_json_object_from_file(questions_file_path)

# Create a mapping of question IDs to their answer labels
question_labels = {}
for question in questions:
    if question["type"] == "preference":
        question_labels[question["id"]] = {
            answer["id"]: answer["text"].strip(";") for answer in question["answers"]
        }

# Create grouped bar plots with enhanced labels
# Modify the plotting to ensure horizontal labels with word-wrap
# Modify the plotting to ensure horizontal labels with proper word-wrap
# Function to apply word-wrap to long labels
def wrap_text(text, line_length=30):
    """
    Wraps text at a specified line length by inserting newlines.

    Args:
        text (str): The text to wrap.
        line_length (int): Maximum line length before wrapping.

    Returns:
        str: The wrapped text with newlines.
    """
    words = text.split()
    wrapped_lines = []
    current_line = []

    for word in words:
        if len(" ".join(current_line + [word])) <= line_length:
            current_line.append(word)
        else:
            wrapped_lines.append(" ".join(current_line))
            current_line = [word]

    # Add the last line
    if current_line:
        wrapped_lines.append(" ".join(current_line))

    return "\n".join(wrapped_lines)


# Plotting with wrapped labels
for qid in preference_question_ids:
    plt.figure(figsize=(10, 6))
    bar_width = 0.2
    bar_positions = []
    modes = list(data.keys())
    labels = list(preference_frequencies[qid][modes[0]].index)  # Assuming all modes have the same labels for qid

    # Sort the labels based on their identifiers (e.g., "13_1", "13_2", "13_3")
    labels.sort(key=lambda x: int(x.split('_')[-1]))  # Sort by numeric part after '_'

    # Map labels to their text and apply word-wrap
    mapped_labels = [
        f"{qid}_{label.split('_')[-1]}: {question_labels[qid][label]}" for label in labels
    ]
    wrapped_labels = [wrap_text(label, line_length=40) for label in mapped_labels]

    # Grouped bar positions
    for i, mode in enumerate(modes):
        positions = np.arange(len(labels)) + (i * bar_width)
        bar_positions.append(positions)
        freq = preference_frequencies[qid][mode].reindex(labels, fill_value=0)  # Fill missing labels with 0
        bars = plt.bar(positions, freq.values, bar_width, label=f"{mode}", alpha=0.7)

        # Annotate percentage
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{int(yval)}%" if yval >= 1 else "<1%", ha="center", va="bottom", fontsize=10 if yval >= 1 else 8)
                #plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.3f}%", ha="center", va="bottom", fontsize=10)



    # Add labels and legend
    plt.xticks(
        np.arange(len(labels)) + bar_width, wrapped_labels, rotation=0, ha="center"
    )
    plt.xlabel("Answer Options")
    plt.ylabel("Percentage (%)")
    plt.title(f"Grouped Preference Distribution for Question {qid}: {preference_labels[qid]}")
    plt.legend(title="Mode")
    plt.tight_layout()

    # Force y-axis to always tick with 10% intervals
    plt.yticks(np.arange(0, 110, 10))  # Y-axis with values from 0 to 100

    # Add subticks for percentage thresholds on Y-axis
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(5))
    plt.gca().tick_params(axis="y", which="minor", length=5, color="#cccccc")

    # Customize grid: Disable vertical grid lines, enable horizontal ones
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    plt.tight_layout()
    plt.show()
