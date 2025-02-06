import json, os
import pandas as pd

# Flag to check if the dataset should be mostly neutral
isNeutral = False  # Set this to True if you want "neutral", False for "biased"

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

# Load questions to retrieve answer labels
def load_json_object_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json_object = json.load(file)
            return json_object
    except json.JSONDecodeError as e:
        raise ValueError(f"The file '{file_path}' contains invalid JSON.") from e

# Define wrap_text function to apply word-wrap to long labels
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

    if current_line:
        wrapped_lines.append(" ".join(current_line))
    return "\n".join(wrapped_lines)

# Create a mapping of question IDs to their answer labels
questions = load_json_object_from_file(questions_file_path)
question_labels = {}
for question in questions:
    if question["type"] == "preference":
        # Create a mapping using the answer's id as the key.
        # Assuming answer IDs are formatted in a way that aligns with our frequency keys.
        question_labels[question["id"]] = {
            answer["id"]: answer["text"].strip(";") for answer in question["answers"]
        }

# Create CSV content
csv_rows = []

# Header row with the modes
header = [""] + list(files.keys())
csv_rows.append("\t".join(header))

# Iterate through each question
for qid in preference_question_ids:
    # Add the question text row (first cell with the question text, rest are empty)
    question_text = preference_labels.get(qid, f"Question {qid}")
    csv_rows.append("\t".join([question_text, "", "", ""]))

    # For each answer option in this question, create a row with percentages
    # Assuming the keys in question_labels[qid] correspond to the labels in the frequency Series.
    for answer_key, answer_text in question_labels[qid].items():
        row = [answer_text]  # First cell is the answer text
        for mode in files.keys():
            # Look up the percentage for this answer in the frequency data;
            # If missing, default to 0.
            percentage = preference_frequencies[qid][mode].get(answer_key, 0)
            row.append(f"{percentage:.2f}%")
        csv_rows.append("\t".join(row))

# Join all rows into CSV output
csv_output = "\n".join(csv_rows)

# Optionally, save the CSV to a file
csv_filename = f"survey_results_{'neutral' if isNeutral else 'biased'}.csv"
with open(csv_filename, 'w', encoding="utf-8") as f:
    f.write(csv_output)

print(f"CSV data saved to {csv_filename}")
