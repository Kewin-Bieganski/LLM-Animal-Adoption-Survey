import os
import json
import random
import time
import re
import csv
from datetime import datetime
from enum import Enum
import openai
from openai import OpenAI
import google.generativeai
from pprint import pprint

# Flag to check if the dataset should be mostly neutral
isNeutral = False  # Set this to True for "neutral", False for "biased""

# Append "neutral" or "biased" based on the isNeutral flag
questions_file_path = f"questions_{'neutral' if isNeutral else 'biased'}.json"

weights_file_path = "weights.json"

pre_prompt = "Ta ankieta ma na celu ocenę Twoich predyspozycji adopcyjnych zwierząt ze schroniska.\n"
instructions_prompt = (
    "Aby odpowiedzieć na poniższe pytania, należy podać odpowiedzi w formie tablicy, "
    "na przykład: [a, b, c, ...]. Jako odpowiedzi użyj tylko tablicy, każda inna odpowiedź "
    "zostanie odrzucona.\n\n"
)

# Enter you OpenAI key here
openai_api_key = "openai_api_key_here"
openai_ai_model = "gpt-4o-mini"

# Enter you Google API key here
google_api_key = "google_api_key_here"
google_ai_model = "gemini-1.5-flash"

timeout = 60
time_between_prompts = 4

class QuestionType(Enum):
    POINTED = "pointed"
    PREFERENCE = "preference"

class APIConnectionStatus(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"

class CycleStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"

class ResponseMode(Enum):
    RANDOM = "Random"
    OPENAI_GPT_4O_MINI = "OpenAI (gpt-4o-mini)"
    GOOGLE_GEMINI_1_5_FLASH = "Google (gemini-1.5-flash)"

def main():
    print("### Adoption Survey Automation ###")

    # Mode selection with error handling
    mode = None
    for _ in range(3):  # Allow up to 3 attempts
        print("Select the mode for running the survey:")
        print("1. Random")
        print("2. OpenAI (gpt-4o-mini)")
        print("3. Google (gemini-1.5-flash)")
        mode_input = input("Enter the mode (1, 2, or 3): ").strip()
        if mode_input in {"1", "2", "3"}:
            mode = {
                "1": ResponseMode.RANDOM,
                "2": ResponseMode.OPENAI_GPT_4O_MINI,
                "3": ResponseMode.GOOGLE_GEMINI_1_5_FLASH
            }[mode_input]
            break
        print("Invalid mode. Please select a valid option.")
    else:
        print("Too many invalid attempts. Exiting.")
        return

    # Cycle count selection with error handling
    cycles = 0
    for _ in range(3):  # Allow up to 3 attempts
        try:
            cycles = int(input("Enter the number of cycles to perform: ").strip())
            if cycles > 0:
                break
            print("Number of cycles must be greater than zero.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    else:
        print("Too many invalid attempts. Exiting.")
        return

    print(f"Running {cycles} cycle(s) in mode {mode.value}.")

    # Load questions and weights
    print("### Loading Questions and Weights ###")
    questions = load_json_object_from_file(questions_file_path)
    weights = load_json_object_from_file(weights_file_path)
    print("Questions and weights loaded successfully.")

    # Calculate maximum points
    max_points = calculate_max_points(questions)
    print(f"Max points (non-weighted): {max_points}")

    # Calculate maximum weighted points
    max_weighted_points = calculate_max_weighted_points(questions, weights)
    print(f"Max points (weighted): {max_weighted_points}")

    for cycle in range(1, cycles + 1):
        print(f"\n### Cycle {cycle}/{cycles} ###")
        cycle_status = CycleStatus.SUCCESS  # Default status

        try:
            # Generate the prompt and mapping
            generated_prompt, mapping = generate_prompt_with_mapping(pre_prompt, instructions_prompt, questions)
            print("Generated Prompt:")
            print(generated_prompt)

            # Handle different modes
            if mode == ResponseMode.RANDOM:
                received_answer = generate_random_received_answer(questions)
                api_connection_status = APIConnectionStatus.SUCCESS
                elapsed_time = 0
            elif mode == ResponseMode.OPENAI_GPT_4O_MINI:
                received_answer, api_connection_status, elapsed_time = send_prompt_to_OpenAI_API(
                    openai_api_key, openai_ai_model, generated_prompt, timeout
                )
            elif mode == ResponseMode.GOOGLE_GEMINI_1_5_FLASH:
                received_answer, api_connection_status, elapsed_time = send_prompt_to_Google_API(
                    google_api_key, google_ai_model, generated_prompt, timeout
                )

            # Log the response
            print("API Response Status:", api_connection_status)
            print("Elapsed Time (s):", elapsed_time)
            print("Received Answer:", received_answer)

            # Skip the cycle if the API call fails
            if api_connection_status != APIConnectionStatus.SUCCESS:
                cycle_status = CycleStatus.ERROR
                print(f"Cycle {cycle} failed due to API connection issues.")
                continue

            # Process the answers
            answers = extract_raw_alphabetic_answers_from_text(received_answer)
            print("Extracted Answers:", answers)

            unmapped_answers = unmap_answers(mapping, answers)
            print("Unmapped Answers:", unmapped_answers)

            pointed_answers, preference_answers = split_unmapped_answers(unmapped_answers, questions)
            print("Pointed Answers:", pointed_answers)
            print("Preference Answers:", preference_answers)

            points = convert_unmapped_answers_to_points(pointed_answers, questions)
            print("Points:", points)

            summed_points = sum(points)
            print("Summed Points:", summed_points)

            weighted_points_arrays = apply_weights_to_points(points, weights)
            print("Weighted Points Arrays:")
            pprint(weighted_points_arrays)

            weighted_points_arrays_summed = sum_weighted_points(weighted_points_arrays)
            print("Summed Weighted Points:")
            pprint(weighted_points_arrays_summed)

            non_weighted_percentage, weighted_percentages = calculate_percentage_scores(
                points, weighted_points_arrays, max_points, max_weighted_points
            )
            print("Non-weighted percentage:", non_weighted_percentage)
            print("Weighted percentages:", weighted_percentages)

            # Prepare results for the current cycle
            results = [{
                "cycle": cycle,
                "mode": mode.value,
                "api_connection_status": api_connection_status.value,
                "cycle_status": cycle_status.value,
                "generated_prompt": generated_prompt,
                "mapping": mapping,
                "received_answer": received_answer,
                "elapsed_time": elapsed_time,
                "answers": answers,
                "unmapped_answers": unmapped_answers,
                "pointed_answers": pointed_answers,
                "preference_answers": preference_answers,
                "points": points,
                "summed_points": summed_points,
                "non_weighted_percentage": non_weighted_percentage,
                "weighted_points_arrays": weighted_points_arrays,
                "summed_weighted_points": weighted_points_arrays_summed,
                "weighted_percentages": weighted_percentages
            }]

            print("Results:")
            pprint(results)

            # Save results to CSV
            save_status = save_results_to_csv(results, mode)
            print(f"Save status: {'Success' if save_status else 'Failed'}")

        except Exception as e:
            cycle_status = CycleStatus.ERROR
            print(f"Error during cycle {cycle}: {e}")
            results = [{
                "cycle": cycle,
                "mode": mode.value,
                "api_connection_status": APIConnectionStatus.ERROR.value,
                "cycle_status": cycle_status.value,
                "generated_prompt": None,
                "mapping": None,
                "received_answer": None,
                "elapsed_time": None,
                "answers": None,
                "unmapped_answers": None,
                "pointed_answers": None,
                "preference_answers": None,
                "points": None,
                "summed_points": None,
                "non_weighted_percentage": None,
                "weighted_points_arrays": None,
                "summed_weighted_points": None,
                "weighted_percentages": None
            }]

            try:
                save_status = save_results_to_csv(results, mode)
                print(f"Save status: {'Success' if save_status else 'Failed'}")
            except Exception as e:
                print(f"Failed to save results for cycle {cycle}: {e}")

        # Optional: Sleep between cycles
        time.sleep(time_between_prompts)

    print("\n### Survey Completed ###")
    print(f"All {cycles} cycle(s) have been processed.")

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

def calculate_max_points(questions):
    """
    Calculates the maximum possible points for non-weighted questions.

    Args:
        questions (list): A list of questions where each question contains a list of possible answers with points.

    Returns:
        int: The total maximum possible points for non-weighted questions.
    """
    max_points = sum([max(q["answers"], key=lambda a: a["points"])["points"] for q in questions])
    return max_points


def calculate_max_weighted_points(questions, weights):
    """
    Calculates the maximum possible points for each weighted points array.

    Args:
        questions (list): A list of questions where each question contains a list of possible answers with points.
        weights (dict): A dictionary where each key is a name for a weight vector, and the value is a list of weights for each question.

    Returns:
        dict: A dictionary where the key is the weight vector name, and the value is the maximum possible weighted points.
    """
    max_weighted_points = {}
    for name, weight_vector in weights.items():
        # Sum the max points for each question, each weighted by the corresponding weight in the weight_vector
        max_weighted_points[name] = sum(
            max(q["answers"], key=lambda a: a["points"])["points"] * w
            for q, w in zip(questions, weight_vector)
        )
    return max_weighted_points

def generate_prompt_with_mapping(pre_prompt, instructions_prompt, questions, max_answer_options=26):
    """
    Generates a prompt where questions and answers are randomized and labeled with letters for AI/human readability.
    Returns the prompt and a mapping to interpret the responses.
    Ensures the number of answer options does not exceed the limit of alphabet letters (default: 26).
    """
    if max_answer_options > 26:
        raise ValueError("Maximum number of answer options cannot exceed 26 (alphabet limit).")

    questions_prompt = ""
    mapping = []  # To track randomized positions for questions and answers

    for question in questions:
        # Shuffle the answers
        randomized_answers = question["answers"][:]
        random.shuffle(randomized_answers)

        if len(randomized_answers) > max_answer_options:
            raise ValueError(
                f"Question {question['id']} exceeds the maximum number of allowed answers ({max_answer_options})."
            )

        # Build the question string with answers labeled as letters
        questions_prompt += f"{question['id']}. {question['question']}\n"
        answer_map = {}
        for idx, ans in enumerate(randomized_answers):
            option_label = chr(97 + idx)  # 'a', 'b', 'c', etc.
            questions_prompt += f"{option_label}. {ans['text']}\n"
            answer_map[option_label] = ans["id"]

        # Append to mapping
        mapping.append({
            "question_id": question["id"],
            "answer_mapping": answer_map
        })

    # Combine all parts of the prompt
    generated_prompt = pre_prompt + instructions_prompt + questions_prompt
    return generated_prompt, mapping

def generate_random_received_answer(questions):
    """
    Generates a random answer string for the given questions.

    Args:
        questions (list): A list of questions, each containing possible answers.

    Returns:
        str: A string representation of randomly generated answers (e.g., "[a, c, b, ...]").
    """
    random_answers = []
    for question in questions:
        # Determine the number of possible answers for the current question
        num_answers = len(question["answers"])
        # Generate a random answer label (e.g., "a", "b", "c", ...)
        random_answer = chr(97 + random.randint(0, num_answers - 1))  # ASCII 'a' + index
        random_answers.append(random_answer)

    # Return the answers as a string formatted like a list
    return f"[{', '.join(random_answers)}]"

def send_prompt_to_OpenAI_API(openai_api_key, openai_ai_model, prompt, timeout=60):
    """
    Sends a prompt to OpenAI API and returns the received answer, status, and elapsed time.

    Args:
        openai_api_key (str): OpenAI API key.
        openai_ai_model (str): The model to use (e.g., "gpt-4o-mini").
        prompt (str): The prompt to send to the API.
        timeout (int): The maximum time to wait for a response (in seconds).

    Returns:
        tuple: A tuple containing the received answer (str), status (APIConnectionStatus), and elapsed time (float).
    """
    client = OpenAI(api_key=openai_api_key)

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=openai_ai_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            timeout=timeout
        )
        elapsed_time = time.time() - start_time

        if elapsed_time > timeout:
            return "", APIConnectionStatus.TIMEOUT, elapsed_time

        received_answer = response.choices[0].message.content
        return received_answer, APIConnectionStatus.SUCCESS, elapsed_time

    except openai.error.Timeout:
        elapsed_time = time.time() - start_time
        return "", APIConnectionStatus.TIMEOUT, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error: {e}")
        return "", APIConnectionStatus.ERROR, elapsed_time

def send_prompt_to_Google_API(google_api_key, google_ai_model, prompt, timeout=60):
    """
    Sends a prompt to Google API and returns the received answer, status, and elapsed time.

    Args:
        google_api_key (str): Google API key.
        google_ai_model (str): The model to use (e.g., "gemini-1.5-flash").
        prompt (str): The prompt to send to the API.
        timeout (int): The maximum time to wait for a response (in seconds).

    Returns:
        tuple: A tuple containing the received answer (str), status (APIConnectionStatus), and elapsed time (float).
    """
    try:
        google.generativeai.configure(api_key=google_api_key, transport="rest")
        model = google.generativeai.GenerativeModel(google_ai_model)

        start_time = time.time()
        response = model.generate_content(prompt)
        elapsed_time = time.time() - start_time

        if elapsed_time > timeout:
            return "", APIConnectionStatus.TIMEOUT, elapsed_time

        received_answer = response.text
        return received_answer, APIConnectionStatus.SUCCESS, elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error: {e}")
        return "", APIConnectionStatus.ERROR, elapsed_time

def extract_raw_alphabetic_answers_from_text(raw_text):
    """
    Extracts answers from the provided raw text.

    Args:
        raw_text (str): The text containing potential answers in array format.

    Returns:
        list: A list of extracted answers (e.g., ["a", "b", "c"]). If no valid answers are found, returns an empty list.
    """
    # Define a regex pattern to match arrays in the text, including quoted elements
    pattern = r"\[([\"'a-zA-Z,\s]+)\]"

    # Search for the pattern in the raw text
    match = re.search(pattern, raw_text)
    if match:
        # Extract the content inside the brackets
        answers = match.group(1)
        # Split the content into individual answers, removing quotes and spaces
        return [answer.strip().strip('"').strip("'") for answer in answers.split(",") if answer.strip()]
    return []

def unmap_answers(mapping, responses):
    """
    Unmaps randomized answers to their original answer IDs.

    Args:
        mapping (list): A list of dictionaries containing question IDs and randomized answer mappings.
        responses (list): A list of selected options (e.g., ["a", "b", "c"]).

    Returns:
        list: A list of unmapped answer IDs (e.g., ["1_3", "2_1", "3_2"]).
    """
    unmapped_answers = []
    for question_map, response in zip(mapping, responses):
        answer_mapping = question_map["answer_mapping"]
        if response not in answer_mapping:
            raise ValueError(f"Invalid response '{response}' for question ID {question_map['question_id']}")
        unmapped_answers.append(answer_mapping[response])
    return unmapped_answers

def split_unmapped_answers(unmapped_answers, questions):
    """
    Splits unmapped answers into pointed and preference answers based on the question type.

    Args:
        unmapped_answers (list): A list of unmapped answer IDs (e.g., ["1_3", "2_1", "3_2"]).
        questions (list): The original list of questions.

    Returns:
        tuple: A tuple containing two lists: pointed_answers and preference_answers.
    """
    pointed_answers = []
    preference_answers = []

    for unmapped_answer in unmapped_answers:
        question_id = int(unmapped_answer.split("_")[0])
        question = next(q for q in questions if q["id"] == question_id)
        if question["type"] == QuestionType.POINTED.value:
            pointed_answers.append(unmapped_answer)
        elif question["type"] == QuestionType.PREFERENCE.value:
            preference_answers.append(unmapped_answer)

    return pointed_answers, preference_answers

def convert_unmapped_answers_to_points(unmapped_answers, questions):
    """
    Converts unmapped answers to their corresponding points.

    Args:
        unmapped_answers (list): A list of unmapped answer IDs (e.g., ["1_3", "2_1", "3_2"]).
        questions (list): The original list of questions.

    Returns:
        list: A list of points corresponding to the unmapped answers.
    """
    points = []
    for unmapped_answer in unmapped_answers:
        question_id = int(unmapped_answer.split("_")[0])
        answer_id = unmapped_answer

        # Find the question
        question = next(q for q in questions if q["id"] == question_id)

        # Find the answer and get its points
        answer = next(a for a in question["answers"] if a["id"] == answer_id)
        points.append(answer["points"])

    return points

def apply_weights_to_points(points, weight_vectors):
    """
    Applies multiple weight vectors (simplified structure) to the points and returns weighted points for each vector.

    Args:
        points (list): A list of points for each response (e.g., [3, 2, 1, ...]).
        weight_vectors (dict): A dictionary where keys are weight names (e.g., "W1") and values are weight arrays.

    Returns:
        dict: A dictionary where keys are weight vector names and values are weighted points arrays.
    """
    weighted_points = {}

    for name, weights in weight_vectors.items():
        if len(points) != len(weights):
            raise ValueError(f"Mismatch between points ({len(points)}) and weights ({len(weights)}) for {name}.")

        # Apply weights to points
        weighted_points[name] = [p * w for p, w in zip(points, weights)]

    return weighted_points

def sum_weighted_points(weighted_points):
    """
    Sums the elements of each weighted point vector.

    Args:
        weighted_points (dict): A dictionary where keys are weight names (e.g., "W1")
                                and values are lists of weighted points.

    Returns:
        dict: A dictionary where keys are the weight names with "_sum" appended,
              and values are the summed elements of the respective vectors.
    """
    summed_points = {}
    for name, points in weighted_points.items():
        summed_points[name] = sum(points)
    return summed_points

def calculate_percentage_scores(points, weighted_points_arrays, max_points, max_weighted_points):
    """
    Calculates percentage scores for non-weighted and weighted points.

    Args:
        points (list): A list of non-weighted points.
        weighted_points_arrays (dict): A dictionary of weighted points arrays.
        max_points (int): Maximum points for non-weighted questions.
        max_weighted_points (dict): Maximum points for each weighted points array.

    Returns:
        tuple: A tuple containing:
            - non_weighted_percentage (float): Percentage score for non-weighted points.
            - weighted_percentages (dict): Percentage scores for each weighted points array.
    """
    # Calculate the non-weighted percentage
    summed_points = sum(points)
    non_weighted_percentage = (summed_points / max_points) * 100 if max_points > 0 else 0

    # Calculate the weighted percentages
    weighted_percentages = {
        name: (sum(weighted_points_arrays[name]) / max_weight) * 100
        for name, max_weight in max_weighted_points.items()
    }

    return non_weighted_percentage, weighted_percentages

def save_results_to_csv(results, mode, file_prefix="survey_results"):
    """
    Saves survey results to a CSV file. Each mode has its own file.

    Args:
        results (list): A list of dictionaries containing results for a single cycle.
        mode (ResponseMode): The mode in which the survey is running (enum value).
        file_prefix (str): Prefix for the CSV file name.

    Returns:
        bool: True if the operation succeeded, False otherwise.
    """
    # Skip saving if results are empty or cycle status is error
    if not results or results[0]["cycle_status"] == CycleStatus.ERROR.value:
        print("Skipping saving results due to empty results or cycle error.")
        return False

    # Determine the file name based on the mode
    file_name = f"{file_prefix}_{mode.value.replace(' ', '_').lower()}.csv"

    # Define headers for the CSV
    headers = [
        "index", "timestamp", "cycle_number", "mode", "api_connection_status",
        "cycle_status", "generated_prompt", "mapping", "received_answer",
        "elapsed_time", "answers", "unmapped_answers", "pointed_answers",
        "preference_answers", "points", "summed_points", "non_weighted_percentage"
    ]

    # Extend headers for weighted points arrays and summed values
    weighted_points_headers = []
    weighted_summed_headers = []
    weighted_percentage_headers = []
    for key in results[0].get("weighted_points_arrays", {}).keys():
        weighted_points_headers.append(f"weighted_points_arrays_{key}")
        weighted_summed_headers.append(f"weighted_points_arrays_summed_{key}")
        weighted_percentage_headers.append(f"weighted_percentages_{key}")

    headers.extend(weighted_points_headers)
    headers.extend(weighted_summed_headers)
    headers.extend(weighted_percentage_headers)

    try:
        # Check if the file exists to determine if headers need to be written
        write_headers = not os.path.exists(file_name)

        with open(file_name, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)

            # Write headers if this is a new file
            if write_headers:
                writer.writeheader()

            for index, result in enumerate(results, start=1):
                row = {
                    "index": index,
                    "timestamp": datetime.now().isoformat(),
                    "cycle_number": result["cycle"],
                    "mode": mode.value,
                    "api_connection_status": result["api_connection_status"],
                    "cycle_status": result["cycle_status"],
                    "generated_prompt": result["generated_prompt"],
                    "mapping": result["mapping"],
                    "received_answer": result["received_answer"],
                    "elapsed_time": round(result["elapsed_time"], 4) if result["elapsed_time"] is not None else None,
                    "answers": result["answers"],
                    "unmapped_answers": result["unmapped_answers"],
                    "pointed_answers": result["pointed_answers"],
                    "preference_answers": result["preference_answers"],
                    "points": result["points"],
                    "summed_points": round(result["summed_points"], 4) if result["summed_points"] is not None else None,
                    "non_weighted_percentage": round(result["non_weighted_percentage"], 4) if result["non_weighted_percentage"] is not None else None,
                }

                # Add weighted points arrays
                for key in weighted_points_headers:
                    col_name = key.split("_")[-1]
                    row[key] = [round(val, 4) for val in result["weighted_points_arrays"].get(col_name, [])]

                # Add summed weighted points
                for key in weighted_summed_headers:
                    col_name = key.split("_")[-1]
                    row[key] = round(result["summed_weighted_points"].get(col_name, 0), 4)

                # Add weighted percentages
                for key in weighted_percentage_headers:
                    col_name = key.split("_")[-1]
                    row[key] = round(result["weighted_percentages"].get(col_name, 0), 4)

                writer.writerow(row)

        print(f"Data saved to {file_name} successfully.")
        return True

    except Exception as e:
        print(f"Error saving data to {file_name}: {e}")
        return False

if __name__ == "__main__":
    main()