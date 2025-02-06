import math

# Initial settings
start_weight = 100  # Weight of the first question
num_questions = 12  # Number of questions in the test

# Function to create linear weights (decreasing by a constant value)
def linear_weights(start_weight, decrease, num_questions):
    return [(start_weight - i * decrease) / 100 for i in range(num_questions)]

# Function to create exponential weights (non-linear decrease)
def exponential_weights(start_weight, num_questions, coefficient=0.1):
    return [(start_weight * math.exp(-coefficient * i)) / 100 for i in range(num_questions)]

# Function to create logarithmic weights
def logarithmic_weights(start_weight, num_questions, coefficient=0.2):
    return [(start_weight / (1 + coefficient * i)) / 100 for i in range(num_questions)]

# Linear decrease by 5 points
linear_decrease_5 = 5
linear_weights_5 = linear_weights(start_weight, linear_decrease_5, num_questions)

# Linear decrease by 8.33 points
linear_decrease_8_33 = 8.33
linear_weights_8_33 = linear_weights(start_weight, linear_decrease_8_33, num_questions)

# Exponential weights
exponential_weights = exponential_weights(start_weight, num_questions)

# Logarithmic weights
logarithmic_weights = logarithmic_weights(start_weight, num_questions)

# Question order in the test
N3 = [1, 2, 6, 9, 10, 7, 4, 5, 11, 8, 3, 12]

# Function to assign weights to questions based on the order N3
def assign_weights_to_questions(weights, order):
    return [weights[i-1] for i in order]

# Assigning weights to questions based on N3 order
linear_weights_5_sorted = assign_weights_to_questions(linear_weights_5, N3)
linear_weights_8_33_sorted = assign_weights_to_questions(linear_weights_8_33, N3)
exponential_weights_sorted = assign_weights_to_questions(exponential_weights, N3)
logarithmic_weights_sorted = assign_weights_to_questions(logarithmic_weights, N3)

# Function to round weights to 4 decimal places
def round_weights(weights):
    return [round(weight, 4) for weight in weights]

# Displaying the weights for different methods after sorting according to N3 and rounding
print("1. Linear weights with 5-point decrease (sorted according to N3):")
print(round_weights(linear_weights_5_sorted))

print("\n2. Linear weights with 8.33-point decrease (sorted according to N3):")
print(round_weights(linear_weights_8_33_sorted))

print("\n3. Exponential weights (sorted according to N3):")
print(round_weights([weight for weight in exponential_weights_sorted]))

print("\n4. Logarithmic weights (sorted according to N3):")
print(round_weights([weight for weight in logarithmic_weights_sorted]))

# Summary sentence
summary = (
    "Based on the order of elements in the vector, weights were also created, "
    "which reflect different methods of decreasing points with each subsequent question.\n\n"
    "1. Linear weights with a 5-point decrease: Weights decrease by 5 points for each subsequent question, starting from a maximum value of 100.\n"
    "2. Linear weights with an 8.33-point decrease: Weights decrease by 8.33 points for each subsequent question, starting from a maximum value of 100.\n"
    "3. Exponential weights: Weights decrease faster at the beginning but then stabilize at a lower level.\n"
    "4. Logarithmic weights: Weights decrease more subtly, slower than in the exponential case."
)

# Displaying the full summary
print("\nWeight summary:\n")
print(summary)
input("Press Enter to continue...")
