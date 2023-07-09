import re

with open(r"Assinging_VAD_scores_BERT\\Learning_log\\XGBoost\\2.txt", 'r') as f:
    data = f.read()

# Find all the iterations and objectives
matches = re.findall(r"iteration (\d+):.*?objective: ([\de\-\.]+)", data, re.DOTALL)

# Extract iteration and objective into two separate lists and convert to appropriate data types
iterations, objectives = zip(*[(int(iteration), float(objective)) for iteration, objective in matches])

# Find the iteration with the minimum objective
min_index = objectives.index(min(objectives))
min_iteration = iterations[min_index]

print(f"The iteration with the smallest objective is: {min_iteration}")