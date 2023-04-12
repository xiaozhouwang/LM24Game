import pandas as pd
from random import sample, seed, choice
import re

def generate_middle_steps(expression: str) -> str:
    # Define a list to store the steps
    steps = [expression]

    # Replace parentheses with brackets
    expression = expression.replace("(", "[").replace(")", "]")

    # Define a helper function to evaluate the expression within brackets
    def eval_expr(match):
        expr = match.group(1)
        return str(eval(expr))

    # Keep evaluating the innermost expressions until no brackets are left
    while "[" in expression:
        expression = re.sub(r'\[(.*?)\]', eval_expr, expression)
        steps.append(expression)

    # Evaluate the final expression
    result = eval(expression)
    steps.append(str(result))

    # Generate the output string
    output = "=".join(steps)
    return output

data = pd.read_parquet("24game_data.parquet")
data['has_solution'] = data['solution'].apply(lambda x: int(x != ''))
data['num_solutions'] = data.groupby('input').transform('size')
gpt4_input_candidates = data[(data['has_solution']==1) & (data['num_solutions']==1)]['input'].unique().tolist()
gpt4_input_examples = data[(data['has_solution']==1) & (data['num_solutions'] != 1)]['input'].unique().tolist()
N_candidates = 20
candidates = sample(gpt4_input_candidates, N_candidates)


def sample_examples_for_gpt4(N=20, random_seed=5, candidate_idx=0):
    seed(random_seed)
    examples = sample(gpt4_input_examples, N)
    prompt = f"Here is {N} examples and their solutions, please figure out the solution to the input given to you:\n"
    for example in examples:
        prompt += f"Input: {example}\n"
        possible_solutions = data[data['input'] == example]
        prompt += f"Solution: {generate_middle_steps(possible_solutions['solution'].iloc[choice(range(len(possible_solutions)))])}\n"
    prompt += f"Your input: {candidates[candidate_idx]}\n"
    prompt += "Your solution:"
    return prompt

for idx in range(20):
    print("+++++++++++")
    print(sample_examples_for_gpt4(random_seed=idx, candidate_idx=idx))

# find the GPT4 answers here https://shareg.pt/in5gKsC (Spoiler: GPT4 failed on all of them)
# https://sharegpt.com/c/kd3UA8B with more hints, and step by step guidance, still struggles more often than not, but sometimes can solve the problems.
## and got switched to GPT3.5 due to message cap on the last message so stopped.
## including middle steps don't seem to help https://sharegpt.com/c/T7PFJw6