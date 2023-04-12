import pandas as pd
from random import sample, seed, choice
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
        prompt += f"Solution: {possible_solutions['solution'].iloc[choice(range(len(possible_solutions)))]}=24\n"
    prompt += f"Your input: {candidates[candidate_idx]}\n"
    prompt += "Your solution:"
    return prompt

for idx in range(20):
    print("+++++++++++")
    print(sample_examples_for_gpt4(random_seed=idx, candidate_idx=idx))

# find the GPT4 answers here https://shareg.pt/in5gKsC (Spoiler: GPT4 failed on all of them)
# https://sharegpt.com/c/kd3UA8B with more hints, and step by step guidance, still struggles more often than not, but sometimes can solve the problems.
## and got switched to GPT3.5 due to message cap on the last message so stopped.