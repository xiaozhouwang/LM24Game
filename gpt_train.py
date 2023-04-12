import pandas as pd
import numpy as np
from random import sample, seed, choice
from itertools import permutations
import torch
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from nanoGPT_model import GPT, GPTConfig


seed(42)

def get_train_valid_test_split():
    data = pd.read_parquet("24game_data.parquet")
    data['has_solution'] = data['solution'].apply(lambda x: int(x != ''))
    data['num_solutions'] = data.groupby('input').transform('size')
    """get test set"""
    """pick tougher ones with one possible solution, and add some without solutions as test set"""
    one_possible_solution_set = data[(data['has_solution']==1) & (data['num_solutions']==1)]['input'].unique().tolist()
    no_solution_set = data[data['has_solution']==0]['input'].unique().tolist()
    test_set = sample(one_possible_solution_set, 50)
    test_set += sample(no_solution_set, 50)
    test_set = get_all_input_permutations(test_set)
    """sample validation set"""
    test_data = data[data['input'].isin(set(test_set))]
    train_valid_data = data[~data['input'].isin(set(test_set))]
    valid_set = sample(train_valid_data['input'].tolist(), 10)
    valid_set = get_all_input_permutations(valid_set)
    valid_data = train_valid_data[train_valid_data['input'].isin(set(valid_set))]
    train_data = train_valid_data[~train_valid_data['input'].isin(set(valid_set))]
    print(f"train data size: {train_data.shape[0]}, validation data size: {valid_data.shape[0]}, test data size: {test_data.shape[0]}")
    return train_data, valid_data, test_data

def get_all_input_permutations(input_list):
    """given an input list, get all permutations of all inputs to avoid data leakage"""
    all_permutation_inputs = set()
    for i in input_list:
        p = permutations([x for x in i.split(" ")])
        all_permutation_inputs.update(set([' '.join(x) for x in p]))
    return list(all_permutation_inputs)

class Tokenizer:
    def __init__(self):
        self.vocabulary = ['PAD'] + [str(x) for x in range(1, 10)] + ['+', '-', '*', '/', '=', '(', ')']
        self.map = dict(zip(self.vocabulary, list(range(len(self.vocabulary))))) # Padding is index 0
        self.inverse_map = {v:k for k, v in self.map.items()}

    def encode(self, string_input):
        """
        '1 2 3 4' --> [0, 1, 2, 3]
        '(3+5)*(1+2)' --> [14, 2, 9, 4, 15, 11, 14, 0, 9, 1, 15]
        """
        return [self.map[x] for x in string_input.replace(" ", "")]
    
    def decode(self, output_list):
        """
        [14, 2, 9, 4, 15, 11, 14, 0, 9, 1, 15] --> '(3+5)*(1+2)'
        """
        return ''.join([self.inverse_map[x] for x in output_list])

class Dataset:
    def __init__(self, data_frame):
        self.inputs = data_frame['input'].tolist()
        self.solutions = data_frame['solution'].tolist()
        self.tokenizer = Tokenizer()
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, item):
        input = self.tokenizer.encode(self.inputs[item])
        solution = self.tokenizer.encode(self.solutions[item])

        full_sequence = input + solution
        full_sequence = [0]*(16 - len(full_sequence)) + full_sequence # pad to seq len 16
        full_sequence = np.array(full_sequence, dtype=np.int32)

        return full_sequence[:-1], full_sequence[1:]


"""initialize model, and train the model"""
if __name__ == "__main__":
    model_args = dict(block_size=256, n_layer=12, n_head=4, n_embd=256)
    run_name = f'GPT_{model_args["block_size"]}_{model_args["n_layer"]}_{model_args["n_head"]}_{model_args["n_embd"]}'
    wandb.init(project="24Game", name=run_name)

    MAXEPOCH = 1000
    learning_rate = 1e-3
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    compile = True
    BATCHSIZE=512
    device = 'cuda'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    BestEpoch = 0
    BestLoss = np.inf
    train_data, validation_data, test_data = get_train_valid_test_split()

    model = GPT(GPTConfig(**model_args)).to(device)
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    optimizer.zero_grad()
    if compile:
        model = torch.compile(model)
    
    train_dataloader = DataLoader(dataset=Dataset(train_data), batch_size=BATCHSIZE, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset=Dataset(validation_data), batch_size=BATCHSIZE, shuffle=False, num_workers=4)

    def get_batch_loss(batch_data, device=device):
        batch_inputs, batch_outputs = batch_data
        batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)
        _, loss = model(batch_inputs, batch_outputs)
        return loss

    for epoch in range(MAXEPOCH):
        epoch_loss = 0
        for batch_data in tqdm(train_dataloader):
            loss = get_batch_loss(batch_data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)
        print(f"epoch: {epoch}, training loss: {epoch_loss}")
        wandb.log({'train_loss': epoch_loss})

        """test loss"""
        epoch_loss = 0
        with torch.no_grad():
            for batch_inputs, batch_outputs in tqdm(test_dataloader):
                loss = get_batch_loss(batch_data)
                epoch_loss += loss.item()
            epoch_loss /= len(test_dataloader)
        print(f"epoch: {epoch}, test loss: {epoch_loss}")
        wandb.log({'test_sim': epoch_loss, 'epoch': epoch})

        if epoch_loss < BestLoss:
            BestLoss = epoch_loss
            BestEpoch = epoch + 1
            print(f"save best model at {BestLoss} with epoch {BestEpoch}")
            torch.save(model.state_dict(), f"{run_name}.pt")

        if epoch - 5 > BestEpoch:
            print(f"early stop at {epoch+1}.")
            break
