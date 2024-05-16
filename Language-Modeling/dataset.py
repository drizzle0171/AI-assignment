import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        self.seq_len = 30
        with open(input_file, 'r') as f:
            raw_data = f.read()
        characters = sorted(list(set(raw_data)))
        vocab = {ch: i for i, ch in enumerate(characters)}
        data = torch.tensor([vocab[ch] for ch in raw_data], dtype=torch.long)
        
        len_data = data.shape[0]
        data = data[:self.seq_len * (len_data // self.seq_len) + 1]
        
        self.data = data
        self.vocab = vocab
        
    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        input = self.data[start_idx:end_idx]
        target = self.data[start_idx+1:end_idx+1]
        return input, target

if __name__ == '__main__':
    
    file_pth = './shakespeare_train.txt'
    ds = Shakespeare(file_pth)
    print(ds[0])
    # result
    # (tensor([17, 45, 54, 55, 56,  2, 14, 45, 56, 45, 62, 41, 50,  9,  1, 13, 41, 42, 51, 54, 41,  2, 59, 41,  2, 52, 54, 51, 39, 41]), 
    # tensor([45, 54, 55, 56,  2, 14, 45, 56, 45, 62, 41, 50, 9, 1, 13, 41, 42, 51, 54, 41, 2, 59, 41,  2, 52, 54, 51, 39, 41, 41]))