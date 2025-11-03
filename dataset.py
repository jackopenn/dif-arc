import os
import json
from functools import partial

import grain
import numpy as np
from tqdm import tqdm


class JsonDataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, file_path):
        self.data = [json.loads(line.strip()) for line in tqdm(open(file_path, 'r').readlines())] 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


class Parse(grain.transforms.Map):
    def map(self, record):
        return {
            "x": np.asarray(record["x"]),
            "y": np.asarray(record["y"]),
            "colour_aug": np.asarray(record["colour_aug"]),
            "d8_aug": np.asarray([record["d8_aug"]]),
            "example_idx": np.asarray([record["example_idx"]]),
            "aug_puzzle_idx": np.asarray([record["aug_puzzle_idx"]]),
        }

        
class Pad(grain.transforms.Map):
    def __init__(self, max_grid_size = 30):
        self.max_grid_size = max_grid_size
        
    def _pad(self, x):
        return np.pad(
            x,
            pad_width=((0, self.max_grid_size - x.shape[0]), (0, self.max_grid_size - x.shape[1])),
            mode="constant",
            constant_values=10
        )
    
    def map(self, record):
        return {
            **record,
            "x": self._pad(record["x"]).flatten(),
            "y": self._pad(record["y"]).flatten(),
        }
    

def get_data_loader(data_dir, batch_size):
    data_source = JsonDataSource(data_dir)
    sampler = grain.samplers.IndexSampler(len(data_source), seed=0, shuffle=True)
    operations = [
        Parse(),
        Pad(),
        grain.transforms.Batch(batch_size=batch_size)
    ]
    return grain.DataLoader(data_source=data_source, operations=operations, sampler=sampler)