import os
import grain
import jax
from jax.nn.initializers import xavier_normal
import jax.numpy as jnp
import numpy as np
import json
from functools import partial


class JsonDataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, folder):
        self.file_paths = [
            os.path.join(folder, f) 
            for f in os.listdir(folder) 
            if f.endswith('.json')
        ]
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        with open(self.file_paths[index], 'r') as f:
            obj = json.load(f)
            obj['puzzle_id'] = self.file_paths[index].split('/')[-1].split('.')[0]
            return obj


class Parse(grain.transforms.Map):
    def map(self, record):
        x = {
            xput: [jnp.array(sample[xput]) for sample in record["train"]]
            for xput in ('input', 'output')
        }
        y = {
            xput: [jnp.array(sample[xput]) for sample in record["test"]]
            for xput in ('input', 'output')
        }
        return x, y, record['puzzle_id']

        
class D8Augmentation(grain.transforms.RandomMap):
    def __init__(self):
        self.ops = (
            lambda x: x,
            partial(jnp.rot90, k=1),
            partial(jnp.rot90, k=2),
            partial(jnp.rot90, k=3),
            jnp.fliplr,
            jnp.flipud,
            jnp.transpose,
            lambda x: jnp.transpose(jnp.rot90(x, k=1)),
        )

    def random_map(self, x, rng: np.random.Generator):
        op_idx = rng.integers(0, 8).item()
        fn = self.ops[op_idx]
        x = jax.tree.map(fn, x)
        x['op_idx'] = op_idx
        return x


class ColourAugmentation(grain.transforms.RandomMap):
    def random_map(self, x, rng: np.random.Generator):
        colours = rng.permutation(10)
        x = jax.tree.map(lambda x: colours[x], x)
        x['colour_permutation'] = colours
        return x


class Stack(grain.transforms.Map):
    def __init__(self, max_grid_size = 30, max_train_samples = 10, max_test_samples = 3):
        self.max_grid_size = max_grid_size
        self.max_train_samples = max_train_samples
        self.max_test_samples = max_test_samples
        
    def _pad_and_expand(self, x):
        padded = jnp.pad(
            x,
            pad_width=((0, self.max_grid_size - x.shape[0]), (0, self.max_grid_size - x.shape[1])),
            mode="constant",
            constant_values=-1
        )
        expanded = padded[jnp.newaxis, ...]
        return expanded
    
    def map(self, x):
        pad_sample = jnp.full((1, self.max_grid_size, self.max_grid_size), -1)
        train_pad_samples = [pad_sample for _ in range(self.max_train_samples - len(x["train"]['input']))]
        test_pad_samples = [pad_sample for _ in range(self.max_test_samples - len(x["test"]['input']))]
        return {
            split: {
                xput: jnp.vstack([self._pad_and_expand(sample) for sample in x[split][xput]] + pad_samples)
                for xput in ('input', 'output')
            }
            for split, pad_samples in zip(("train", "test"), (train_pad_samples, test_pad_samples))
        }
    

def get_data_loader(data_dir, batch_size):
    data_source = JsonDataSource(data_dir)
    sampler = grain.samplers.IndexSampler(len(data_source), seed=0)
    operations = [
        Parse(),
        # D8Augmentation(),
        # ColourAugmentation(),
        # Stack(),
        # grain.transforms.Batch(batch_size=2)
    ]
    return grain.DataLoader(data_source=data_source, operations=operations, sampler=sampler)