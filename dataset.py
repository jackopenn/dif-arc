import json

import jax
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
            # "puzzle_id": np.asarray([record["puzzle_id"]]),
            "puzzle_idx": np.asarray([record["puzzle_idx"]]),
        }


class TranslateAndPad(grain.transforms.RandomMap):
    def __init__(self, translate, max_grid_size=30):
        self.translate = translate
        self.max_grid_size = max_grid_size
        
    def _pad(self, x, pad_width, pad_value):
        return np.pad(
            x,
            pad_width=pad_width,
            mode="constant",
            constant_values=pad_value
        )
    
    def random_map(self, record, rng):
        x, y = record["x"], record["y"]

        # top left corner
        if self.translate:
            if self.translate == "random":
                pad_c = rng.integers(0, self.max_grid_size - max(x.shape[0], y.shape[0]) + 1)
                pad_r = rng.integers(0, self.max_grid_size - max(x.shape[1], y.shape[1]) + 1)
            elif self.translate == "fixed":
                pad_c = np.random.default_rng(seed=record["aug_puzzle_idx"]).integers(0, self.max_grid_size - max(x.shape[0], y.shape[0]) + 1)
                pad_r = np.random.default_rng(seed=record["aug_puzzle_idx"]).integers(0, self.max_grid_size - max(x.shape[1], y.shape[1]) + 1)
        else:
            pad_c = 0
            pad_r = 0
        
        padded_x = self._pad(x, ((pad_c, self.max_grid_size - x.shape[0] - pad_c), (pad_r, self.max_grid_size - x.shape[1] - pad_r)), 11)
        padded_y = self._pad(y, ((pad_c, self.max_grid_size - y.shape[0] - pad_c), (pad_r, self.max_grid_size - y.shape[1] - pad_r)), 11)

        border_c = pad_c + x.shape[0]
        border_r = pad_r + x.shape[1]

        if border_c < self.max_grid_size:
            padded_x[border_c, pad_r:border_r] = 10
            padded_y[border_c, pad_r:border_r] = 10
        if border_r < self.max_grid_size:
            padded_x[pad_c:border_c, border_r] = 10
            padded_y[pad_c:border_c, border_r] = 10
    
        return {
            **record,
            "x": padded_x.flatten(),
            "y": padded_y.flatten(),
        }
    

def get_data_loader(data_dir, batch_size, translate, max_grid_size, repeat=True, drop_remainder=True, shard_by_jax_process=False):
    per_process_batch_size = batch_size // jax.process_count()
    data_source = JsonDataSource(data_dir)
    sampler = grain.samplers.IndexSampler(
        len(data_source),
        seed=0,
        shuffle=True,
        num_epochs=None if repeat else 1,
        shard_options=grain.sharding.ShardByJaxProcess(drop_remainder=False) if shard_by_jax_process else grain.sharding.NoSharding()
    )
    operations = [
        Parse(),
        TranslateAndPad(translate=translate, max_grid_size=max_grid_size),
        grain.transforms.Batch(batch_size=per_process_batch_size, drop_remainder=drop_remainder)
    ]
    return grain.DataLoader(data_source=data_source, operations=operations, sampler=sampler)