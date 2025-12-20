import json

import jax
import grain
import numpy as np
from tqdm import tqdm


class JsonDataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = [json.loads(line.strip()) for line in tqdm(open(file_path, 'r').readlines())]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __repr__(self) -> str:
        return f"JsonDataSource(file_path={self.file_path})"


class Parse(grain.transforms.Map):
    def map(self, record):
        return {
            "x": np.asarray(record["x"]),
            "y": np.asarray(record["y"]),
            "colour_aug": np.asarray(record["colour_aug"]),
            "d8_aug": np.asarray([record["d8_aug"]]),
            "example_idx": np.asarray([record["example_idx"]]),
            "aug_puzzle_idx": np.asarray(record["aug_puzzle_idx"]),
            # "puzzle_id": np.asarray([record["puzzle_id"]]),
            "puzzle_idx": np.asarray([record["puzzle_idx"]]),
        }



# def np_grid_to_seq_translational_augment(inp: np.ndarray, out: np.ndarray, do_translation: bool):
#     # PAD: 0, <eos>: 1, digits: 2 ... 11
#     # Compute random top-left pad
#     if do_translation:
#         pad_r = np.random.randint(0, ARCMaxGridSize - max(inp.shape[0], out.shape[0]) + 1)
#         pad_c = np.random.randint(0, ARCMaxGridSize - max(inp.shape[1], out.shape[1]) + 1)
#     else:
#         pad_r = pad_c = 0

#     # Pad grid
#     result = []
#     for grid in [inp, out]:
#         nrow, ncol = grid.shape
#         grid = np.pad(grid + 2, ((pad_r, ARCMaxGridSize - pad_r - nrow), (pad_c, ARCMaxGridSize - pad_c - ncol)), constant_values=0)

#         # Add <eos>
#         eos_row, eos_col = pad_r + nrow, pad_c + ncol
#         if eos_row < ARCMaxGridSize:
#             grid[eos_row, pad_c:eos_col] = 1
#         if eos_col < ARCMaxGridSize:
#             grid[pad_r:eos_row, eos_col] = 1

#         result.append(grid.flatten())

#     return result

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

    def _translate_pad_and_add_border(
        self,
        grid: np.ndarray,
        *,
        pad_r: int,
        pad_c: int,
        pad_value: int,
        border_value: int,
    ) -> np.ndarray:
        """Pad grid into max_grid_size canvas and add an EOS-like border.

        This mirrors the logic in the commented reference implementation above:
        - Use shared (pad_r, pad_c) offsets for translation
        - Compute border coordinates using the *current grid's* shape
        """
        nrow, ncol = grid.shape
        padded = self._pad(
            grid,
            (
                (pad_r, self.max_grid_size - pad_r - nrow),
                (pad_c, self.max_grid_size - pad_c - ncol),
            ),
            pad_value,
        )

        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < self.max_grid_size:
            padded[eos_row, pad_c:eos_col] = border_value
        if eos_col < self.max_grid_size:
            padded[pad_r:eos_row, eos_col] = border_value
        return padded
    
    def random_map(self, record, rng):
        x, y = record["x"], record["y"]

        # top left corner
        max_rows = max(x.shape[0], y.shape[0])
        max_cols = max(x.shape[1], y.shape[1])
        if self.translate == "random":
            pad_r = rng.integers(0, self.max_grid_size - max_rows + 1)
            pad_c = rng.integers(0, self.max_grid_size - max_cols + 1)
        elif self.translate == "fixed":
            seed = int(np.asarray(record["aug_puzzle_idx"]).reshape(-1)[0])
            local_rng = np.random.default_rng(seed=seed)
            pad_r = local_rng.integers(0, self.max_grid_size - max_rows + 1)
            pad_c = local_rng.integers(0, self.max_grid_size - max_cols + 1)
        else:
            pad_c = 0
            pad_r = 0
        
        # pad: 11, border: 10 (viewer expects these sentinel values)
        padded_x = self._translate_pad_and_add_border(
            x, pad_r=pad_r, pad_c=pad_c, pad_value=11, border_value=10
        )
        padded_y = self._translate_pad_and_add_border(
            y, pad_r=pad_r, pad_c=pad_c, pad_value=11, border_value=10
        )
    
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