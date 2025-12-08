import os
import json
import random
from dataclasses import dataclass

import numpy as np
import grain
from tqdm import tqdm
from datasets.dataset import TranslateAndPad


@dataclass(frozen=True)
class ExamplePointer:
    puzzle_id: str
    aug_idx: int
    example_idx: int


class ArcExampleDataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, data_dir):
        self.file_paths = sorted(
            (f[:-len(".jsonl")], os.path.join(data_dir, f))
            for f in os.listdir(data_dir)
            if f.endswith(".jsonl")
        )
        self._data = {
            puzzle_id: [json.loads(line) for line in open(file_path)]
            for puzzle_id, file_path in tqdm(self.file_paths, desc="loading data")
        }
        self._index: list[ExamplePointer] = []
        self._puzzle_aug_examples: dict[str, list[list[int]]] = {}
        self._build_index()
        self._puzzle_ids = list(self._puzzle_aug_examples.keys())

    def _build_index(self):
        for puzzle_id, aug_puzzles in tqdm(self._data.items(), desc="building index"):
            aug_entries: list[list[int]] = []
            for aug_idx, aug_puzzle in enumerate(aug_puzzles):
                example_indices: list[int] = []
                for example_idx, _ in enumerate(aug_puzzle["examples"]):
                    pointer = ExamplePointer(
                        puzzle_id=puzzle_id,
                        aug_idx=aug_idx,
                        example_idx=example_idx,
                    )
                    self._index.append(pointer)
                    example_indices.append(len(self._index) - 1)
                if example_indices:
                    aug_entries.append(example_indices)
            if aug_entries:
                self._puzzle_aug_examples[puzzle_id] = aug_entries

    def __len__(self):
        return len(self._index)

    def __getitem__(self, index: int):
        pointer = self._index[index]
        aug_puzzle = self._data[pointer.puzzle_id][pointer.aug_idx]
        example = aug_puzzle["examples"][pointer.example_idx]
        return {
            "x": np.asarray(example["x"], dtype=np.int32),
            "y": np.asarray(example["y"], dtype=np.int32),
            "colour_aug": np.asarray(aug_puzzle["colour_aug"], dtype=np.int32),
            "d8_aug": np.asarray([aug_puzzle["d8_aug"]]),
            "aug_puzzle_idx": np.asarray([aug_puzzle["aug_puzzle_idx"]]),
            "puzzle_idx": np.asarray([aug_puzzle["puzzle_idx"]]),
            "example_idx": np.asarray([pointer.example_idx]),
        }

    @property
    def puzzle_ids(self) -> list[str]:
        return self._puzzle_ids

    @property
    def puzzle_aug_examples(self) -> dict[str, list[list[int]]]:
        return self._puzzle_aug_examples



class PuzzleBatchSampler(grain.samplers.Sampler):
    """Sampler that replicates the original puzzle-wise batching logic."""

    def __init__(self, data_source: ArcExampleDataSource, batch_size: int, seed: int = 0):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._data_source = data_source
        self._batch_size = batch_size
        self._rng = random.Random(seed)
        self._puzzle_ids = data_source.puzzle_ids
        if not self._puzzle_ids:
            raise ValueError("Dataset is empty")
        self._puzzle_queue: list[str] = []
        self._record_keys: list[int] = []
    def __len__(self) -> int:
        # Effectively infinite stream of batches.
        return np.iinfo(np.int64).max

    def __getitem__(self, index: int) -> grain.RecordMetadata:
        while index >= len(self._record_keys):
            self._append_batch()
        record_key = self._record_keys[index]
        return grain.RecordMetadata(index=index, record_key=record_key)

    def _append_batch(self) -> None:
        self._record_keys.extend(self._sample_single_batch())

    def _next_puzzle_id(self) -> str:
        if not self._puzzle_queue:
            self._puzzle_queue = self._puzzle_ids[:]
            self._rng.shuffle(self._puzzle_queue)
        return self._puzzle_queue.pop()

    def _sample_single_batch(self) -> list[int]:
        indices: list[int] = []
        remaining = self._batch_size
        while remaining > 0:
            puzzle_id = self._next_puzzle_id()
            aug_examples = self._rng.choice(self._data_source.puzzle_aug_examples[puzzle_id])
            if not aug_examples:
                continue
            if len(aug_examples) <= remaining:
                indices.extend(aug_examples)
                remaining -= len(aug_examples)
            else:
                indices.extend(self._rng.sample(aug_examples, remaining))
                remaining = 0
        return indices


def get_data_loader(
    data_dir,
    batch_size: int = 768,
    translate: str = "fixed",
    max_grid_size: int = 30,
    drop_remainder: bool = True,
    seed: int = 0,
    validation: bool = False,
) -> grain.DataLoader:

    data_source = ArcExampleDataSource(data_dir)

    if validation:
        sampler = grain.samplers.IndexSampler(
            len(data_source),
            seed=seed,
            shuffle=False,
            num_epochs=1,
            shard_options=grain.sharding.NoSharding(),
        )
        batch_drop = False
    else:
        sampler = PuzzleBatchSampler(
            data_source=data_source,
            batch_size=batch_size,
            seed=seed,
        )
        batch_drop = drop_remainder

    operations = [
        TranslateAndPad(translate=translate, max_grid_size=max_grid_size),
        grain.transforms.Batch(batch_size=batch_size, drop_remainder=batch_drop),
    ]
    return grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        shard_options=grain.sharding.NoSharding(),
        worker_count=0,
    )