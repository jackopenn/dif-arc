import os
import json
from functools import partial
import random

import jax
from jax import numpy as jnp
from tqdm import tqdm
from sws import Config, run


def get_config():
    cfg = Config()
    cfg.input_dir = "data"
    cfg.output_dir = lambda: f"data/arc-aug-{cfg.n_augs}"
    cfg.subsets = ["training", "evaluation"]
    cfg.test_set = "evaluation"
    cfg.n_augs = 10
    cfg.bg_cololour_aug = False # False: keep background black
    cfg.seed = 69420    
    return cfg


def d8_aug(puzzle_sample, op_idx):
    ops = [
        lambda x: x,
        partial(jnp.rot90, k=1),
        partial(jnp.rot90, k=2),
        partial(jnp.rot90, k=3),
        jnp.fliplr,
        jnp.flipud,
        jnp.transpose,
        lambda x: jnp.transpose(jnp.rot90(x, k=1)),
    ]
    return ops[op_idx](puzzle_sample)


def colour_aug(puzzle_sample, colours):
    return colours[puzzle_sample]


def main(cfg):
    puzzles = []
    n_train_puzzles = 0
    n_test_puzzles = 0
    n_train_examples = 0
    n_test_examples = 0
    for subset in cfg.subsets:
        input_path = f"{cfg.input_dir}/{subset}"
        file_paths = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith('.json')
        ]
        for file_path in file_paths:
            puzzle = json.load(open(file_path, 'r'))
            puzzle['puzzle_id'] = file_path.split('/')[-1].split('.')[0]
            n_train_puzzles += 1
            if subset == cfg.test_set:
                n_test_puzzles += 1

            examples = []
            for split in ['train', 'test']:
                for pair in puzzle[split]:
                    if subset == cfg.test_set and split == "test":
                        n_test_examples += 1
                    else:
                        n_train_examples += 1
                    examples.append({
                        "x": jnp.asarray(pair['input']),
                        "y": jnp.asarray(pair['output']),
                        "split": split
                    })

            puzzles.append({
                "puzzle_id": puzzle['puzzle_id'],
                "examples": examples,
                "subset": subset
            })
    
    key = jax.random.key(cfg.seed)
    aug_puzzles = []
    aug_puzzle_idx = 0
    puzzle_idx = 0
    for puzzle in tqdm(puzzles, desc="augmenting"):
        # no augs
        base = puzzle.copy() 
        base["puzzle_idx"] = puzzle_idx
        base["aug_puzzle_idx"] = aug_puzzle_idx
        base["d8_aug"] = 0 
        base["colour_aug"] = jnp.arange(10)
        aug_puzzles.append(base)
        puzzle_metas = {(base['puzzle_id'], base['d8_aug'], str(base["colour_aug"]))}

        # keep trying augs until unique n_augs
        current_augs = 0
        while current_augs < cfg.n_augs:
            key, op_key, colour_key = jax.random.split(key, 3)
            op_idx = jax.random.randint(op_key, (), 0, 8).item()
            if cfg.bg_cololour_aug:
                colours = jax.random.permutation(colour_key, jnp.arange(10))
            else:
                # keep background black (0)
                colours = jnp.concatenate([jnp.array([0]), jax.random.permutation(colour_key, jnp.arange(1, 10))])

            aug_puzzle = {
                "puzzle_id": puzzle['puzzle_id'],
                "subset": puzzle['subset'],
                "puzzle_idx": puzzle_idx,
                "aug_puzzle_idx": aug_puzzle_idx,
                "d8_aug": op_idx,
                "colour_aug": colours,
                "examples": [
                    {
                        "x": colour_aug(d8_aug(example['x'], op_idx), colours),
                        "y": colour_aug(d8_aug(example['y'], op_idx), colours),
                        "split": example['split']
                    } for example in puzzle['examples']
                ]
            }
            aug_meta = (aug_puzzle['puzzle_id'], aug_puzzle['d8_aug'], str(aug_puzzle["colour_aug"]))
            if aug_meta not in puzzle_metas:
                aug_puzzles.append(aug_puzzle)
                puzzle_metas.add(aug_meta)
                current_augs += 1
                aug_puzzle_idx += 1

    flat_aug_puzzles = []
    n_train_aug_puzzles = 0
    n_test_aug_puzzles = 0
    for puzzle in aug_puzzles:
        n_train_aug_puzzles += 1
        if puzzle['subset'] == cfg.test_set:
            n_test_aug_puzzles += 1
        for idx, example in enumerate(puzzle['examples']):
            flat_aug_puzzles.append({
                "puzzle_id": puzzle['puzzle_id'],
                "subset": puzzle['subset'],
                "d8_aug": puzzle['d8_aug'],
                "colour_aug": puzzle['colour_aug'],
                "puzzle_idx": puzzle['puzzle_idx'],
                "aug_puzzle_idx": puzzle['aug_puzzle_idx'],
                "example_idx": idx,
                "x": example['x'],
                "y": example['y'],
                "split": example['split']
            })
    
    random.seed(cfg.seed)
    random.shuffle(flat_aug_puzzles)

    os.makedirs(cfg.output_dir, exist_ok=True)
    train_output_path = f"{cfg.output_dir}/train.jsonl"
    test_output_path = f"{cfg.output_dir}/test.jsonl"
    n_train_aug_examples = 0
    n_test_aug_examples = 0
    with open(train_output_path, 'w') as train_file:
        with open(test_output_path, 'w') as test_file:
            for aug_puzzle in tqdm(flat_aug_puzzles, desc="writing"):
                if aug_puzzle['subset'] == cfg.test_set and aug_puzzle["split"] == "test":
                    out_file = test_file
                    n_test_aug_examples += 1
                else:
                    out_file = train_file
                    n_train_aug_examples += 1
                out_puzzle = {
                    "puzzle_id": aug_puzzle['puzzle_id'],
                    "subset": aug_puzzle['subset'],
                    "split": aug_puzzle['split'],
                    "d8_aug": int(aug_puzzle['d8_aug']),
                    "colour_aug": aug_puzzle['colour_aug'].tolist(),
                    "puzzle_idx": int(aug_puzzle['puzzle_idx']),
                    "aug_puzzle_idx": int(aug_puzzle['aug_puzzle_idx']),
                    "example_idx": int(aug_puzzle['example_idx']),
                    "x": aug_puzzle['x'].tolist(),
                    "y": aug_puzzle['y'].tolist()
                }
                json.dump(out_puzzle, out_file)
                out_file.write("\n")

    metadata = {
        "config": cfg.to_dict(),
        "train": {
            "num_puzzles": n_train_puzzles,
            "num_examples": n_train_examples,
            "num_aug_puzzles": n_train_aug_puzzles,
            "num_aug_examples": n_train_aug_examples,
        },
        "test": {
            "num_puzzles": n_test_puzzles,
            "num_examples": n_test_examples,
            "num_aug_puzzles": n_test_aug_puzzles,
            "num_aug_examples": n_test_aug_examples,
        }
    }
    with open(f"{cfg.output_dir}/metadata.json", 'w') as file:
        json.dump(metadata, file)


if __name__ == "__main__":
    run(main)