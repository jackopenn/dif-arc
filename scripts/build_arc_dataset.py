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
    cfg.output_dir = "data"
    cfg.split = "training"
    cfg.n_augs = 100
    cfg.seed = 69420
    return cfg


def d8_aug(puzzle, op_idx):
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
    return {
        **puzzle,
        "x": ops[op_idx](puzzle["x"]), 
        "y": ops[op_idx](puzzle["y"]),
        "d8_aug": op_idx
    }


def colour_aug(puzzle, colours):
    return {
        **puzzle,
        "x": colours[puzzle["x"]],
        "y": colours[puzzle["x"]],
        "colour_aug": colours
    }


def main(cfg):
    input_path = f"{cfg.input_dir}/{cfg.split}" 
    file_paths = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path) 
        if f.endswith('.json')
    ]
    puzzles = []
    for file_path in file_paths:
        puzzle = json.load(open(file_path, 'r'))
        puzzle['puzzle_id'] = file_path.split('/')[-1].split('.')[0]
        puzzles.append(puzzle)

    flat_puzzles = []
    for puzzle in puzzles:
        for pair in puzzle['train']:
            flat_puzzles.append({
                "puzzle_id": puzzle['puzzle_id'],
                "x": jnp.asarray(pair['input']),
                "y": jnp.asarray(pair['output'])
            })
    
    key = jax.random.key(cfg.seed)
    aug_puzzles = []
    for puzzle in tqdm(flat_puzzles, desc="augmenting"):
        # no augs
        base = puzzle.copy() 
        base["d8_aug"] = -1
        base["colour_aug"] = jnp.arange(10)
        aug_puzzles.append(base)
        puzzle_metas = {(base['puzzle_id'], base['d8_aug'], str(base["colour_aug"]))}

        # keep trying augs until unique n_augs
        current_augs = 0
        while current_augs < cfg.n_augs:
            key, op_key, colour_key = jax.random.split(key, 3)
            op_idx = jax.random.randint(op_key, (), 0, 8).item()
            colours = jax.random.permutation(colour_key, jnp.arange(10))
            aug_puzzle = colour_aug(d8_aug(puzzle.copy(), op_idx), colours)
            aug_meta = (aug_puzzle['puzzle_id'], aug_puzzle['d8_aug'], str(aug_puzzle["colour_aug"]))
            if aug_meta not in puzzle_metas:
                aug_puzzles.append(aug_puzzle)
                puzzle_metas.add(aug_meta)
                current_augs += 1
    
    random.seed(cfg.seed)
    random.shuffle(aug_puzzles)

    output_path = f"{cfg.output_dir}/{cfg.split}_n_augs={cfg.n_augs}.jsonl"
    with open(output_path, 'w') as file:
        for aug_puzzle in tqdm(aug_puzzles, desc="writing"):
            out_puzzle = {
                **aug_puzzle,
                "x": aug_puzzle["x"].tolist(),
                "y": aug_puzzle["y"].tolist(),
                "colour_aug": aug_puzzle["colour_aug"].tolist()
            }
            json.dump(out_puzzle, file)
            file.write("\n")


if __name__ == "__main__":
    run(main)