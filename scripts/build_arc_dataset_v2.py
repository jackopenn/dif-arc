from collections import defaultdict
import os
import json
from functools import partial
import random
import hashlib

import jax
import numpy as np
from tqdm import tqdm
from sws import Config, run



def get_config():
    cfg = Config()
    cfg.input_dir = "data/arc-agi-2"
    cfg.output_dir = lambda: f"{cfg.input_dir}-aug-concept-{cfg.n_augs}-v2"
    cfg.subsets = ["training", "evaluation", "concept"]
    cfg.test_set = "evaluation"
    cfg.n_augs = 1000
    cfg.bg_colour_aug = False # False: keep background black
    cfg.seed = 69420
    cfg.aug_retry_factor = 10
    return cfg


def d8_aug(puzzle_sample, op_idx):
    ops = [
        lambda x: x,
        partial(np.rot90, k=1),
        partial(np.rot90, k=2),
        partial(np.rot90, k=3),
        np.fliplr,
        np.flipud,
        np.transpose,
        lambda x: np.fliplr(np.rot90(x, k=1)),
    ]
    return ops[op_idx](puzzle_sample)


def colour_aug(puzzle_sample, colours):
    return colours[puzzle_sample]


def grid_hash(grid: np.ndarray):
    assert grid.ndim == 2

    buffer = [x.to_bytes() for x in grid.shape]
    buffer.append(grid.tobytes())
    
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def puzzle_hash(puzzle: dict):
    # Hash the puzzle for checking equivalence
    hashes = []
    for example in puzzle['examples']:
        hashes.append(f"{grid_hash(np.asarray(example['x']))}|{grid_hash(np.asarray(example['y']))}")
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def inverse_d8_aug(puzzle_sample, op_idx):
    ops = [
        lambda x: x,
        partial(np.rot90, k=-1),
        partial(np.rot90, k=-2),
        partial(np.rot90, k=-3),
        np.fliplr,
        np.flipud,
        np.transpose,
        lambda x: np.fliplr(np.rot90(x, k=1)),
    ]
    return ops[op_idx](puzzle_sample)


def inverse_colour_aug(puzzle_sample, colours):
    colours = np.argsort(colours)
    return colours[puzzle_sample]


def crop(grid):
    # gpt5 made this
    H, W = grid.shape
    safe = (grid < 10).astype(np.int32)
    S = np.cumsum(np.cumsum(safe, 0), 1)
    hs, ws = np.arange(1, H+1)[:, None], np.arange(1, W+1)[None, :]
    areas = hs * ws
    ma = np.where(S == areas, areas, -1)
    idx = np.argmax(ma)
    max_area = ma.reshape(-1)[idx]

    def no():
        return np.array(0, np.int32), np.array(0, np.int32)
    def yes():
        h_idx, w_idx = np.divmod(idx, W)
        return h_idx + 1, w_idx + 1
    
    # h, w = jax.lax.cond(max_area <= 0, no, yes)
    if max_area <= 0:
        h, w = no()
    else:
        h, w = yes()
        
    return grid[:h, :w]


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
                        "x": np.asarray(pair['input']),
                        "y": np.asarray(pair['output']),
                        "split": split
                    })

            puzzles.append({
                "puzzle_id": puzzle['puzzle_id'],
                "examples": examples,
                "subset": subset
            })
    
    key = jax.random.key(cfg.seed)
    aug_puzzles = defaultdict(list)
    aug_puzzle_idx = 0
    for puzzle_idx, puzzle in enumerate(tqdm(puzzles, desc="augmenting")):
        # no augs
        # print(puzzle)
        base = puzzle.copy() 
        base["puzzle_idx"] = puzzle_idx
        base["aug_puzzle_idx"] = aug_puzzle_idx
        base["d8_aug"] = 0 
        base["colour_aug"] = np.arange(10).tolist()
        base["examples"] = [
            {
                "x": example["x"].tolist(),
                "y": example["y"].tolist(),
                "split": example["split"]
            } for example in base["examples"]
        ]
        aug_puzzles[base["puzzle_id"]].append(base)
        puzzle_hashes = {puzzle_hash(base)}

        # first augment has differnt idx to base
        aug_puzzle_idx += 1

        # keep trying augs until unique n_augs
        for i in range(cfg.n_augs * cfg.aug_retry_factor):
            # print(i)
            key, op_key, colour_key = jax.random.split(key, 3)
            op_idx = jax.random.randint(op_key, (), 0, 8).item()
            if cfg.bg_colour_aug:
                colours = jax.random.permutation(colour_key, np.arange(10))
            else:
                # keep background black (0)
                colours = np.concatenate([np.array([0]), jax.random.permutation(colour_key, np.arange(1, 10))])

            # print(op_idx, colours)
            aug_puzzle = {
                "puzzle_id": puzzle['puzzle_id'],
                "subset": puzzle['subset'],
                "puzzle_idx": puzzle_idx,
                "aug_puzzle_idx": int(aug_puzzle_idx),
                "d8_aug": int(op_idx),
                "colour_aug": colours.tolist(),
                "examples": [
                    {
                        "x": colour_aug(d8_aug(example['x'], op_idx), colours).tolist(),
                        "y": colour_aug(d8_aug(example['y'], op_idx), colours).tolist(),
                        "split": example['split']
                    } for example in puzzle['examples']
                ]
            }

            # print(aug_puzzles)
            hashed_puzzle = puzzle_hash(aug_puzzle)
            # print(hashed_puzzle)
            if hashed_puzzle not in puzzle_hashes:
                # print(op_idx, colours)
                # aug_puzzles.append(aug_puzzle)
                aug_puzzles[aug_puzzle["puzzle_id"]].append(aug_puzzle)
                puzzle_hashes.add(hashed_puzzle)
                aug_puzzle_idx += 1

            if len(puzzle_hashes) >= cfg.n_augs + 1:
                break

        if len(puzzle_hashes) < cfg.n_augs + 1:
            print(f"WARNING: only {len(puzzle_hashes)} augs found for puzzle {puzzle['puzzle_id']}")

    train_puzzles = defaultdict(list)
    test_puzzles = defaultdict(list)
    for puzzle_id, aug_puzzles in aug_puzzles.items():
        for aug_puzzle in aug_puzzles:
            if aug_puzzle['subset'] == cfg.test_set:
                aug_puzzle_test = aug_puzzle.copy()
                aug_puzzle_test['examples'] = list(filter(lambda x: x['split'] == 'test', aug_puzzle_test['examples']))
                test_puzzles[puzzle_id].append(aug_puzzle_test)

                aug_puzzle_train = aug_puzzle.copy()
                aug_puzzle_train['examples'] = list(filter(lambda x: x['split'] == 'train', aug_puzzle_train['examples']))
                train_puzzles[puzzle_id].append(aug_puzzle_train)
            else:
                train_puzzles[puzzle_id].append(aug_puzzle)
        




    train_output_path = f"{cfg.output_dir}/train"
    test_output_path = f"{cfg.output_dir}/test"
    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(test_output_path, exist_ok=True)
    n_train_aug_puzzles = 0
    n_test_aug_puzzles = 0
    n_train_aug_examples = 0
    n_test_aug_examples = 0
    for puzzle_id, aug_puzzles in tqdm(train_puzzles.items(), desc="writing train"):

        with open(f"{train_output_path}/{puzzle_id}.jsonl", 'w') as file:
            for aug_puzzle in aug_puzzles:
                json.dump(aug_puzzle, file)
                file.write("\n")
                n_train_aug_examples += len(aug_puzzle['examples'])
                n_train_aug_puzzles += 1
    for puzzle_id, aug_puzzles in tqdm(test_puzzles.items(), desc="writing test"):
        with open(f"{test_output_path}/{puzzle_id}.jsonl", 'w') as file:
            for aug_puzzle in aug_puzzles:
                json.dump(aug_puzzle, file)
                file.write("\n")
                n_test_aug_examples += len(aug_puzzle['examples'])
                n_test_aug_puzzles += 1
    
    

    # flat_aug_puzzles = []
    # n_train_aug_puzzles = 0
    # n_test_aug_puzzles = 0
    # for puzzle in aug_puzzles:
    #     n_train_aug_puzzles += 1
    #     if puzzle['subset'] == cfg.test_set:
    #         n_test_aug_puzzles += 1
    #     for idx, example in enumerate(puzzle['examples']):
    #         flat_aug_puzzles.append({
    #             "puzzle_id": puzzle['puzzle_id'],
    #             "subset": puzzle['subset'],
    #             "d8_aug": puzzle['d8_aug'],
    #             "colour_aug": puzzle['colour_aug'],
    #             "puzzle_idx": puzzle['puzzle_idx'],
    #             "aug_puzzle_idx": puzzle['aug_puzzle_idx'],
    #             "example_idx": idx,
    #             "x": example['x'],
    #             "y": example['y'],
    #             "split": example['split']
    #         })
    
    # random.seed(cfg.seed)
    # random.shuffle(flat_aug_puzzles)

    # os.makedirs(cfg.output_dir, exist_ok=True)
    # train_output_path = f"{cfg.output_dir}/train.jsonl"
    # test_output_path = f"{cfg.output_dir}/test.jsonl"
    # n_train_aug_examples = 0
    # n_test_aug_examples = 0
    # with open(train_output_path, 'w') as train_file:
    #     with open(test_output_path, 'w') as test_file:
    #         for aug_puzzle in tqdm(flat_aug_puzzles, desc="writing"):
    #             if aug_puzzle['subset'] == cfg.test_set and aug_puzzle["split"] == "test":
    #                 out_file = test_file
    #                 n_test_aug_examples += 1
    #             else:
    #                 out_file = train_file
    #                 n_train_aug_examples += 1
    #             out_puzzle = {
    #                 "puzzle_id": aug_puzzle['puzzle_id'],
    #                 "subset": aug_puzzle['subset'],
    #                 "split": aug_puzzle['split'],
    #                 "d8_aug": int(aug_puzzle['d8_aug']),
    #                 "colour_aug": aug_puzzle['colour_aug'].tolist(),
    #                 "puzzle_idx": int(aug_puzzle['puzzle_idx']),
    #                 "aug_puzzle_idx": int(aug_puzzle['aug_puzzle_idx']),
    #                 "example_idx": int(aug_puzzle['example_idx']),
    #                 "x": aug_puzzle['x'].tolist(),
    #                 "y": aug_puzzle['y'].tolist()
    #             }
    #             json.dump(out_puzzle, out_file)
    #             out_file.write("\n")

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
    print("writing metadata ...")
    with open(f"{cfg.output_dir}/metadata.json", 'w') as file:
        json.dump(metadata, file)

    print("done (sometimes python doesn't exit, but it's done)")

if __name__ == "__main__":
    run(main)