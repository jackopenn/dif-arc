from math import ceil

import jax
from jax import numpy as jnp
import numpy as np
from flax import nnx, struct
from tqdm import tqdm

from scripts.build_arc_dataset import inverse_d8_aug, inverse_colour_aug, crop, grid_hash

@struct.dataclass
class Carry:
    z: jax.Array
    y: jax.Array
    x_input: jax.Array
    aug_puzzle_idx: jax.Array
    y_true: jax.Array
    step: jax.Array
    halted: jax.Array


def init_carry(batch, z_init, y_init):
    """initialize the carry with the initial data"""
    batch_size = batch['x'].shape[0]
    hidden_dim = z_init.shape[-1]
    z_init = jnp.broadcast_to(z_init, (batch_size, 916, hidden_dim))
    y_init = jnp.broadcast_to(y_init, (batch_size, 916, hidden_dim))
    # z_init = shard_data(z_init)
    # y_init = shard_data(y_init)
    return Carry(
        z=z_init,                                         # (batch_size, 901, hidden_dim)
        y=y_init,                                         # (batch_size, 901, hidden_dim)
        x_input=batch['x'],                               # (batch_size, 900)
        aug_puzzle_idx=batch['aug_puzzle_idx'],           # (batch_size,)
        y_true=batch['y'],                                # (batch_size, 900)
        step=jnp.zeros((batch_size, ), dtype=jnp.int32),  # (batch_size,)
        halted=jnp.zeros((batch_size, ), dtype=jnp.bool_) # (batch_size,)
    )
  
@nnx.jit(static_argnames=["N_supervision", "n", "T" ])
def eval_step(model, carry, N_supervision, n, T):
    def latent_recursion(model, x, y, z, n):
        for _ in range(n):
            z = model(x, y, z)
        y = model(y, z)
        return y, z

    x = model.input_embedding(carry.x_input, carry.aug_puzzle_idx)
    y, z = carry.y, carry.z
    for _ in range(N_supervision):
        for _ in range(T):
            y, z = latent_recursion(model, x, y, z, n)
    y_logits = model.output_head(y)
    y_preds = jnp.argmax(y_logits, axis=-1)
    return y_preds


def get_top_k_preds(example_preds, k):
    # example_preds is a dictionary of predictions and their counts
    # return the top k predictions
    example_preds = dict(sorted(example_preds.items(), key=lambda x: x[1], reverse=True))
    top_k = []
    current_k = 0
    for pred, count in example_preds.items():
        top_k.append(pred)
        current_k += count
        if current_k >= k:
            break
    return top_k



def evaluate(model, data_loader_factory, y_init, z_init, N_supervision, n, T, pass_ks, shard_data, batch_size):
    
    # preds = {
    #     "abcde1g7": {
    #         "0": {
    #             "y_true": ...,
    #             "y_preds": {
    #               "pred_1": count,
    #               "pred_2": count,
    #               ...
    #               "pred_n": count
    #         }
    #     }
    # }
    preds = {}
    data_loader = data_loader_factory()
    for batch in tqdm(data_loader, desc="evaluating", total=ceil(len(data_loader._data_source) / batch_size)):
        batch = shard_data(batch)
        
        carry = init_carry(batch, z_init, y_init)

        y_preds = eval_step(model, carry, N_supervision, n, T)

        y_preds = jax.experimental.multihost_utils.process_allgather(y_preds, tiled=True)
        y_trues = jax.experimental.multihost_utils.process_allgather(batch['y'], tiled=True)
        puzzle_idxs = jax.experimental.multihost_utils.process_allgather(batch['puzzle_idx'], tiled=True)
        example_idxs = jax.experimental.multihost_utils.process_allgather(batch['example_idx'], tiled=True)
        d8_augs = jax.experimental.multihost_utils.process_allgather(batch['d8_aug'], tiled=True)
        colour_augs = jax.experimental.multihost_utils.process_allgather(batch['colour_aug'], tiled=True)
        
        y_preds = np.array(y_preds.reshape(batch['x'].shape[0], 30, 30))
        y_trues = np.array(y_trues.reshape(batch['x'].shape[0], 30, 30))
        
        for i in range(batch['x'].shape[0]):
            # Unwrap scalars from batched fields
            puzzle_idx = int(puzzle_idxs[i][0])
            example_idx = int(example_idxs[i][0])
            d8_aug = int(d8_augs[i][0])
            colour_aug = colour_augs[i]
            y_pred = y_preds[i]
            
            y_pred = crop(y_pred)
            y_pred = grid_hash(inverse_d8_aug(inverse_colour_aug(y_pred, colour_aug), d8_aug))

            if puzzle_idx not in preds:
                preds[puzzle_idx] = {}
            if example_idx not in preds[puzzle_idx]:
                preds[puzzle_idx][example_idx] = {"y_true": None, "y_preds": dict()}
                y_true = y_trues[i]
                y_true = crop(y_true)
                y_true = grid_hash(inverse_d8_aug(inverse_colour_aug(y_true, colour_aug), d8_aug))
                preds[puzzle_idx][example_idx]['y_true'] = y_true

            if y_pred not in preds[puzzle_idx][example_idx]['y_preds']:
                preds[puzzle_idx][example_idx]['y_preds'][y_pred] = 1
            else:
                preds[puzzle_idx][example_idx]['y_preds'][y_pred] += 1

    # passes = {
    #     "abcde1g7": {
    #         k_1: [True, False],
    #         k_2: [True, False],
    #         ...
    #         k_n: [True, False]
    #     }
    # }
    passes = {}
    for puzzle_idx, data in tqdm(preds.items(), desc="computing passes"):
        for example_idx, example in data.items():
            y_true = example['y_true']
            for k in pass_ks:
                top_k_preds = get_top_k_preds(example['y_preds'], k)
                if puzzle_id not in passes:
                    passes[puzzle_idx] = {}
                if k not in passes[puzzle_idx]:
                    passes[puzzle_idx][k] = []
                passes[puzzle_idx][k].append(y_true in top_k_preds)
                
    # passes_reduced = {
    #     k_1: n_true,
    #     k_2: n_true,
    #     ...
    #     k_n: n_true
    # }
    passes_reduced = {}
    for puzzle_idx, ks in tqdm(passes.items(), desc="computing passes reduced"):
        for k, vs in ks.items():
            passes_reduced[k] = passes_reduced.get(k, 0) + int(all(vs))

    print(passes_reduced)
    
    n_puzzles = len(passes)
    passes_reduced = {f"pass_{k}": n_true / n_puzzles for k, n_true in passes_reduced.items()}

    return passes_reduced