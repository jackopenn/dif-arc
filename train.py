import time
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from flax import nnx, struct
import optax
import json
import os
from sws import Config, run
import wandb
from functools import partial

from dataset import get_data_loader
from modelling.model import Model
from utils import MetricLogger

def get_config():
    
    def get_puzzle_vocab_size(data_dir):
        return json.load(open(os.path.join(data_dir, "metadata.json"), 'r'))['train']['num_aug_puzzles']
    
    cfg = Config()
    cfg.seed = 69420
    
    cfg.model.vocab_size = 10 + 1 # +1 for padding
    cfg.model.hidden_dim = 16
    cfg.model.intermediate_dim = lambda: 1 * cfg.model.hidden_dim
    cfg.model.num_layers = 1
    cfg.model.num_attention_heads = 1
    cfg.model.num_key_value_heads = 1
    cfg.model.head_dim = lambda: cfg.model.hidden_dim // cfg.model.num_attention_heads
    cfg.model.act_fn = "swish"
    cfg.model.tie_embeddings = False
    cfg.model.rope_theta = 10000
    cfg.model.puzzle_vocab_size = lambda: get_puzzle_vocab_size(cfg.data.data_dir)
    
    cfg.recursion.N_supervision = 16
    cfg.recursion.n = 1
    cfg.recursion.T = 1
    
    cfg.optim.weight_decay = 0.1
    cfg.optim.b1 = 0.9
    cfg.optim.b2 = 0.95


    # TODO: embeddings have diff lr
    cfg.schedule.init_value = 0
    cfg.schedule.peak_value = 1e-4  
    cfg.schedule.warmup_steps = 100

    cfg.max_steps = 100000

    cfg.data.data_dir = "data/arc-aug-10"
    cfg.data.batch_size = 4

    cfg.parallel.n_devices = 1

    return cfg


def main(cfg):
    key = jax.random.key(cfg.seed)
    
    model = Model(**cfg.model.to_dict(), rngs=nnx.Rngs(key))
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(optax.warmup_constant_schedule(**cfg.schedule.to_dict()), **cfg.optim.to_dict()),
        wrt=nnx.Param,
    )

    shard_data = lambda data: data
    if cfg.parallel.n_devices > 1:
        mesh = jax.make_mesh((cfg.parallel.n_devices,), ("data",))
        jax.set_mesh(mesh)

        repl_sharding = NamedSharding(mesh, P())
        data_sharding = NamedSharding(mesh, P("data", None))

        _, model_state = nnx.split(model)
        sharded_model_state = jax.lax.with_sharding_constraint(model_state, repl_sharding)
        nnx.update(model, sharded_model_state)
        
        _, optimizer_state = nnx.split(optimizer)
        sharded_optimizer_state = jax.lax.with_sharding_constraint(optimizer_state, repl_sharding)
        nnx.update(optimizer, sharded_optimizer_state)

        shard_data = lambda data: jax.tree.map(lambda x: jax.device_put(x, data_sharding), data)


    @struct.dataclass
    class Carry:
        z: jax.Array
        y: jax.Array
        x_input: jax.Array
        aug_puzzle_idx: jax.Array
        y_true: jax.Array
        step: jax.Array
        halted: jax.Array
    
    def init_carry(batch, z_init, y_init, hidden_dim):
        """initialize the carry with the initial data"""
        batch_size = batch['x'].shape[0]
        return Carry(
            z=jnp.broadcast_to(z_init, (batch_size, 901, hidden_dim)), # (batch_size, 900, hidden_dim)
            y=jnp.broadcast_to(y_init, (batch_size, 901, hidden_dim)), # (batch_size, 900, hidden_dim)
            x_input=batch['x'],                                        # (batch_size, 900)
            aug_puzzle_idx=batch['aug_puzzle_idx'],                    # (batch_size,)
            y_true=batch['y'],                                         # (batch_size, 900)
            step=jnp.zeros((batch_size, ), dtype=jnp.int32),           # (batch_size,)
            halted=jnp.zeros((batch_size, ), dtype=jnp.bool_),         # (batch_size,)
        )
    
    def pre_update_carry(carry, batch, z_init, y_init):
        """update the carry with new data from batch (if halted)"""
        return Carry(
            z=jnp.where(carry.halted[..., jnp.newaxis, jnp.newaxis], z_init[jnp.newaxis, jnp.newaxis, ...], carry.z),
            y=jnp.where(carry.halted[..., jnp.newaxis, jnp.newaxis], y_init[jnp.newaxis, jnp.newaxis, ...], carry.y),
            x_input=jnp.where(carry.halted[..., jnp.newaxis], batch['x'], carry.x_input),
            aug_puzzle_idx=jnp.where(carry.halted[..., jnp.newaxis], batch['aug_puzzle_idx'], carry.aug_puzzle_idx),
            y_true=jnp.where(carry.halted[..., jnp.newaxis], batch['y'], carry.y_true),
            step=jnp.where(carry.halted, 0, carry.step),
            halted=jnp.where(carry.halted, False, carry.halted),
        )
    
    def post_update_carry(carry, q_logits, z, y, N_supervision):
        """update the halt flag if step >= N_supervision or q_logits > 0"""
        step = carry.step + 1
        halted = jnp.where(step >= N_supervision, True, carry.halted)
        halted = jnp.where(q_logits.reshape(-1) > 0, True, halted)
        return Carry(
            z=z,
            y=y,
            x_input=carry.x_input,
            aug_puzzle_idx=carry.aug_puzzle_idx,
            y_true=carry.y_true,
            step=step,
            halted=halted,
        )
        
    def latent_recursion(model, x, y, z, n):
        for _ in range(n):
            z = model(x, y, z)
        y = model(y, z)
        return y, z
    
    
    def loss_fn(model, x_input, aug_puzzle_idx, y, z, y_true, n):
        # forward pass
        x = model.input_embedding(x_input, aug_puzzle_idx)
        y, z = latent_recursion(model, x, y, z, n)
        y_logits, q_logits = model.output_head(y), model.q_head(z)
        # compute losses
        y_loss = optax.softmax_cross_entropy_with_integer_labels(
            y_logits.reshape(-1, y_logits.shape[-1]).astype(jnp.float32),
            y_true.reshape(-1)
        ).mean(where=y_true.reshape(-1) < 10)
        q_loss = optax.sigmoid_binary_cross_entropy(
            q_logits,
            (jnp.argmax(y_logits, axis=-1) == y_true)
        ).mean()
        loss = y_loss + q_loss
        return loss, (y, z, y_loss, q_loss, y_logits, q_logits)
    
    
    @nnx.jit(static_argnames=["N_supervision", "n", "T"])
    def train_step(model, optimizer, carry, batch, y_init, z_init, N_supervision, n, T):
        # update carry (if halted, update with init and batch)
        carry = pre_update_carry(carry, batch, z_init, y_init)
        # 1 N_supervision step
        x = model.input_embedding(carry.x_input, carry.aug_puzzle_idx)
        # deep recursion loop (no grads)
        z, y = carry.z, carry.y
        for _ in range(T-1):
            y, z = latent_recursion(model, x, y, z, n)
        # 1-step approx BPTT
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, (y, z, y_loss, q_loss, y_logits, q_logits)), grads = grad_fn(
            model, carry.x_input, carry.aug_puzzle_idx, y, z, carry.y_true, n
        )
        optimizer.update(model, grads)

        # compute metrics (10 = padding)
        correct = jnp.argmax(y_logits, axis=-1) == carry.y_true
        cell_acc = correct.mean(where=carry.y_true < 10)
        puzzle_acc = correct.all(axis=-1, where=carry.y_true < 10).mean()
        metrics = {
            "loss": loss,
            "y_loss": y_loss,
            "q_loss": q_loss,
            "cell_acc": cell_acc,
            "puzzle_acc": puzzle_acc,
            "y_max": jnp.max(jnp.abs(y)),
            "z_max": jnp.max(jnp.abs(z)),
            "y_norm": jnp.sqrt(jnp.mean(y**2)),
            "z_norm": jnp.sqrt(jnp.mean(z**2)),
            "n_supervision_steps": carry.step.mean(),
        }

        # update halt flag
        carry = post_update_carry(carry, q_logits, z, y, N_supervision)
        return carry, metrics, q_logits
    
    # init logging 
    wandb.init(project="arc", entity="jackpenn", config=cfg.to_dict())
    train_logger = MetricLogger(batch_size=cfg.data.batch_size, wandb=wandb)

    # init latents
    y_key, z_key = jax.random.split(key, 2)
    initializer = jax.nn.initializers.truncated_normal(stddev=1.0)
    y_init = initializer(y_key, (cfg.model.hidden_dim,), jnp.bfloat16)
    z_init = initializer(z_key, (cfg.model.hidden_dim,), jnp.bfloat16) 
    y_init, z_init = shard_data(y_init), shard_data(z_init)

    # init data loader
    train_data_loader = get_data_loader(cfg.data.data_dir + "/train.jsonl", cfg.data.batch_size)
    train_iter = (shard_data(batch) for batch in train_data_loader)

    t0 = time.time()
    for step, batch in enumerate(train_iter):
        batch = shard_data(batch)
        if step == 0:
            carry = init_carry(batch, z_init, y_init, cfg.model.hidden_dim)
        carry, metrics, q_logits = train_step(
            model, optimizer, carry, batch, y_init, z_init,
            cfg.recursion.N_supervision, cfg.recursion.n, cfg.recursion.T
        )
        # print(carry.step, carry.halted, q_logits.reshape(4))
        step_time = time.time() - t0
        train_logger.log({**metrics, "step_time": step_time})
        t0 = time.time()

if __name__ == "__main__":
    run(main)