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

# TODO: log epochs
# TODO: add eval loop (match augs in eval set)

def main(cfg):
    key = jax.random.key(cfg.seed)
    
    model = Model(**cfg.model.to_dict(), rngs=nnx.Rngs(key))
    # TODO: need to check this works
    tx = optax.partition(
        {
            "embed": optax.adamw(
                optax.warmup_constant_schedule(**cfg.embed_schedule.to_dict()),**cfg.optim.to_dict()
            ),
            "other": optax.adamw(
                optax.warmup_constant_schedule(**cfg.other_schedule.to_dict()), **cfg.optim.to_dict()
            )
        },
        lambda state: jax.tree.map_with_path(lambda path, _: "embed" if path[0].key == "embed" else "other", state)
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    shard_data = lambda data: data
    if cfg.parallel.n_devices > 1:
        mesh = jax.make_mesh((cfg.parallel.n_devices,), ("data",))
        jax.set_mesh(mesh)

        repl_sharding = NamedSharding(mesh, P())
        data_sharding = NamedSharding(mesh, P("data",))

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
        z_init = jnp.broadcast_to(z_init, (batch_size, 901, hidden_dim))
        y_init = jnp.broadcast_to(y_init, (batch_size, 901, hidden_dim))
        if cfg.parallel.n_devices > 1:
            z_init = jax.device_put(z_init, data_sharding)
            y_init = jax.device_put(y_init, data_sharding)
        return Carry(
            z=z_init,                                         # (batch_size, 901, hidden_dim)
            y=y_init,                                         # (batch_size, 901, hidden_dim)
            x_input=batch['x'],                               # (batch_size, 900)
            aug_puzzle_idx=batch['aug_puzzle_idx'],           # (batch_size,)
            y_true=batch['y'],                                # (batch_size, 900)
            step=jnp.zeros((batch_size, ), dtype=jnp.int32),  # (batch_size,)
            halted=jnp.zeros((batch_size, ), dtype=jnp.bool_) # (batch_size,)
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
    

    def post_update_carry(carry, q_logits, z, y, N_supervision, halt_explore_prob, key):
        """update the halt flag if step >= N_supervision or q_logits > 0"""
        step = carry.step + 1
        halted = jnp.where(step >= N_supervision, True, carry.halted)
        if cfg.recursion.act:
            halted = jnp.where(q_logits.reshape(-1) > 0, True, halted)
        if halt_explore_prob > 0:
            key, subkey, subkey2 = jax.random.split(key, 3)
            min_halt_steps = (jax.random.uniform(subkey, halted.shape) < halt_explore_prob) * jax.random.randint(subkey2, step.shape, minval=2, maxval=N_supervision + 1)
            halted = halted & (step >= min_halt_steps)
        return Carry(
            z=z,
            y=y,
            x_input=carry.x_input,
            aug_puzzle_idx=carry.aug_puzzle_idx,
            y_true=carry.y_true,
            step=step,
            halted=halted,
        ), key

    def stable_max_with_integer_labels(logits, labels, axis=-1):
        # s_logits = jnp.where(logits >= 0.0, logits + 1.0, jnp.reciprocal(1.0 - logits + 1e-30))
        # log_probs = jnp.log(s_logits / (jnp.sum(s_logits, axis=axis, keepdims=True)))
        # log_probs_2 = jnp.take_along_axis(log_probs, jnp.expand_dims(labels, axis), axis=axis).squeeze(axis)
        # return -log_probs_2
        eps = 1e-10
        s_logits = jnp.where(logits >= 0, jnp.log(logits + 1 + eps), -jnp.log(1 - logits + eps))
        return optax.softmax_cross_entropy_with_integer_labels(s_logits, labels, axis)


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
        y_preds = jnp.argmax(y_logits, axis=-1)
        # compute losses
        # y_loss = optax.softmax_cross_entropy_with_integer_labels(
        y_loss = stable_max_with_integer_labels(
            y_logits.reshape(-1, y_logits.shape[-1]).astype(jnp.float32),
            y_true.reshape(-1)
        ).mean(where=y_true.reshape(-1) < 10)
        if cfg.recursion.act:
            # TODO: only compute for halted ?
            q_loss = optax.sigmoid_binary_cross_entropy(
                q_logits.reshape(-1),
                (y_preds == y_true).all(axis=-1, where=y_true < 10)
            ).mean()
        else:
            q_loss = 0
        loss = y_loss + 0.5*q_loss
        return loss, (y, z, y_loss, q_loss, y_preds, q_logits)
    
    
    @nnx.jit(static_argnames=["N_supervision", "n", "T", "halt_explore_prob"])
    def train_step(model, optimizer, carry, batch, y_init, z_init, N_supervision, n, T, halt_explore_prob, key):
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
        (loss, (y, z, y_loss, q_loss, y_preds, q_logits)), grads = grad_fn(
            model, carry.x_input, carry.aug_puzzle_idx, y, z, carry.y_true, n
        )
        optimizer.update(model, grads)

        # update halt flag
        carry, key = post_update_carry(carry, q_logits, z, y, N_supervision, halt_explore_prob, key)

        # compute metrics (10 = padding)
        cell_correct = y_preds == carry.y_true # (batch_size, 900)
        puzzle_correct = cell_correct.all(axis=-1, where=carry.y_true < 10)
        cell_acc = cell_correct.mean(where=(carry.y_true < 10) & (carry.halted[..., jnp.newaxis]))
        puzzle_acc = puzzle_correct.mean(where=carry.halted)
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
            "n_supervision_steps": carry.step.mean(where=carry.halted),
        }
        if cfg.recursion.act:
            q_acc = ((q_logits.reshape(-1) > 0) == puzzle_correct).mean(where=carry.halted)
            metrics["q_acc"] = q_acc

        return carry, metrics, key
    
    # init logging 
    if cfg.wandb:
        wandb.init(project="arc", entity="jackpenn", config=cfg.to_dict())
        train_logger = MetricLogger(cfg.data.batch_size, wandb)
    else:
        train_logger = MetricLogger(cfg.data.batch_size, None)

    # init latents
    key, y_key, z_key = jax.random.split(key, 3)
    initializer = jax.nn.initializers.truncated_normal(stddev=1.0)
    y_init = initializer(y_key, (cfg.model.hidden_dim,), jnp.bfloat16)
    z_init = initializer(z_key, (cfg.model.hidden_dim,), jnp.bfloat16) 

    # init data loader
    train_data_loader = get_data_loader(cfg.data.data_dir + "/train.jsonl", cfg.data.batch_size)
    train_iter = (shard_data(batch) for batch in train_data_loader)

    # init profiler
    profiler_options = jax.profiler.ProfileOptions()
    profiler_options.host_tracer_level = 3
    trace_dir = "profile"

    t0 = time.time()
    for step, batch in enumerate(train_iter):
        batch = shard_data(batch)
        if step == 0:
            carry = init_carry(batch, z_init, y_init, cfg.model.hidden_dim)
        if step == 10: 
            jax.profiler.start_trace(trace_dir, profiler_options=profiler_options)
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            carry, metrics, key = train_step(
                model, optimizer, carry, batch, y_init, z_init,
                cfg.recursion.N_supervision, cfg.recursion.n, cfg.recursion.T,
                cfg.recursion.halt_explore_prob, key
            )
        if step == 15:
            jax.profiler.stop_trace()
        step_time = time.time() - t0
        train_logger.log({**metrics, "step_time": step_time})
        t0 = time.time()

if __name__ == "__main__":
    run(main)