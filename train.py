import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from flax import nnx
import optax
import json
import os
from sws import Config, run
import wandb
from functools import partial

from dataset import get_data_loader
from modelling.model import Model

def get_config():
    
    def get_puzzle_vocab_size(data_dir):
        return json.load(open(os.path.join(data_dir, "metadata.json"), 'r'))['train']['num_aug_puzzles']
    
    cfg = Config()
    cfg.seed = 69420
    
    cfg.model.vocab_size = 10 + 1 # +1 for padding
    cfg.model.hidden_dim = 512
    cfg.model.intermediate_dim = lambda: 4 * cfg.model.hidden_dim
    cfg.model.num_layers = 2
    cfg.model.num_attention_heads = 8
    cfg.model.num_key_value_heads = 8
    cfg.model.head_dim = lambda: cfg.model.hidden_dim // cfg.model.num_attention_heads
    cfg.model.act_fn = "swish"
    cfg.model.tie_embeddings = False
    cfg.model.rope_theta = 10000
    cfg.model.puzzle_vocab_size = lambda: get_puzzle_vocab_size(cfg.data.data_dir)
    
    cfg.recursion.N_supervision = 4
    cfg.recursion.n = 4
    cfg.recursion.T = 2
    
    cfg.optim.weight_decay = 0.1
    cfg.optim.b1 = 0.9
    cfg.optim.b2 = 0.95


    # TODO: embeddings have diff lr
    cfg.schedule.init_value = 0
    cfg.schedule.peak_value = 1e-4  
    cfg.schedule.warmup_steps = 100

    cfg.max_steps = 100000

    cfg.data.data_dir = "data/arc-aug-10"
    cfg.data.batch_size = 768

    cfg.parallel.n_devices = 8

    return cfg


def main(cfg):
    key = jax.random.key(cfg.seed)
    
    model = Model(**cfg.model.to_dict(), rngs=nnx.Rngs(key))
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(optax.warmup_constant_schedule(**cfg.schedule.to_dict()), **cfg.optim.to_dict()),
        wrt=nnx.Param,
    )

    shard_batch = lambda batch: batch
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

        shard_batch = lambda batch: jax.tree.map(lambda x: jax.device_put(x, data_sharding), batch)

    
    def latent_recursion(model, x, y, z, n):
        for _ in range(n):
            z = model(x, y, z)
        y = model(y, z)
        return y, z
    
    
    def deep_recursion(model, x, y, z, n, T):
        y, z = latent_recursion(model, x, y, z, n)
        return (y, z), model.output_head(y), model.Q_head(z)
    
    
    def loss_fn(model, x_input, aug_puzzle_idx, y_true, y,z, n, T):
        x = model.input_embedding(x_input, aug_puzzle_idx)
        (y, z), y_hat, q_hat = deep_recursion(model, x, y, z, n, T)

        y_loss = optax.softmax_cross_entropy_with_integer_labels(
            y_hat.reshape(-1, y_hat.shape[-1]).astype(jnp.float32),
            y_true.reshape(-1)
        ).mean(where=y_true.reshape(-1) < 10)
        # TODO: add Q loss
        # Q_loss = optax.sigmoid_binary_cross_entropy(q_hat, (y_hat == y_true)).mean()
        Q_loss = 0
        loss = y_loss + Q_loss
        # metrics
        y_hat_max = jnp.argmax(y_hat, axis=-1)
        correct = y_hat_max == y_true
        cell_acc = correct.mean(where=y_true < 10)
        puzzle_acc = correct.all(axis=-1, where=y_true < 10).mean()
        
        metrics = {
            "loss": loss,
            "y_loss": y_loss,
            "Q_loss": Q_loss,
            "cell_acc": cell_acc,
            "puzzle_acc": puzzle_acc,
            "y_max": jnp.max(jnp.abs(y)),
            "z_max": jnp.max(jnp.abs(z)),
            "y_norm": jnp.sqrt(jnp.mean(y**2)),
            "z_norm": jnp.sqrt(jnp.mean(z**2)),
        }
        return loss, (metrics, y, z)
    
    
    @nnx.jit(static_argnames=["N_supervision", "n", "T"])
    def train_step(model, optimizer, batch, y_init, z_init, N_supervision, n, T):
        metrics = []
        x_input = batch['x']
        y_true = batch['y'] 
        aug_puzzle_idx = batch['aug_puzzle_idx']
        y = y_init
        z = z_init
        x = model.input_embedding(x_input, aug_puzzle_idx)
        for s in range(N_supervision):
            # deep recursion loop (no grads)
            for _ in range(T-1):
                y, z = latent_recursion(model, x, y, z, n)
            # 1-step approx BPTT
            grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
            (_, (step_metrics, y, z)), grads = grad_fn(model, x_input, aug_puzzle_idx, y_true, y, z, n, T)
            optimizer.update(model, grads)
            metrics.append(step_metrics)
        return metrics
       
    wandb.init(project="arc", entity="jackpenn", config=cfg.to_dict())

    y_key, z_key = jax.random.split(key, 2)
    initializer = jax.nn.initializers.truncated_normal(stddev=1.0)
    y_init = initializer(y_key, (cfg.model.hidden_dim,), jnp.bfloat16)
    z_init = initializer(z_key, (cfg.model.hidden_dim,), jnp.bfloat16) 

    train_data_loader = get_data_loader(cfg.data.data_dir + "/train.jsonl", cfg.data.batch_size)
    train_iter = (shard_batch(batch) for batch in train_data_loader)

    for step, batch in enumerate(train_iter):
        batch = shard_batch(batch)
        metrics= train_step(model, optimizer, batch, y_init, z_init, cfg.recursion.N_supervision, cfg.recursion.n, cfg.recursion.T,)

        print(f"step {step}: ", end="")
        for metric_name, metric in metrics[-1].items():
            print(f"{metric_name}: {metric}", end=", ")
        print()

        # TODO async logging
        log_metrics = {}
        for supervision_step, step_metrics in enumerate(metrics):
            for metric_name, metric in step_metrics.items():
                log_metrics[f"train/ss_{supervision_step}_{metric_name}"] = metric.item()
        
        wandb.log(log_metrics)


if __name__ == "__main__":
    run(main)