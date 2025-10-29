import jax
import jax.numpy as jnp
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
    cfg.model.hidden_dim = 16
    cfg.model.intermediate_dim = 64
    cfg.model.num_layers = 1
    cfg.model.num_attention_heads = 1
    cfg.model.num_key_value_heads = 1
    cfg.model.head_dim = lambda: cfg.model.hidden_dim // cfg.model.num_attention_heads
    cfg.model.act_fn = "swish"
    cfg.model.tie_embeddings = True
    cfg.model.rope_theta = 10000
    
    cfg.model.puzzle_vocab_size = lambda: get_puzzle_vocab_size(cfg.data.data_dir)
    
    cfg.recursion.N_supervision = 16
    cfg.recursion.n = 4
    cfg.recursion.T = 2
    
    cfg.optim.learning_rate = 1e-4
    cfg.optim.warmup_steps = 100
    cfg.optim.max_steps = 20000
    cfg.optim.weight_decay = 0.1

    cfg.data.data_dir = "data/arc-aug-10"
    cfg.data.batch_size = 16
    return cfg

def main(cfg):
    key = jax.random.key(cfg.seed)
    model = Model(**cfg.model.to_dict(), rngs=nnx.Rngs(key))
    schedule_fn = optax.warmup_constant_schedule(
        init_value=0,
        peak_value=cfg.optim.learning_rate,
        warmup_steps=cfg.optim.warmup_steps,
    )
    tx = optax.adamw(schedule_fn, weight_decay=cfg.optim.weight_decay)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    train_data_loader = get_data_loader(cfg.data.data_dir + "/train.jsonl", cfg.data.batch_size)
    
    
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
            y_true.reshape(-1),
            where=y_true.reshape(-1) < 10
        ).mean()
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
            # print(s)
            # print(y)
            # print(z)
            # print()
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

    for step, batch in enumerate(train_data_loader):
        # bbatch = {
        #     "aug_puzzle_idx": jnp.array([[1283]]),
        #     "x": jnp.array([[8, 8, 8, 8, 6, 8, 6, 6, 6]]),
        #     "y": jnp.array([[5, 5, 1, 5, 1, 5, 1, 5, 5]])
        # }
        metrics= train_step(model, optimizer, batch, y_init, z_init, cfg.recursion.N_supervision, cfg.recursion.n, cfg.recursion.T,)

        print(f"step {step}: ", end="")
        for metric_name, metric in metrics[-1].items():
            print(f"{metric_name}: {metric}", end=", ")
        print()

        log_metrics = {}
        for supervision_step, step_metrics in enumerate(metrics):
            for metric_name, metric in step_metrics.items():
                log_metrics[f"train/ss_{supervision_step}_{metric_name}"] = metric.item()
        
        wandb.log(log_metrics)


if __name__ == "__main__":
    run(main)