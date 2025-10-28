import jax
import jax.numpy as jnp
from flax import nnx
import optax
from sws import Config

from dataset import get_data_loader
from modelling.model import Model

cfg = Config()
cfg.seed = 69420

cfg.model.vocab_size = 10
cfg.model.hidden_dim = 32
cfg.model.intermediate_dim = 128
cfg.model.num_layers = 2
cfg.model.num_attention_heads = 1
cfg.model.num_key_value_heads = 1
cfg.model.head_dim = lambda: cfg.model.hidden_dim // cfg.model.num_attention_heads
cfg.model.act_fn = "swish"
cfg.model.tie_embeddings = True
cfg.model.rope_theta = 10000

cfg.recursion.N_supervision = 4
cfg.recursion.n = 6
cfg.recursion.T = 2

cfg.optim.learning_rate = 1e-4
cfg.optim.weight_decay = 0.01

cfg.data.train_data_dir = "data/training_n_augs=100.jsonl"
cfg.data.batch_size = 16

cfg = cfg.finalize()

key = jax.random.key(cfg.seed)
model = Model(**cfg.model.to_dict(), rngs=nnx.Rngs(key))
optimizer = nnx.Optimizer(model, optax.adamw(**cfg.optim.to_dict()), wrt=nnx.Param)
initializer = jax.nn.initializers.truncated_normal()
train_data_loader = get_data_loader(cfg.data.train_data_dir, cfg.data.batch_size)


def loss_fn(model, x_input, y_true, y,z, n, T):
    def latent_recursion(x, y, z, n):
        for _ in range(n):
            z = model(x, y, z)
        y = model(y, z)
        return y, z

    def deep_recursion(x, y, z, n, T):
        for _ in range(T-1):
            y, z = latent_recursion(x, y, z, n)
        y, z = jax.lax.stop_gradient(y), jax.lax.stop_gradient(z)
        y, z = latent_recursion(x, y, z, n)
        return (y, z), model.output_head(y), model.Q_head(z)
    
    x = model.input_embedding(x_input)
    (y, z), y_hat, q_hat = deep_recursion(x, y, z, n, T)
    y_loss = optax.softmax_cross_entropy_with_integer_labels(y_hat, y_true).mean()
    # Q_loss = optax.sigmoid_binary_cross_entropy(q_hat, (y_hat == y_true)).mean()
    Q_loss = 0
    loss = y_loss + Q_loss
    return loss, (y_loss, Q_loss, y, z)


@nnx.jit(static_argnames=["N_supervision", "n", "T"])
def train_step(model, optimizer, batch, N_supervision, n, T, key):
    x_input = batch['x']
    y_true = batch['y']
    key, y_key, z_key = jax.random.split(key, 3)
    y = initializer(y_key, (32,), jnp.bfloat16)
    z = initializer(z_key, (32,), jnp.bfloat16)
    losses = []
    for _ in range(N_supervision):
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, (y_loss, Q_loss, y, z)), grads = grad_fn(model, x_input, y_true, y, z, n, T)
        optimizer.update(model, grads)
        losses.append((y_loss, Q_loss))
    return losses, key


for step, batch in enumerate(train_data_loader):
    break

for step in range(10000):
    losses, key = train_step(model, optimizer, batch, cfg.recursion.N_supervision, cfg.recursion.n, cfg.recursion.T, key)
    losses = [f"{idx}:{y.item()}" for idx, (y, q) in enumerate(losses)]
    print(step, losses)
