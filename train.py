import jax
from jax import numpy as jnp
from flax import nnx

from model import Model, ModelConfig

model_config = ModelConfig(
    vocab_size=10000,
    hidden_dim=256,
    num_layers=12,
    num_attention_heads=16,
    num_key_value_heads=16,
    head_dim=16,
    act_fn=jax.nn.gelu,
    intermediate_dim=1024,
)

model  = Model(model_config)