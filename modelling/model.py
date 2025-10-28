from functools import reduce, partial
import jax
import jax.numpy as jnp
from flax import nnx

from modelling.layers import TransformerBlock

class Model(nnx.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        intermediate_dim,
        num_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        act_fn,
        tie_embeddings,
        rope_theta,
        rngs
    ):
    
        self.embed = nnx.Embed(vocab_size, hidden_dim, dtype=jnp.bfloat16, rngs=rngs)
        self.unembed = self.embed.attend if tie_embeddings else nnx.Linear(hidden_dim, vocab_size, dtype=jnp.bfloat16, rngs=rngs)
        self.layers = nnx.List([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_dim=intermediate_dim,
                act_fn=act_fn,
                rope_theta=rope_theta,
                rngs=rngs
            )
            for _ in range(num_layers)
        ])
        self.Q_head = nnx.Linear(hidden_dim, 1, rngs=rngs)
    
    def input_embedding(self, x):
        return self.embed(x)

    def output_head(self, x):
        return self.unembed(x)

    def __call__(self, *x):
        x = reduce(jnp.add, x)
        for layer in self.layers:
            x = layer(x)
        return x
