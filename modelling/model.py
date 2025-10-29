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
        puzzle_vocab_size,
        rngs
    ):
        self.puzzle_emb = nnx.Embed(puzzle_vocab_size, hidden_dim, dtype=jnp.bfloat16, rngs=rngs)
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
        self.q_head = nnx.Linear(hidden_dim, 1, rngs=rngs)
    
    def input_embedding(self, x, aug_puzzle_idx):
        return jnp.concatenate([self.puzzle_emb(aug_puzzle_idx), self.embed(x)], axis=1) # * jnp.sqrt(x.shape[-1])

    def output_head(self, x):
        return self.unembed(x[:, 1:, :])

    def Q_head(self, x):
        return self.q_head(x[:, 0, :])

    def __call__(self, *x):
        x = reduce(jnp.add, x)
        for layer in self.layers:
            x = layer(x)
        return x
