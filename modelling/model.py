from functools import reduce

from jax import numpy as jnp
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
        use_bias,
        puzzle_emb_len,
        input_size,
        rngs
    ):
        self.puzzle_emb_len = puzzle_emb_len
        self.input_size = input_size
        self.rope_theta = rope_theta
        self.seq_len = puzzle_emb_len + input_size * input_size
        if self.rope_theta == "learned":
            self.pos_embed = nnx.Embed(
                self.seq_len,
                hidden_dim,
                dtype=jnp.bfloat16,
                embedding_init=nnx.initializers.truncated_normal(
                    stddev=jnp.reciprocal(jnp.sqrt(hidden_dim))
                ),
                rngs=rngs
            )
        self.puzzle_emb = nnx.Embed(
            puzzle_vocab_size,
            hidden_dim,
            dtype=jnp.bfloat16,
            embedding_init=nnx.initializers.truncated_normal(stddev=0),
            rngs=rngs
        )
        self.embed = nnx.Embed(
            vocab_size,
            hidden_dim,
            dtype=jnp.bfloat16,
            embedding_init=nnx.initializers.truncated_normal(
                stddev=jnp.reciprocal(jnp.sqrt(hidden_dim))
            ),
            rngs=rngs)
        self.unembed = (
            self.embed.attend
            if tie_embeddings
            else nnx.Linear(hidden_dim, vocab_size, use_bias=use_bias, dtype=jnp.bfloat16, rngs=rngs)
        )
        self.layers = nnx.List([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_dim=intermediate_dim,
                act_fn=act_fn,
                rope_theta=rope_theta,
                use_bias=use_bias,
                rngs=rngs
            )
            for _ in range(num_layers)
        ])
        self.q_head_layer = nnx.Linear(
            hidden_dim,
            1,
            use_bias=True,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.constant(-5),
            rngs=rngs
        )
    
    def input_embedding(self, x, aug_puzzle_idx):
        puzzle_emb = self.puzzle_emb(aug_puzzle_idx)[:, jnp.newaxis, :]
        B, _, D = puzzle_emb.shape
        if self.puzzle_emb_len > 1:
            pad_len = self.puzzle_emb_len - 1
            puzzle_emb = jnp.pad(puzzle_emb, ((0, 0), (0, pad_len), (0, 0)), mode='constant', constant_values=0)
        seq_emb = self.embed(x)
        embedding = jnp.concatenate([puzzle_emb, seq_emb], axis=1)
        if self.rope_theta == "learned":
            embedding = 0.707106781 * (embedding + self.pos_embed(jnp.arange(embedding.shape[1])))
        embedding = embedding * jnp.sqrt(D)
        return embedding


    def output_head(self, x):
        x = x[:, self.puzzle_emb_len:, :]
        return self.unembed(x)

    def q_head(self, x):
        return self.q_head_layer(x[:, 0, :])

    def __call__(self, *x):
        x = reduce(jnp.add, x)
        for layer in self.layers:
            x = layer(x)
        return x
