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
        vision_mode,
        patch_size,
        input_size,
        rngs
    ):
        self.puzzle_emb_len = puzzle_emb_len
        self.vision_mode = vision_mode
        self.input_size = input_size
        self.rope_theta = rope_theta
        if not self.vision_mode:
            if not self.rope_theta:
                self.pos_embed = nnx.Embed(
                    900 + puzzle_emb_len,
                    hidden_dim,
                    dtype=jnp.bfloat16,
                    embedding_init=nnx.initializers.truncated_normal(
                        stddev=jnp.reciprocal(jnp.sqrt(hidden_dim))
                    ),
                    rngs=rngs
                )
        else:
            self.patch_proj = nnx.Conv(
                hidden_dim,
                hidden_dim,
                (patch_size, patch_size),
                stride=patch_size,
                use_bias=use_bias,
                dtype=jnp.bfloat16,
                rngs=rngs
            )
            if not self.rope_theta:
                self.pos_embed_horizontal = nnx.Embed(
                    input_size,
                    hidden_dim // 2,
                    dtype=jnp.bfloat16,
                    embedding_init=nnx.initializers.truncated_normal(stddev=0),
                    rngs=rngs
                )
                self.pos_embed_vertical = nnx.Embed(
                    input_size,
                    hidden_dim // 2,
                    dtype=jnp.bfloat16,
                    embedding_init=nnx.initializers.truncated_normal(stddev=0),
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
        puzzle_emb = self.puzzle_emb(aug_puzzle_idx) # [batch_size, 1, hidden_dim]
        if self.puzzle_emb_len > 1:
            pad_len = self.puzzle_emb_len - 1
            puzzle_emb = jnp.pad(puzzle_emb, ((0, 0), (0, pad_len), (0, 0)), mode='constant', constant_values=0) # [batch_size, puzzle_emb_len, hidden_dim]
        if not self.vision_mode:
            seq_emb = self.embed(x) # [batch_size, seq_len, hidden_dim]
            embedding = jnp.concatenate([puzzle_emb, seq_emb], axis=1)
            if not self.rope_theta:
                embedding = 0.707106781 * (embedding + self.pos_embed(jnp.arange(embedding.shape[1])))
        else:
            image_emb = self.embed(x) # [batch_size, input_size, input_size, hidden_dim]
            if not self.rope_theta:
                horizontal_pos_emb = jnp.concatenate([jnp.zeros(self.input_size, image_emb.shape[-1] // 2), self.pos_embed_horizontal(jnp.arange(self.input_size))], axis=-1) # [input_size, hidden_dim]
                vertical_pos_emb = jnp.concatenate([self.pos_embed_vertical(jnp.arange(self.input_size)), jnp.zeros(self.input_size, image_emb.shape[-1] // 2)], axis=-1) # [input_size, hidden_dim]
                image_emb = image_emb + horizontal_pos_emb[None, :, None, :] + vertical_pos_emb[None, None, :, :] # [batch_size, input_size, input_size, hidden_dim]
            image_emb = self.patch_proj(image_emb) # [batch_size, input_size//patch_size, input_size//patch_size, hidden_dim]
            image_emb = image_emb.reshape(image_emb.shape[0], -1, image_emb.shape[-1])
            embedding = jnp.concatenate([puzzle_emb, image_emb], axis=1)
        
        # scale
        embedding = embedding * jnp.sqrt(embedding.shape[-1])
        return embedding


    def output_head(self, x):
        return self.unembed(x[:, self.puzzle_emb_len:, :])

    def q_head(self, x):
        return self.q_head_layer(x[:, 0, :])

    def __call__(self, *x):
        x = reduce(jnp.add, x)
        for layer in self.layers:
            x = layer(x)
        return x
