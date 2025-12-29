from functools import partial
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from flax import nnx
from jax.experimental.pallas.ops.tpu import splash_attention

class GLU(nnx.Module):
    def __init__(self, hidden_dim, intermediate_dim, act_fn, use_bias, rngs):
        self.up_proj = nnx.Linear(hidden_dim, intermediate_dim, use_bias=use_bias, dtype=jnp.bfloat16, kernel_init=nnx.initializers.truncated_normal(stddev=1/(hidden_dim**0.5)), rngs=rngs)
        self.gate_proj = nnx.Linear(hidden_dim, intermediate_dim, use_bias=use_bias, dtype=jnp.bfloat16, kernel_init=nnx.initializers.truncated_normal(stddev=1/(hidden_dim**0.5)), rngs=rngs)
        self.down_proj = nnx.Linear(intermediate_dim, hidden_dim, use_bias=use_bias, dtype=jnp.bfloat16, kernel_init=nnx.initializers.truncated_normal(stddev=1/(hidden_dim**0.5)), rngs=rngs)
        self.act_fn = nnx.swish if act_fn == "swish" else nnx.gelu if act_fn == "gelu" else None

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nnx.Module):
    def __init__(self, hidden_dim, num_attention_heads, num_key_value_heads, head_dim, rope_theta, use_bias, puzzle_emb_len, rngs, vision_rope=False):
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rope_theta = rope_theta
        self.vision_rope = vision_rope
        self.puzzle_emb_len = puzzle_emb_len
        self.q_proj = nnx.LinearGeneral(hidden_dim, (num_attention_heads, head_dim), use_bias=use_bias, dtype=jnp.bfloat16, kernel_init=nnx.initializers.truncated_normal(stddev=1/(hidden_dim**0.5)), rngs=rngs)
        self.k_proj = nnx.LinearGeneral(hidden_dim, (num_key_value_heads, head_dim), use_bias=use_bias, dtype=jnp.bfloat16, kernel_init=nnx.initializers.truncated_normal(stddev=1/(hidden_dim**0.5)), rngs=rngs)
        self.v_proj = nnx.LinearGeneral(hidden_dim, (num_key_value_heads, head_dim), use_bias=use_bias, dtype=jnp.bfloat16, kernel_init=nnx.initializers.truncated_normal(stddev=1/(hidden_dim**0.5)), rngs=rngs)
        self.o_proj = nnx.LinearGeneral((num_attention_heads, head_dim), hidden_dim, axis=(-2, -1), use_bias=use_bias, dtype=jnp.bfloat16, kernel_init=nnx.initializers.truncated_normal(stddev=1/((num_attention_heads*head_dim)**0.5)), rngs=rngs)

    def __call__(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.rope_theta:
            qp, kp = q[:, :self.puzzle_emb_len], k[:, :self.puzzle_emb_len]
            qs, ks = q[:, self.puzzle_emb_len:], k[:, self.puzzle_emb_len:]
            seq_len = qs.shape[1]
            if self.vision_rope:
                hw = int(seq_len ** 0.5)
                qs = apply_rope_2d(qs, hw, hw, base_frequency=self.rope_theta)
                ks = apply_rope_2d(ks, hw, hw, base_frequency=self.rope_theta)
            else:
                positions = jnp.arange(seq_len)[None, :]
                qs = apply_rope(qs, positions, base_frequency=self.rope_theta)
                ks = apply_rope(ks, positions, base_frequency=self.rope_theta)
            q = jnp.concatenate([qp, qs], axis=1)
            k = jnp.concatenate([kp, ks], axis=1)

        # tmp disbale this - slow af
        if False and jax.default_backend() == "tpu":
            q = jax.lax.with_sharding_constraint(q, P("data", None, None, None))
            k = jax.lax.with_sharding_constraint(k, P("data", None, None, None))
            v = jax.lax.with_sharding_constraint(v, P("data", None, None, None))

            with jax.named_scope("repeat_kv_and_swap_axes"):
                k = jnp.repeat(k, self.num_attention_heads // self.num_key_value_heads, axis=2)
                v = jnp.repeat(v, self.num_attention_heads // self.num_key_value_heads, axis=2)
                
                q = jnp.swapaxes(q, 1, 2)
                k = jnp.swapaxes(k, 1, 2)
                v = jnp.swapaxes(v, 1, 2)

            with jax.named_scope("attention"):
                @jax.shard_map(
                    in_specs=(
                        P("data", None, None, None),
                        P("data", None, None, None),
                        P("data", None, None, None),
                    ),
                    out_specs=P("data", None, None, None),
                    check_vma=False
                )
                def splash_attention_fn(q, k, v):
                    seq_len = q.shape[-2]
                    mask = splash_attention.FullMask((seq_len, seq_len))
                    mask = splash_attention.MultiHeadMask(masks=(mask,) * self.num_attention_heads)
                    return jax.vmap(
                        splash_attention.make_splash_mha(
                            mask=mask,
                            head_shards=1,
                            q_seq_shards=1,
                        ),
                        in_axes=(0, 0, 0)
                    )(q, k, v)

                att = splash_attention_fn(q, k, v)

            att = jnp.swapaxes(att, 1, 2)
        else:
            att = jax.nn.dot_product_attention(
                query=q, key=k, value=v, is_causal=False,
                implementation="cudnn" if jax.default_backend() == "gpu" else "xla"
            )
        return self.o_proj(att)
    

class TransformerBlock(nnx.Module):
    def __init__(self,hidden_dim, num_attention_heads, num_key_value_heads, head_dim, intermediate_dim, act_fn, rope_theta, use_bias, rngs, puzzle_emb_len, vision_rope=False):
        self.attention = Attention(hidden_dim, num_attention_heads, num_key_value_heads, head_dim, rope_theta, use_bias, puzzle_emb_len, rngs, vision_rope)
        self.norm_1 = nnx.RMSNorm(hidden_dim, dtype=jnp.bfloat16, use_scale=False, epsilon=1e-5, rngs=rngs)
        self.mlp = GLU(hidden_dim, intermediate_dim, act_fn, use_bias, rngs)
        self.norm_2 = nnx.RMSNorm(hidden_dim, dtype=jnp.bfloat16, use_scale=False, epsilon=1e-5, rngs=rngs)

    def __call__(self, x):
        # x = x + self.attention(self.norm_1(x))
        # x = x + self.mlp(self.norm_2(x))
        x = self.norm_1(x + self.attention(x))
        x = self.norm_2(x + self.mlp(x))
        return x

def apply_rope(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: int,
    scale_factor: float = 1.0,
) -> jax.Array:
    head_dim = inputs.shape[-1]
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = base_frequency**fraction

    sinusoid_inp = (
        positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    )
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    if scale_factor < 1.0:
        raise ValueError(f'scale_factor must be >= 1.0, got {scale_factor}')
    sinusoid_inp /= scale_factor

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


def apply_rope_2d(
    inputs: jax.Array,
    height: int,
    width: int,
    *,
    base_frequency: float = 10000.0,
) -> jax.Array:
    """Apply 2D RoPE for images. Splits head_dim: first half for rows, second half for cols.
    
    Args:
        inputs: Shape (batch, seq_len, num_heads, head_dim) where seq_len = height * width
        height: Grid height
        width: Grid width
        base_frequency: RoPE base frequency (theta)
    
    Returns:
        Rotated inputs with same shape
    """
    head_dim = inputs.shape[-1]
    half_dim = head_dim // 2
    
    # Frequency bands for each half
    fraction = 2 * jnp.arange(0, half_dim // 2) / half_dim
    timescale = base_frequency ** fraction  # (half_dim // 2,)
    
    # Create 2D position grid
    row_pos = jnp.arange(height)
    col_pos = jnp.arange(width)
    
    # Compute sinusoids for rows and cols
    row_sinusoid = row_pos[:, None] / timescale[None, :]  # (height, half_dim//2)
    col_sinusoid = col_pos[:, None] / timescale[None, :]  # (width, half_dim//2)
    
    # Broadcast to full grid: (height, width, half_dim//2)
    row_sinusoid = jnp.broadcast_to(row_sinusoid[:, None, :], (height, width, half_dim // 2))
    col_sinusoid = jnp.broadcast_to(col_sinusoid[None, :, :], (height, width, half_dim // 2))
    
    # Flatten to (seq_len, half_dim//2)
    row_sinusoid = row_sinusoid.reshape(-1, half_dim // 2)
    col_sinusoid = col_sinusoid.reshape(-1, half_dim // 2)
    
    # Compute sin/cos: (seq_len, half_dim//2)
    row_sin, row_cos = jnp.sin(row_sinusoid), jnp.cos(row_sinusoid)
    col_sin, col_cos = jnp.sin(col_sinusoid), jnp.cos(col_sinusoid)
    
    # Reshape for broadcasting: (1, seq_len, 1, half_dim//2)
    row_sin = row_sin[None, :, None, :]
    row_cos = row_cos[None, :, None, :]
    col_sin = col_sin[None, :, None, :]
    col_cos = col_cos[None, :, None, :]
    
    # Split input into 4 quarters: [row_first, row_second, col_first, col_second]
    row_half, col_half = jnp.split(inputs, 2, axis=-1)
    row_first, row_second = jnp.split(row_half, 2, axis=-1)
    col_first, col_second = jnp.split(col_half, 2, axis=-1)
    
    # Apply rotation to each half
    row_first_out = row_first * row_cos - row_second * row_sin
    row_second_out = row_second * row_cos + row_first * row_sin
    col_first_out = col_first * col_cos - col_second * col_sin
    col_second_out = col_second * col_cos + col_first * col_sin
    
    out = jnp.concatenate([row_first_out, row_second_out, col_first_out, col_second_out], axis=-1)
    return out.astype(inputs.dtype)