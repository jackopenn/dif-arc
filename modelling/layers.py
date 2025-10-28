import jax
import jax.numpy as jnp
from flax import nnx

class GLU(nnx.Module):
    def __init__(self, hidden_dim, intermediate_dim, act_fn, rngs):
        self.up_proj = nnx.Linear(hidden_dim, intermediate_dim, dtype=jnp.bfloat16, rngs=rngs)
        self.gate_proj = nnx.Linear(hidden_dim, intermediate_dim, dtype=jnp.bfloat16, rngs=rngs)
        self.down_proj = nnx.Linear(intermediate_dim, hidden_dim, dtype=jnp.bfloat16, rngs=rngs)
        self.act_fn = nnx.swish if act_fn == "swish" else nnx.gelu if act_fn == "gelu" else None

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nnx.Module):
    def __init__(self, hidden_dim, num_attention_heads, num_key_value_heads, head_dim, rope_theta,rngs):
        self.q_proj = nnx.LinearGeneral(hidden_dim, (num_attention_heads, head_dim), dtype=jnp.bfloat16, rngs=rngs)
        self.k_proj = nnx.LinearGeneral(hidden_dim, (num_key_value_heads, head_dim), dtype=jnp.bfloat16, rngs=rngs)
        self.v_proj = nnx.LinearGeneral(hidden_dim, (num_key_value_heads, head_dim), dtype=jnp.bfloat16, rngs=rngs)
        self.o_proj = nnx.LinearGeneral((num_attention_heads, head_dim), hidden_dim, axis=(-2, -1), dtype=jnp.bfloat16, rngs=rngs)

    def __call__(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.rope_theta:
            positions = jnp.arange(x.shape[1])[None, :]
            q = apply_rope(q, positions, self.rope_theta)
            k = apply_rope(k, positions, self.rope_theta)
        att = jax.nn.dot_product_attention(query=q, key=k, value=v, is_causal=False)
        return self.o_proj(att)
    

class TransformerBlock(nnx.Module):
    def __init__(self,hidden_dim, num_attention_heads, num_key_value_heads, head_dim, intermediate_dim, act_fn, rope_theta, rngs):
        self.attention = Attention(hidden_dim, num_attention_heads, num_key_value_heads, head_dim, rope_theta, rngs)
        self.norm_1 = nnx.RMSNorm(head_dim, rngs=rngs)
        self.mlp = GLU(hidden_dim, intermediate_dim, act_fn, rngs)
        self.norm_2 = nnx.RMSNorm(head_dim, rngs=rngs)

    def __call__(self, x):
        x = x + self.attention(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
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