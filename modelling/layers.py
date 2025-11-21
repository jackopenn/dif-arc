from functools import partial
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from flax import nnx
from jax.experimental.pallas.ops.tpu import splash_attention

class GLU(nnx.Module):
    def __init__(self, hidden_dim, intermediate_dim, act_fn, use_bias, rngs):
        self.up_proj = nnx.Linear(hidden_dim, intermediate_dim, use_bias=use_bias, dtype=jnp.bfloat16, rngs=rngs)
        self.gate_proj = nnx.Linear(hidden_dim, intermediate_dim, use_bias=use_bias, dtype=jnp.bfloat16, rngs=rngs)
        self.down_proj = nnx.Linear(intermediate_dim, hidden_dim, use_bias=use_bias, dtype=jnp.bfloat16, rngs=rngs)
        self.act_fn = nnx.swish if act_fn == "swish" else nnx.gelu if act_fn == "gelu" else None

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nnx.Module):
    def __init__(self, hidden_dim, num_attention_heads, num_key_value_heads, head_dim, rope_theta, use_bias, rngs):
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rope_theta = rope_theta
        self.q_proj = nnx.LinearGeneral(hidden_dim, (num_attention_heads, head_dim), use_bias=use_bias, dtype=jnp.bfloat16, rngs=rngs)
        self.k_proj = nnx.LinearGeneral(hidden_dim, (num_key_value_heads, head_dim), use_bias=use_bias, dtype=jnp.bfloat16, rngs=rngs)
        self.v_proj = nnx.LinearGeneral(hidden_dim, (num_key_value_heads, head_dim), use_bias=use_bias, dtype=jnp.bfloat16, rngs=rngs)
        self.o_proj = nnx.LinearGeneral((num_attention_heads, head_dim), hidden_dim, axis=(-2, -1), use_bias=use_bias, dtype=jnp.bfloat16, rngs=rngs)

        

    def __call__(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.rope_theta:
            positions = jnp.arange(x.shape[1])[None, :]
            q = apply_rope(q, positions, base_frequency=self.rope_theta)
            k = apply_rope(k, positions, base_frequency=self.rope_theta)

        if jax.default_backend() == "tpu":
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
                    return jax.vmap(
                        splash_attention.make_splash_mha(
                            mask=splash_attention.FullMask((seq_len, seq_len)),
                            head_shards=1,
                            q_seq_shards=1,
                        ),
                        in_axes=(0, 0, 0)
                    )(q, k, v)

                att = splash_attention_fn(q, k, v)

            att = jnp.swapaxes(att, 1, 2)
        else:
            att = jax.nn.dot_product_attention(query=q, key=k, value=v, is_causal=False)
        return self.o_proj(att)
    

class TransformerBlock(nnx.Module):
    def __init__(self,hidden_dim, num_attention_heads, num_key_value_heads, head_dim, intermediate_dim, act_fn, rope_theta, use_bias, rngs):
        self.attention = Attention(hidden_dim, num_attention_heads, num_key_value_heads, head_dim, rope_theta, use_bias, rngs)
        self.norm_1 = nnx.RMSNorm(hidden_dim, rngs=rngs)
        self.mlp = GLU(hidden_dim, intermediate_dim, act_fn, use_bias, rngs)
        self.norm_2 = nnx.RMSNorm(hidden_dim, rngs=rngs)

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