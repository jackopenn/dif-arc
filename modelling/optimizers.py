from typing import Optional, Any, Union, Callable
import jax
from jax import numpy as jnp
import optax
from optax._src import base, numerics, utils, combine, transform
from optax._src.transform import ScaleByAdamState

### copied from optax


def scale_by_adam_atan2(
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
        nu = optax.tree.zeros_like(params)  # Second moment
        return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)


    def update_fn(updates, state, params=None):
        del params
        mu = optax.tree.update_moment(updates, state.mu, b1, 1)
        nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_increment(state.count)
        if nesterov:
            mu_hat = jax.tree.map(
                lambda m, g: b1 * m + (1 - b1) * g,
                optax.tree.bias_correction(mu, b1,
                                            numerics.safe_increment(count_inc)),
                optax.tree.bias_correction(updates, b1, count_inc),
            )
        else:
            mu_hat = optax.tree.bias_correction(mu, b1, count_inc)
        # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
        # unclear why. Other Nadam implementations also omit the extra b2 factor.
        nu_hat = optax.tree.bias_correction(nu, b2, count_inc)
        updates = jax.tree.map(
           # arctan2 change from https://arxiv.org/pdf/2407.05872
            lambda m, v: None if m is None else (jnp.arctan2(m, jnp.sqrt(v))),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        mu = optax.tree.cast(mu, mu_dtype)
        return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def adamw_atan2(
    learning_rate: base.ScalarOrSchedule,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    mu_dtype: Optional[Any] = None,
    weight_decay: base.ScalarOrSchedule = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformationExtraArgs:
  
    return combine.chain(
        scale_by_adam_atan2(
            b1=b1,
            b2=b2,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        ),
        transform.add_decayed_weights(weight_decay, mask), # TODO: remove decay
        transform.scale_by_learning_rate(learning_rate),
    )
