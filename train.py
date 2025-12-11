import time
import grain
from math import ceil
import os
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from flax import nnx, struct
import optax
from sws import run
import wandb
import orbax.checkpoint as ocp
from datetime import datetime
from dataset import get_data_loader
from modelling.model import Model
from modelling.optimizers import adamw_atan2, sign_sgdw
from utils import MetricLogger
from evaluate import evaluate

 # TODO: fix crop function to also crop top/left based on bottom/right border. tmp solution is translate=False on val set.
 
def main(cfg):

    if cfg.parallel.n_devices > 1:
        jax.distributed.initialize()

    key = jax.random.key(cfg.seed)
    
    model = Model(**cfg.model.to_dict(), rngs=nnx.Rngs(key))

    opt_fn = adamw_atan2 if cfg.optim.use_atan2 else optax.adamw
    tx = optax.partition(
        {
            "embed": sign_sgdw(
                optax.warmup_constant_schedule(**cfg.embed_schedule.to_dict()), cfg.optim.weight_decay
            ),
            "other": opt_fn(
                optax.warmup_constant_schedule(**cfg.other_schedule.to_dict()),
                b1=cfg.optim.b1,
                b2=cfg.optim.b2,
                weight_decay=cfg.optim.weight_decay,
            )
        },
        lambda state: jax.tree.map_with_path(lambda path, _: "embed" if path[0].key == "puzzle_emb" else "other", state)
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    
    shard_data = lambda data: data
    if cfg.parallel.n_devices > 1:
        mesh = jax.make_mesh((cfg.parallel.n_devices,), ("data",))
        jax.set_mesh(mesh)

        if jax.process_index() == 0:
            print(f"{mesh=}")

        repl_sharding = NamedSharding(mesh, P())
        data_sharding = NamedSharding(mesh, P("data",))

        _, model_state = nnx.split(model)
        sharded_model_state = jax.lax.with_sharding_constraint(model_state, repl_sharding)
        # fsdp on puzzle embs
        sharded_model_state.puzzle_emb = jax.lax.with_sharding_constraint(sharded_model_state.puzzle_emb, data_sharding)
        nnx.update(model, sharded_model_state)
        
        _, optimizer_state = nnx.split(optimizer)
        sharded_optimizer_state = jax.lax.with_sharding_constraint(optimizer_state, repl_sharding)
        nnx.update(optimizer, sharded_optimizer_state)

        shard_data = lambda data: jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), data)


    @struct.dataclass
    class Carry:
        z: jax.Array
        y: jax.Array
        x_input: jax.Array
        aug_puzzle_idx: jax.Array
        y_true: jax.Array
        step: jax.Array
        halted: jax.Array
    
    
    def init_carry(batch, z_init, y_init):
        """initialize the carry with the initial data"""
        batch_size = batch['x'].shape[0]
        hidden_dim = z_init.shape[-1]
        if cfg.model.vision_mode:
            seq_len = cfg.model.input_size // cfg.model.patch_size * cfg.model.input_size // cfg.model.patch_size
        else:
            seq_len = 900
        seq_len = seq_len + cfg.model.puzzle_emb_len
        z_init = jnp.broadcast_to(z_init, (batch_size, seq_len, hidden_dim))
        y_init = jnp.broadcast_to(y_init, (batch_size, seq_len, hidden_dim))
        if cfg.parallel.n_devices > 1:
            z_init = jax.device_put(z_init, data_sharding)
            y_init = jax.device_put(y_init, data_sharding)
        return Carry(
            z=z_init,                                         # (batch_size, seq_len, hidden_dim)
            y=y_init,                                         # (batch_size, seq_len, hidden_dim)
            x_input=batch['x'],                               # (batch_size, 900)
            aug_puzzle_idx=batch['aug_puzzle_idx'],           # (batch_size,)
            y_true=batch['y'],                                # (batch_size, 900)
            step=jnp.zeros((batch_size, ), dtype=jnp.int32),  # (batch_size,)
            halted=jnp.zeros((batch_size, ), dtype=jnp.bool_) # (batch_size,)
        )
    

    def pre_update_carry(carry, batch, z_init, y_init):
        """update the carry with new data from batch (if halted)"""
        return Carry(
            z=jnp.where(carry.halted[..., jnp.newaxis, jnp.newaxis], z_init[jnp.newaxis, jnp.newaxis, ...], carry.z),
            y=jnp.where(carry.halted[..., jnp.newaxis, jnp.newaxis], y_init[jnp.newaxis, jnp.newaxis, ...], carry.y),
            x_input=jnp.where(carry.halted[..., jnp.newaxis], batch['x'], carry.x_input),
            aug_puzzle_idx=jnp.where(carry.halted[..., jnp.newaxis], batch['aug_puzzle_idx'], carry.aug_puzzle_idx),
            y_true=jnp.where(carry.halted[..., jnp.newaxis], batch['y'], carry.y_true),
            step=jnp.where(carry.halted, 0, carry.step),
            halted=jnp.where(carry.halted, False, carry.halted),
        )
    

    def post_update_carry(carry, q_logits, z, y, N_supervision, halt_explore_prob, key):
        """update the halt flag if step >= N_supervision or q_logits > 0"""
        step = carry.step + 1
        halted = step >= N_supervision
        if cfg.recursion.act:
            halted = halted | (q_logits.reshape(-1) > 0)
        if halt_explore_prob > 0:
            key, subkey, subkey2 = jax.random.split(key, 3)
            min_halt_steps = (
                (jax.random.uniform(subkey, halted.shape) < halt_explore_prob)
                * jax.random.randint(subkey2, step.shape, minval=2, maxval=N_supervision + 1)
            )
            halted = halted & (step >= min_halt_steps)
        return Carry(
            z=z,
            y=y,
            x_input=carry.x_input,
            aug_puzzle_idx=carry.aug_puzzle_idx,
            y_true=carry.y_true,
            step=step,
            halted=halted,
        ), key

    
    def stablemax_cross_entropy_with_integer_labels(logits, labels, eps=1e-30):
        pos = jnp.log(jnp.maximum(logits, 0.) + 1.0 + eps)
        neg = -jnp.log(jnp.maximum(1.0 - logits, eps))

        s_logits = jnp.where(logits >= 0, pos, neg)
        return optax.softmax_cross_entropy_with_integer_labels(s_logits, labels)


    def latent_recursion(model, x, y, z, n):
        def latent_recursion_body(z, _):
            return model(x, y, z), None
        with jax.named_scope("latent_recursion_scan"):
            z, _ = jax.lax.scan(latent_recursion_body, z, None, length=n, unroll=True)
        with jax.named_scope("latent_recursion_last"):
            y = model(y, z)
        return y, z


    def deep_recursion(model, x, y, z, n, T):
        def deep_recursion_body(carry, _):
            y, z = carry
            y, z = latent_recursion(model, x, y, z, n)
            return (y, z), None
        with jax.named_scope("deep_recursion_scan"):
            (y, z), _ = jax.lax.scan(deep_recursion_body, (y, z), None, length=T, unroll=True)
        return y, z


    def loss_fn(model, x_input, aug_puzzle_idx, y, z, y_true, n):
        # forward pass
        x = model.input_embedding(x_input, aug_puzzle_idx)
        y, z = latent_recursion(model, x, y, z, n)
        y_logits, q_logits = model.output_head(y), model.q_head(z)
        y_preds = jnp.argmax(y_logits, axis=-1)
        # compute losses
        y_loss = stablemax_cross_entropy_with_integer_labels(
            y_logits.reshape(-1, y_logits.shape[-1]).astype(jnp.float32),
            y_true.reshape(-1)
        ).mean(where=y_true.reshape(-1) < 11)
        if cfg.recursion.act:
            # TODO: only compute for halted ?
            q_loss = optax.sigmoid_binary_cross_entropy(
                q_logits.reshape(-1),
                (y_preds == y_true).all(axis=-1, where=y_true < 11)
            ).mean()
        else:
            q_loss = 0
        loss = y_loss + 0.5 * q_loss # 0.5* why?
        return loss, (y, z, y_loss, q_loss, y_preds, q_logits)
    
    
    @nnx.jit(static_argnames=["N_supervision", "n", "T", "halt_explore_prob"])
    def train_step(model, optimizer, carry, batch, y_init, z_init, N_supervision, n, T, halt_explore_prob, ema_model, key):
        # update carry (if halted, update with init and batch)
        carry = pre_update_carry(carry, batch, z_init, y_init)
        # 1 N_supervision step
        x = model.input_embedding(carry.x_input, carry.aug_puzzle_idx)
        # deep recursion loop (no grads)
        z, y = carry.z, carry.y
        y, z = deep_recursion(model, x, y, z, n, T-1)
        # 1-step approx BPTT
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, (y, z, y_loss, q_loss, y_preds, q_logits)), grads = grad_fn(
            model, carry.x_input, carry.aug_puzzle_idx, y, z, carry.y_true, n
        )
        optimizer.update(model, grads)

        # update halt flag
        carry, key = post_update_carry(carry, q_logits, z, y, N_supervision, halt_explore_prob, key)

        # compute metrics (11 = padding)
        cell_correct = y_preds == carry.y_true # (batch_size, 900)
        puzzle_correct = cell_correct.all(axis=-1, where=carry.y_true < 11)
        cell_acc = cell_correct.mean(where=(carry.y_true < 11) & (carry.halted[..., jnp.newaxis]))
        puzzle_acc = puzzle_correct.mean(where=carry.halted)
        metrics = {
            "loss": loss,
            "y_loss": y_loss,
            "q_loss": q_loss,
            "cell_acc": cell_acc,
            "puzzle_acc": puzzle_acc,
            "y_max": jnp.max(jnp.abs(y)),
            "z_max": jnp.max(jnp.abs(z)),
            "y_norm": jnp.sqrt(jnp.mean(y**2)),
            "z_norm": jnp.sqrt(jnp.mean(z**2)),
            "n_supervision_steps": carry.step.mean(where=carry.halted),
        }
        if cfg.recursion.act:
            q_acc = ((q_logits.reshape(-1) > 0) == puzzle_correct).mean(where=carry.halted)
            metrics["q_acc"] = q_acc

        if ema_model is not None:
            new_ema_state = jax.tree.map(
                lambda new, old: (
                    None if new is None else (1.0 - cfg.ema_weight) * new + cfg.ema_weight * old
                ),
                nnx.state(model),
                nnx.state(ema_model),
                is_leaf=lambda x: x is None,
            )
            nnx.update(ema_model, new_ema_state)

        return carry, metrics, key
    
    # init logging 
    if jax.process_index() == 0:
        if cfg.wandb:
            wandb.init(project="arc", entity="jackpenn", config=cfg.to_dict())
            train_logger = MetricLogger(cfg.data.train_batch_size, prefix="train", buffer=True, wandb=wandb)
            val_logger = MetricLogger(cfg.data.eval_batch_size, prefix="val", buffer=False, wandb=wandb)
        else:
            train_logger = MetricLogger(cfg.data.train_batch_size, prefix="train", buffer=True, wandb=None)
            val_logger = MetricLogger(cfg.data.eval_batch_size, prefix="val", buffer=False, wandb=None)

    # count params
    _, params = nnx.split(model, nnx.Param)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    puzzle_emb_params = model.puzzle_emb.embedding.size
    num_params -= puzzle_emb_params
    print(f"{num_params=}, {puzzle_emb_params=}")
    del params
    
    # init latents
    key, y_key, z_key = jax.random.split(key, 3)
    initializer = jax.nn.initializers.truncated_normal(stddev=1.0)
    y_init = initializer(y_key, (cfg.model.hidden_dim,), jnp.bfloat16)
    z_init = initializer(z_key, (cfg.model.hidden_dim,), jnp.bfloat16) 
    

    # init data loader
    train_data_loader = get_data_loader(
        cfg.data.data_dir + "/train.jsonl",
        cfg.data.train_batch_size,
        translate=cfg.data.translate,
        max_grid_size=cfg.data.max_grid_size,
        repeat=True,
        drop_remainder=True,
        shard_by_jax_process=True
    )
    val_data_loader_factory = lambda: get_data_loader(
        cfg.data.data_dir + "/test.jsonl",
        cfg.data.eval_batch_size,
        translate=False,
        max_grid_size=cfg.data.max_grid_size,
        repeat=False,
        drop_remainder=True,
        shard_by_jax_process=True
    ) # tmp drop remainder because of sharding ( so eval on n lik 99% subset)

    # init checkpoint manager
    # ckpt_dir = ocp.test_utils.erase_and_create_empty(f'{os.getcwd()}/checkpoints/')
    ckpt_dir = f'{os.getcwd()}/checkpoints/'
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=1, cleanup_tmp_directories=True)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, options=ckpt_options)

    # init profiler
    profiler_options = jax.profiler.ProfileOptions()
    profiler_options.host_tracer_level = 3
    profile_dir = f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


    train_iter = iter(train_data_loader)
    if cfg.restore_from_checkpoint:
        abstract_model_state = nnx.state(nnx.eval_shape(lambda: model))
        abstract_optim_state = nnx.state(nnx.eval_shape(lambda: optimizer))
        restore_args = ocp.args.Composite(
            z_init=ocp.args.ArrayRestore(z_init),
            y_init=ocp.args.ArrayRestore(y_init),
            model_state=ocp.args.StandardRestore(abstract_model_state),
            optim_state=ocp.args.StandardRestore(abstract_optim_state),
            # data_loader=grain.checkpoint.CheckpointRestore(train_iter),
        )
        if cfg.use_ema:
            restore_args.ema_model = ocp.args.StandardRestore(abstract_model_state)
        restored = ckpt_mngr.restore(ckpt_mngr.latest_step(), args=restore_args)
        step = ckpt_mngr.latest_step() + 1
        nnx.update(model, restored.model_state)
        nnx.update(optimizer, restored.optim_state)
        if cfg.use_ema:
            ema_model = nnx.clone(model)
            nnx.update(ema_model, restored.ema_model)

        z_init = restored.z_init
        y_init = restored.y_init
        # train_iter = restored.data_loader
    else:
        step = 0
        if cfg.use_ema:
            ema_model = nnx.clone(model)

    carry = init_carry(shard_data(next(train_iter)), z_init, y_init)

    t0 = time.perf_counter()
    while step < cfg.max_steps:
        batch = shard_data(next(train_iter))

        if jax.process_index() == 0 and step == 10: 
            jax.profiler.start_trace(profile_dir, profiler_options=profiler_options)
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            carry, metrics, key = train_step(
                model, optimizer, carry, batch, y_init, z_init,
                cfg.recursion.N_supervision, cfg.recursion.n, cfg.recursion.T,
                cfg.recursion.halt_explore_prob,
                ema_model if cfg.use_ema else None,
                key
            )
        if jax.process_index() == 0 and step == 15:
            jax.profiler.stop_trace()
            if cfg.wandb:
                wandb.log_artifact(f"{os.getcwd()}/{profile_dir}/", name=f"run_{wandb.run.id}_profile", type="profile")

        step_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        
        if jax.process_index() == 0:
            train_logger.log({**metrics, "step_time": step_time, "step": step})
        
        if step > 0 and step % cfg.eval.eval_every == 0:
            val_metrics = evaluate(
                ema_model if cfg.use_ema else model,
                val_data_loader_factory, y_init, z_init,
                cfg.recursion.N_supervision, cfg.recursion.n, cfg.recursion.T,
                cfg.eval.pass_ks, shard_data, cfg.data.eval_batch_size
            )
            if jax.process_index() == 0:
                val_logger.log({**val_metrics, "step_time": step_time, "step": step})
        
        if step > 0 and step % cfg.log_every == 0:
            args = ocp.args.Composite(
                z_init=ocp.args.ArraySave(z_init),
                y_init=ocp.args.ArraySave(y_init),
                model_state=ocp.args.StandardSave(nnx.state(model)),
                optim_state=ocp.args.StandardSave(nnx.state(optimizer)),
                # data_loader=grain.checkpoint.CheckpointSave(train_iter),
            )
            if cfg.use_ema:
                args.ema_model = ocp.args.StandardSave(nnx.state(ema_model))
            ckpt_mngr.save(step, args=args)
            if jax.process_index() == 0 and cfg.wandb:
                ckpt_mngr.wait_until_finished()
                wandb.log_model(f"{ckpt_dir}/{step}", name=f"{wandb.run.id}_model", aliases=[f"step_{step}"])
        
        step += 1


if __name__ == "__main__":
    run(main)
