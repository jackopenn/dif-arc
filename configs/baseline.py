import os
import json

from sws import Config

def get_config():
    
    def get_puzzle_vocab_size(data_dir):
        return json.load(open(os.path.join(data_dir, "metadata.json"), 'r'))['train']['num_aug_puzzles']
    
    cfg = Config()
    cfg.seed = 69420
    
    cfg.model.vocab_size = 10 + 1 # +1 for padding
    cfg.model.hidden_dim = 512
    cfg.model.intermediate_dim = lambda: 4 * cfg.model.hidden_dim
    cfg.model.num_layers = 2
    cfg.model.num_attention_heads = 8
    cfg.model.num_key_value_heads = 8
    cfg.model.head_dim = lambda: cfg.model.hidden_dim // cfg.model.num_attention_heads
    cfg.model.act_fn = "swish"
    cfg.model.tie_embeddings = False
    cfg.model.use_bias = False
    cfg.model.rope_theta = 10000 # none = learned
    cfg.model.puzzle_vocab_size = lambda: get_puzzle_vocab_size(cfg.data.data_dir)
    cfg.model.puzzle_emb_len = 16

    cfg.recursion.N_supervision = 16
    cfg.recursion.n = 6
    cfg.recursion.T = 3
    cfg.recursion.act = True
    cfg.recursion.halt_explore_prob = 0.1
    
    cfg.optim.weight_decay = 0.1
    cfg.optim.b1 = 0.9
    cfg.optim.b2 = 0.95

    init_value = 0
    warmup_steps = 2000

    cfg.embed_schedule.init_value = init_value
    cfg.embed_schedule.peak_value = 1e-2
    cfg.embed_schedule.warmup_steps = warmup_steps
    
    cfg.other_schedule.init_value = init_value
    cfg.other_schedule.peak_value = 1e-4
    cfg.other_schedule.warmup_steps = warmup_steps

    cfg.max_steps = 100_000

    cfg.data.data_dir = "data/arc-aug-100"
    cfg.data.batch_size = 512

    cfg.parallel.n_devices = 8
    
    cfg.wandb = True
    
    cfg.eval.pass_ks = [1, 2, 5, 10, 100]
    cfg.eval.eval_every = 10_000

    return cfg