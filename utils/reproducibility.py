import numpy as np
import random
import torch


def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """DataLoader worker_init_fn: seed every RNG a worker may use.

    Combined with a seeded torch.Generator passed to the DataLoader, this
    makes data order and augmentation randomness reproducible even with
    num_workers > 0 (workers are re-seeded deterministically each epoch from
    the loader's generator via its per-epoch base_seed).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_rng_states():
    """Snapshot of all host RNG states relevant to training."""
    states = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states['torch_cuda'] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states):
    """Restore a snapshot produced by get_rng_states()."""
    random.setstate(states['python'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch_cpu'])
    if 'torch_cuda' in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['torch_cuda'])
