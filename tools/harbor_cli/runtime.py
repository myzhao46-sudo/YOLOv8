"""Configure the actual CPU pools, after pinning the local Ultralytics source."""
from pathlib import Path
import torch
from tools.harbor_poc.core import setup as legacy_setup,read,save

def setup(root):
    legacy_setup(root)
    cfg=read(Path(root)/'config.json')
    torch.set_num_threads(cfg['num_threads'])
    torch.set_num_interop_threads(cfg['num_interop_threads'])
    save(Path(root)/f'runtime_threads_{__import__("os").getpid()}.json',{
        'torch_num_threads':torch.get_num_threads(),'torch_num_interop_threads':torch.get_num_interop_threads(),
        'train_workers':cfg['workers'],'validation_workers':cfg['val_workers'],
        'persistent_workers_effective':bool(cfg['workers'] and cfg['persistent_workers']),
        'prefetch_factor_effective':cfg['prefetch_factor'] if cfg['workers'] else None,
        'cache_images':cfg['cache_images'],'CUDA_VISIBLE_DEVICES':__import__('os').environ.get('CUDA_VISIBLE_DEVICES')})
