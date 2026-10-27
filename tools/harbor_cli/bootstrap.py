"""Adapt the proven B->E0 bootstrap to CLI paths/config without editing legacy files."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.harbor_poc import checked_bootstrap as original
from tools.harbor_poc.core import read,save,sha,WORK
from tools.harbor_cli.runtime import setup

EXTRA=['tools/train_harbor_expert.py','tools/harbor_cli/__init__.py','tools/harbor_cli/bootstrap.py','tools/harbor_cli/data.py',
    'tools/harbor_cli/inference.py','tools/harbor_cli/runtime.py','tools/harbor_cli/train_checked.py','tools/harbor_cli/launcher.py','tools/harbor_cli/predict.py']

def main(path):
    job=read(path);root=Path(job['root']);cfg=job['config']
    original.B=Path(job['base_weights']);original.DER=Path(job['data_root_dir']);original.OLD=Path(job['reference_experiment'])
    original.SOURCES=list(dict.fromkeys(original.SOURCES+EXTRA));original.setup=setup
    original.main(root)
    # Complete CLI config replaces only this NEW experiment's bootstrap defaults.
    save(root/'config.json',cfg)
    man=read(root/'experiment_manifest.json');man.update(run=job['run'],scope='CLI fresh B->E0->harbor expert; configured epoch budget, no test',
        cli_entry=str(WORK/'tools/train_harbor_expert.py'),data_rng_version=cfg['data_rng_version'],initialization_parity_input_size=640,
        comparison_note='Same inherited-channel supervised method; CLI data flips are stateless per seed/epoch/presentation, not identical to legacy global RNG augmentation')
    save(root/'experiment_manifest.json',man)
    protocol=read(root/'evaluation_protocol.json');protocol.update(imgsz=cfg['imgsz'],nms_backend='explicit local torchvision.ops.nms',nms_wall_clock_limit='disabled',
        inference_source_sha256=sha(WORK/'tools/harbor_cli/inference.py'),primary_selection=cfg['selection'],
        cached_regression_note='Reference caches are historical640 protocol, used for metric arithmetic checks; not a new model baseline at a different CLI image_size')
    save(root/'evaluation_protocol.json',protocol)
    save(root/'cli_effective_config.json',cfg)
    print('CLI initialization ready; batch/workers/CPU pools now come from the explicit CLI config.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);a=p.parse_args();main(a.job)
