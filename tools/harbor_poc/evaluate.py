# -*- coding: utf-8 -*-
"""Re-evaluate a named checkpoint on a fixed manifest, with explicit fresh output."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.harbor_poc.core import *
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--base',action='store_true');p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    setup(a.root);a.out.mkdir(parents=True,exist_ok=False);man=read(a.root/'experiment_manifest.json');cfg=read(a.root/'config_A_seed42.json')
    if a.base:
        assert sha(a.weights)==man['base_sha256'];model=load_base(man)
    else:model,_=load_expert(a.weights)
    rr=read(Path(man['derived_root'])/f'{a.split}_manifest.json');result,_=evaluate_to(model,rr,cfg,a.out,a.base)
    print(json.dumps(result,indent=2))
