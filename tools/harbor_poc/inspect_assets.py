# -*- coding: utf-8 -*-
"""Inventory existing assets in a separate experiment; never mutate source weights."""
import hashlib, json, os, platform, subprocess, sys
from pathlib import Path
ROOT=Path(r'D:\RSdataset\_experiments\harbor_poc_20261007_124509')
WORK=Path(__file__).resolve().parents[2]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'runtime_settings')
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['YOLO_OFFLINE']='true'
sys.path.insert(0,str(WORK))
from tools.clean_pipeline.runtime import setup_runtime, assert_local_imports
setup_runtime()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(name,d): (ROOT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str),encoding='utf8')
if __name__=='__main__':
    manifest={'run_id':ROOT.name,'experiment_root':str(ROOT),'worktree':str(WORK),'branch':'harbor_expert_poc',
      'original_workspace':r'C:\Users\DOCTOR\Documents\GitHub\YOLOv8_distill_pipeline_clean',
      'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
      'python':sys.executable,'audit_root':r'D:\RSdataset\_audit\harbor\_audit\_20261006_210643',
      'dataset_roots':{'rgb':r'D:\RSdataset\DIOR','sar':r'D:\RSdataset\Sar_det100k'},
      'path_correction':'SAR source is Sar_det100k; earlier audit wording attributing Sar/_det100k to request is not reused.',
      'derived_root':r'D:\RSdataset\_derived\harbor_poc_20261007_124509',
      'base_weights':r'C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\best.pt',
      'text_encoder':r'C:\Users\DOCTOR\Documents\GitHub\YOLOv8\mobileclip_blt.ts','status':'inspecting'}
    save('experiment_manifest.json',manifest)
    (ROOT/'HANDOFF.md').write_text('# Harbor expert POC\n\nStatus: inspecting assets; no training started.\n\n'+ '\n'.join(f'- {k}: `{v}`' for k,v in manifest.items())+'\n\nNext: inspect fused harbor channel, prepare fixed derived subsets, verify E0 then short training and formal training. Never use test for selection.\n',encoding='utf8')
    import torch, torchvision, ultralytics, numpy, PIL
    env={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'torchvision':torchvision.__version__,
      'ultralytics':ultralytics.__version__,'numpy':numpy.__version__,'pillow':PIL.__version__,'cuda':torch.version.cuda,
      'gpu':torch.cuda.get_device_name(0),'gpu_bytes':torch.cuda.get_device_properties(0).total_memory,'platform':platform.platform(),
      'local_imports':assert_local_imports()}
    save('environment.json',env)
    (ROOT/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf8')
    manifest['base_sha256']=sha(manifest['base_weights']);manifest['text_encoder_sha256']=sha(manifest['text_encoder'])
    ck=torch.load(manifest['base_weights'],map_location='cpu',weights_only=False)
    m=ck.get('ema') or ck['model'];m=m.float();h=m.model[-1]
    info={'checkpoint_keys':list(ck),'class':str(type(m)),'head':str(type(h)),'names':m.names,'yaml':m.yaml,
      'args':getattr(m,'args',None),'train_args':ck.get('train_args'),'epoch':ck.get('epoch'),
      'head_flags':{k:getattr(h,k,None) for k in ['nc','no','nl','reg_max','embed','is_fused','end2end','stride']},
      'parameter_count':sum(p.numel() for p in m.parameters()),'head_repr':str(h),
      'blocks':[{'index':i,'type':str(type(b)),'source':b.f,'parameters':sum(p.numel() for p in b.parameters())} for i,b in enumerate(m.model)],
      'tensor_shapes':{k:list(v.shape) for k,v in m.state_dict().items()},
      'classifier_shapes':[list(b[-1].weight.shape) for b in h.cv3]}
    save('base_architecture.json',info);save('experiment_manifest.json',manifest)
    print(json.dumps({k:info[k] for k in ['class','head','names','head_flags','parameter_count','classifier_shapes','blocks']},default=str,indent=2))
    print('sha256',manifest['base_sha256']); print('head',str(h.cv4));print('yaml',m.yaml)
