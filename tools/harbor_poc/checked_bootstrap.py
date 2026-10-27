"""New-root bootstrap, inherited E0 parity and cached metric regression only.

No formal training, no full-val model inference, no test manifest loading.
"""
import argparse,copy,json,os,platform,shutil,subprocess,sys,datetime,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.harbor_poc.core import *
from tools.harbor_poc.metrics_v2 import evaluate,boundary_checks,VERSION

OLD=Path(r'D:\RSdataset\_experiments\harbor_poc_20261007_124509')
DER=Path(r'D:\RSdataset\_derived\harbor_poc_20261007_124509')
B=Path(r'C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\best.pt')
B_SHA='50f306b0e2e50de49fe632377b8b4adcdd8c4f6eef50af3cff075a6a9b9415bd'
SOURCES=['tools/harbor_poc/core.py','tools/harbor_poc/train.py','tools/harbor_poc/metrics_v2.py','tools/harbor_poc/train_checked.py',
    'tools/harbor_poc/checked_launcher.py','tools/harbor_poc/checked_predict.py','tools/harbor_poc/checked_bootstrap.py','tools/harbor_poc/status.py',
    'tools/eval/eval_teacher_ship_external.py','tools/clean_pipeline/runtime.py','ultralytics/ultralytics/utils/loss.py',
    'ultralytics/ultralytics/utils/tal.py','ultralytics/ultralytics/nn/modules/head.py','ultralytics/ultralytics/nn/tasks.py',
    'tools/harbor_poc/checked_inference.py','ultralytics/ultralytics/utils/nms.py']
def now():return datetime.datetime.now().astimezone().isoformat()
def git(*args):return subprocess.check_output(['git','-C',str(WORK),*args],text=True,encoding='utf-8').strip()

def main(root):
    if (root/'experiment_manifest.json').exists():raise FileExistsError('Bootstrap refuses existing experiment manifest')
    root.mkdir(exist_ok=True,parents=True);os.environ['MPLCONFIGDIR']=str(root/'runtime_settings/matplotlib');setup(root)
    assert sha(B)==B_SHA,'Original B hash mismatch'
    data_hashes={f'{s}_manifest.json':sha(DER/f'{s}_manifest.json') for s in ['train','val']}
    recorded=read(DER/'manifest_hashes.json');assert all(recorded[k]==v for k,v in data_hashes.items())
    train=read(DER/'train_manifest.json');val=read(DER/'val_manifest.json')
    counts={}
    for split,rr in [('train',train),('val',val)]:
        for mod in ['rgb','sar']:
            subset=[r for r in rr if r['modality']==mod];counts[mod+'/'+split]={'images':len(subset),'boxes':sum(len(r['harbor']) for r in subset)}
            for r in subset:
                assert Path(r['image']).is_file() and Path(r['annotation_source']).is_file() and r['harbor']
                for o in r['harbor']:
                    x1,y1,x2,y2=o['xyxy'];assert all(np.isfinite(o['xyxy'])) and 0<=x1<x2<=r['width'] and 0<=y1<y2<=r['height']
    assert counts=={'rgb/train':{'images':318,'boxes':1117},'sar/train':{'images':1290,'boxes':3199},'rgb/val':{'images':328,'boxes':1205},'sar/val':{'images':161,'boxes':391}}
    man={'experiment_root':str(root),'worktree':str(WORK),'branch':git('branch','--show-current'),'base_commit':git('rev-parse','HEAD'),'python':sys.executable,
        'base_weights':str(B),'base_sha256':B_SHA,'derived_root':str(DER),'created':now(),'run':'A_seed43','test_used':False,
        'scope':'seed43 supervised harbor chain check: max10 formal epochs here; user continues same150-epoch run',
        'comparison_note':'Corrected metrics and explicit formal RNG reset; not a seed-only ablation against historical A_seed42'}
    save(root/'experiment_manifest.json',man);save(root/'data_identity.json',{'derived_root':str(DER),'manifest_sha256':data_hashes,'counts':counts,
        'audit':r'D:\RSdataset\audit\harbor\data_readiness20261023_145913','test_read_by_model':False})
    cfg=read(OLD/'config_A_seed42.json');cfg.update(seed=43,config_id='A_checked_seed43',initial_weights=str(root/'initialization/E0.pt'),
        loss_normalization='official batch-scaled loss.sum()/actual accumulation group sample count',
        second_seed_policy='This is seed43 checked run with corrected metrics and formal RNG reset; no automatic campaign or test',
        formal_rng_reset='after short check and dataset/model construction, before epoch1',metric_version=VERSION)
    save(root/'config.json',cfg)
    protocol=read(OLD/'evaluation_protocol.json')
    for key in ['ship_combination','seed_replication_policy','mapping','manifest_hashes','data_summary']:protocol.pop(key,None)
    protocol.update(version=VERSION,created=now(),locked_before_training=True,manifest_hashes=data_hashes,mapping={'B_harbor_source':1,'E_harbor':0},
        AP_implementation='metrics_v2.evaluate: stable score greedy; later GT wins exact IoU tie; 101 recall levels precision envelope mean; unachieved recall=0; area=all/maxDets300/noignore',
        AP_source_sha256=sha(WORK/'tools/harbor_poc/metrics_v2.py'),test_use='NOT USED by this workflow, including completion export',
        primary_selection='EMA RGB/SAR AP50-95 equal mean; strict greater improvement; earliest epoch wins ties',
        limits='harbor-positive validation subset only; not full/default COCO, no negative-scene quality claim, no IR/gating/fusion/distillation')
    save(root/'evaluation_protocol.json',protocol)
    tests=boundary_checks()
    # Actual local NMS max_det check with 301 disjoint high-confidence boxes.
    from ultralytics.utils.nms import non_max_suppression
    toy=torch.zeros(1,5,301);toy[0,0]=torch.arange(301)*10+5;toy[0,1]=5;toy[0,2:4]=2;toy[0,4]=torch.linspace(.99,.5,301)
    out=non_max_suppression(toy,conf_thres=.001,iou_thres=.7,max_det=300,nc=1);assert len(out[0])==300
    tests['actual_local_NMS_301_disjoint_returns_300']=True;save(root/'validation_checks/metric_boundaries.json',tests)
    independent=read(OLD/'baseline_checks/B_independent_20261007_225216/independent_metrics.json');comparison={}
    cachepaths={'B':OLD/'evaluations/B_val/predictions.json','E0':OLD/'evaluations/E0_val/predictions.json',
        'A_best_selected_under_original_protocol':OLD/'runs/A_seed42/best_val_predictions.json'}
    priorpred=None
    for label,path in cachepaths.items():
        pred=read(path);met=evaluate(val,pred);err={}
        for mod in ['rgb','sar']:
            ref=independent[label][mod];err[mod]=met[mod]['ap50_95']-ref['coco_style_ap50_95']
            assert abs(err[mod])<1e-12,(label,mod,err)
            assert np.allclose(met[mod]['ap_by_iou'],[x['coco_style_ap_101'] for x in ref['per_iou']],atol=1e-12,rtol=0)
            fixed=ref['per_iou'][0]['fixed_conf_025']
            assert all(met[mod][k]==fixed[k] for k in ['tp','fp','fn'])
        comparison[label]={'source':str(path),'sha256':sha(path),'metrics':met,'difference_from_independent_audit_AP95':err}
        if label=='B':priorpred=pred
        if label=='E0':assert pred==priorpred
    save(root/'validation_checks/cache_regression.json',{'passed':True,'B_E0_candidates_exact':True,'comparisons':comparison,'inference_performed':False})
    # Preserve actual code, including previously untracked files. No git changes/staging/commit.
    snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True)
    codehash={}
    for rel in SOURCES:
        path=WORK/rel;dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);codehash[rel]=sha(path)
    save(snapshot/'code_hashes.json',codehash)
    (snapshot/'git_diff.patch').write_text(git('diff','HEAD'),encoding='utf-8')
    (snapshot/'git_status.txt').write_text(git('status','--porcelain=v1','--untracked-files=all'),encoding='utf-8')
    untracked=git('ls-files','--others','--exclude-standard').splitlines();save(snapshot/'untracked_files.json',{'paths':untracked,'hashes':{p:sha(WORK/p) for p in untracked if (WORK/p).is_file()}})
    import ultralytics,torchvision,PIL
    save(root/'environment.json',{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,
        'Pillow':PIL.__version__,'ultralytics':ultralytics.__version__,'ultralytics_path':ultralytics.__file__,'torch_path':torch.__file__,
        'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
        'gpu_free_total_bytes_at_bootstrap':torch.cuda.mem_get_info(),'platform':platform.platform(),'cudnn':torch.backends.cudnn.version(),
        'TF32':False,'amp':False,'precision':'FP32','workers':0,'matplotlib_available':importlib.util.find_spec('matplotlib') is not None})
    original=torch.load(B,map_location='cpu',weights_only=False)
    load_branch='ema' if original.get('ema') is not None else 'model';base=load_base(man)
    save(root/'initialization/base_identity.json',{'path':str(B),'sha256':sha(B),'loaded_checkpoint_key':load_branch,'structure':type(base).__name__,
        'head':type(base.model[-1]).__name__,'names':base.names,'parameters':sum(p.numel() for p in base.parameters()),'text_semantics_fused':base.model[-1].is_fused,
        'has_pe':hasattr(base,'pe'),'Conv_BN_fusion_performed_this_run':False})
    del original
    expert,inherit=detection_from_base(base);save(root/'initialization/inheritance_manifest.json',inherit)
    fm=freeze(expert);assert fm['total_parameters']==11135987 and fm['trainable_parameters']==4085571;save(root/'freeze_manifest.json',fm)
    save(root/'initialization/frozen_digest.json',{'sha256':frozen_digest(expert),'scope':'ALL frozen parameters and ALL registered buffers; raw and EMA must remain identical'})
    torch.save({'model':copy.deepcopy(expert).cpu(),'ema':None,'epoch':-1,'train_args':expert.args,'version':'8.4.32','source_sha256':B_SHA},root/'initialization/E0.pt')
    man.update(E0=str(root/'initialization/E0.pt'),E0_sha256=sha(root/'initialization/E0.pt'));save(root/'experiment_manifest.json',man)
    sample=[]
    for mod in ['rgb','sar']:sample.extend([r for r in val if r['modality']==mod][:2])
    save(root/'initialization/fixed_val_samples.json',{'selection':'first2 RGB + first2 SAR in fixed val manifest; no prediction/error selection','keys':[r['key'] for r in sample]})
    ds=HarborDataset(sample,640);reload,_=load_expert(root/'initialization/E0.pt');base=base.cuda();expert=expert.cuda();reload=reload.cuda()
    from tools.eval.eval_teacher_ship_external import raw_yoloeseg_fused_forward
    parity=[]
    with torch.inference_mode():
        for i,r in enumerate(sample):
            x=ds[i][0].unsqueeze(0).cuda().float()/255;b=raw_yoloeseg_fused_forward(base,x)[0];target=torch.cat((b[:,:4],b[:,5:6]),1)
            e=expert(x)[0];rr=reload(x)[0];diff=float((target-e).abs().max());rdiff=float((rr-e).abs().max());assert diff==0 and rdiff==0
            parity.append({'key':r['key'],'B_harbor_to_E0_decoded_prediction_max_error':diff,'save_reload_max_error':rdiff})
    save(root/'initialization/parity.json',{'passed':True,'checks':parity,'scope':'decoded pixel boxes and harbor scores before NMS; no reset prompts/fusion'})
    print('BOOTSTRAP PASSED',json.dumps({'root':str(root),'base_sha256':B_SHA,'E0_sha256':man['E0_sha256'],'metric_cache_checks':True,'parity':True,'counts':counts}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
