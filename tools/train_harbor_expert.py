"""Explicit single-GPU harbor fine-tuning CLI (fresh B -> E0 -> expert).

Parsing/dry-run/status use only the standard library. The user starts training
by running this command without --dry_run. No campaign/gate/fusion/test flow.
"""
import argparse,datetime,json,os,re,subprocess,sys
from pathlib import Path
WORK=Path(__file__).resolve().parents[1]

DEFAULTS={'seed':43,'epochs':150,'patience':30,'imgsz':640,'batch':4,'gradient_accumulation':4,'val_batch_size':4,
    'workers':0,'val_workers':0,'num_threads':4,'num_interop_threads':1,'prefetch_factor':2,'persistent_workers':True,'pin_memory':True,
    'cache_images':'shared','lr0':1e-4,'lrf':.1,'warmup_epochs':3,'weight_decay':.0005,'gradient_clip':10.,
    'rgb_repeat':4,'sar_repeat':1,'flip_h':.5,'flip_v':.5,'box_gain':7.5,'cls_gain':.5,'dfl_gain':1.5,
    'optimizer':'AdamW','amp':False}
DEFAULT_B=Path(r'C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\best.pt')
DEFAULT_DATA=Path(r'D:\RSdataset\_derived\harbor_poc_20261007_124509')
DEFAULT_REFERENCE=Path(r'D:\RSdataset\_experiments\harbor_poc_20261007_124509')

def boolean(value):
    if isinstance(value,bool):return value
    if value.lower() in ('1','true','yes','on'):return True
    if value.lower() in ('0','false','no','off'):return False
    raise argparse.ArgumentTypeError('Expected True or False')
def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--base_weights','--model',type=Path,help='Original fused ship/harbor/tank B, identity checked before initialization')
    p.add_argument('--data_root_dir',type=Path,help='Fixed derived directory with train/val_manifest.json (not original XML root)')
    p.add_argument('--reference_experiment',type=Path,help='Prior cached metric regression assets')
    p.add_argument('--run_root_dir',type=Path,default=Path(r'D:\RSdataset\experiments'))
    p.add_argument('--experiment_dir',type=Path,help='Exact new output directory, or existing CLI-run root with --resume')
    p.add_argument('--run_id_note',help='Safe directory/run name prefix; default harbor_seed43')
    p.add_argument('--device',default=None,help='One physical GPU index, e.g. 0; mapped to CUDA_VISIBLE_DEVICES before importing torch')
    flags=[('seed','seed',int),('epochs','epochs',int),('patience','patience',int),('image_size','imgsz',int),
      ('batch_size','batch',int),('grad_accumulation_steps','gradient_accumulation',int),('val_batch_size','val_batch_size',int),
      ('num_workers','workers',int),('val_num_workers','val_workers',int),('num_threads','num_threads',int),
      ('num_interop_threads','num_interop_threads',int),('prefetch_factor','prefetch_factor',int),
      ('persistent_workers','persistent_workers',boolean),('pin_memory','pin_memory',boolean),('cache_images','cache_images',str),
      ('learning_rate','lr0',float),('lrf','lrf',float),('warmup_epochs','warmup_epochs',int),('weight_decay','weight_decay',float),
      ('gradient_clip','gradient_clip',float),('rgb_repeat','rgb_repeat',int),('sar_repeat','sar_repeat',int),
      ('horizontal_flip','flip_h',float),('vertical_flip','flip_v',float),('box_loss_gain','box_gain',float),
      ('cls_loss_gain','cls_gain',float),('dfl_loss_gain','dfl_gain',float),('optimizer','optimizer',str),('amp','amp',boolean)]
    for flag,dest,kind in flags:
        aliases=['--'+flag]
        if flag=='image_size':aliases.append('--imgsz')
        p.add_argument(*aliases,dest=dest,type=kind,default=None,help=f'Fresh-run default: {DEFAULTS[dest]}')
    p.add_argument('--resume',action='store_true',help='Resume this CLI version using --experiment_dir; supplied config overrides must match saved values')
    p.add_argument('--pause_after_epoch',type=int,help='Optional human epoch boundary; separate from total --epochs')
    p.add_argument('--dry_run',action='store_true',help='Print effective config/commands only; no directories, torch imports, model or training')
    p.add_argument('--status',action='store_true',help='Read-only query for --experiment_dir')
    p.add_argument('--tail',type=int,default=12)
    return p

def validate(v):
    positive=['epochs','patience','imgsz','batch','gradient_accumulation','val_batch_size','num_threads','num_interop_threads','prefetch_factor','rgb_repeat','sar_repeat']
    if any(v[k]<1 for k in positive):raise ValueError('Counts/sizes must be positive')
    if v['workers']<0 or v['val_workers']<0:raise ValueError('Worker counts cannot be negative')
    if v['imgsz']%32:raise ValueError('image_size must be a multiple of32')
    if v['warmup_epochs']<0 or v['epochs']<v['warmup_epochs']+2:raise ValueError('epochs must be at least warmup_epochs+2 for the cosine schedule')
    if v['optimizer']!='AdamW':raise ValueError('Current tested optimizer path is AdamW; auto selection is disabled')
    if v['amp']:raise ValueError('Current verified numerical path is FP32; --amp True is not supported yet')
    if v['cache_images'] not in ('none','ram','shared'):raise ValueError('cache_images must be none, ram or shared')
    if v['cache_images']=='ram' and (v['workers'] or v['val_workers']):raise ValueError('Use shared/none cache with workers>0 on Windows, avoiding per-worker RAM copies')
    if not 0<=v['flip_h']<=1 or not 0<=v['flip_v']<=1:raise ValueError('Flip probabilities must be within[0,1]')
    if v['lr0']<=0 or not 0<=v['lrf']<=1 or v['weight_decay']<0 or v['gradient_clip']<=0:raise ValueError('Invalid optimizer/scheduler parameters')
    if min(v['box_gain'],v['cls_gain'],v['dfl_gain'])<0:raise ValueError('Loss gains cannot be negative')

def build_config(v,root,run):
    return {'config_id':run,'seed':v['seed'],'epochs':v['epochs'],'patience':v['patience'],'imgsz':v['imgsz'],
      'batch':v['batch'],'workers':v['workers'],'val_workers':v['val_workers'],'val_batch_size':v['val_batch_size'],
      'num_threads':v['num_threads'],'num_interop_threads':v['num_interop_threads'],'prefetch_factor':v['prefetch_factor'],
      'persistent_workers':v['persistent_workers'],'pin_memory':v['pin_memory'],'cache_images':v['cache_images'],
      'gradient_accumulation':v['gradient_accumulation'],'effective_batch':v['batch']*v['gradient_accumulation'],'nbs':v['batch']*v['gradient_accumulation'],
      'optimizer':'AdamW','lr0':v['lr0'],'lrf':v['lrf'],'warmup_epochs':v['warmup_epochs'],'weight_decay':v['weight_decay'],
      'betas':[.9,.999],'gradient_clip':v['gradient_clip'],'precision':'FP32','amp':False,
      'scheduler':'linear warmup then cosine over the originally configured epoch budget',
      'loss_gains':{'box':v['box_gain'],'cls':v['cls_gain'],'dfl':v['dfl_gain']},
      'loss_normalization':'official batch-scaled loss.sum()/actual accumulation group sample count',
      'rgb_repeat':v['rgb_repeat'],'sar_repeat':v['sar_repeat'],'sampling':'fixed repetitions with seed+epoch shuffled image order',
      'augmentation':{'horizontal_flip':v['flip_h'],'vertical_flip':v['flip_v'],'mosaic':0,'mixup':0,'hsv':0,'scale_jitter':0,'rotation':0},
      'expanded_freeze':False,'freeze_preset':'p5_head','ema':{'decay':.9999,'tau_steps':2000,'fixed_buffers':True},
      'selection':'EMA equal-weight RGB/SAR val AP50-95; earlier epoch wins ties','negative_images':0,
      'initial_weights':str(root/'initialization/E0.pt'),'formal_rng_reset':'after short check and model/dataset construction, immediately before epoch1',
      'metric_version':'harbor_coco101_v2_20261024','data_rng_version':'harbor_stateless_flip_v1'}

def resolve(a):
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('This entry is single-GPU, not DDP; do not use torchrun nproc>1')
    if a.resume:
        if not a.experiment_dir:raise ValueError('--resume requires --experiment_dir')
        root=a.experiment_dir.resolve();path=root/'cli_job.json'
        if not path.exists():raise ValueError('Not a new CLI run. Legacy 10-epoch checkpoints keep their existing Continue_Training.ps1 entry.')
        job=json.loads(path.read_text(encoding='utf-8'));v=job['cli_parameters']
        for key in DEFAULTS:
            supplied=getattr(a,key)
            if supplied is not None and supplied!=v[key]:raise ValueError('Resume config cannot change '+key)
        for key in ['base_weights','data_root_dir','reference_experiment']:
            supplied=getattr(a,key)
            if supplied is not None and str(supplied.resolve())!=job[key]:raise ValueError('Resume asset cannot change '+key)
        if a.run_id_note is not None and a.run_id_note!=job['run']:raise ValueError('Resume run name differs')
        job['device']=a.device if a.device is not None else job['device']
    else:
        v={key:getattr(a,key) if getattr(a,key) is not None else default for key,default in DEFAULTS.items()};validate(v)
        run=a.run_id_note or f'harbor_seed{v["seed"]}'
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',run):raise ValueError('Use a simple ASCII run_id_note, max80 characters')
        root=(a.experiment_dir or a.run_root_dir/(run+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
        if root.exists():raise FileExistsError('Fresh run needs a new experiment_dir: '+str(root))
        job={'schema':'harbor_cli_v1','root':str(root),'run':run,'device':a.device or '0',
          'base_weights':str((a.base_weights or DEFAULT_B).resolve()),'data_root_dir':str((a.data_root_dir or DEFAULT_DATA).resolve()),
          'reference_experiment':str((a.reference_experiment or DEFAULT_REFERENCE).resolve()),'cli_parameters':v,
          'config':build_config(v,root,run),'argv':sys.argv,'created':datetime.datetime.now().astimezone().isoformat()}
    if not str(job['device']).isdigit():raise ValueError('device must be one physical GPU index, e.g.0')
    if a.pause_after_epoch is not None and not 1<=a.pause_after_epoch<=job['config']['epochs']:raise ValueError('Invalid pause_after_epoch')
    return job

def run_logged(command,cwd,env,path):
    with path.open('x',encoding='utf-8',buffering=1) as f:
        f.write(json.dumps({'command':command,'started':datetime.datetime.now().astimezone().isoformat()})+'\n')
        child=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        for line in child.stdout:print(line,end='',flush=True);f.write(line)
        code=child.wait();f.write(json.dumps({'exit_code':code,'finished':datetime.datetime.now().astimezone().isoformat()})+'\n')
    return code

def main(argv=None):
    p=parser();a=p.parse_args(argv)
    if a.status:
        if not a.experiment_dir:p.error('--status requires --experiment_dir')
        path=a.experiment_dir/'cli_job.json'
        if not path.exists():p.error('Not a CLI run directory')
        job=json.loads(path.read_text(encoding='utf-8'))
        return subprocess.call([sys.executable,'-B','-X','utf8',str(WORK/'tools/harbor_cli/launcher.py'),'--root',str(a.experiment_dir),'--run',job['run'],'--status','--tail',str(a.tail)],cwd=WORK)
    try:job=resolve(a)
    except (ValueError,OSError) as e:p.error(str(e))
    root=Path(job['root']);config=job['config'];cmd=[sys.executable,'-B','-u','-X','utf8','-X','faulthandler',str(WORK/'tools/harbor_cli/launcher.py'),'--root',str(root),'--run',job['run'],'--config',str(root/'config.json')]
    if a.resume:cmd.append('--resume')
    if a.pause_after_epoch is not None:cmd+=['--pause-after-epoch',str(a.pause_after_epoch)]
    if a.dry_run:
        print(json.dumps({'dry_run':True,'experiment_root':str(root),'config':config,'device':job['device'],'launch_command':cmd,
          'bootstrap_required':not a.resume,'torch_imported':'torch' in sys.modules,'training_started':False,
          'effective_persistent_train_workers':bool(config['workers'] and config['persistent_workers'])},ensure_ascii=False,indent=2));return 0
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=job['device'],WANDB_MODE='disabled',OMP_NUM_THREADS=str(config['num_threads']),MKL_NUM_THREADS=str(config['num_threads']),
      PYTHONHASHSEED=str(config['seed']),PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',YOLO_CONFIG_DIR=str(root/'runtime_settings'),MPLCONFIGDIR=str(root/'runtime_settings/matplotlib'),YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
    if not a.resume:
        root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
        (root/'cli_job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
        (root/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        resume_command=f'& "{sys.executable}" -B -u -X utf8 "{WORK / "tools/train_harbor_expert.py"}" --experiment_dir "{root}" --resume'
        status_command=f'& "{sys.executable}" -B -X utf8 "{WORK / "tools/train_harbor_expert.py"}" --experiment_dir "{root}" --status --tail 12'
        (root/'RUN_COMMANDS.md').write_text('# CLI run commands\n\nRun root: `'+str(root)+'`\n\n'
            'Resume the same configuration:\n```powershell\n'+resume_command+'\n```\n\n'
            'Read-only status:\n```powershell\n'+status_command+'\n```\n',encoding='utf-8')
        boot=[sys.executable,'-B','-u','-X','utf8',str(WORK/'tools/harbor_cli/bootstrap.py'),'--job',str(root/'cli_job.json')]
        code=run_logged(boot,WORK,env,root/'logs/bootstrap.log')
        if code:return code
    print('Experiment:',root,'\nLogs:',root/'logs',flush=True)
    return subprocess.call(cmd,cwd=WORK,env=env)

if __name__=='__main__':sys.exit(main())
