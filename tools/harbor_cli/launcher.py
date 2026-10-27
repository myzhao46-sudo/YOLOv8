"""Foreground Windows launcher and lightweight read-only status. No torch import.

Tee child output without a shell pipeline; preserve native exit codes. Named mutex
plus PID creation identity protects a run. User terminal remains the owner.
"""
import argparse,ctypes,datetime,hashlib,json,os,subprocess,sys,threading
from pathlib import Path
from ctypes import wintypes
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.harbor_poc.status import process_info,read_json,atomic_json,compact_metrics

def now():return datetime.datetime.now().astimezone().isoformat()
def snapshot(root,run):
    rd=root/'runs'/run;p=read_json(rd/'progress.json');a=read_json(rd/'active_session.json')
    live=process_info(a.get('training_pid'),a.get('training_creation_filetime'))
    status='running' if live['alive'] else p.get('status','not_started')
    if not live['alive'] and a.get('status')=='failed':status='failed'
    if not live['alive'] and status=='running':status='not_running_unconfirmed_exit'
    best=read_json(rd/'best_val_metrics.json');last=p.get('latest',{}).get('metrics',{})
    return {'status':status,'checked_at':now(),'run':run,'process':live,'epoch':p.get('last_completed_epoch'),
        'best_epoch':p.get('best_epoch'),'latest_validation':compact_metrics(last),'best_validation':compact_metrics(best),
        'checkpoint':str(rd/'weights/last.pt'),'best':str(rd/'weights/best.pt'),'log':a.get('log'),
        'error':p.get('error',a.get('error')),'exit_code':a.get('exit_code')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',default='A_seed43')
    p.add_argument('--config',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--pause-after-epoch',type=int)
    p.add_argument('--status',action='store_true');p.add_argument('--tail',type=int,default=0);p.add_argument('--json',action='store_true')
    a=p.parse_args();root=a.root.resolve();rd=root/'runs'/a.run
    if a.status:
        s=snapshot(root,a.run)
        if a.json: print(json.dumps(s,ensure_ascii=False,indent=2))
        else:
            print(f"{s['status']} | {s['run']} | epoch={s['epoch']} best={s['best_epoch']} | {s['checked_at']}")
            for label in ['latest_validation','best_validation']:
                for mod,m in s[label].items(): print(f"{label} {mod}: AP50={m['ap50']*100:.4f}% AP50-95={m['ap50_95']*100:.4f}% P/R={m['precision']:.4f}/{m['recall']:.4f} TP/FP/FN={m['tp']}/{m['fp']}/{m['fn']}")
            print('last: '+s['checkpoint']+'\nbest: '+s['best']+'\nlog: '+str(s['log'])+'\nerror: '+str(s['error']))
        if a.tail and s['log'] and Path(s['log']).exists():
            from collections import deque
            with Path(s['log']).open(encoding='utf-8') as f:print(''.join(deque(f,maxlen=a.tail)),end='')
        return 0
    cfg=a.config or root/'config.json';config=read_json(cfg)
    if not config:raise ValueError('Missing config')
    if read_json(rd/'progress.json').get('status')=='completed':raise RuntimeError('Run already completed; refusing resume')
    old=read_json(rd/'active_session.json')
    if process_info(old.get('training_pid'),old.get('training_creation_filetime'))['alive']:raise RuntimeError('Trainer already alive')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.argtypes=[ctypes.c_void_p,wintypes.BOOL,wintypes.LPCWSTR];kernel.CreateMutexW.restype=wintypes.HANDLE
    name='Local\\harbor_checked_'+hashlib.sha256((str(root).lower()+'/'+a.run).encode()).hexdigest()
    mutex=kernel.CreateMutexW(None,False,name)
    if not mutex or ctypes.get_last_error()==183:raise RuntimeError('Same-run launcher mutex already exists')
    rd.mkdir(parents=True,exist_ok=True);logs=root/'logs';logs.mkdir(exist_ok=True)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');sid=a.run+'_'+stamp
    logpath=logs/(sid+'.log');statepath=logs/(sid+'.json')
    cmd=[sys.executable,'-B','-u','-X','utf8','-X','faulthandler',str(Path(__file__).with_name('train_checked.py')),
        '--root',str(root),'--run',a.run,'--config',str(cfg),'--session',sid]
    if a.resume:cmd.append('--resume')
    if a.pause_after_epoch is not None:cmd+=['--pause-after-epoch',str(a.pause_after_epoch)]
    env=os.environ.copy();env.update(PYTHONHASHSEED=str(config['seed']),PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',
        YOLO_CONFIG_DIR=str(root/'runtime_settings'),YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',MPLCONFIGDIR=str(root/'runtime_settings/matplotlib'),
        OMP_NUM_THREADS=str(config['num_threads']),MKL_NUM_THREADS=str(config['num_threads']),WANDB_MODE='disabled')
    state={'session':sid,'status':'starting','started':now(),'command':cmd,'cwd':str(Path(__file__).resolve().parents[2]),'log':str(logpath),
        'resume':a.resume,'pause_after_epoch':a.pause_after_epoch,'launcher_pid':os.getpid()}
    with logpath.open('x',encoding='utf-8',buffering=1) as log:
        header=json.dumps(state,ensure_ascii=False);print(header,flush=True);log.write(header+'\n')
        child=subprocess.Popen(cmd,cwd=state['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace',bufsize=1)
        info=process_info(child.pid);state.update(status='running',training_pid=child.pid,training_creation_filetime=info.get('creation_filetime'))
        atomic_json(statepath,state);atomic_json(rd/'active_session.json',state)
        try:
            for line in child.stdout:
                log.write(line)
                try:print(line,end='',flush=True)
                except (BrokenPipeError,OSError):pass
            code=child.wait()
        except KeyboardInterrupt:
            # With no new process group, console Ctrl-C is also delivered to child.
            log.write('\nLauncher interrupted; waiting for child to record failure/last complete epoch.\n')
            code=child.wait()
        status=read_json(rd/'progress.json').get('status')
        if code or status not in ('paused','completed'):
            state.update(status='failed',error=f'Child exit {code}; final state {status}')
            prog=read_json(rd/'progress.json');prog.update(status='failed',error=state['error'],exit_code=code);atomic_json(rd/'progress.json',prog)
        else:state['status']=status
        state.update(exit_code=code,finished=now());atomic_json(statepath,state);atomic_json(rd/'active_session.json',state)
        footer='SESSION EXIT '+json.dumps(state,ensure_ascii=False);log.write(footer+'\n');print(footer,flush=True)
    return code if code else (0 if state['status'] in ('paused','completed') else 1)

if __name__=='__main__':sys.exit(main())
