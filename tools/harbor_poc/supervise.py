# -*- coding: utf-8 -*-
"""Hidden persistent process wrapper preserving child exit code and separate logs."""
import argparse,ctypes,datetime,hashlib,json,os,subprocess,sys
from ctypes import wintypes
from pathlib import Path
from status import atomic_json,process_info,read_json,snapshot
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--config',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.argtypes=[ctypes.c_void_p,wintypes.BOOL,wintypes.LPCWSTR];kernel.CreateMutexW.restype=wintypes.HANDLE
    key=hashlib.sha256((str(a.root.resolve()).lower()+'/'+a.run).encode()).hexdigest()
    mutex=kernel.CreateMutexW(None,False,'Local\\harbor_poc_'+key)
    if not mutex or ctypes.get_last_error()==183:raise RuntimeError('Supervisor already exists or mutex acquisition failed; no duplicate launch.')
    old=read_json(a.root/'active_process.json')
    if old.get('run')==a.run and process_info(old.get('training_pid'),old.get('training_creation_filetime'))['alive']:
        raise RuntimeError('Training process already alive; refusing duplicate launch.')
    work=Path(__file__).resolve().parents[2];logs=a.root/'logs';logs.mkdir(exist_ok=True)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');prefix=logs/f'{a.run}_{stamp}'
    cmd=[sys.executable,'-B','-u','-X','utf8','-X','faulthandler',str(work/'tools/harbor_poc/train.py'),'--root',str(a.root),'--run',a.run,'--config',a.config]
    if a.resume:cmd.append('--resume')
    env=os.environ.copy();env['PYTHONHASHSEED']=str(read_json(a.config).get('seed',42));env['PYTHONUTF8']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    with prefix.with_suffix('.stdout.log').open('x',encoding='utf8') as stdout,prefix.with_suffix('.stderr.log').open('x',encoding='utf8') as stderr:
        child=subprocess.Popen(cmd,cwd=work,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,creationflags=subprocess.CREATE_NO_WINDOW,close_fds=True)
        state={'supervisor_pid':os.getpid(),'training_pid':child.pid,'run':a.run,'start':stamp,'command':cmd,'workdir':str(work),'stdout':str(prefix.with_suffix('.stdout.log')),'stderr':str(prefix.with_suffix('.stderr.log')),'status':'running'}
        child_info=process_info(child.pid);own_info=process_info(os.getpid())
        state.update(resume=a.resume,training_creation_filetime=child_info.get('creation_filetime'),supervisor_creation_filetime=own_info.get('creation_filetime'),lifecycle_at_launch={'supervisor':own_info,'training':child_info})
        statepath=prefix.with_suffix('.process.json');statepath.write_text(json.dumps(state,indent=2),encoding='utf8')
        (a.root/'active_process.json').write_text(json.dumps(state,indent=2),encoding='utf8')
        while True:
            atomic_json(a.root/'latest_status.json',snapshot(a.root))
            try:code=child.wait(timeout=30);break
            except subprocess.TimeoutExpired:pass
        state.update(status='completed' if code==0 else 'failed',exit_code=code,end=datetime.datetime.now().astimezone().isoformat())
        atomic_json(statepath,state);atomic_json(a.root/'active_process.json',state)
        atomic_json(a.root/'latest_status.json',snapshot(a.root))
        if code:
            run_dir=a.root/'runs'/a.run;progress=read_json(run_dir/'progress.json')
            progress.update(status='failed',exit_code=code,error_summary=f'Process exit 0x{code & 0xffffffff:08X}',stderr=state['stderr'],ended=state['end'])
            atomic_json(run_dir/'progress.json',progress)
            handoff=a.root/'HANDOFF.md';content=handoff.read_text(encoding='utf8') if handoff.exists() else ''
            content=content.replace('Status: **formal_training_running**','Status: **failed**')
            content+=f"\n\nSupervisor observed exit {code} (0x{code & 0xffffffff:08X}) at {state['end']}. No automatic retry.\nStderr: `{state['stderr']}`. Read `latest_status.json` before resuming.\n"
            handoff.write_text(content,encoding='utf8')
    return code
if __name__=='__main__':sys.exit(main())
