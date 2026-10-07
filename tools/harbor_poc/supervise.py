# -*- coding: utf-8 -*-
"""Hidden persistent process wrapper preserving child exit code and separate logs."""
import argparse,datetime,json,os,subprocess,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--config',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    work=Path(__file__).resolve().parents[2];logs=a.root/'logs';logs.mkdir(exist_ok=True)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');prefix=logs/f'{a.run}_{stamp}'
    cmd=[sys.executable,'-B','-u','-X','utf8',str(work/'tools/harbor_poc/train.py'),'--root',str(a.root),'--run',a.run,'--config',a.config]
    if a.resume:cmd.append('--resume')
    env=os.environ.copy();env['PYTHONHASHSEED']='42';env['PYTHONUTF8']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    with prefix.with_suffix('.stdout.log').open('w',encoding='utf8') as stdout,prefix.with_suffix('.stderr.log').open('w',encoding='utf8') as stderr:
        child=subprocess.Popen(cmd,cwd=work,stdout=stdout,stderr=stderr,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        state={'supervisor_pid':os.getpid(),'training_pid':child.pid,'run':a.run,'start':stamp,'command':cmd,'workdir':str(work),'stdout':str(prefix.with_suffix('.stdout.log')),'stderr':str(prefix.with_suffix('.stderr.log')),'status':'running'}
        statepath=prefix.with_suffix('.process.json');statepath.write_text(json.dumps(state,indent=2),encoding='utf8')
        (a.root/'active_process.json').write_text(json.dumps(state,indent=2),encoding='utf8')
        code=child.wait();state.update(status='completed' if code==0 else 'failed',exit_code=code,end=datetime.datetime.now().isoformat())
        statepath.write_text(json.dumps(state,indent=2),encoding='utf8');(a.root/'active_process.json').write_text(json.dumps(state,indent=2),encoding='utf8')
    return code
if __name__=='__main__':sys.exit(main())
