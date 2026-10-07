# -*- coding: utf-8 -*-
"""Continue a healthy A run, then predeclared seed replication and locked final test."""
import argparse,datetime,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.harbor_poc.core import read,save,sha,WORK

def main(root):
    plan={'created':datetime.datetime.now().astimezone().isoformat(),'pid':os.getpid(),
      'rule':'If completed A seed42 improves BOTH modality validation AP50-95 over B, stop configuration search and replicate A seed43. Otherwise stop for validation-only diagnosis and bounded config B decision.',
      'seed_rule':'Seed43 is validation replication only; selected final checkpoint remains seed42 best macro validation epoch. No selection between seeds on test.',
      'test_rule':'Write final_selection.json with hashes BEFORE final_evaluate.py; after test never tune. Any unfavorable test result is reported unchanged.',
      'max_configs':3,'status':'waiting_A_seed42'}
    save(root/'campaign_plan.json',plan)
    while not (root/'runs/A_seed42/result.json').exists():
        if (root/'runs/A_seed42/failure.json').exists():raise RuntimeError('A_seed42 failed; inspect persisted failure, do not duplicate training')
        time.sleep(30)
    a=read(root/'runs/A_seed42/result.json');b=read(root/'evaluations/B_val/metrics.json')
    if not all(a['metrics'][m]['ap50_95']>b[m]['ap50_95'] for m in ['rgb','sar']):
        plan.update(status='validation_review_needed',A=a,B=b);save(root/'campaign_plan.json',plan);return 2
    cfg=read(root/'config_A_seed42.json');cfg['seed']=43;save(root/'config_A_seed43.json',cfg)
    plan.update(status='replicating_seed43',A_best_epoch=a['best_epoch'],A_metrics=a['metrics']);save(root/'campaign_plan.json',plan)
    command=[sys.executable,'-B','-u','-X','utf8',str(WORK/'tools/harbor_poc/supervise.py'),'--root',str(root),'--run','A_seed43','--config',str(root/'config_A_seed43.json')]
    proc=subprocess.run(command,cwd=WORK,creationflags=subprocess.CREATE_NO_WINDOW)
    if proc.returncode:raise RuntimeError(f'Seed43 failed with exit code {proc.returncode}')
    replica=read(root/'runs/A_seed43/result.json')
    selection={'run_id':'A_seed42','checkpoint':a['best_path'],'checkpoint_sha256':a['best_sha256'],'epoch':a['best_epoch'],
      'test_not_used_for_selection':True,'selected_at':datetime.datetime.now().astimezone().isoformat(),
      'reason':'Predeclared RGB/SAR macro AP50-95 selection within A seed42; both modalities improved on validation, no configuration B/C needed.',
      'validation':a['metrics'],'seed43_validation':replica['metrics'],'seed43_best_epoch':replica['best_epoch'],
      'seed_selection':'seed42 main model fixed before replication; seed43 checks validation sensitivity, not test cherry-picking',
      'false_positive_handling':'Report fixed-threshold FP/TP/FN and recall for both seeds; no arbitrary acceptability threshold or score calibration.'}
    save(root/'final_selection.json',selection);plan.update(status='final_test_running');save(root/'campaign_plan.json',plan)
    logs=root/'logs'
    for script in ['final_evaluate.py','report.py']:
        with (logs/(script+'.stdout.log')).open('w',encoding='utf8') as so,(logs/(script+'.stderr.log')).open('w',encoding='utf8') as se:
            cmd=[sys.executable,'-B','-u','-X','utf8',str(WORK/'tools/harbor_poc'/script),'--root',str(root)]
            ret=subprocess.run(cmd,cwd=WORK,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        save(logs/(script+'.exit.json'),{'command':cmd,'exit_code':ret.returncode})
        if ret.returncode:raise RuntimeError(f'{script} failed: {ret.returncode}')
    plan.update(status='complete',finished=datetime.datetime.now().astimezone().isoformat());save(root/'campaign_plan.json',plan)
    return 0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:code=main(a.root)
    except BaseException as e:
        save(a.root/'campaign_failure.json',{'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'exit_code':1,'pid':os.getpid()});traceback.print_exc();code=1
    sys.exit(code)
