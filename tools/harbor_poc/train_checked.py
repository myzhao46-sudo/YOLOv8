"""Single harbor supervised run, versioned metrics and epoch-boundary pause/resume.

No test, campaign, fusion, gate or distillation entry points. The pause limit is
runtime control, never part of the scientific config or 150-epoch scheduler.
"""
import argparse,copy,csv,datetime,hashlib,json,math,os,random,shutil,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.harbor_poc.core import *
from tools.harbor_poc.checked_inference import get_preds
from tools.harbor_poc.train import make_opt,make_scheduler,make_loss,short_check,assert_freeze,cpu_copy
from tools.harbor_poc.metrics_v2 import evaluate,VERSION
from tools.harbor_poc.status import process_info

def now():return datetime.datetime.now().astimezone().isoformat()
def digest(value):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            a=v.detach().cpu().contiguous().numpy();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        elif isinstance(v,np.ndarray):h.update(str(v.dtype).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(tuple,list)):
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()
def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def atomic_copy(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp');shutil.copyfile(source,tmp);os.replace(tmp,target)
def check_sources(root):
    expected=read(root/'source_snapshot/code_hashes.json')
    for rel,h in expected.items():
        if sha(WORK/rel)!=h:raise RuntimeError('Executable source changed since this run was fixed: '+rel)
    return expected
def manifests(root):
    man=read(root/'experiment_manifest.json');der=Path(man['derived_root']);expected=read(root/'data_identity.json')
    for name,h in expected['manifest_sha256'].items():
        if sha(der/name)!=h:raise RuntimeError('Fixed train/val manifest changed: '+name)
    return man,read(der/'train_manifest.json'),read(der/'val_manifest.json')
def write_history(rd,history):
    save(rd/'history.json',history)
    fields=['epoch','checkpoint_epoch','lr','total_loss','box_loss','cls_loss','dfl_loss','rgb_ap50','rgb_ap50_95','rgb_precision','rgb_recall','rgb_tp','rgb_fp','rgb_fn',
        'sar_ap50','sar_ap50_95','sar_precision','sar_recall','sar_tp','sar_fp','sar_fn','macro_ap50_95','best_epoch','bad_epochs','epoch_optimizer_steps','optimizer_steps','seconds','gpu_max_allocated','frozen_passed']
    tmp=rd/'history.csv.tmp'
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for row in history:
            x={k:row[k] for k in ['epoch','checkpoint_epoch','lr','total_loss','best_epoch','bad_epochs','epoch_optimizer_steps','optimizer_steps','seconds','gpu_max_allocated']}
            x.update(zip(['box_loss','cls_loss','dfl_loss'],row['loss']));x['macro_ap50_95']=row['metrics']['macro_ap50_95'];x['frozen_passed']=row['frozen_check']['passed']
            for mod in ['rgb','sar']:
                for k in ['ap50','ap50_95','precision','recall','tp','fp','fn']:x[mod+'_'+k]=row['metrics'][mod][k]
            w.writerow(x)
    os.replace(tmp,rd/'history.csv')
def plot_loss(rd,history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4));epochs=[x['epoch'] for x in history]
    for key,label in [(None,'total'),(0,'box'),(1,'cls'),(2,'DFL')]:
        ax.plot(epochs,[x['total_loss'] if key is None else x['loss'][key] for x in history],marker='.',label=label)
    ax.set(xlabel='Completed epoch (150-epoch schedule)',ylabel='Image-weighted supervised loss');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(rd/'loss_curve.png',dpi=160);plt.close(fig)
def save_checkpoint(path,m,ema,opt,scheduler,epoch,steps,cfg,history,best,best_epoch,bad,root,frozen):
    payload={'model':cpu_copy(m),'ema':cpu_copy(ema),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'updates':steps,
        'config':cfg,'history':history,'best_fitness':best,'best_epoch':best_epoch,'bad_epochs':bad,'rng':rng_state(),'scaler':None,'precision':'FP32',
        'version':'8.4.32','date':now(),'parent':cfg['initial_weights'],'base_sha256':read(root/'experiment_manifest.json')['base_sha256'],
        'protocol_sha256':sha(root/'evaluation_protocol.json'),'data_identity':read(root/'data_identity.json'),'code_hashes':check_sources(root),
        'initial_frozen_digest':frozen,'checkpoint_schema':'harbor_checked_v1','metric_version':VERSION,
        'runtime_control_note':'pause-after-epoch is deliberately NOT stored as config or resume limit',
        'train_args':{'task':'detect','imgsz':cfg['imgsz'],'single_cls':True}}
    path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)
def export_and_check(root,rd,val,stage,epoch):
    """Same exporter for paused stage-check and truly completed inference artifact."""
    cfg=read(rd/'effective_config.json');best,bck=load_expert(rd/'weights/best.pt');freeze(best,cfg['expanded_freeze'])
    dest=rd/'inference' if stage=='completed' else rd/'stage_check'/f'epoch_{epoch:04d}'
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/('harbor_expert.pt' if stage=='completed' else 'harbor_expert_stage.pt')
    payload={'model':cpu_copy(best),'ema':None,'epoch':bck['epoch'],'names':{0:'harbor'},'task':'detect','stage':stage,
        'source_checkpoint':str(rd/'weights/best.pt'),'source_checkpoint_sha256':sha(rd/'weights/best.pt'),
        'source_B_sha256':bck['base_sha256'],'metric_version':VERSION,'train_args':{'task':'detect','imgsz':640,'single_cls':True}}
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)
    keys=read(root/'initialization/fixed_val_samples.json')['keys'];lookup={r['key']:r for r in val};sample=[lookup[k] for k in keys]
    reference,_=get_preds(best,sample,640,4);best=best.cpu();del best;torch.cuda.empty_cache()
    exported,_=load_expert(path);actual,_=get_preds(exported,sample,640,4)
    assert actual==reference,'Export/reload same-input candidates differ'
    # Trace back to the selected full-val epoch cache, with a recorded numerical tolerance.
    bestrow=bck['history'][bck['best_epoch']-1];cache=read(bestrow['validation_predictions']);max_box=max_score=0.
    for k in keys:
        assert len(cache[k])==len(reference[k])
        for p,q in zip(cache[k],reference[k]):
            max_box=max(max_box,max(abs(a-b) for a,b in zip(p['xyxy'],q['xyxy'])));max_score=max(max_score,abs(p['score']-q['score']))
    assert max_box<=.001 and max_score<=1e-6,(max_box,max_score)
    save(dest/'fixed_val_predictions.json',actual)
    info={'stage':stage,'not_final_when_paused':stage!='completed','weights':str(path),'sha256':sha(path),'names':{'0':'harbor'},'structure':'DetectionModel/Detect, inherited fused harbor classifier',
        'source_B':read(root/'experiment_manifest.json')['base_weights'],'source_B_sha256':bck['base_sha256'],'best_checkpoint_sha256':sha(rd/'weights/best.pt'),
        'best_epoch':bck['best_epoch'],'best_validation':bestrow['metrics'],'protocol_sha256':sha(root/'evaluation_protocol.json'),
        'reloaded_EMA_inference_same_input_candidates_exact':True,'cached_full_val_max_box_error':max_box,'cached_full_val_max_score_error':max_score,
        'fixed_val_keys':keys,'test_used':False,'loading':'tools.harbor_poc.core.load_expert (pinned local repository); model defaults to EMA in training checkpoint'}
    save(dest/'export_manifest.json',info)
    (dest/'LOADING.md').write_text('Single-class 0=harbor; RGB/SAR prototype, FP32, PIL RGB letterbox640/114 then /255.\n'
        'Requires this pinned local worktree, not arbitrary Ultralytics releases.\n'
        f'Use tools/harbor_poc/checked_predict.py --root "{root}" --weights "{path}" --image <image> --out <new-directory>.\n'
        f'Stage: {stage}. A paused export is a stage-check artifact, not completed-training quality evidence.\n',encoding='utf-8')
    del exported,bck;torch.cuda.empty_cache()
    return info

def run(a):
    root=a.root.resolve();setup(root);check_sources(root);cfg=read(a.config);man,train,val=manifests(root);rd=root/'runs'/a.run
    # Also protect direct trainer invocation, independent of the launcher mutex.
    import msvcrt
    rd.mkdir(parents=True,exist_ok=True);a._run_lock=(rd/'trainer.lock').open('a+b')
    if a._run_lock.seek(0,2)==0:a._run_lock.write(b'0');a._run_lock.flush()
    a._run_lock.seek(0)
    try:msvcrt.locking(a._run_lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise RuntimeError('Another trainer holds this run lock')
    a._lock_acquired=True
    if cfg['epochs']!=150 or cfg['amp'] or cfg['precision']!='FP32':raise ValueError('Expected locked 150-epoch FP32 scientific config')
    if a.pause_after_epoch is not None and not 1<=a.pause_after_epoch<=cfg['epochs']:raise ValueError('Invalid human epoch pause bound')
    if (rd/'progress.json').exists() and read(rd/'progress.json').get('status')=='completed':raise RuntimeError('Completed run cannot be resumed')
    if not a.resume and (rd/'weights/last.pt').exists():raise RuntimeError('Fresh run refuses existing checkpoint')
    if a.resume and not (rd/'weights/last.pt').is_file():raise FileNotFoundError('Resume requires this run last.pt')
    oldprocess=read(rd/'process.json') if (rd/'process.json').exists() else {}
    if process_info(oldprocess.get('pid'),oldprocess.get('creation_filetime'))['alive']:raise RuntimeError('Run trainer already alive')
    rd.mkdir(parents=True,exist_ok=True);(rd/'sessions').mkdir(exist_ok=True)
    process={'pid':os.getpid(),'creation_filetime':process_info(os.getpid()).get('creation_filetime'),'session':a.session,'resume':a.resume,'argv':sys.argv,'started':now(),'status':'running'}
    save(rd/'process.json',process);save(rd/'sessions'/f'{a.session}.json',process)
    if a.resume and read(rd/'effective_config.json')!=cfg:raise RuntimeError('Resume config changed')
    save(rd/'effective_config.json',cfg)
    if not a.resume:
        seed_all(cfg['seed']);short_check(root,cfg,train)
    if not read(root/'short_training/update_validation.json')['passed']:raise RuntimeError('Short check not passed')
    ck=None
    if a.resume:m,ck=load_expert(rd/'weights/last.pt',ema=False)
    else:m,_=load_expert(cfg['initial_weights'],ema=False)
    fm=freeze(m,cfg['expanded_freeze']);save(rd/'freeze_manifest.json',fm)
    assert fm['total_parameters']==11135987 and fm['trainable_parameters']==4085571
    baseline=read(root/'initialization/frozen_digest.json')['sha256'];assert frozen_digest(m)==baseline
    m=m.cuda();ema=copy.deepcopy(m).eval();opt=make_opt(m,cfg);scheduler=make_scheduler(opt,cfg);criterion=make_loss(m,cfg)
    history=[];best=-1.;best_epoch=bad=steps=start_epoch=0
    td=HarborDataset(train,cfg['imgsz'],True);vd=HarborDataset(val,cfg['imgsz'],False)
    indices=[i for i,r in enumerate(train) for _ in range(cfg['rgb_repeat'] if r['modality']=='rgb' else cfg['sar_repeat'])]
    batches=math.ceil(len(indices)/cfg['batch']);expected_steps=math.ceil(batches/cfg['gradient_accumulation'])
    save(rd/'sampling_manifest.json',{'unique_images':len(train),'epoch_presentations':len(indices),'rgb_presentations':sum(train[i]['modality']=='rgb' for i in indices),
        'sar_presentations':sum(train[i]['modality']=='sar' for i in indices),'microbatches':batches,'optimizer_steps_per_epoch':expected_steps,
        'full_group_samples':cfg['batch']*cfg['gradient_accumulation'],'last_group_samples':len(indices)-(expected_steps-1)*cfg['batch']*cfg['gradient_accumulation'],
        'normalization':'official batch-scaled loss.sum()/actual samples in accumulation group; last group is not divided by nominal16'})
    if ck:
        assert ck['config']==cfg and ck['protocol_sha256']==sha(root/'evaluation_protocol.json')
        assert ck['code_hashes']==check_sources(root) and ck['data_identity']==read(root/'data_identity.json')
        assert ck['metric_version']==VERSION and ck['base_sha256']==man['base_sha256'] and ck['initial_frozen_digest']==baseline
        ema.load_state_dict(ck['ema'].state_dict());opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        history=ck['history'];best=ck['best_fitness'];best_epoch=ck['best_epoch'];bad=ck['bad_epochs'];steps=ck['updates'];start_epoch=ck['epoch']+1
        restore_rng(ck['rng'])
        pairs={'model':(m.state_dict(),ck['model'].state_dict()),'EMA':(ema.state_dict(),ck['ema'].state_dict()),
            'optimizer':(opt.state_dict(),ck['optimizer']),'scheduler':(scheduler.state_dict(),ck['scheduler']),'RNG':(rng_state(),ck['rng'])}
        restoration={k:{'actual_sha256':digest(x),'checkpoint_sha256':digest(y),'equal':digest(x)==digest(y)} for k,(x,y) in pairs.items()}
        assert all(x['equal'] for x in restoration.values()),restoration
        save(rd/'sessions'/f'{a.session}_resume_verification.json',{'passed':True,'restored':restoration,'checkpoint_path':str(rd/'weights/last.pt'),
            'checkpoint_sha256':sha(rd/'weights/last.pt'),'checkpoint_epoch_0based':ck['epoch'],'last_completed_human_epoch':start_epoch,'next_human_epoch':start_epoch+1,
            'updates':steps,'best_fitness':best,'best_epoch':best_epoch,'bad_epochs':bad,'lr_for_next_epoch':[g['lr'] for g in opt.param_groups],
            'scaler':None,'precision':'FP32','epoch_boundary_only':True,'cross_hardware_bitwise_equivalence_claimed':False})
        # Checkpoint is the transaction source after any interrupted epoch/log write.
        write_history(rd,history)
        if best_epoch==start_epoch:atomic_copy(rd/'weights/last.pt',rd/'weights/best.pt')
        if not (rd/'weights/best.pt').exists():raise RuntimeError('Best checkpoint missing and cannot reconstruct from last')
        print(f'RESUMED last completed={start_epoch}, next={start_epoch+1}, updates={steps}, lr={opt.param_groups[0]["lr"]:.12g}',flush=True)
    else:
        # Critical stage boundary: short-check presence/length cannot shift formal RNG.
        seed_all(cfg['seed']);save(rd/'formal_rng_start.json',{'seed':cfg['seed'],'sha256':digest(rng_state()),'reset_after':'short check, model/optimizer/dataset construction; immediately before formal loop'})
    if a.pause_after_epoch is not None and a.pause_after_epoch<=start_epoch:raise RuntimeError('Pause bound must be later than saved completed epoch')
    del ck
    all_trainable={n:p for n,p in m.named_parameters() if p.requires_grad};opt_ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(opt_ids)==len(set(opt_ids))==len(all_trainable) and set(opt_ids)=={id(p) for p in all_trainable.values()}
    optimizer_info={'optimizer':'AdamW','parameter_groups':[{'lr':g['lr'],'weight_decay':g['weight_decay'],'tensor_count':len(g['params']),
        'parameters':sum(p.numel() for p in g['params']),'betas':g['betas'],'eps':g['eps']} for g in opt.param_groups],
        'trainable_tensor_count':len(all_trainable),'every_trainable_in_optimizer_exactly_once':True}
    save(rd/'optimizer_manifest.json',optimizer_info);save(rd/'sessions'/f'{a.session}_optimizer_manifest.json',optimizer_info)
    prior=history[-1]['total_seconds'] if history else 0.;started=time.time();status='paused';reason='pause_after_epoch'
    if start_epoch>=cfg['epochs']:status='completed';reason='epoch_limit'
    elif bad>=cfg['patience']:status='completed';reason='early_stopping'
    save(rd/'progress.json',{'status':'running','last_completed_epoch':start_epoch,'best_epoch':best_epoch,'planned_epochs':cfg['epochs'],'pause_after_epoch':a.pause_after_epoch})
    for epoch in range(start_epoch,start_epoch if bad>=cfg['patience'] else cfg['epochs']):
        t=time.time();train_mode(m);opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        oldtrain={n:p.detach().cpu().clone() for n,p in all_trainable.items()};order=indices.copy();random.Random(cfg['seed']+epoch).shuffle(order)
        loader=torch.utils.data.DataLoader(td,batch_size=cfg['batch'],sampler=order,num_workers=cfg['workers'],collate_fn=collate,pin_memory=True)
        loss_sum=np.zeros(3);seen=0;epochsteps=0;grad_min=float('inf');grad_max=norm_max=0.;lr=opt.param_groups[0]['lr'];groups=[]
        for bi,item in enumerate(loader):
            n=len(item['img']);item['img']=item['img'].cuda(non_blocking=True).float()/255;vec,det=criterion(m(item['img']),item)
            group_start=(bi//cfg['gradient_accumulation'])*cfg['gradient_accumulation']
            group_samples=min(cfg['batch']*cfg['gradient_accumulation'],len(order)-group_start*cfg['batch'])
            loss=vec.sum()/group_samples
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss epoch{epoch+1} batch{bi+1}')
            loss.backward();loss_sum+=np.asarray(det.detach().cpu())*n;seen+=n
            if (bi+1)%cfg['gradient_accumulation']==0 or bi+1==len(loader):
                grads=[p.grad for p in all_trainable.values() if p.grad is not None]
                if not grads or any(not torch.isfinite(g).all() for g in grads):raise FloatingPointError('Missing/nonfinite gradients')
                maximum=max(float(g.abs().max()) for g in grads)
                if maximum<=0:raise RuntimeError('All gradients zero')
                norm=float(torch.nn.utils.clip_grad_norm_(list(all_trainable.values()),cfg['gradient_clip']))
                if not math.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
                opt.step();opt.zero_grad(set_to_none=True);steps+=1;epochsteps+=1;update_ema(ema,m,steps)
                grad_min=min(grad_min,maximum);grad_max=max(grad_max,maximum);norm_max=max(norm_max,norm);groups.append(group_samples)
            if bi%100==0:save(rd/'heartbeat.json',{'pid':os.getpid(),'human_epoch':epoch+1,'batch':bi+1,'batches':len(loader),'updates':steps,'timestamp':now()})
        assert epochsteps==expected_steps and seen==len(indices)
        frozen=assert_freeze(m,ema,baseline)
        changes={n:float((p.detach().cpu()-oldtrain[n]).abs().max()) for n,p in all_trainable.items()}
        assert max(changes.values())>0;del oldtrain
        pred,_=get_preds(ema,val,cfg['imgsz'],cfg['batch'],dataset=vd);met=evaluate(val,pred,details=True);matches=met.pop('per_image_matches')
        vdpath=rd/'validation'/f'epoch_{epoch+1:04d}_{a.session}';save(vdpath/'predictions.json',pred);save(vdpath/'matches_iou50.json',matches);save(vdpath/'metrics.json',met)
        fitness=met['macro_ap50_95'];improved=fitness>best
        if improved:best=fitness;best_epoch=epoch+1;bad=0
        else:bad+=1
        scheduler.step();losses=(loss_sum/seen).tolist()
        row={'epoch':epoch+1,'checkpoint_epoch':epoch,'session':a.session,'lr':lr,'next_lr':opt.param_groups[0]['lr'],'loss':losses,'total_loss':sum(losses),
            'metrics':met,'optimizer_steps':steps,'epoch_optimizer_steps':epochsteps,'gradient_max_min':grad_min,'gradient_max':grad_max,'gradient_norm_max_preclip':norm_max,
            'finite_loss_and_gradients':True,'seconds':time.time()-t,'total_seconds':prior+time.time()-started,'gpu_max_allocated':torch.cuda.max_memory_allocated(),
            'gpu_max_reserved':torch.cuda.max_memory_reserved(),'best_epoch':best_epoch,'best_fitness':best,'bad_epochs':bad,'frozen_check':frozen,
            'trainable_parameter_max_changes':changes,'actual_accumulation_group_samples':dict((str(x),groups.count(x)) for x in sorted(set(groups))),
            'validation_predictions':str(vdpath/'predictions.json'),'validation_predictions_sha256':sha(vdpath/'predictions.json'),'metric_version':VERSION}
        history.append(row)
        save_checkpoint(rd/'weights/last.pt',m,ema,opt,scheduler,epoch,steps,cfg,history,best,best_epoch,bad,root,baseline)
        if improved:
            atomic_copy(rd/'weights/last.pt',rd/'weights/best.pt');save(rd/'best_val_metrics.json',met);save(rd/'best_identity.json',{'epoch':best_epoch,'validation_predictions':row['validation_predictions'],'checkpoint_sha256':sha(rd/'weights/best.pt')})
        write_history(rd,history)
        save(rd/'progress.json',{'status':'running','last_completed_epoch':epoch+1,'best_epoch':best_epoch,'best_fitness':best,'bad_epochs':bad,'latest':row,'checkpoint':str(rd/'weights/last.pt')})
        print(f'EPOCH {epoch+1}/150 loss={losses} total={sum(losses):.6f} lr={lr:.12g} AP95_RGB/SAR={met["rgb"]["ap50_95"]:.6f}/{met["sar"]["ap50_95"]:.6f} best={best:.6f}@{best_epoch} updates={steps} seconds={row["seconds"]:.1f}',flush=True)
        if bad>=cfg['patience']:status='completed';reason='early_stopping';break
        if epoch+1==cfg['epochs']:status='completed';reason='epoch_limit';break
        if a.pause_after_epoch is not None and epoch+1>=a.pause_after_epoch:break
    # Verify this session's final raw/EMA checkpoint before releasing training models.
    lastm,lastck=load_expert(rd/'weights/last.pt',ema=False)
    same_model=digest(lastm.state_dict())==digest(m.state_dict());same_ema=digest(lastck['ema'].state_dict())==digest(ema.state_dict());assert same_model and same_ema
    save(rd/'sessions'/f'{a.session}_save_reload_verification.json',{'passed':True,'raw_state_equal':same_model,'EMA_state_equal':same_ema,'checkpoint_epoch_0based':lastck['epoch'],'updates':lastck['updates']})
    del m,ema,opt,criterion,td,lastm,lastck;torch.cuda.empty_cache()
    plot_loss(rd,history);export=export_and_check(root,rd,val,status,len(history))
    assert sha(man['base_weights'])==man['base_sha256'],'Original B changed'
    finish={'status':status,'termination':reason,'last_completed_epoch':len(history),'next_human_epoch':len(history)+1 if status=='paused' else None,'planned_epochs':cfg['epochs'],
        'best_epoch':best_epoch,'best_fitness':best,'bad_epochs':bad,'optimizer_steps':steps,'latest':history[-1],
        'last_path':str(rd/'weights/last.pt'),'best_path':str(rd/'weights/best.pt'),'last_sha256':sha(rd/'weights/last.pt'),'best_sha256':sha(rd/'weights/best.pt'),
        'export':export,'base_sha256_after':sha(man['base_weights']),'test_used':False,'finished':now(),'exit_code':0}
    save(rd/'progress.json',finish);save(rd/('result.json' if status=='completed' else 'pause_result.json'),finish)
    process.update(status=status,finished=now(),exit_code=0);save(rd/'process.json',process);save(rd/'sessions'/f'{a.session}.json',process)
    print('RUN '+status.upper()+' '+json.dumps({k:finish[k] for k in ['last_completed_epoch','best_epoch','termination','optimizer_steps','exit_code']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--pause-after-epoch',type=int);p.add_argument('--session',required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        rd=a.root/'runs'/a.run;rd.mkdir(parents=True,exist_ok=True)
        fail={'status':'failed','pid':os.getpid(),'session':a.session,'error':repr(e),'traceback':traceback.format_exc(),'time':now(),'exit_code':1,
            'restore_scope':'last completed epoch only, no batch-level guarantee'}
        save(rd/'sessions'/f'{a.session}_failure.json',fail)
        if getattr(a,'_lock_acquired',False):
            progress=read(rd/'progress.json') if (rd/'progress.json').exists() else {};progress.update(fail);save(rd/'progress.json',progress)
        traceback.print_exc();sys.exit(1)
