# -*- coding: utf-8 -*-
"""Conservative supervised box fine-tuning with explicit BN freeze and resumable state."""
import argparse,copy,csv,datetime,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.harbor_poc.core import *

def timestamp():return datetime.datetime.now().astimezone().isoformat()
def handoff(root,run,status,detail):
    man=read(root/'experiment_manifest.json');run_dir=root/'runs'/run
    resume=f'& "{sys.executable}" -B -X utf8 "{WORK}/tools/harbor_poc/train.py" --root "{root}" --run "{run}" --config "{run_dir}/effective_config.json" --resume'
    (root/'HANDOFF.md').write_text('# Harbor expert POC — current handoff\n\n'+f'Updated: {timestamp()}\nStatus: **{status}**\n\n'+
      '\n'.join(f'- {k}: `{man[k]}`' for k in ['experiment_root','worktree','branch','base_commit','python','base_weights','base_sha256','derived_root'])+
      f'\n- run_id: `{run}`\n- PID: `{os.getpid()}`\n- run directory: `{run_dir}`\n- checkpoint: `{run_dir}/weights/last.pt`\n- progress: `{run_dir}/progress.json`\n\n'+
      json.dumps(detail,ensure_ascii=False,indent=2)+'\n\n## Resume if process has stopped\n\nFirst inspect recorded PID and progress. Never start a duplicate process. Use resume only with identical config; new config is a new run from a recorded parent.\n\n```powershell\n'+resume+'\n```\n\n'+
      'Completed: audit reused, fixed derived manifests and evaluation protocol, B/E0 validation and parity checks.\nNext: finish config A, assess validation evidence, maximum 3 configurations; chosen config second-seed validation before test. Test inference has NOT run unless final_selection.json/test outputs explicitly exist. Thermal IR/gating/distillation not implemented.\n',encoding='utf8')
    note=root/'LOW_INTERACTION.md'
    if note.exists():
        with (root/'HANDOFF.md').open('a',encoding='utf8') as f:f.write('\n\n'+note.read_text(encoding='utf8'))

def cpu_copy(m):
    x=copy.deepcopy(m).cpu().eval()
    if hasattr(x,'criterion'):delattr(x,'criterion')
    return x
def checkpoint(path,m,ema,opt,scheduler,epoch,steps,cfg,history,best,best_epoch,bad,root):
    payload={'model':cpu_copy(m),'ema':cpu_copy(ema),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
      'epoch':epoch,'updates':steps,'config':cfg,'history':history,'best_fitness':best,'best_epoch':best_epoch,'bad_epochs':bad,
      'train_args':{'task':'detect','imgsz':cfg['imgsz'],'data':str(Path(read(root/'experiment_manifest.json')['derived_root'])/'harbor.yaml'),'single_cls':True},
      'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},
      'scaler':None,'precision':'FP32','version':'8.4.32','date':timestamp(),'parent':cfg['initial_weights'],
      'base_sha256':read(root/'experiment_manifest.json')['base_sha256'],'protocol_sha256':sha(root/'evaluation_protocol.json'),
      'data_hashes':read(Path(read(root/'experiment_manifest.json')['derived_root'])/'manifest_hashes.json')}
    path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)

def make_opt(m,cfg):
    decay=[];nodecay=[]
    for n,p in m.named_parameters():
        if p.requires_grad:(nodecay if p.ndim<2 or n.endswith('.bias') else decay).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':cfg['weight_decay']},{'params':nodecay,'weight_decay':0}],lr=cfg['lr0'],betas=tuple(cfg['betas']))
def make_scheduler(opt,cfg):
    def factor(epoch):
        if epoch<cfg['warmup_epochs']:return .1+.9*(epoch+1)/cfg['warmup_epochs']
        progress=(epoch-cfg['warmup_epochs'])/max(cfg['epochs']-cfg['warmup_epochs']-1,1)
        return cfg['lrf']+(1-cfg['lrf'])*(1+math.cos(math.pi*min(progress,1)))/2
    return torch.optim.lr_scheduler.LambdaLR(opt,factor)
def make_loss(m,cfg):
    from ultralytics.utils.loss import v8DetectionLoss
    old=m.args;m.args=SimpleNamespace(**cfg['loss_gains']);criterion=v8DetectionLoss(m);m.args=old
    return criterion
def optimizer_step(m,ema,opt,steps):
    grads=[p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
    if not grads or any(not torch.isfinite(g).all() for g in grads):raise FloatingPointError('Missing or nonfinite gradients')
    maximum=max(float(g.abs().max()) for g in grads)
    if maximum<=0:raise RuntimeError('All gradients zero')
    norm=float(torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],10.0))
    opt.step();opt.zero_grad(set_to_none=True);steps+=1;update_ema(ema,m,steps)
    return steps,maximum,norm
def assert_freeze(m,ema,before):
    raw=frozen_digest(m);em=frozen_digest(ema)
    if raw!=before or em!=before:raise RuntimeError(f'Frozen model/EMA state changed: {before} {raw} {em}')
    for b in m.modules():
        if isinstance(b,torch.nn.modules.batchnorm._BatchNorm) and b.training:raise RuntimeError('BN training mode unexpectedly enabled')
    return {'model_frozen_sha256':raw,'ema_frozen_sha256':em,'passed':True}

def short_check(root,cfg,train_records):
    target=root/'short_training';target.mkdir(exist_ok=False);seed_all(cfg['seed'])
    chosen=[]
    for mod in ['rgb','sar']:chosen.extend([r for r in train_records if r['modality']==mod][:16])
    ds=HarborDataset(chosen,cfg['imgsz'],True);loader=torch.utils.data.DataLoader(ds,batch_size=cfg['batch'],shuffle=False,num_workers=0,collate_fn=collate)
    m,_=load_expert(cfg['initial_weights']);fm=freeze(m,cfg['expanded_freeze']);m=m.cuda();ema=copy.deepcopy(m).eval()
    before=frozen_digest(m);trainable={n:p.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad}
    opt=make_opt(m,cfg);sched=make_scheduler(opt,cfg);criterion=make_loss(m,cfg);train_mode(m);steps=0;losses=[];gn=[]
    for item in loader:
        item['img']=item['img'].cuda().float()/255
        vec,det=criterion(m(item['img']),item);loss=vec.sum()/len(item['img'])
        if not torch.isfinite(loss):raise FloatingPointError('Short-check nonfinite loss')
        loss.backward();steps,g,norm=optimizer_step(m,ema,opt,steps);gn.append(g);losses.append(det.detach().cpu().tolist())
    changes={n:float((p.detach().cpu()-trainable[n]).abs().max()) for n,p in m.named_parameters() if n in trainable}
    assert max(changes.values())>0
    check=assert_freeze(m,ema,before);checkpoint(target/'last.pt',m,ema,opt,sched,0,steps,cfg,[],0,0,0,root)
    restored,ck=load_expert(target/'last.pt',ema=False);freeze(restored,cfg['expanded_freeze']);restored=restored.cuda();ropt=make_opt(restored,cfg);ropt.load_state_dict(ck['optimizer']);rsched=make_scheduler(ropt,cfg);rsched.load_state_dict(ck['scheduler'])
    state_equal=all(torch.equal(v.cpu(),restored.state_dict()[k].cpu()) for k,v in m.state_dict().items());assert state_equal
    m.eval();restored.eval();x=ds[0][0].unsqueeze(0).cuda().float()/255
    with torch.no_grad():reload_error=float((m(x)[0]-restored(x)[0]).abs().max())
    assert reload_error==0
    # Compare the NEXT optimizer update after restoring full state; no augmentation/RNG involved in this paired check.
    item=next(iter(loader));item['img']=item['img'].cuda().float()/255
    for model,optimizer in [(m,opt),(restored,ropt)]:
        train_mode(model);optimizer.zero_grad(set_to_none=True);v,_=make_loss(model,cfg)(model(item['img']),item);(v.sum()/len(item['img'])).backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10);optimizer.step()
    next_error=max(float((v.cpu()-restored.state_dict()[k].cpu()).abs().max()) for k,v in m.state_dict().items());assert next_error<=1e-7,next_error
    result={'passed':True,'optimizer_steps':steps,'finite_loss':True,'nonzero_gradients':gn,'losses':losses,'parameter_max_deltas':changes,
      'frozen_validation':check,'reloaded_state_equal':state_equal,'reload_prediction_max_error':reload_error,'restored_next_optimizer_update_max_error':next_error,
      'formal_training_initializes_from':'E0, not this short-trained checkpoint','short_accumulation':1,'formal_accumulation':cfg['gradient_accumulation']}
    save(target/'update_validation.json',result);print('SHORT TRAINING PASSED',json.dumps({k:result[k] for k in ['passed','optimizer_steps','reload_prediction_max_error','restored_next_optimizer_update_max_error']}),flush=True)
    del m,ema,restored,opt,ropt,criterion;torch.cuda.empty_cache()

def run(root,run_id,cfg_path,resume=False):
    setup(root);cfg=read(cfg_path);seed_all(cfg['seed']);man=read(root/'experiment_manifest.json');derived=Path(man['derived_root']);run_dir=root/'runs'/run_id
    if not resume:run_dir.mkdir(parents=True,exist_ok=False)
    elif not run_dir.is_dir():raise FileNotFoundError(run_dir)
    save(run_dir/'effective_config.json',cfg);save(run_dir/'process.json',{'pid':os.getpid(),'start':timestamp(),'run_id':run_id,'resume':resume,'argv':sys.argv,'status':'running'})
    (run_dir/'code_diff.patch').write_text(subprocess.check_output(['git','diff','HEAD'],cwd=WORK,text=True,encoding='utf8'),encoding='utf8')
    train=read(derived/'train_manifest.json');val=read(derived/'val_manifest.json')
    if not resume and not (root/'short_training/update_validation.json').exists():
        handoff(root,run_id,'short_training',{'epoch':0});short_check(root,cfg,train)
    if not read(root/'short_training/update_validation.json')['passed']:raise RuntimeError('Short check failed')
    ck=None
    if resume:m,ck=load_expert(run_dir/'weights/last.pt',ema=False)
    else:m,_=load_expert(cfg['initial_weights'],ema=False)
    fm=freeze(m,cfg['expanded_freeze']);save(run_dir/'freeze_manifest.json',fm);m=m.cuda();ema=copy.deepcopy(m).eval()
    before=frozen_digest(m);trainable_before={n:p.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad}
    opt=make_opt(m,cfg);scheduler=make_scheduler(opt,cfg);criterion=make_loss(m,cfg)
    history=[];best=-1;best_epoch=0;bad=0;steps=0;start_epoch=0
    if ck:
        assert ck['config']==cfg,'resume config must be identical'
        assert ck['protocol_sha256']==sha(root/'evaluation_protocol.json'),'protocol changed'
        assert ck['data_hashes']==read(derived/'manifest_hashes.json'),'data changed'
        ema.load_state_dict(ck['ema'].state_dict());opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        history=ck['history'];best=ck['best_fitness'];best_epoch=ck['best_epoch'];bad=ck['bad_epochs'];steps=ck['updates'];start_epoch=ck['epoch']+1
        random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy']);torch.set_rng_state(ck['rng']['torch']);torch.cuda.set_rng_state_all(ck['rng']['cuda'])
    td=HarborDataset(train,cfg['imgsz'],True);vd=HarborDataset(val,cfg['imgsz'],False)
    indices=[i for i,r in enumerate(train) for _ in range(cfg['rgb_repeat'] if r['modality']=='rgb' else cfg['sar_repeat'])]
    save(run_dir/'sampling_manifest.json',{'unique_images':len(train),'epoch_images':len(indices),'rgb_presentations':sum(train[i]['modality']=='rgb' for i in indices),
       'sar_presentations':sum(train[i]['modality']=='sar' for i in indices),'indices_before_shuffle':indices,'shuffle_seed':'seed+epoch','gate_reserved_images':0})
    start=time.time();prior=history[-1]['total_seconds'] if history else 0;completed=False
    handoff(root,run_id,'formal_training_running',{'start_epoch':start_epoch,'planned_epochs':cfg['epochs'],'best_macro_ap50_95':best})
    for epoch in range(start_epoch,cfg['epochs']):
        t=time.time();torch.cuda.reset_peak_memory_stats();train_mode(m);opt.zero_grad(set_to_none=True);order=indices.copy();random.Random(cfg['seed']+epoch).shuffle(order)
        loader=torch.utils.data.DataLoader(td,batch_size=cfg['batch'],sampler=order,num_workers=cfg['workers'],collate_fn=collate,pin_memory=True)
        losses=[];grad_max=0.;norm_max=0.;epochsteps=0;lr=opt.param_groups[0]['lr'];modseen=Counter() if False else {'rgb':0,'sar':0}
        for bi,item in enumerate(loader):
            item['img']=item['img'].cuda(non_blocking=True).float()/255;vec,det=criterion(m(item['img']),item)
            group_start=(bi//cfg['gradient_accumulation'])*cfg['gradient_accumulation'];div=min(cfg['gradient_accumulation'],len(loader)-group_start)
            loss=vec.sum()/len(item['img'])/div
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss epoch{epoch+1} batch{bi}')
            loss.backward();losses.append(det.detach().cpu().tolist())
            for i in item['indices']:modseen[train[i]['modality']]+=1
            if (bi+1)%cfg['gradient_accumulation']==0 or bi+1==len(loader):
                steps,g,norm=optimizer_step(m,ema,opt,steps);epochsteps+=1;grad_max=max(grad_max,g);norm_max=max(norm_max,norm)
            if bi%100==0:save(run_dir/'heartbeat.json',{'pid':os.getpid(),'epoch_in_progress':epoch+1,'batch':bi+1,'batches':len(loader),'optimizer_steps':steps,'timestamp':timestamp(),'status':'training'})
        freeze_result=assert_freeze(m,ema,before)
        pred,_=get_preds(ema,val,cfg['imgsz'],cfg['batch'],dataset=vd);met=metrics(val,pred);fitness=met['macro_ap50_95']
        improved=fitness>best
        if improved:best=fitness;best_epoch=epoch+1;bad=0
        else:bad+=1
        scheduler.step()
        row={'epoch':epoch+1,'loss':np.mean(losses,axis=0).tolist(),'lr':lr,'metrics':met,'optimizer_steps':steps,'epoch_optimizer_steps':epochsteps,
          'gradient_max':grad_max,'gradient_norm_max_preclip':norm_max,'seconds':time.time()-t,'total_seconds':prior+time.time()-start,
          'gpu_max_allocated':torch.cuda.max_memory_allocated(),'gpu_max_reserved':torch.cuda.max_memory_reserved(),'best_epoch':best_epoch,'best_fitness':best,
          'patience_used':bad,'modality_presentations':modseen,'frozen_check':freeze_result}
        history.append(row);save(run_dir/'history.json',history)
        with (run_dir/'epochs.csv').open('w',newline='',encoding='utf-8-sig') as f:
            names=['epoch','lr','box_loss','cls_loss','dfl_loss','rgb_ap50','rgb_ap50_95','sar_ap50','sar_ap50_95','macro_ap50_95','rgb_fp','sar_fp','optimizer_steps','seconds','gpu_max_allocated']
            w=csv.DictWriter(f,fieldnames=names);w.writeheader()
            for h in history:w.writerow({'epoch':h['epoch'],'lr':h['lr'],'box_loss':h['loss'][0],'cls_loss':h['loss'][1],'dfl_loss':h['loss'][2],
              **{f'{mod}_{k}':h['metrics'][mod][k] for mod in ['rgb','sar'] for k in ['ap50','ap50_95','fp']},'macro_ap50_95':h['metrics']['macro_ap50_95'],
              **{k:h[k] for k in ['optimizer_steps','seconds','gpu_max_allocated']}})
        checkpoint(run_dir/'weights/last.pt',m,ema,opt,scheduler,epoch,steps,cfg,history,best,best_epoch,bad,root)
        if improved:
            import shutil
            shutil.copy2(run_dir/'weights/last.pt',run_dir/'weights/best.pt');save(run_dir/'best_val_predictions.json',pred);save(run_dir/'best_val_metrics.json',met)
        save(run_dir/'progress.json',{'status':'running','run_id':run_id,'pid':os.getpid(),'last_completed_epoch':epoch+1,'best_epoch':best_epoch,'best_fitness':best,'latest':row,'checkpoint':str(run_dir/'weights/last.pt')})
        handoff(root,run_id,'formal_training_running',{'last_completed_epoch':epoch+1,'best_epoch':best_epoch,'best_macro_ap50_95':best,'latest_metrics':met,'total_seconds':row['total_seconds']})
        print(f"EPOCH {epoch+1}/{cfg['epochs']} rgb_AP={met['rgb']['ap50_95']:.5f} sar_AP={met['sar']['ap50_95']:.5f} macro={fitness:.5f} best={best:.5f}@{best_epoch} FP={met['rgb']['fp']}/{met['sar']['fp']} loss={row['loss']} seconds={row['seconds']:.1f}",flush=True)
        if bad>=cfg['patience']:completed=True;reason='early_stopping';break
    else:completed=True;reason='epoch_limit'
    changes={n:float((p.detach().cpu()-trainable_before[n]).abs().max()) for n,p in m.named_parameters() if n in trainable_before}
    save(run_dir/'parameter_update_validation.json',{'frozen':assert_freeze(m,ema,before),'trainable_max_changes':changes,'optimizer_steps':steps,'passed':max(changes.values())>0})
    del m,ema,opt,criterion,td;torch.cuda.empty_cache()
    chosen,_=load_expert(run_dir/'weights/best.pt');result,_=evaluate_to(chosen,val,cfg,run_dir/'reloaded_best_val',dataset=vd)
    assert abs(result['macro_ap50_95']-best)<1e-9,(result['macro_ap50_95'],best)
    finish={'status':'completed' if completed else 'incomplete','termination':reason,'epochs':len(history),'best_epoch':best_epoch,'best_fitness':best,'metrics':result,
      'training_seconds':history[-1]['total_seconds'],'best_path':str(run_dir/'weights/best.pt'),'last_path':str(run_dir/'weights/last.pt'),'best_sha256':sha(run_dir/'weights/best.pt'),
      'last_sha256':sha(run_dir/'weights/last.pt'),'exit_code':0,'finished':timestamp()}
    save(run_dir/'result.json',finish);save(run_dir/'progress.json',finish);save(run_dir/'process.json',{'pid':os.getpid(),'status':'completed','exit_code':0,'finished':timestamp()})
    handoff(root,run_id,'formal_run_completed_validation_review_pending',finish);print('RUN COMPLETED',json.dumps(finish,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:run(a.root,a.run,a.config,a.resume)
    except BaseException as e:
        failure={'status':'failed','pid':os.getpid(),'exit_code':1,'error':repr(e),'traceback':traceback.format_exc(),'time':timestamp()}
        save(a.root/'runs'/a.run/'failure.json',failure);save(a.root/'runs'/a.run/'process.json',failure);handoff(a.root,a.run,'failed',failure)
        traceback.print_exc();sys.exit(1)
