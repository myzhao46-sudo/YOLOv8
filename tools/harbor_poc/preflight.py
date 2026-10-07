# -*- coding: utf-8 -*-
import argparse,copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.harbor_poc.core import *

def main(root):
    setup(root);man=read(root/'experiment_manifest.json');derived=Path(man['derived_root']);seed_all(42)
    cfg={'config_id':'A','seed':42,'imgsz':640,'batch':4,'workers':0,'gradient_accumulation':4,'effective_batch':16,'nbs':16,
      'optimizer':'AdamW','lr0':0.0001,'lrf':0.1,'warmup_epochs':3,'weight_decay':0.0005,'betas':[.9,.999],
      'epochs':150,'patience':30,'precision':'FP32','amp':False,'scheduler':'3-epoch linear warmup then cosine to lrf',
      'gradient_clip':10.0,'loss_gains':{'box':7.5,'cls':.5,'dfl':1.5},'loss_normalization':'official loss.sum()/micro_batch/accumulation',
      'rgb_repeat':4,'sar_repeat':1,'sampling':'each unique image appears every epoch; RGB repeated4, SAR1; fixed seeded shuffle',
      'augmentation':{'horizontal_flip':.5,'vertical_flip':.5,'mosaic':0,'mixup':0,'hsv':0,'scale_jitter':0,'rotation':0},
      'expanded_freeze':False,'max_formal_configurations':3,'ema':{'decay':.9999,'tau_steps':2000,'fixed_buffers':True},
      'selection':'equal-weight RGB/SAR validation harbor AP50-95; earliest epoch wins exact ties',
      'second_seed_policy':'After configuration selection and before ANY test inference, train same chosen config seed43; use validation replication to describe consistency; recommend seed42 main model, never choose seed by test.',
      'negative_images':0,'initial_weights':str(root/'initialization/E0.pt')}
    save(root/'config_A_seed42.json',cfg)
    protocol={'locked_before_training':True,'dataset_scope':'harbor-positive subset, no empty-image negatives; within-scene FP only',
      'manifest_hashes':read(derived/'manifest_hashes.json'),'data_summary':read(derived/'data_summary.json'),
      'mapping':{'B_harbor_source':1,'E_harbor':0,'joint_ship':0,'joint_harbor':1},'imgsz':640,
      'preprocessing':'existing eval_teacher_ship_external.make_letterbox_rgb: PIL RGB bilinear square letterbox114, scale/255, no EXIF rotation or TTA',
      'harbor_candidate_extraction':'B selects source class1 probability BEFORE per-class NMS; E selects class0. No inter-class argmax competition for harbor. Identical fixed response protocol for B/E0/E.',
      'candidate_conf':.001,'report_conf':.25,'nms_iou':.7,'max_det_per_class':300,'report_match_iou':.5,'AP_iou':list(np.linspace(.5,.95,10)),
      'AP_implementation':'existing eval_late_fusion_ship_bridge.match_class + compute_ap (score-ordered greedy unmatched GT,101-point precision-envelope interpolation/trapezoid); class names supplied separately as harbor',
      'AP_source_sha256':sha(WORK/'tools/eval/eval_late_fusion_ship_bridge.py'),'calibration':'none; raw original scores only',
      'primary_selection':cfg['selection'],'seed_replication_policy':cfg['second_seed_policy'],'test_use':'B/E0/selected seed42 E only after configuration+epoch selection locked; no test tuning',
      'ship_combination':'B native3class NMS at .001,.7,global300 then select original ship0; union untouched ship list with independently capped E harbor list. No cross-class suppression/shared final cap.',
      'limits':'No SAR joint GT claims; no negative-scene coverage; thermal infrared untouched; upstream training overlap unknown.'}
    save(root/'evaluation_protocol.json',protocol)
    base=load_base(man);expert,inherit=detection_from_base(base);save(root/'initialization/inheritance_manifest.json',inherit)
    fm=freeze(expert);save(root/'freeze_manifest.json',fm)
    (root/'initialization').mkdir(exist_ok=True)
    torch.save({'model':copy.deepcopy(expert).cpu(),'ema':None,'epoch':-1,'train_args':expert.args,'version':'8.4.32','source_sha256':man['base_sha256']},root/'initialization/E0.pt')
    reload,_=load_expert(root/'initialization/E0.pt');val=read(derived/'val_manifest.json');ds=HarborDataset(val,640)
    base=base.cuda();expert=expert.cuda();reload=reload.cuda();parity=[]
    from tools.eval.eval_teacher_ship_external import raw_yoloeseg_fused_forward
    with torch.inference_mode():
        for idx in [0,1,len(val)-2,len(val)-1]:
            x=ds[idx][0].unsqueeze(0).cuda().float()/255
            b=raw_yoloeseg_fused_forward(base,x)[0];e=expert(x)[0];r=reload(x)[0];target=torch.cat((b[:,:4],b[:,5:6]),1)
            diff=float((target-e).abs().max());rd=float((r-e).abs().max());assert diff<=.001 and rd==0,(diff,rd)
            parity.append({'image':val[idx]['key'],'max_abs_raw_B_harbor_E0':diff,'max_abs_reload':rd})
    save(root/'initialization/parity.json',{'raw_checks':parity,'class_channel_selection':'B1->E0','passed':True})
    del reload;expert=expert.cpu();torch.cuda.empty_cache()
    mb,pb=evaluate_to(base,val,cfg,root/'evaluations/B_val',base=True,dataset=ds);base=base.cpu();torch.cuda.empty_cache()
    me,pe=evaluate_to(expert,val,cfg,root/'evaluations/E0_val',dataset=ds)
    save(root/'initialization/validation_equivalence.json',{'B':mb,'E0':me,'raw_parity':parity})
    print('BASELINE',json.dumps({'B':mb,'E0':me,'frozen_fraction':fm['frozen_fraction'],'trainable_parameters':fm['trainable_parameters']},indent=2),flush=True)
    man['status']='baseline_evaluated_ready_for_short_training';man['E0']=str(root/'initialization/E0.pt');man['E0_sha256']=sha(root/'initialization/E0.pt');save(root/'experiment_manifest.json',man)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
