"""Same greedy matching/AP inputs as the existing evaluator, with cached IoUs."""
import numpy as np

def prepare(records,predictions,class_id):
    ground_truth={};total=0;selected={r['image_key'] for r in records}
    for r in records:
        gt=np.asarray([o['xyxy'] for o in r['ground_truth'] if o['class_id']==class_id],dtype=np.float64).reshape(-1,4)
        ground_truth[r['image_key']]=gt;total+=len(gt)
    flat=[]
    for k in selected:
        for p in predictions[k]:
            if p['class_id']==class_id:flat.append((k,p))
    flat.sort(key=lambda item:item[1]['score'],reverse=True)
    ious=[];orders=[]
    from tools.eval.eval_teacher_ship_external import box_iou_one_to_many
    for key,p in flat:
        v=box_iou_one_to_many(np.asarray(p['xyxy'],dtype=np.float64),ground_truth[key])
        ious.append(v);orders.append(np.argsort(-v))
    return {'flat':flat,'ious':ious,'orders':orders,'total_gt':total,'selected':selected}

def match(prepared,threshold,minimum_score=None):
    flat=prepared['flat'];used={k:set() for k in prepared['selected']}
    selected=[i for i,(_,p) in enumerate(flat) if minimum_score is None or p['score']>=minimum_score]
    tp=np.zeros(len(selected),dtype=np.float64);fp=np.ones(len(selected),dtype=np.float64);scores=np.empty(len(selected),dtype=np.float64)
    for outidx,idx in enumerate(selected):
        key,p=flat[idx];scores[outidx]=p['score'];ious=prepared['ious'][idx]
        for gt in prepared['orders'][idx]:
            if ious[gt]<threshold:break
            if int(gt) not in used[key]:
                used[key].add(int(gt));tp[outidx]=1.;fp[outidx]=0.;break
    n=int(tp.sum());return {'total_gt':prepared['total_gt'],'prediction_count':len(selected),'tp':n,'fp':int(fp.sum()),'fn':prepared['total_gt']-n,
                           'true_positive':tp,'false_positive':fp,'scores':scores}

def validate(root):
    from pathlib import Path
    import json,time
    from tools.eval.eval_late_fusion_ship_bridge import match_class,compute_ap
    root=Path(root);man=json.loads((root/'experiment_manifest.json').read_text());rr=json.loads((Path(man['derived_root'])/'val_manifest.json').read_text())
    pred=json.loads((root/'runs/A_seed42/best_val_predictions.json').read_text());tests=[];start=time.time()
    for mod in ['rgb','sar']:
        records=[{'image_key':r['key'],'ground_truth':[{'class_id':0,'xyxy':o['xyxy']} for o in r['harbor']]} for r in rr if r['modality']==mod]
        cache=prepare(records,pred,0)
        for iou in np.linspace(.5,.95,10):
            ref=match_class(records,pred,0,float(iou));new=match(cache,float(iou))
            for field in ['total_gt','prediction_count','tp','fp','fn']:assert ref[field]==new[field]
            for field in ['true_positive','false_positive','scores']:assert np.array_equal(ref[field],new[field]),(mod,iou,field)
            assert compute_ap(ref['total_gt'],ref['true_positive'],ref['false_positive'])==compute_ap(new['total_gt'],new['true_positive'],new['false_positive'])
        ref=match_class(records,pred,0,.5,.25);new=match(cache,.5,.25)
        for field in ['true_positive','false_positive','scores']:assert np.array_equal(ref[field],new[field])
        tests.append({'modality':mod,'candidates':len(cache['flat']),'all_10_AP_match_arrays_bitwise_equal':True,'fixed_threshold_arrays_equal':True})
    return {'passed':True,'tests':tests,'seconds':time.time()-start,'scope':'Arithmetic and iteration order preserved; cache pairwise IoUs once instead of recalculating at each IoU threshold.'}
