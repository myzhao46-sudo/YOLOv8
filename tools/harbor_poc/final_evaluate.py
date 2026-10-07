# -*- coding: utf-8 -*-
"""Locked final test + ship-preserving union; requires explicit pre-test selection file."""
import argparse,copy,datetime,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.harbor_poc.core import *
from PIL import ImageDraw,ImageFont

def visualizations(records,pred_b,pred_e,folder):
    folder.mkdir(parents=True,exist_ok=True)
    for mod in ['rgb','sar']:
        scores=[]
        for r in records:
            if r['modality']!=mod:continue
            # Per-image diagnostic only, not selection of evaluation membership or model.
            from tools.eval.eval_late_fusion_ship_bridge import match_class
            v=[{'image_key':r['key'],'ground_truth':[{'class_id':0,'xyxy':o['xyxy']} for o in r['harbor']]}]
            b=match_class(v,pred_b,0,.5,.25);e=match_class(v,pred_e,0,.5,.25)
            scores.append((e['tp']-b['tp'],e['fp']-b['fp'],r))
        chosen=[('recall_gain',max(scores,key=lambda x:(x[0],-x[1]))[2]),('false_positive_cost',max(scores,key=lambda x:(x[1],-x[0]))[2])]
        for kind,r in chosen:
            with Image.open(r['image']) as im:orig=im.convert('RGB')
            panels=[]
            for label,pred in [('B',pred_b),('E',pred_e)]:
                im=orig.copy();draw=ImageDraw.Draw(im)
                for o in r['harbor']:draw.rectangle(o['xyxy'],outline=(0,255,0),width=2)
                for p in pred[r['key']]:
                    if p['score']<.25:continue
                    b=p['xyxy'];draw.rectangle(b,outline=(255,50,0),width=2);draw.text((b[0],max(0,b[1]-10)),f"{p['score']:.2f}",fill=(255,255,0))
                im.thumbnail((800,800));panels.append(im)
            canvas=Image.new('RGB',(panels[0].width*2,panels[0].height+24),(20,20,20));canvas.paste(panels[0],(0,24));canvas.paste(panels[1],(panels[0].width,24));draw=ImageDraw.Draw(canvas)
            draw.text((5,5),'B | green=GT red=prediction >=0.25',fill='white');draw.text((panels[0].width+5,5),'E | same input and threshold',fill='white')
            canvas.save(folder/f"{mod}_{kind}_{r['key']}.png")

def main(root):
    setup(root);man=read(root/'experiment_manifest.json');selection=read(root/'final_selection.json');cfg=read(root/'runs'/selection['run_id']/'effective_config.json')
    assert selection['test_not_used_for_selection'] is True
    assert sha(selection['checkpoint'])==selection['checkpoint_sha256']
    out=root/'final_test';out.mkdir(exist_ok=False)
    save(out/'started.json',{'time':datetime.datetime.now().astimezone().isoformat(),'selection':selection,'pid':os.getpid()})
    records=read(Path(man['derived_root'])/'test_manifest.json');ds=HarborDataset(records,cfg['imgsz'])
    base=load_base(man);mb,pb=evaluate_to(base,records,cfg,out/'B',base=True,dataset=ds,ship_native=True);del base;torch.cuda.empty_cache()
    e0,_=load_expert(root/'initialization/E0.pt');m0,p0=evaluate_to(e0,records,cfg,out/'E0',dataset=ds);del e0;torch.cuda.empty_cache()
    expert,ck=load_expert(selection['checkpoint']);me,pe=evaluate_to(expert,records,cfg,out/'E',dataset=ds)
    ship=read(out/'B/B_native_ship_predictions.json');combined={}
    for r in records:combined[r['key']]=copy.deepcopy(ship[r['key']])+[{**p,'class_id':1} for p in pe[r['key']]]
    # No second NMS for already class-NMSed lists; exactly preserves B's entire original ship path.
    same=all([p for p in combined[k] if p['class_id']==0]==ship[k] for k in ship);assert same
    save(out/'B_ship_E_harbor_union.json',combined)
    save(out/'ship_preservation.json',{'all_inputs':len(records),'unchanged':same,'max_score_delta':0,'max_box_delta':0,'changed_ship_candidates':0,
      'class_budget':'300 independently per head; no shared cap after union','metric_change_from_union':'exactly zero for any fixed ship truth because same candidates; strict SAR ship GT evaluation not attempted',
      'joint_truth_limit':'Original DIOR ship labels retained but not certified complete/clean for formal joint AP; SAR ship labels incomplete. This checks prediction preservation only.'})
    export_dir=root/'recommended';export_dir.mkdir(exist_ok=True)
    export=export_dir/'harbor_expert.pt'
    torch.save({'model':cpu_model(expert),'ema':None,'epoch':ck['epoch'],'train_args':ck['train_args'],'version':'8.4.32','source_checkpoint_sha256':selection['checkpoint_sha256']},export)
    # Actual standalone export reload and fixed test evaluation, compare metrics and every candidate.
    del expert;torch.cuda.empty_cache();reloaded,_=load_expert(export);mr,pr=evaluate_to(reloaded,records,cfg,out/'reloaded_export',dataset=ds)
    assert all(pr[k]==pe[k] for k in pe),'export predictions differ'
    delta={mod:{metric:100*(me[mod][metric]-mb[mod][metric]) for metric in ['ap50','ap50_95','precision','recall']} for mod in ['rgb','sar']}
    for mod in delta:delta[mod].update(fp_change=me[mod]['fp']-mb[mod]['fp'],tp_change=me[mod]['tp']-mb[mod]['tp'],fn_change=me[mod]['fn']-mb[mod]['fn'])
    result={'B':mb,'E0':m0,'E':me,'delta_percentage_points':delta,'recommended_weights':str(export),'recommended_sha256':sha(export),
      'structure':'DetectionModel / Detect; upstream YOLOE-v8s backbone + neck + fused harbor classification row, no masks',
      'classes':{'0':'harbor'},'modalities':['rgb','sar'],'reload_predictions_identical':True,'B_sha256_unchanged':sha(man['base_weights'])==man['base_sha256'],
      'status':'complete','selection':selection}
    assert result['B_sha256_unchanged'];save(out/'comparison.json',result);visualizations(records,pb,pe,out/'examples')
    print(json.dumps(result,indent=2),flush=True)
def cpu_model(m):
    m=copy.deepcopy(m).cpu().eval()
    if hasattr(m,'criterion'):delattr(m,'criterion')
    return m
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
