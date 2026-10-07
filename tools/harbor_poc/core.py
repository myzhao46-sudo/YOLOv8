# -*- coding: utf-8 -*-
"""Detection-only harbor POC built on pinned local Ultralytics primitives."""
import copy,hashlib,json,math,os,random,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image

WORK=Path(__file__).resolve().parents[2]
def setup(root):
    os.environ['YOLO_CONFIG_DIR']=str(Path(root)/'runtime_settings');os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_OFFLINE']='true'
    sys.path.insert(0,str(WORK))
    from tools.clean_pipeline.runtime import setup_runtime
    setup_runtime()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf8');tmp.replace(p)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def load_base(man):
    ck=torch.load(man['base_weights'],map_location='cpu',weights_only=False);m=(ck.get('ema') or ck['model']).float().eval()
    assert m.names=={0:'ship',1:'harbor',2:'tank'} and type(m).__name__=='YOLOESegModel'
    assert m.model[-1].is_fused and all(b[-1].out_channels==3 for b in m.model[-1].cv3)
    for p in m.parameters():p.requires_grad_(False)
    return m
def detection_from_base(base):
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.head import Detect
    m=copy.deepcopy(base);before=m.state_dict();m.__class__=DetectionModel;h=m.model[-1];h.__class__=Detect
    removed=[]
    for name in ['cv4','cv5','proto','savpe','reprta']:
        if hasattr(h,name):removed.append(name);delattr(h,name)
    for classifier in h.cv3:
        old=classifier[-1];new=torch.nn.Conv2d(old.in_channels,1,1,bias=True)
        with torch.no_grad():new.weight.copy_(old.weight[1:2]);new.bias.copy_(old.bias[1:2])
        classifier[-1]=new
    h.nc=1;h.no=4*h.reg_max+1;h.end2end=False;h.shape=None;h.type='ultralytics.nn.modules.head.Detect'
    m.names={0:'harbor'};m.task='detect';m.yaml=copy.deepcopy(base.yaml);m.yaml['nc']=1;m.yaml['head'][-1][-2]='Detect';m.yaml['head'][-1][-1]=['nc'];m.yaml['scale']='s'
    m.args={'task':'detect','imgsz':640,'single_cls':True,'model':'B_harbor_inherited','data':'harbor.yaml'}
    for name in ['pe','criterion','text_model','clip_model']:
        if hasattr(m,name):delattr(m,name)
    mapping=[];after=m.state_dict()
    for k,v in before.items():
        if k not in after:status='removed_unused_segmentation_or_prompt_module'
        elif after[k].shape==v.shape:status='copied_exact';assert torch.equal(after[k],v)
        else:status='harbor_row_1_to_row_0';assert torch.equal(after[k],v[1:2])
        mapping.append({'tensor':k,'original_shape':list(v.shape),'new_shape':list(after[k].shape) if k in after else None,'original_numel':v.numel(),'new_numel':after[k].numel() if k in after else 0,'status':status})
    report={'source_model':type(base).__name__,'source_head':type(base.model[-1]).__name__,'target_model':type(m).__name__,
        'target_head':type(h).__name__,'removed_modules':removed,'new_random_parameters':0,'semantics':'Already fused YOLOE harbor classifier row1 copied to detector row0 at all three scales. No new text encoder/embedding; fused coefficients preserve upstream harbor semantics.',
        'source_parameters':sum(p.numel() for p in base.parameters()),'target_parameters':sum(p.numel() for p in m.parameters()),'tensor_mapping':mapping}
    return m.eval(),report
def freeze(m,expanded=False):
    train_blocks={21,22} if not expanded else {18,19,21,22}
    for name,p in m.named_parameters():p.requires_grad_(int(name.split('.')[1]) in train_blocks and '.dfl.' not in name)
    entries=[{'name':n,'shape':list(p.shape),'parameters':p.numel(),'trainable':p.requires_grad} for n,p in m.named_parameters()]
    total=sum(x['parameters'] for x in entries);train=sum(x['parameters'] for x in entries if x['trainable'])
    return {'train_blocks':sorted(train_blocks),'reason':'Train final P5 neck C2f fusion and inherited class/box heads; freeze entire backbone and earlier multiscale neck. Fixed DFL projection.',
      'total_parameters':total,'trainable_parameters':train,'frozen_fraction':1-train/total,
      'bn_policy':'ALL BatchNorm running_mean/running_var/num_batches_tracked fixed in eval mode, even in trainable blocks; affine trainable only in selected blocks. Reassert after every model.train().',
      'ema_policy':'Average only trainable parameters; frozen params and all buffers copied unchanged, preventing equal-value EMA roundoff drift.', 'parameters':entries,
      'modules':[{'name':n,'type':type(b).__name__,'direct_parameters':sum(p.numel() for p in b.parameters(recurse=False)),
                  'direct_trainable_parameters':sum(p.numel() for p in b.parameters(recurse=False) if p.requires_grad)} for n,b in m.named_modules()]}
def train_mode(m):
    m.train()
    for block in m.model:
        if not any(p.requires_grad for p in block.parameters()):block.eval()
    for b in m.modules():
        if isinstance(b,torch.nn.modules.batchnorm._BatchNorm):b.eval()
def frozen_digest(m):
    frozen={n for n,p in m.named_parameters() if not p.requires_grad};buffers=set(dict(m.named_buffers()));h=hashlib.sha256()
    for n,v in m.state_dict().items():
        if n in frozen or n in buffers:h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def update_ema(ema,model,steps):
    decay=.9999*(1-math.exp(-steps/2000));ep=dict(ema.named_parameters())
    with torch.no_grad():
        for n,p in model.named_parameters():
            if p.requires_grad:ep[n].mul_(decay).add_(p,alpha=1-decay)
        eb=dict(ema.named_buffers())
        for n,b in model.named_buffers():eb[n].copy_(b)
def load_expert(path,ema=True):
    ck=torch.load(path,map_location='cpu',weights_only=False);m=ck.get('ema') if ema else None
    if m is None:m=ck['model']
    return m.float().eval(),ck

class HarborDataset(torch.utils.data.Dataset):
    def __init__(self,records,size,augment=False):
        self.records=records;self.size=size;self.augment=augment
        from tools.eval.eval_teacher_ship_external import make_letterbox_rgb
        self.images=[];self.boxes=[]
        for r in records:
            im,meta=make_letterbox_rgb(Path(r['image']),size);self.images.append(im)
            # All selected source images are square; still apply actual letterbox explicitly.
            boxes=[]
            for o in r['harbor']:
                x1,y1,x2,y2=o['xyxy'];sc=meta['scale'];px=meta['pad_x'];py=meta['pad_y']
                boxes.append([((x1+x2)*sc/2+px)/size,((y1+y2)*sc/2+py)/size,(x2-x1)*sc/size,(y2-y1)*sc/size])
            self.boxes.append(np.asarray(boxes,dtype=np.float32).reshape(-1,4))
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        im=self.images[i];boxes=self.boxes[i].copy()
        if self.augment:
            if random.random()<.5:im=im[:,::-1];boxes[:,0]=1-boxes[:,0]
            if random.random()<.5:im=im[::-1];boxes[:,1]=1-boxes[:,1]
        return torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1))),torch.from_numpy(boxes),i
def collate(batch):
    im,bb,ii=zip(*batch)
    return {'img':torch.stack(im),'bboxes':torch.cat(bb),'cls':torch.zeros(sum(len(b) for b in bb),1),
            'batch_idx':torch.cat([torch.full((len(b),),j) for j,b in enumerate(bb)]),'indices':list(ii)}

def get_preds(model,records,size,batch,base=False,dataset=None,ship_native=False):
    from tools.eval.eval_teacher_ship_external import raw_yoloeseg_fused_forward,unletterbox_xyxy
    from ultralytics.utils.nms import non_max_suppression
    dataset=dataset or HarborDataset(records,size);loader=torch.utils.data.DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0,collate_fn=collate)
    pred={r['key']:[] for r in records};ships={r['key']:[] for r in records};model.eval().to('cuda')
    with torch.inference_mode():
        for item in loader:
            x=item['img'].to('cuda',non_blocking=True).float()/255
            raw=raw_yoloeseg_fused_forward(model,x)[0] if base else model(x)[0]
            harbor=torch.cat((raw[:,:4],raw[:,5:6]),1) if base else raw
            outputs=non_max_suppression(harbor.clone(),conf_thres=.001,iou_thres=.7,max_det=300,nc=1)
            native=non_max_suppression(raw.clone(),conf_thres=.001,iou_thres=.7,max_det=300,nc=3) if ship_native and base else None
            for j,idx in enumerate(item['indices']):
                r=records[idx];sc=min(size/r['width'],size/r['height']);meta={'orig_w':r['width'],'orig_h':r['height'],'scale':sc,'pad_x':(size-round(r['width']*sc))/2,'pad_y':(size-round(r['height']*sc))/2}
                for row in outputs[j].cpu().numpy():pred[r['key']].append({'xyxy':unletterbox_xyxy(row[:4],meta),'score':float(row[4]),'class_id':0})
                if native is not None:
                    for row in native[j].cpu().numpy():
                        if int(row[5])==0:ships[r['key']].append({'xyxy':unletterbox_xyxy(row[:4],meta),'score':float(row[4]),'class_id':0})
    return pred,ships
def metrics(records,pred,details=False):
    from tools.eval.eval_late_fusion_ship_bridge import match_class,compute_ap
    from tools.eval.eval_teacher_ship_external import box_iou_one_to_many
    result={}
    for mod in ['rgb','sar']:
        rr=[r for r in records if r['modality']==mod]
        view=[{'image_key':r['key'],'ground_truth':[{'class_id':0,'xyxy':o['xyxy']} for o in r['harbor']]} for r in rr]
        aps=[]
        for t in np.linspace(.5,.95,10):
            mm=match_class(view,pred,0,float(t));aps.append(compute_ap(mm['total_gt'],mm['true_positive'],mm['false_positive']))
        fixed=match_class(view,pred,0,.5,minimum_score=.25)
        result[mod]={'images':len(rr),'gt':fixed['total_gt'],'ap50':aps[0],'ap50_95':float(np.mean(aps)),
            'precision':fixed['tp']/max(fixed['tp']+fixed['fp'],1),'recall':fixed['tp']/max(fixed['total_gt'],1),
            'tp':fixed['tp'],'fp':fixed['fp'],'fn':fixed['fn'],'candidates':sum(len(pred[r['key']]) for r in rr),'ap_by_iou':aps}
    result['macro_ap50_95']=float(np.mean([result[x]['ap50_95'] for x in ['rgb','sar']]))
    if details:
        matches={}
        for r in records:
            gt=np.asarray([o['xyxy'] for o in r['harbor']]).reshape(-1,4);used=set();rows=[]
            for p in sorted(pred[r['key']],key=lambda p:-p['score']):
                ious=box_iou_one_to_many(np.asarray(p['xyxy']),gt);matched=None
                for idx in np.argsort(-ious):
                    if ious[idx]<.5:break
                    if int(idx) not in used:matched=int(idx);used.add(matched);break
                rows.append({**p,'tp_iou50':matched is not None,'gt_index':matched,'at_report_threshold':p['score']>=.25})
            matches[r['key']]={'predictions':rows,'gt':gt.tolist(),'unmatched_gt_at_candidate_floor':[j for j in range(len(gt)) if j not in used]}
        result['per_image_matches']=matches
    return result
def evaluate_to(model,records,cfg,out,base=False,dataset=None,ship_native=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.time()
    pred,ship=get_preds(model,records,cfg['imgsz'],cfg['batch'],base,dataset,ship_native)
    result=metrics(records,pred,True);matches=result.pop('per_image_matches');result['seconds']=time.time()-start
    save(out/'metrics.json',result);save(out/'predictions.json',pred);save(out/'matches_iou50.json',matches)
    if ship_native:save(out/'B_native_ship_predictions.json',ship)
    return result,pred
