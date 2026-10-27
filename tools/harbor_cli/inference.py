"""Complete configurable validation data loading with unchanged candidate rules."""
import torch
import torchvision
from pathlib import Path
from tools.harbor_poc.core import read
from tools.harbor_cli.data import HarborDataset,get_loader

_CONFIG=None
def configure(cfg):
    global _CONFIG
    _CONFIG=cfg

def get_preds(model,records,size,batch,base=False,dataset=None):
    if _CONFIG is None:raise RuntimeError('Validation loader config not set')
    cfg=_CONFIG
    from tools.eval.eval_teacher_ship_external import raw_yoloeseg_fused_forward,unletterbox_xyxy
    from ultralytics.utils.nms import non_max_suppression
    own=dataset is None
    dataset=dataset or HarborDataset(records,size,cache=cfg['cache_images'],seed=cfg['seed'])
    loader=get_loader(dataset,cfg['val_batch_size'],cfg['val_workers'],cfg['prefetch_factor'],cfg['persistent_workers'],cfg['pin_memory'])
    predictions={r['key']:[] for r in records};seen=[];model.eval().cuda()
    try:
        with torch.inference_mode():
            for item in loader:
                x=item['img'].cuda(non_blocking=cfg['pin_memory']).float()/255
                raw=raw_yoloeseg_fused_forward(model,x)[0] if base else model(x)[0]
                harbor=torch.cat((raw[:,:4],raw[:,5:6]),1) if base else raw
                output=non_max_suppression(harbor.clone(),conf_thres=.001,iou_thres=.7,max_det=300,nc=1,max_time_img=float('inf'))
                for j,idx in enumerate(item['indices']):
                    r=records[idx];seen.append(r['key']);scale=min(size/r['width'],size/r['height'])
                    meta={'orig_w':r['width'],'orig_h':r['height'],'scale':scale,'pad_x':(size-round(r['width']*scale))/2,'pad_y':(size-round(r['height']*scale))/2}
                    predictions[r['key']]=[{'xyxy':unletterbox_xyxy(p[:4],meta),'score':float(p[4]),'class_id':0} for p in output[j].cpu().numpy()]
        assert len(seen)==len(records) and set(seen)==set(predictions)
    finally:
        if own:dataset.close()
    return predictions,{}
