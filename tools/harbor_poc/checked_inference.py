"""Complete single-class NMS evaluation, no wall-clock-dependent skipped images."""
from pathlib import Path
import torch
import torchvision  # Explicit fast backend for local Ultralytics NMS.
from tools.harbor_poc.core import HarborDataset,collate

def get_preds(model,records,size,batch,base=False,dataset=None):
    from tools.eval.eval_teacher_ship_external import raw_yoloeseg_fused_forward,unletterbox_xyxy
    from ultralytics.utils.nms import non_max_suppression
    dataset=dataset or HarborDataset(records,size)
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0,collate_fn=collate)
    predictions={r['key']:[] for r in records};seen=[];model.eval().to('cuda')
    with torch.inference_mode():
        for item in loader:
            x=item['img'].cuda().float()/255
            raw=raw_yoloeseg_fused_forward(model,x)[0] if base else model(x)[0]
            harbor=torch.cat((raw[:,:4],raw[:,5:6]),1) if base else raw
            output=non_max_suppression(harbor.clone(),conf_thres=.001,iou_thres=.7,max_det=300,nc=1,max_time_img=float('inf'))
            assert len(output)==len(item['indices'])
            for j,idx in enumerate(item['indices']):
                r=records[idx];seen.append(r['key']);scale=min(size/r['width'],size/r['height'])
                meta={'orig_w':r['width'],'orig_h':r['height'],'scale':scale,'pad_x':(size-round(r['width']*scale))/2,'pad_y':(size-round(r['height']*scale))/2}
                predictions[r['key']]=[{'xyxy':unletterbox_xyxy(p[:4],meta),'score':float(p[4]),'class_id':0} for p in output[j].cpu().numpy()]
    assert len(seen)==len(records) and set(seen)==set(predictions)
    return predictions,{}
