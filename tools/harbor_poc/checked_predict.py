"""Pinned single-image single-class harbor inference, no test/batch auto-discovery."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.harbor_poc.core import setup,load_expert,save,sha
from tools.harbor_poc.checked_inference import get_preds
from PIL import Image,ImageDraw

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--modality',choices=['rgb','sar'],default='rgb');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Choose a new inference output directory')
    setup(a.root);model,ck=load_expert(a.weights);assert model.names=={0:'harbor'}
    with Image.open(a.image) as im:w,h=im.size
    record={'key':'single_image','image':str(a.image),'width':w,'height':h,'harbor':[],'modality':a.modality}
    pred,_=get_preds(model,[record],640,1)
    save(a.out/'predictions.json',{'weights':str(a.weights),'weights_sha256':sha(a.weights),'names':{'0':'harbor'},'image':str(a.image),
        'protocol':'PIL RGB letterbox640 fill114 /255 FP32; conf>.001,NMS.7,max_det300; original pixel xyxy','candidates':pred['single_image']})
    with Image.open(a.image) as im:im=im.convert('RGB')
    draw=ImageDraw.Draw(im)
    for box in pred['single_image']:
        if box['score']>=.25:draw.rectangle(box['xyxy'],outline='red',width=2);draw.text(tuple(box['xyxy'][:2]),f"harbor {box['score']:.3f}",fill='red')
    im.save(a.out/'preview.jpg');print('harbor candidates:',len(pred['single_image']),'output:',a.out)
if __name__=='__main__':main()
