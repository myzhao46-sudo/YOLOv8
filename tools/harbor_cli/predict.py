"""Pinned CLI-run expert inference with the run's actual image/loader config."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.harbor_poc.core import load_expert,save,read,sha
from tools.harbor_cli.runtime import setup
from tools.harbor_cli.inference import get_preds,configure
from PIL import Image,ImageDraw

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--modality',choices=['rgb','sar'],default='rgb');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Choose a new output directory')
    setup(a.root);cfg=read(a.root/'config.json');configure(cfg);model,_=load_expert(a.weights);assert model.names=={0:'harbor'}
    with Image.open(a.image) as im:w,h=im.size
    record={'key':'single_image','image':str(a.image),'width':w,'height':h,'harbor':[],'modality':a.modality}
    pred,_=get_preds(model,[record],cfg['imgsz'],cfg['val_batch_size'])
    save(a.out/'predictions.json',{'weights':str(a.weights),'sha256':sha(a.weights),'image':str(a.image),'names':{'0':'harbor'},'imgsz':cfg['imgsz'],'predictions':pred['single_image']})
    with Image.open(a.image) as im:im=im.convert('RGB')
    draw=ImageDraw.Draw(im)
    for box in pred['single_image']:
        if box['score']>=.25:draw.rectangle(box['xyxy'],outline='red',width=2)
    im.save(a.out/'preview.jpg');print(a.out)
if __name__=='__main__':main()
