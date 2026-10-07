# -*- coding: utf-8 -*-
"""Harbor-positive subset preparation. Source files are read-only; no model selection."""
import argparse,csv,hashlib,json,shutil,xml.etree.ElementTree as ET
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def normalized(box,w,h):
    x1,y1,x2,y2=box;return [(x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h]
def reverse(v,w,h):
    x,y,bw,bh=v;return [(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
def collect(root):
    man=json.loads((root/'experiment_manifest.json').read_text(encoding='utf8'));out=Path(man['derived_root']);audit=Path(man['audit_root'])
    records=[]
    with (audit/'dior/image_annotation_manifest.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if int(row['harbor_boxes_raw'])==0:continue
            ann=Path(row['HBB_annotation']);doc=ET.parse(ann).getroot(); objs=[]
            for n,o in enumerate(doc.findall('object'),1):
                name=o.findtext('name');b=o.find('bndbox')
                box=[float(b.findtext(k)) for k in ['xmin','ymin','xmax','ymax']]
                objs.append({'original_id':f'{ann.stem}:object[{n}]','original_category_id':None,'name':name,'raw_box':box,'box_type':'xyxy','xyxy':box})
            records.append({'dataset':'DIOR','modality':'rgb','split':row['splits'],'original_image_id':row['image_id'],
                            'original_image':row['image_path'],'annotation_source':str(ann),'objects':objs})
    sr=Path(man['dataset_roots']['sar'])
    for split in ['train','val','test']:
        jp=sr/'Annotations'/f'{split}.json';doc=json.loads(jp.read_text());cats={a['id']:a['name'] for a in doc['categories']}
        by=defaultdict(list)
        for a in doc['annotations']:by[a['image_id']].append(a)
        for im in doc['images']:
            aa=by[im['id']]
            if not any(cats[a['category_id']]=='harbor' for a in aa):continue
            objs=[]
            for a in aa:
                x,y,w,h=a['bbox'];objs.append({'original_id':a['id'],'original_category_id':a['category_id'],'name':cats[a['category_id']],
                    'raw_box':a['bbox'],'box_type':'xywh','xyxy':[x,y,x+w,y+h],'original_record':a})
            records.append({'dataset':'SARDet-100K','modality':'sar','split':split,'original_image_id':im['id'],
                'original_image':str(sr/'Images'/split/im['file_name']),'annotation_source':str(jp),'objects':objs})
    for r in records:
        p=Path(r['original_image']);r['key']=f"{r['modality']}__{r['split']}__{r['original_image_id']}__{p.stem}"
        r['exclusions']=[];r['label_changes']=[]
        with Image.open(p) as im:
            im.load();arr=np.asarray(im.convert('RGB'));r['width'],r['height']=im.size
        r['source_sha256']=sha(p);r['decoded_sha256']=hashlib.sha256(str(arr.shape).encode()+arr.tobytes()).hexdigest()
        grey=cv2.cvtColor(cv2.resize(arr,(32,32)),cv2.COLOR_RGB2GRAY).astype(np.float32)
        dct=cv2.dct(grey)[:8,:8].flatten();bits=dct>np.median(dct[1:]);r['phash']=int(''.join('1' if x else '0' for x in bits),2)
        r['harbor']=[];seen=set()
        for o in r['objects']:
            if o['name']!='harbor':continue
            b=o['xyxy'];x1,y1,x2,y2=b
            if not all(np.isfinite(b)) or x2<=x1 or y2<=y1 or min(x1,y1)<0 or x2>r['width'] or y2>r['height']:
                r['exclusions'].append({'reason':'quarantine_entire_image_unverified_harbor_geometry','record_id':o['original_id'],'box':b})
            if tuple(b) in seen:
                r['label_changes'].append({'reason':'exact_duplicate_harbor_geometry','record_id':o['original_id']});continue
            seen.add(tuple(b));v=normalized(b,r['width'],r['height']);o['yolo']=v
            assert np.max(np.abs(np.array(reverse(v,r['width'],r['height']))-b))<1e-5
            r['harbor'].append(o)
        r['image']=str(out/'images'/r['split']/(r['key']+p.suffix.lower()))
        r['label']=str(out/'labels'/r['split']/(r['key']+'.txt'))
    policy={'scope':'harbor positive subset only; zero reviewed empty-image negatives. FP measured within harbor-positive scenes.',
      'coordinates':'unchanged source pixel xyxy for DIOR; x,y,x+w,y+h for SAR; continuous-edge convention for YOLO. No +/-1 applied; original endpoint metadata unconfirmed. Chosen to preserve source geometry and audit overlays.',
      'invalid_harbor':'quarantine ENTIRE image if any harbor box invalid/out of true bounds; no speculative clipping or missing-target background',
      'duplicates':'decoded exact duplicates: keep highest split priority test>val>train then lexicographic key; cross-split near candidates phash<=4 and resized RGB MAE<=0.015: conservatively quarantine lower priority image. Other near clues logged only.',
      'gate':'no training images reserved this prototype; future 5-fold cross-fitting grouped by detected duplicate cluster using original train only, fixed seed42, out-of-fold model candidates; no val/test for gate fitting.',
      'class_mapping':{'expert':{'0':'harbor'},'future_joint':{'0':'ship','1':'harbor'}},'negative_images':0,'upstream_overlap':'unknown'}
    save(out/'preparation_policy.json',policy);save(out/'records_pre_duplicate.json',records)
    preview=root/'data_previews';preview.mkdir(exist_ok=True)
    for mod in ['rgb','sar']:
        chosen=[r for r in records if r['modality']==mod and r['split']=='train' and not r['exclusions']][:3]
        for r in chosen:
            with Image.open(r['original_image']) as im:im=im.convert('RGB')
            draw=ImageDraw.Draw(im)
            for o in r['harbor']:
                b=reverse(o['yolo'],r['width'],r['height']);draw.rectangle(b,outline='red',width=2);draw.text((b[0],max(0,b[1]-11)),str(o['original_id']),fill='cyan')
            im.save(preview/f"{r['key']}.png")
    print('Preview preparation',len(records),'records, quarantined',sum(bool(r['exclusions']) for r in records),flush=True)

def finalize(root):
    man=json.loads((root/'experiment_manifest.json').read_text(encoding='utf8'));out=Path(man['derived_root'])
    records=json.loads((out/'records_pre_duplicate.json').read_text(encoding='utf8'));priority={'train':0,'val':1,'test':2}
    groups=defaultdict(list);duplicates=[];near=[]
    for r in records:groups[r['decoded_sha256']].append(r)
    for group in groups.values():
        if len(group)<2:continue
        group.sort(key=lambda r:(-priority[r['split']],r['key']));keep=group[0]
        for r in group[1:]:r['exclusions'].append({'reason':'exact_decoded_duplicate','kept_key':keep['key']})
        duplicates.append([r['key'] for r in group])
    thumbs={}
    for i,a in enumerate(records):
        for b in records[i+1:]:
            if a['modality']!=b['modality'] or a['decoded_sha256']==b['decoded_sha256']:continue
            dist=(a['phash']^b['phash']).bit_count()
            if dist>4:continue
            for r in [a,b]:
                if r['key'] not in thumbs:
                    with Image.open(r['original_image']) as im:thumbs[r['key']]=np.asarray(im.convert('RGB').resize((64,64))).astype(float)/255
            mae=float(np.abs(thumbs[a['key']]-thumbs[b['key']]).mean());excluded=None
            if a['split']!=b['split'] and mae<=.015:
                drop,keep=(a,b) if priority[a['split']]<priority[b['split']] else (b,a)
                drop['exclusions'].append({'reason':'cross_split_near_duplicate_conservative','kept_key':keep['key'],'mae':mae});excluded=drop['key']
            near.append({'a':a['key'],'b':b['key'],'phash_hamming':dist,'rgb_mae':mae,'excluded':excluded})
    kept=[r for r in records if not r['exclusions']];counts={}
    for mod in ['rgb','sar']:
        for split in ['train','val','test']:
            allr=[r for r in records if r['modality']==mod and r['split']==split];rr=[r for r in kept if r in allr]
            counts[f'{mod}/{split}']={'raw_images':len(allr),'raw_harbor_boxes':sum(len(r['harbor']) for r in allr),'images':len(rr),
                'harbor_boxes':sum(len(r['harbor']) for r in rr),'positive_images':len(rr),'negative_images':0,'excluded_images':len(allr)-len(rr),
                'excluded_harbor_boxes':sum(len(r['harbor']) for r in allr)-sum(len(r['harbor']) for r in rr)}
    for r in kept:
        ip,lp=Path(r['image']),Path(r['label']);ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(r['original_image'],ip)
        lp.write_text('\n'.join('0 '+' '.join(f'{v:.10f}' for v in o['yolo']) for o in r['harbor'])+'\n',encoding='utf8')
    for split in ['train','val','test']:
        save(out/f'{split}_manifest.json',[r for r in kept if r['split']==split])
        (out/f'{split}.txt').write_text('\n'.join(r['image'] for r in kept if r['split']==split)+'\n',encoding='utf8')
    save(out/'all_source_records.json',records);save(out/'duplicate_review.json',{'exact_groups':duplicates,'near_clues':near,'rule_fixed_before_inference':True})
    save(out/'data_summary.json',counts)
    import yaml
    (out/'harbor.yaml').write_text(yaml.safe_dump({'path':str(out),'train':str(out/'train.txt'),'val':str(out/'val.txt'),'test':str(out/'test.txt'),'names':{0:'harbor'}},sort_keys=False),encoding='utf8')
    hashes={p.name:sha(p) for p in out.glob('*manifest.json')};save(out/'manifest_hashes.json',hashes)
    print(json.dumps({'counts':counts,'exact_groups':len(duplicates),'near_clues':len(near),'hashes':hashes},indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--stage',choices=['preview','finalize'],required=True);a=ap.parse_args()
    (collect if a.stage=='preview' else finalize)(a.root)
