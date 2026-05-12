#!/usr/bin/env python
"""
Evaluate a YOLOE checkpoint on a specified dataset.

Handles YOLOE text-embedding initialization:
  Strategy A: Call model.get_text_pe() (uses mobileclip internally)
  Strategy B: Try ultralytics.nn.text_model.build_text_model directly
  Strategy C: Deterministic pseudo-embeddings (last resort, results degraded)

Usage (Windows):
  cd YOLOv8/ultralytics
  python eval_bestpt_on_extratest.py

Usage (Linux):
  cd /root/autodl-tmp/YOLOv8
  python eval_bestpt_on_extratest.py \
      --model ultralytics/ultralytics/best.pt \
      --data  ultralytics/tmp_data/tank_extratest_eval.yaml \
      --conf 0.1 --iou 0.5 --imgsz 640 --device 0
"""

from __future__ import annotations
import argparse, sys, os
from pathlib import Path
from collections import defaultdict

_script_dir = os.path.dirname(os.path.abspath(__file__))
for _c in [_script_dir, os.path.join(_script_dir, "ultralytics")]:
    if os.path.isdir(os.path.join(_c, "ultralytics")):
        sys.path.insert(0, _c)
        break

import yaml, numpy as np, torch

DEFAULT_MODEL  = r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\best.pt" 
DEFAULT_DATA   = r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\datasets\tank_extratest_eval_reinforce.yaml"
DEFAULT_CONF   = 0.1
DEFAULT_IOU    = 0.5
DEFAULT_IMGSZ  = 640
DEFAULT_DEVICE = "cpu"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default=DEFAULT_MODEL)
    p.add_argument("--data",    default=DEFAULT_DATA)
    p.add_argument("--conf",    type=float, default=DEFAULT_CONF)
    p.add_argument("--iou",     type=float, default=DEFAULT_IOU)
    p.add_argument("--imgsz",   type=int,   default=DEFAULT_IMGSZ)
    p.add_argument("--device",  default=DEFAULT_DEVICE)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# =====================================================================
# YAML / remap
# =====================================================================
def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f: return yaml.safe_load(f)

def parse_names(raw):
    if isinstance(raw, list): return {i: n for i, n in enumerate(raw)}
    if isinstance(raw, dict): return {int(k): str(v) for k, v in raw.items()}
    raise ValueError(f"Cannot parse names: {raw}")

def build_class_remap(dn, mn):
    lk = {n.strip().lower(): i for i, n in mn.items()}
    rm = {}
    for di, d in dn.items():
        k = d.strip().lower()
        if k not in lk: raise ValueError(f"'{d}'(idx {di}) not in model: {mn}")
        rm[di] = lk[k]
    return rm

def resolve_val_path(cfg, yp):
    yd = Path(yp).resolve().parent
    vr = cfg.get("val")
    if not vr: raise ValueError("No 'val' in YAML")
    vp = Path(vr)
    if vp.is_absolute() and vp.exists(): return vp
    base = cfg.get("path", "")
    if base:
        b = Path(base)
        if not b.is_absolute(): b = yd / b
        c = (b / vr).resolve()
        if c.exists(): return c
    c = (yd / vr).resolve()
    if c.exists(): return c
    if vp.exists(): return vp.resolve()
    raise FileNotFoundError(f"Cannot find val: {vr}")

# =====================================================================
# YOLOE text embedding initialization
# =====================================================================
def init_yoloe(model, class_names):
    """Initialize YOLOE model with text embeddings so inference works."""
    inner = model.model
    nc = len(class_names)
    embed_dim = getattr(list(inner.model)[-1], "embed", 512)
    print(f"  nc={nc}, embed_dim={embed_dim}")

    # Strategy A: use model's own get_text_pe()
    print(f"  Strategy A: inner.get_text_pe({class_names}) ...")
    try:
        pe = inner.get_text_pe(class_names)
        print(f"    get_text_pe -> shape={pe.shape}, dtype={pe.dtype}")
        assert pe.ndim == 3, f"Expected 3D, got {pe.ndim}D"
        inner.set_classes(class_names, pe)
        print(f"    set_classes OK. Names: {dict(model.names)}")
        return model
    except Exception as e:
        print(f"    Failed: {e}")

    # Strategy B: build_text_model directly
    print(f"  Strategy B: build_text_model ...")
    try:
        from ultralytics.nn.text_model import build_text_model
        device = next(inner.model.parameters()).device
        text_model = build_text_model("mobileclip:blt", device=device)
        text_token = text_model.tokenize(class_names)
        txt_feats = text_model.encode_text(text_token).detach()
        pe = txt_feats.reshape(1, nc, embed_dim)
        print(f"    pe shape={pe.shape}")
        inner.set_classes(class_names, pe)
        print(f"    set_classes OK. Names: {dict(model.names)}")
        return model
    except Exception as e:
        print(f"    Failed: {e}")

    # Strategy C: pseudo-embeddings (results will be degraded but inference runs)
    print(f"  Strategy C: pseudo-embeddings (RESULTS WILL BE DEGRADED!) ...")
    rng = torch.Generator().manual_seed(42)
    pe = torch.randn(1, nc, embed_dim, generator=rng)
    pe = pe / pe.norm(dim=-1, keepdim=True)
    try:
        inner.set_classes(class_names, pe)
        print(f"    set_classes with pseudo-pe OK. Names: {dict(model.names)}")
        print(f"    WARNING: Using random embeddings - AP will NOT reflect true model capability!")
        return model
    except Exception as e:
        print(f"    set_classes failed even with pseudo-pe: {e}")
        raise RuntimeError(
            "Cannot initialize YOLOE for inference.\n"
            "You need mobileclip. Try: pip install mobileclip\n"
            "Or run this script on your Linux training server where CLIP is available."
        )

# =====================================================================
# Labels / matching / AP
# =====================================================================
def read_labels(path, remap=None):
    """Read labels supporting both formats:
      - 5-col standard YOLO:  cls cx cy w h
      - 9-col polygon/DOTA:   cls x1 y1 x2 y2 x3 y3 x4 y4
    Returns list of (cls, x1_norm, y1_norm, x2_norm, y2_norm) as axis-aligned bbox.
    """
    boxes = []
    if not os.path.exists(path): return boxes
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5: continue
            raw = int(p[0])
            if remap is not None:
                if raw not in remap: continue
                cls = remap[raw]
            else: cls = raw

            if len(p) >= 9:
                # 9-col polygon: cls x1 y1 x2 y2 x3 y3 x4 y4
                coords = list(map(float, p[1:9]))
                xs = coords[0::2]  # x1, x2, x3, x4
                ys = coords[1::2]  # y1, y2, y3, y4
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                # Store as xyxy directly (flag with None to skip xywh2xyxy)
                boxes.append((cls, x1, y1, x2, y2, "xyxy"))
            else:
                # 5-col standard: cls cx cy w h
                cx, cy, w, h = map(float, p[1:5])
                boxes.append((cls, cx, cy, w, h, "xywh"))
    return boxes

def label_to_xyxy(box_tuple):
    """Convert a label tuple to (cls, x1, y1, x2, y2)."""
    if box_tuple[-1] == "xyxy":
        return box_tuple[0], box_tuple[1], box_tuple[2], box_tuple[3], box_tuple[4]
    else:  # xywh
        cls, cx, cy, w, h = box_tuple[:5]
        return cls, cx-w/2, cy-h/2, cx+w/2, cy+h/2

def iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    i=max(0,x2-x1)*max(0,y2-y1)
    u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
    return i/u if u>0 else 0

def match(preds, gts, thr, tc=None):
    if tc is not None:
        preds=[p for p in preds if p["c"]==tc]
        gts=[g for g in gts if g["c"]==tc]
    preds=sorted(preds,key=lambda x:x["s"],reverse=True)
    gm=[False]*len(gts); tp=fp=0; det=[]
    for p in preds:
        bi,bj=0,-1
        for j,g in enumerate(gts):
            if gm[j] or p["c"]!=g["c"]: continue
            v=iou(p["b"],g["b"])
            if v>bi: bi,bj=v,j
        if bi>=thr and bj>=0:
            tp+=1;gm[bj]=True;det.append((1,p["s"]))
        else: fp+=1;det.append((0,p["s"]))
    fn=sum(1 for m in gm if not m)
    return tp,fp,fn,det

def compute_ap(flags, confs, ngt):
    if ngt==0: return 0.0
    o=np.argsort(-np.array(confs)); t=np.array(flags)[o]
    ct=np.cumsum(t);cf=np.cumsum(1-t)
    r=np.concatenate(([0],ct/ngt,[1])); p=np.concatenate(([1],ct/(ct+cf),[0]))
    for i in range(len(p)-2,-1,-1): p[i]=max(p[i],p[i+1])
    idx=np.where(r[1:]!=r[:-1])[0]
    return float(np.sum((r[idx+1]-r[idx])*p[idx+1]))

# =====================================================================
# Main
# =====================================================================
def main():
    args = parse_args()
    print("="*80)
    print("YOLOE Evaluation with Text Embedding Initialization")
    print("="*80)
    print(f"  Model:  {args.model}")
    print(f"  Data:   {args.data}")
    print(f"  Conf={args.conf}  IoU={args.iou}  ImgSz={args.imgsz}  Device={args.device}")

    # Dataset
    print(f"\n[1] Dataset ...")
    dcfg = load_yaml(args.data)
    dnames = parse_names(dcfg.get("names",{}))
    print(f"  Dataset names: {dnames}")
    vp = resolve_val_path(dcfg, args.data)
    ld = Path(str(vp).replace("images","labels"))
    print(f"  Images: {vp}")
    print(f"  Labels: {ld}")

    # Model
    print(f"\n[2] Loading model ...")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        rm = ckpt.get("model")
        mnames = dict(rm.names) if rm and hasattr(rm,"names") else {0:"ship",1:"harbor",2:"tank"}
    else: mnames = {0:"ship",1:"harbor",2:"tank"}
    del ckpt
    clist = [mnames[i] for i in sorted(mnames)]
    print(f"  Original classes: {clist}")

    from ultralytics import YOLO
    model = YOLO(args.model)
    print(f"  Task: {model.task}, Names: {dict(model.names)}")

    # Init YOLOE
    model = init_yoloe(model, clist)
    fnames = dict(model.names)

    # Remap
    print(f"\n[3] Class remap ...")
    remap = build_class_remap(dnames, fnames)
    print(f"  {remap}")
    for d,m in remap.items():
        print(f"    label[{d}] '{dnames[d]}' -> model[{m}] '{fnames[m]}'")
    evcls = set(remap.values())

    # Images
    ext={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    imgs=sorted(f for f in vp.iterdir() if f.suffix.lower() in ext)
    print(f"\n[4] {len(imgs)} images")
    if not imgs: print("ERROR: none"); sys.exit(1)

    # Sanity check
    sc=set()
    for f in imgs[:5]:
        lf=ld/(f.stem+".txt")
        if lf.exists():
            for ln in open(lf):
                p=ln.strip().split()
                if p: sc.add(int(p[0]))
    print(f"  Label indices: {sorted(sc)}, YAML expects: {sorted(dnames.keys())}")
    miss=sc-set(dnames.keys())
    if miss: print(f"  WARNING: {miss} in labels but not YAML!")

    # Eval
    print(f"\n[5] Evaluating ...")
    tpf=defaultdict(list); cfs=defaultdict(list)
    gtc=defaultdict(int); ctp=defaultdict(int); cfp=defaultdict(int); cfn=defaultdict(int)
    N=len(imgs)

    for ii,ip in enumerate(imgs):
        lf=ld/(ip.stem+".txt")
        rg=read_labels(str(lf),remap)
        gb=[]
        for box_tuple in rg:
            cls, x1, y1, x2, y2 = label_to_xyxy(box_tuple)
            gb.append({"c": cls, "b": (x1, y1, x2, y2)})
        for g in gb: gtc[g["c"]]+=1

        try:
            res=model.predict(str(ip),conf=args.conf,iou=args.iou,
                              imgsz=args.imgsz,device=args.device,verbose=False,save=False)
        except Exception as e:
            if ii==0:
                print(f"\n  FATAL on 1st image: {e}")
                import traceback; traceback.print_exc()
                sys.exit(1)
            continue

        pb=[]
        for r in res:
            bx=r.boxes
            if bx is None or len(bx)==0: continue
            for i in range(len(bx)):
                ci=int(bx.cls[i].item())
                if ci not in evcls: continue
                co=float(bx.conf[i].item())
                if hasattr(bx,"xyxyn") and bx.xyxyn is not None and len(bx.xyxyn)>0:
                    x1,y1,x2,y2=bx.xyxyn[i].tolist()
                else:
                    x1,y1,x2,y2=bx.xyxy[i].tolist()
                    ih,iw=r.orig_shape; x1/=iw; y1/=ih; x2/=iw; y2/=ih
                pb.append({"c":ci,"s":co,"b":(x1,y1,x2,y2)})

        for mc in evcls:
            t,f,n,d=match(pb,gb,args.iou,mc)
            ctp[mc]+=t;cfp[mc]+=f;cfn[mc]+=n
            for fl,co in d: tpf[mc].append(fl); cfs[mc].append(co)

        if (ii+1)%50==0 or ii+1==N:
            print(f"  {ii+1}/{N} ...")

    # Results
    print("\n"+"="*80)
    print("RESULTS")
    print("="*80)
    hdr=f"{'Class':>15}|{'GT':>6}|{'TP':>6}|{'FP':>6}|{'FN':>6}|{'P':>7}|{'R':>7}|{'F1':>7}|{'AP50':>7}"
    print(hdr); print("-"*len(hdr))
    tt=tf=tn=tg=0; aps=[]
    for mc in sorted(evcls):
        nm=fnames[mc]; gt=gtc[mc]; tp=ctp[mc]; fp=cfp[mc]; fn=cfn[mc]
        p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
        f1=2*p*r/(p+r) if p+r else 0
        ap=compute_ap(tpf[mc],cfs[mc],gt)
        print(f"{nm:>15}|{gt:6}|{tp:6}|{fp:6}|{fn:6}|{p:7.4f}|{r:7.4f}|{f1:7.4f}|{ap:7.4f}")
        tt+=tp;tf+=fp;tn+=fn;tg+=gt;aps.append(ap)
    op=tt/(tt+tf) if tt+tf else 0; or_=tt/(tt+tn) if tt+tn else 0
    of=2*op*or_/(op+or_) if op+or_ else 0; ma=np.mean(aps) if aps else 0
    print("-"*len(hdr))
    print(f"{'ALL':>15}|{tg:6}|{tt:6}|{tf:6}|{tn:6}|{op:7.4f}|{or_:7.4f}|{of:7.4f}|{ma:7.4f}")

    print(f"\n  mAP50 = {ma:.4f} ({ma*100:.1f}%)")
    print(f"  Remap: {remap}")
    print(f"  NOTE: P/R/F1 at conf={args.conf}. AP50 = all-point interpolation.")
    if ma < 0.1:
        print(f"\n  WARNING: Very low mAP. If pseudo-embeddings were used,")
        print(f"  install mobileclip (pip install mobileclip) and re-run,")
        print(f"  or run on your Linux server where CLIP is available.")

if __name__=="__main__":
    main()