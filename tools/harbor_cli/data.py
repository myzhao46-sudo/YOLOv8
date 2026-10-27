"""Windows-spawn-compatible image loading with optional shared uint8 cache.

Flip RNG is a function of seed, epoch and presentation position. Workers and
persistent-worker lifetime therefore do not change augmentation or need hidden
worker RNG snapshots. This is a versioned change from the legacy random flips.
"""
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,Sampler
from tools.harbor_poc.core import collate

DATA_VERSION='harbor_stateless_flip_v1'

def read_image(path,size):
    from tools.eval.eval_teacher_ship_external import make_letterbox_rgb
    return make_letterbox_rgb(Path(path),size)

class HarborDataset(Dataset):
    def __init__(self,records,size,augment=False,cache='shared',seed=43,flip_h=.5,flip_v=.5):
        self.size=size;self.augment=augment;self.cache_mode=cache;self.seed=seed;self.flip_h=flip_h;self.flip_v=flip_v
        self.epoch=torch.zeros(1,dtype=torch.int64).share_memory_()
        self.records=[{'image':r['image'],'key':r['key'],'width':r['width'],'height':r['height']} for r in records]
        self.images=None;self.boxes=[];self.loader_cache={}
        if cache not in ('none','ram','shared'):raise ValueError('Unknown cache mode')
        if cache!='none':
            self.images=torch.empty((len(records),size,size,3),dtype=torch.uint8)
            if cache=='shared':self.images.share_memory_()
        for i,r in enumerate(records):
            scale=min(size/r['width'],size/r['height']);px=(size-round(r['width']*scale))/2;py=(size-round(r['height']*scale))/2
            bb=[]
            for o in r['harbor']:
                x1,y1,x2,y2=o['xyxy'];bb.append([((x1+x2)*scale/2+px)/size,((y1+y2)*scale/2+py)/size,(x2-x1)*scale/size,(y2-y1)*scale/size])
            self.boxes.append(np.asarray(bb,dtype=np.float32).reshape(-1,4))
            if self.images is not None:
                im,meta=read_image(r['image'],size)
                if (meta['orig_w'],meta['orig_h'])!=(r['width'],r['height']):raise ValueError('Source dimensions changed: '+r['image'])
                self.images[i].copy_(torch.from_numpy(np.array(im,copy=True)))
    def __getstate__(self):
        # Do not pickle parent-side loaders into worker datasets.
        state=self.__dict__.copy();state['loader_cache']={};return state
    def __len__(self):return len(self.records)
    def set_epoch(self,epoch):self.epoch[0]=epoch
    def __getitem__(self,token):
        if isinstance(token,tuple):idx,presentation=token
        else:idx=presentation=int(token)
        im=self.images[idx].numpy() if self.images is not None else read_image(self.records[idx]['image'],self.size)[0]
        boxes=self.boxes[idx].copy()
        if self.augment:
            key=((self.seed&0xffffffff)<<32)^((int(self.epoch[0])+1)*0x9E3779B1)^((presentation+1)*0x85EBCA77)^(idx*0xC2B2AE3D)
            rng=random.Random(key)
            if rng.random()<self.flip_h:im=im[:,::-1];boxes[:,0]=1-boxes[:,0]
            if rng.random()<self.flip_v:im=im[::-1];boxes[:,1]=1-boxes[:,1]
        # Always own the batch sample: no worker or caller may modify the cache.
        image=np.array(im.transpose(2,0,1),copy=True,order='C')
        return torch.from_numpy(image),torch.from_numpy(boxes),idx
    def close(self):
        for loader,_ in self.loader_cache.values():
            iterator=getattr(loader,'_iterator',None)
            if iterator is not None:iterator._shutdown_workers()
        self.loader_cache.clear()

class PresentationSampler(Sampler):
    def __init__(self,order):self.order=list(order)
    def __iter__(self):return iter((idx,position) for position,idx in enumerate(self.order))
    def __len__(self):return len(self.order)

def init_worker(worker_id):
    from tools.clean_pipeline.runtime import setup_runtime
    setup_runtime();torch.set_num_threads(1)

def get_loader(dataset,batch_size,workers,prefetch_factor,persistent_workers,pin_memory,order=None):
    if workers>0 and dataset.cache_mode=='ram':raise ValueError('Use shared or none cache with Windows worker processes')
    key=(batch_size,workers,prefetch_factor,bool(persistent_workers and workers),bool(pin_memory),order is not None)
    if key in dataset.loader_cache:
        loader,sampler=dataset.loader_cache[key]
        if sampler is not None:sampler.order=list(order)
        return loader
    sampler=PresentationSampler(order) if order is not None else None
    kwargs={'batch_size':batch_size,'num_workers':workers,'collate_fn':collate,'pin_memory':pin_memory,
        'sampler':sampler,'worker_init_fn':init_worker,
        'generator':torch.Generator().manual_seed(dataset.seed+1009+(1 if dataset.augment else 0))}
    if workers>0:kwargs.update(prefetch_factor=prefetch_factor,persistent_workers=persistent_workers,multiprocessing_context='spawn')
    loader=DataLoader(dataset,**kwargs);dataset.loader_cache[key]=(loader,sampler);return loader
