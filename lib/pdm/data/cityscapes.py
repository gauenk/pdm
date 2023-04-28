"""

An inpainting dataset using CityScapes

?? Where from ??

... maybe detectron2 ?


"""

import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
from pathlib import Path

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

def pil_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset_video(dir):
    if os.path.isfile(dir):
        vids = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        vids = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fdir in sorted(fdir):
                path = os.path.join(root, fname)
                vids.append(path)
    return vids

class InpaintingCityScapes(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1,
                 image_size=[256, 256], loader=pil_loader):
        # -- load paths --
        # vid_dir = data_root / "leftImg8bit_sequence"
        # vid_dir = data_root / "gtCoarse"
        imgs = make_dataset_video(data_root)
        if data_len > 0: self.imgs = imgs[:int(data_len)]
        else: self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self,index):

        # -- indices --
        image_index = self.indices[index]

        # -- load burst --
        subvid_name = self.names[image_index]
        clean,frame_nums,loc = read_data(subvid_name,self.iroot,self.nframes,self.bw)

        # -- augmentations --
        if self.nscale_augs > 0:
            aug_idx = random.randint(0,self.nscale_augs-1)
            trans_fxn = self.scale_augs[aug_idx]
            clean = trans_fxn(clean)
        if self.nflip_augs > 0:
            aug_idx = random.randint(0,self.nflip_augs-1)
            trans_fxn = self.flippy_augs[aug_idx]
            clean = trans_fxn(clean)

        # -- flow io --
        size = list(clean.shape[-2:])
        vid_name = "_".join(subvid_name.split("+")[0].split("_")[:-2])
        fflow,bflow = read_flows(FLOW_BASE,self.read_flows,vid_name,
                                 self.noise_info,self.seed,loc,size)

        # -- cropping --
        region = th.IntTensor([])
        in_vids = [clean,fflow,bflow] if self.read_flows else [clean]
        use_region = "region" in self.cropmode or "coords" in self.cropmode
        if use_region:
            region = crop_vid(clean,self.cropmode,self.isize,self.region_temp)
        else:
            in_vids = crop_vid(in_vids,self.cropmode,self.isize,self.region_temp)
            clean = in_vids[0]
            if self.read_flows:
                fflow,bflow = in_vids[1],in_vids[2]

        # -- sample mask --
        mask = self.sample_mask(index)

        # -- apply mask --
        cond_vid = clean*(1. - mask) + mask*torch.randn_like(clean)
        fflow = fflow*(1. - mask) + mask*torch.randn_like(fflow)
        bflow = bflow*(1. - mask) + mask*torch.randn_like(bflow)

        # -- manage flow and output --
        index_th = th.IntTensor([image_index])

        return {'cond_image':cond_vid,'gt_image':clean,'mask_image':mask,
                'index':index_th,'fnums':frame_nums,'rng_state':rng_state,
                'fflow':fflow,'bflow':bflow}

    def sample_mask(self,index):
        # -- pick a segmentation --
        segm = pick_segm(index)
        return segm

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)
