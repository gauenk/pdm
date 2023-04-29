import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import torch as th
import numpy as np
from pathlib import Path
from einops import repeat

from dev_basics.utils import vid_io
from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset_video(dir):
    if os.path.isfile(dir):
        vids = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        length = 0
        vids = {}
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for vid_path in (Path(dir)/"JPEGImages").iterdir():
            vid_path = Path(vid_path)
            vid_name = vid_path.name
            vids[vid_name] = []
            for fname in sorted(vid_path.iterdir()):
                vids[vid_name].append(fname)
    return vids

def get_num_subvids(vids,nframes):
    total = 0
    for name in vids:
        ntotal = len(vids[name])
        nsubs = (ntotal-1)/nframes+1
        total += nsubs
    return total

def compute_name_from_index(vids,nframes):
    names = []
    fstarts = []
    total = 0
    for name in vids:
        ntotal = len(vids[name])
        sub_frames = nframes if nframes > 0 else ntotal
        nsubs = ntotal - (sub_frames-1)
        names.extend([name for _ in range(nsubs)])
        fstarts.extend([t for t in range(nsubs)])
    return names,fstarts

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

def pil_loader_anno(path):
    return Image.open(path)


class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        assert data_len <= 0
        imgs = make_dataset(data_root)
        self.imgs = imgs
        self.names = list(self.imgs.keys())
        self.ntotal = get_num_subvids(self.imgs,nframes)
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

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
        return self.ntotal

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

class InpaintDatasetVideo(data.Dataset):
    def __init__(self, data_root, nframes=0, mask_config={}, data_len=-1,
                 image_size=[256, 256], loader=pil_loader):

        assert data_len <= 0
        imgs = make_dataset_video(data_root)
        self.imgs = imgs
        self.names = list(self.imgs.keys())
        self.nframes = nframes
        l_names,l_frames = compute_name_from_index(self.imgs,nframes)
        self.listed_names = l_names
        self.listed_frames = l_frames
        self.ntotal = len(self.listed_names)

        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.tfs_annos = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
        ])

        self.image_size = image_size
        self.loader = loader
        self.aloader = pil_loader_anno
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def get_paths(self,index):
        def bnd(num,lim):
            if num >= lim: return 2*(lim-1)-num
            else: return num
        name = self.listed_names[index]
        vid_p = self.imgs[name]
        vid_nframes = len(vid_p)
        t_start = self.listed_frames[index]
        t_end = t_start+self.nframes
        paths_t = [vid_p[bnd(t,vid_nframes)] for t in range(t_start,t_end)]
        return paths_t

    def __getitem__(self, index):

        # -- load video & mask --
        paths = self.get_paths(index)
        vid = [self.tfs(self.loader(path)) for path in paths]
        vid = th.stack(vid)
        # mask = self.get_anno_mask(paths)
        mask = self.get_mask(vid.shape[0])

        # -- crop it! [centering mask] --
        # mask,vid = self.crop_them_all(mask,vid)

        # # -- update mask [only 50% can be masked] --
        # mask = self.ensure_nonmasked(mask)

        # -- create derivative quantities --
        cond_image = vid*(1. - mask) + mask*torch.randn_like(vid)
        mask_vid = vid*(1. - mask) + mask

        # -- [debug] view --
        # print("vid.shape: ",vid.shape)
        # print("mask.shape: ",mask.shape)
        # debug_vid = vid*(1. - mask)
        # vid_io.save_video((cond_image+1)/2.,"output/debug/","%d_cond"%index,itype="png")
        # vid_io.save_video((mask_vid+1)/2.,"output/debug/","%d_mask"%index,itype="png")
        # vid_io.save_video((vid+1)/2.,"output/debug/","%d_vid"%index,itype="png")
        # exit(0)

        # # -- crop it! [centering mask] --
        # mask,vid,cond_image,mask_vid = self.crop_them_all(mask,vid,cond_image,mask_vid)

        # -- verify shapes --
        all_vids = [mask,vid,cond_image,mask_vid]
        for _vid in all_vids:
            fmt = "%d_%d" % (vid.shape[-2],self.image_size[0])
            assert _vid.shape[-2] == self.image_size[0],fmt
            fmt = "%d_%d" % (vid.shape[-1],self.image_size[1])
            assert _vid.shape[-1] == self.image_size[1],fmt

        # -- create output --
        ret = {}
        ret['gt_image'] = vid
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_vid
        ret['mask'] = mask
        ret['path'] = [str(p) for p in paths]
        return ret

    def __len__(self):
        return self.ntotal

    def ensure_nonmasked(self,mask):
        self.perc_thresh = 0.5
        perc = (mask.sum()/mask.numel()).item()
        if perc > self.perc_thresh:
            H,W = mask.shape[-2:]
            H2,W2 = H//2,W//2
            sH,sW = H//4,W//4
            mask_c = th.zeros_like(mask)
            mask_c[:,:,sH:sH+H2,sW:sW+W2] = mask[:,:,sH:sH+H2,sW:sW+W2]
        else:
            mask_c = mask
        return mask_c

    def crop_them_all(self,mask,*vids):

        # -- find center of mask --
        args = th.where(mask[0]>0)
        args0 = args[-2]*1.
        args1 = args[-1]*1.
        # print(args0.min().item(),args0.max().item())
        # print(args1.min().item(),args1.max().item())
        cH = th.mean(args0)
        cW = th.mean(args1)

        # -- crop slices --
        H,W = mask.shape[-2:]
        sH,sW = self.image_size

        h_start = int(cH-sH/2)
        h_shift = min(0,h_start) + max(0,h_start + sH - (H-1));
        h_start = h_start - h_shift
        h_end = h_start + sH
        # print(h_start,h_end,h_end-h_start,sH,H)
        msg = (h_start,h_end,h_end-h_start,sH,H)
        assert h_end - h_start == sH and h_end < H,msg
        hslice = slice(h_start,h_start+sH)

        w_start = int(cW-sW/2)
        w_shift = min(0,w_start) + max(0,w_start + sW - (W-1));
        w_start = w_start - w_shift
        w_end = w_start + sW
        # print(w_start,w_end,w_end-w_start,sW,W)
        msg = (h_start,h_end,h_end-h_start,sH,H)
        assert w_end - w_start == sW and w_end < W,msg
        wslice = slice(w_start,w_start+sW)

        # -- exec crop --
        cvids = [mask[...,hslice,wslice]]
        for vid in vids:
            cvids.append(vid[...,hslice,wslice])

        return cvids

    def get_anno_mask(self,ipaths):

        #
        # -- read --
        #

        def reformat(ipath):
            apath = str(ipath).replace("JPEGImages","Annotations")
            apath = Path(apath).with_suffix('.png')
            return apath
        apaths = [reformat(p) for p in ipaths]
        annos = [self.tfs_annos(self.aloader(apath)) for apath in apaths]
        annos = th.stack(annos)

        # -- debug --
        # index = th.randint(0,10,(1,)).item()
        # vid_io.save_video(annos/annos.max(),"output/debug/",
        #             "%d_anno"%(index),itype="png")


        #
        # -- gather & pick uniqs --
        #

        uniqs = th.unique(annos[0]).numpy().tolist()
        if len(uniqs) > 1:
            uniqs = sorted(uniqs)[1:] # remove background class
        label = uniqs[th.randint(0,len(uniqs),(1,))[0]]

        #
        # -- create mask --
        #

        mask = 1.*(annos==label)

        return mask

    def get_mask(self,t):
        mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        mask = th.tensor(mask)*1.
        mask = repeat(mask,'h w 1 -> t 1 h w',t=t)
        # print(mask.shape)
        return mask

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



class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

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
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


