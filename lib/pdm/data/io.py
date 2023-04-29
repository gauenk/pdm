"""

Custom IO


"""

import numpy as np
from functools import partial
from easydict import EasyDict as edict

import pdm.core.util as Util
from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset
from .cityscapes import InpaintingCityScapes
from .dataset import InpaintDatasetVideo


def load(cfg,logger):

    # -- configs --
    # mask_config = {"mask_mode": "hybrid"}
    mask_config = {"mask_mode": "fourdirection"}
    dataloader = {
        "validation_split": 2,
        "args":{
            "batch_size": 4,
            "num_workers": 4,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True
        },
        "val_args":{
            "batch_size": 1,
            "num_workers": 4,
            "shuffle": False,
            "pin_memory": True,
            "drop_last": False
        }
    }
    gl_seed = 123
    image_size = [256,256]

    # -- data --
    mode = "train"
    if mode == "train":
        root = "data/youtube_voc/train"
    if mode == "val":
        root = "data/youtube_voc/valid"
    phase_dataset = InpaintDatasetVideo(root,cfg.nframes,mask_config,
                                        image_size=image_size)
    data_len = len(phase_dataset)
    val_dataset = None
    print(data_len)

    # -- split --
    val_split = dataloader['validation_split']
    phase_dataset,val_dataset = split_dataset(phase_dataset,val_dataset,
                                              val_split,data_len=data_len)

    # -- loader --
    args = dataloader['args']
    worker_init_fn = partial(Util.set_seed, gl_seed=gl_seed)
    loaders = edict()
    loaders.phase = DataLoader(phase_dataset, sampler=None,
                               worker_init_fn=worker_init_fn, **args)
    args.update(dataloader['val_args'])
    loaders.val = DataLoader(phase_dataset, sampler=None,
                             worker_init_fn=worker_init_fn, **args)

    return loaders

def split_dataset(phase_dataset, val_dataset, valid_split, data_len=-1):
    seed = 123
    if valid_split > 0.0:# or 'debug' in opt['name']:
        if isinstance(valid_split, int):
            assert valid_split < data_len, "Validation set size is configured to be larger than entire dataset."
            valid_len = valid_split
        else:
            valid_len = int(data_len * valid_split)
        data_len -= valid_len
        generator = Generator().manual_seed(seed)
        phase_dataset, val_dataset = subset_split(dataset=phase_dataset,
                                                  lengths=[data_len, valid_len],
                                                  generator=generator)
    return phase_dataset, val_dataset

def subset_split(dataset, lengths, generator):
    """
    split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length : offset]))
    return Subsets


