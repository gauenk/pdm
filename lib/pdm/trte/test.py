
# -- helpers --
import copy
dcopy = copy.deepcopy
import numpy as np
import torch as th
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

# -- loss --
import torch.nn.functional as F

# -- helpers --
import pdm
from pdm.core.logger import VisualWriterV2, InfoLoggerV2
# from pdm.models import Palette
from pdm.augmented import Palette

# -- configs --
from dev_basics.configs import ExtractConfig,dcat
econfig = ExtractConfig(__file__) # init extraction
extract_config = econfig.extract_config # rename extraction

# -- load the model --
@econfig.set_init
def run(cfg):

    # -=-=-=-=-=-=-=-=-=-=-
    #
    #        Config
    #
    # -=-=-=-=-=-=-=-=-=-=-

    # -- init --
    econfig.init(cfg)
    device = econfig.optional(cfg,"device","cuda:0")

    # -- unpack local vars --
    local_pairs = {"tr":train_pairs(),
                   "diff":diff_pairs(),
                   "optim":optim_pairs(),
                   "ema":ema_pairs()}
    cfgs = econfig.extract_dict_of_pairs(cfg,local_pairs,restrict=True)
    cfg = dcat(cfg,econfig.flatten(cfgs)) # update cfg

    # -- end init --
    if econfig.is_init: return
    rank = 0

    # -- model --
    model = pdm.load_model(cfg)


    # -- compute loss --


    return loss
