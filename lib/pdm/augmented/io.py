
# -- helpers --
import copy
dcopy = copy.deepcopy
import numpy as np
import torch as th
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

# -- search/normalize/aggregate --
import stnls
from stnls import search,normz,agg

# -- io --
# from ..utils import model_io
from dev_basics import arch_io
import pdm
from .network import Network

# -- configs --
from dev_basics.configs import ExtractConfig,dcat
econfig = ExtractConfig(__file__) # init extraction
extract_config = econfig.extract_config # rename extraction


# -- load the model --
@econfig.set_init
def load_model(cfg):

    # -=-=-=-=-=-=-=-=-=-=-
    #
    #        Config
    #
    # -=-=-=-=-=-=-=-=-=-=-

    # -- init --
    econfig.init(cfg)
    device = econfig.optional(cfg,"device","cuda:0")

    # -- unpack local vars --
    local_pairs = {"io":io_pairs(),
                   "arch":arch_pairs(),
                   "attn":attn_pairs(),
                   "beta":beta_pairs()}
    cfgs = econfig.extract_dict_of_pairs(cfg,local_pairs,restrict=True)
    cfg = dcat(cfg,econfig.flatten(cfgs)) # update cfg
    # cfg.nheads = cfg.num_heads
    dep_pairs = {"normz":stnls.normz.econfig,
                 "agg":stnls.agg.econfig}
    cfgs = dcat(cfgs,econfig.extract_dict_of_econfigs(cfg,dep_pairs))
    cfg = dcat(cfg,econfig.flatten(cfgs))
    cfgs.search = stnls.search.extract_config(cfg)
    cfg = dcat(cfg,econfig.flatten(cfgs))

    # -- end init --
    if econfig.is_init: return

    # -- setup --
    cfgs.attn.nheads = cfgs.arch.num_heads
    cfgs.attn.embed_dim = cfgs.arch.inner_channel

    # -- init model --
    beta_sched = nest_beta(cfgs.beta)
    module = pdm.models.diffusion_stnls.unet
    # module = pdm.models.guided_diffusion_modules.unet
    unet = module.UNet(cfgs.arch,cfgs.attn,
                       cfgs.search,cfgs.normz,cfgs.agg)
    model = Network(unet,beta_sched)
    # losses = [mse_loss]
    # sample_num = cfgs.diff.sample_num

    # phase_loader = phase_loader,
    # val_loader = val_loader,
    # losses = losses,
    # metrics = metrics,
    # logger = phase_logger,
    # writer = phase_writer
    # model = Palette([net], losses, sample_num, cfgs.arch.task, cfgs.optim, cfgs.ema,
    #                 metrics=metrics,)

    # -- load model --
    load_pretrained(model,cfgs.io)

    # -- device --
    model = model.to(device)

    return model

def load_pretrained(model,cfg):
    if cfg.pretrained_load:
        print("Loading model: ",cfg.pretrained_path)
        arch_io.load_checkpoint(model,cfg.pretrained_path,
                                cfg.pretrained_root,cfg.pretrained_type)

def arch_pairs():
    pairs = {"in_channel":6,
             "inner_channel":8,
             "out_channel":3,
             "res_blocks":2,
             "attn_res":[1,2],
             "channel_mults":(1, 2, 4, 8),
             "conv_resample":True,
             "use_checkpoint":False,
             "use_fp16":False,
             "num_heads":1,
             "dropout":0,
             "num_head_channels":-1,
             "num_heads_upsample":-1,
             "use_scale_shift_norm":True,
             "resblock_updown":True,
             "use_new_attention_order":False,
             "task":"inpainting",
             "cond_embed_dim":True,"attn_type":"stnls"}
    return pairs

def io_pairs():
    base = Path("weights/checkpoints/")
    pretrained_path = base / "model/model_best.pt"
    pairs = {"pretrained_load":False,
             "pretrained_path":str(pretrained_path),
             "pretrained_type":"lit",
             "pretrained_root":"."}
    return pairs

def attn_pairs():
    pairs = {"qk_frac":1.,"qkv_bias":True,
             "token_mlp":'leff',"attn_mode":"default",
             "token_projection":'linear',
             "drop_rate_proj":0.,"attn_timer":False,
             "use_flow":True,"use_state_update":False}
    return pairs

# def arch_pairs():
#     pairs = {"in_chans":3,"dd_in":3,
#              "dowsample":"Downsample", "upsample":"Upsample",
#              "embed_dim":None,"input_proj_depth":1,
#              "output_proj_depth":1,"drop_rate_pos":0.,
#              "attn_timer":False,"task":"inpainting",
#     }
#     return pairs

def beta_pairs():
    pairs = {
        "train+schedule": "linear",
        "train+n_timestep": 2000,
        "train+linear_start": 1e-6,
        "train+linear_end": 0.01,
        "test+schedule": "linear",
        "test+n_timestep": 1000,
        "test+linear_start": 1e-4,
        "test+linear_end": 0.09
    }
    return pairs

def nest_beta(beta_cfg):
    nested = {"train":{},"test":{}}
    for raw_key in beta_cfg: # train+*/test+*
        ntype,key = raw_key.split("+")
        nested[ntype][key] = beta_cfg[raw_key]
    return nested

