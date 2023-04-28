
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

    # -=-=-=-=-=-=-=-=-=-=-
    #
    #    Create Trainer
    #
    # -=-=-=-=-=-=-=-=-=-=-

    # -- paths --
    subdir = cfgs.tr.subdir
    root = Path(cfgs.tr.root) / "output" / "train" / subdir
    log_dir = root / "logs" / str(cfgs.tr.uuid)
    tb_log_dir = root / "tb_logs" / str(cfgs.tr.uuid)
    tb_res_dir = root / "tb_res" / str(cfgs.tr.uuid)
    pik_dir = root / "pickles" / str(cfgs.tr.uuid)
    chkpt_dir = root / "checkpoints" / str(cfgs.tr.uuid)
    init_paths(log_dir,pik_dir,chkpt_dir)

    # -- model --
    model = pdm.load_model(cfg)

    # phase_loader = phase_loader,
    # val_loader = val_loader,
    # phase_loader, val_loader = define_dataloader(phase_logger, opt)

    # -- info --
    phase_logger = InfoLoggerV2(rank,"train",log_dir)
    phase_writer = VisualWriterV2(rank,tb_log_dir,tb_res_dir, True, phase_logger)
    phase_logger.info("Create the log file in directory [{log_dir}]")

    # -- data --
    loaders = pdm.data.io.load(cfg,phase_logger)

    # -- palette --
    losses = [mse_loss]
    sample_num = cfgs.diff.sample_num
    metrics = [mae]
    opt = {"batch_size":-1,"distributed":False,
           "path":{"resume_state":False},
           "train":{"n_epoch":cfg.nepochs,
                    "n_iter":cfg.niters,
                    "log_iter":100}}
    limit_batches = 150
    trainer = Palette([model], losses, sample_num,
                      cfgs.diff.task, [cfgs.optim], cfgs.ema,
                      limit_batches=limit_batches,
                      opt=opt,phase_loader=loaders.phase,val_loader=loaders.val,
                      metrics=metrics,logger=phase_logger,writer=phase_writer)

    # -- run --
    trainer.train()

def train_pairs():
    pairs = {"num_workers":4,
             "dset_tr":"tr",
             "dset_val":"val",
             "persistent_workers":True,
             "rand_order_tr":True,
             "gradient_clip_algorithm":"norm",
             "gradient_clip_val":0.5,
             "index_skip_val":5,
             "root":".","seed":123,
             "accumulate_grad_batches":1,
             "ndevices":1,
             "precision":32,
             "limit_train_batches":1.,
             "nepochs":30,
             "niters":100000,
             "uuid":"",
             "swa":False,
             "swa_epoch_start":0.8,
             "nsamples_at_testing":1,
             "isize":"128_128",
             "subdir":"",
             "save_epoch_list":"",
    }
    return pairs

def diff_pairs():
    pairs = {"sample_num":8,"task":"inpainting"}
    return pairs

def ema_pairs():
    pairs = {
        "ema_start": 1,
        "ema_iter": 1,
        "ema_decay": 0.9999
    }
    return pairs

def optim_pairs():
    pairs = { "lr": 5e-5, "weight_decay": 0}
    return pairs

def mse_loss(output, target):
    return F.mse_loss(output, target)

def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output

def init_paths(log_dir,pik_dir,chkpt_dir):

    # -- init log dir --
    print("Log Dir: ",log_dir)
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    log_subdirs = ["train"]
    for sub in log_subdirs:
        log_subdir = log_dir / sub
        if not log_subdir.exists():
            log_subdir.mkdir(parents=True)

    # -- prepare save directory for pickles --
    if not pik_dir.exists():
        pik_dir.mkdir(parents=True)

def get_checkpoint(checkpoint_dir,uuid,nepochs):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return ""
    chosen_ckpt = ""
    for epoch in range(nepochs):
        ckpt_fn = checkpoint_dir / ("%s-epoch=%02d.ckpt" % (uuid,epoch))
        if ckpt_fn.exists(): chosen_ckpt = ckpt_fn
    assert ((chosen_ckpt == "") or chosen_ckpt.exists())
    if chosen_ckpt != "":
        print("Resuming training from {%s}" % (str(chosen_ckpt)))
    return str(chosen_ckpt)


