"""

Basic training script

"""


# -- sys --
import os
import numpy as np
import pandas as pd
from easydict import EasyDict as edict

# -- testing --
from pdm.trte import train

# -- caching results --
import cache_io

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    def clear_fxn(num,cfg): return False
    exps,uuids = cache_io.train_stages.run("exps/baseline/train.cfg",
                                           ".cache_io_exps/baseline/train/")
    # 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
    # train.max_iter = 184375
    cfg = edict({
        "dname":"youtube",
        "device":"cuda:0",
        "ndevices":1,
        "use_amp":False,
        "log_root":"./output/train/baseline/logs",
        "chkpt_root":"./output/train/baseline/checkpoints",
        "nepochs":40,
        "subdir":"baseline",
        "lr_init":0.0001,
        "weight_decay":0.,
        "arch_subname":"vit_l",
        "ws":15,
        "wt":3,
        "limit_nbatches_tr":2000,#330000,
        "limit_nbatches_val":5,
        "limit_nbatches_te":5,
        "nframes":5,
    })
    results = cache_io.run_exps([cfg],train.run,#uuids=uuids,
                                name=".cache_io/baseline/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/baseline/train.pkl",
                                records_reload=True,use_wandb=False)

if __name__ == "__main__":
    main()
