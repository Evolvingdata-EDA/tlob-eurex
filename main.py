import random
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import hydra

import instruments_eurex  # noqa: F401  (registers Eurex instrument specs in tlob)
from tlob import constants as cst
from tlob.config import Config
from tlob.run import run


@hydra.main(config_path="", config_name="config")
def hydra_app(config: Config):
    set_reproducibility(config.experiment.seed)
    if torch.cuda.is_available():
        gpu_id = int(config.experiment.gpu_id)
        n_gpus = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= n_gpus:
            raise ValueError(f"experiment.gpu_id={gpu_id} but only {n_gpus} GPU(s) visible")
        cst.DEVICE = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        accelerator = "gpu"
        devices = [gpu_id]
    else:
        cst.DEVICE = "cpu"
        accelerator = "cpu"
        devices = "auto"
    print(f"Using device: {cst.DEVICE}")
    run(config, accelerator, devices)


def set_reproducibility(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_torch():
    torch.set_default_dtype(cst.TENSOR_DTYPE)
    torch.autograd.set_detect_anomaly(False)
    if cst.PRECISION == 32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('highest')


if __name__ == "__main__":
    set_torch()
    hydra_app()
