from itertools import product

import numpy as np
import pandas as pd
import torch

from fedga.core.FedAvgX import CLIMB, FedAvg, FedGA, FedNTD, FedProx
from fedga.models import MLP, LeNet, resnet20
from fedga.utils.config import get_config
from fedga.utils.data import fed_dataloader

FED_LEARNER = {
    "FedAvg": FedAvg,
    "FedProx": FedProx,
    "CLIMB": CLIMB,
    "FedNTD": FedNTD,
    "FedGA": FedGA,
}

args = get_config()
args.FL.PROJECT = "test-fedga"
args.CL.MODEL.DEVICE = "cuda"
args.CL.OPTIM.TYPE = "sgd"
args.CL.OPTIM.MOMENTUM = 0
args.FL.LOG = 20
args.CL.OPTIM.LR_DECAY = 1


dss = ["MNIST", "CIFAR10", "SVHN", "CIFAR100", "Tiny-ImageNet"]
fed_algs = ["FedAvg", "FedProx", "CLIMB", "FedNTD", "FedGA"]
alphas = [0.05, 0.1, 0.5, 1, 10]
seeds = range(5)

for job in product(dss, fed_algs, alphas, seeds):
    args._box_config["frozen_box"] = False
    (
        args.DATA.NAME,
        args.FL.ALG,
        args.DATA.IB.ALPHA,
        args.DATA.SEED,
    ) = job
    # args.DATA.TRAIN.WORKERS = 1
    # args.DATA.TEST.WORKERS = 2

    # extract data specific parameters
    data = args.DATA.NAME
    alpha = args.DATA.IB.ALPHA
    conf = (
        pd.read_csv("configs/sgd-param.csv")
        .set_index(["Dataset", "a"])
        .xs((data, alpha))
    )
    args.CL.MODEL.NET = conf["model"]
    args.FL.EPOCH.GLOBAL = conf["epochs"].item()
    args.CL.OPTIM.LR = conf["LR"].item()

    if args.FL.ALG == "FedProx":
        # grid search from [1, 0.1, 0.01, 0.001]
        args.FL.PARAM = conf["proximal"].item()
    elif args.FL.ALG == "FedRL":
        # recommended value in the paper
        args.FL.PARAM = {}
        args.FL.PARAM.ALPHA = 1  # type: ignore
        args.FL.PARAM.BETA = 0.1  # type: ignore
    elif args.FL.ALG == "CLIMB":
        # grid search dlr in [4, 2, 1, 0.5, 0.1, 0.05]
        # eps in [1, 0.1, 0.01, 0.001]
        args.FL.PARAM = {}
        args.FL.PARAM.LR_DUAL = conf["dlr"].item()  # type: ignore
        args.FL.PARAM.EPSILON = conf["tol"].item()  # type: ignore
    elif args.FL.ALG == "FedNTD":
        # recommended value in the paper
        args.FL.PARAM = {}
        args.FL.PARAM.TAU = 1  # type: ignore
        args.FL.PARAM.BETA = 1  # type: ignore

    np.random.seed(args.DATA.SEED)
    torch.manual_seed(args.DATA.SEED)
    dataloaders, test_dls, shape, n_cls = fed_dataloader(args)

    if args.CL.MODEL.NET == "MLP":
        model = MLP(shape[0] * shape[1] * shape[2], n_cls).to(args.CL.MODEL.DEVICE)
    elif args.CL.MODEL.NET == "LeNet":
        model = LeNet(*shape, num_classes=n_cls).to(args.CL.MODEL.DEVICE)
    elif args.CL.MODEL.NET == "ResNet":
        model = resnet20(num_classes=n_cls).to(args.CL.MODEL.DEVICE)
    else:
        raise ValueError(f"Unknown model: {args.CL.MODEL.NET}")

    fed = FED_LEARNER[args.FL.ALG](dataloaders, test_dls, model, args)
    fed.train()
