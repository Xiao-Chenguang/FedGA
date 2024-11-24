import time
from copy import deepcopy

import torch
import wandb
from torch import Tensor, nn
from torch.nn.functional import kl_div
from torch.nn.utils import clip_grad_norm_  # type: ignore
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchmetrics.classification.accuracy import MulticlassAccuracy

from fedga.utils.log import get_logger
from fedga.utils.metrics import ib_metrics
from fedga.utils.tools import agg_model

from .api import Federation

OPTIM = {
    "sgd": SGD,
    "adam": Adam,
}


class FedAvg(Federation):
    def __init__(self, dataloaders, test_loader, model, args) -> None:
        super().__init__(dataloaders, test_loader, model, args)
        self.prox = 0

    def client_fit(self, selected_clients):
        fit_start = time.time()
        updates = []
        loss = 0
        for i, client in enumerate(selected_clients):
            client_start = time.time()
            update, client_loss = trainX(
                self.global_model,
                self.dataloaders[client],
                self.args,
                self.global_round,
                client,
                prox=self.prox,
            )
            updates.append(update)
            loss += client_loss
            client_end = time.time()
            self.logger.debug(
                f"\tclient {client:3d} trained in: {client_end - client_start:.2f}s"
            )
            if self.global_round % self.args.FL.CEVAL == 0:
                test_start = time.time()
                self.test_update(update, client)
                test_end = time.time()
                self.logger.debug(
                    f"\tclient {client:3d} tested in: {test_end - test_start:.2f}s"
                )
        loss /= len(selected_clients)
        wandb.log({"T-Loss": loss}, commit=False)
        self.logger.info(
            f"Epoche [{self.global_round+1:4d}/{self.args.FL.EPOCH.GLOBAL}] >> train loss: {loss:.5f}"
        )
        fit_end = time.time()
        self.logger.debug(f"clients trained in: {fit_end - fit_start:.2f}s")
        return updates

    def server_agg(self, updates, weights):
        agg_start = time.time()
        self.global_grad = agg_model(updates)

        self.global_grad * self.args.FL.LR
        self.global_model + self.global_grad
        agg_end = time.time()
        self.logger.debug(f"aggregated in: {agg_end - agg_start:.2f}s")

    def eval(self):
        eval_start = time.time()
        lossfn = nn.CrossEntropyLoss()
        self.global_model.eval()
        loss = 0
        test_size = len(self.test_loader.dataset)
        pred = []
        gtrue = []
        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, dict):
                    x, y = batch["image"], batch["label"]
                else:
                    x, y = batch
                batch_size = len(y)
                x, y = x.to(self.args.CL.MODEL.DEVICE), y.to(self.args.CL.MODEL.DEVICE)
                logits = self.global_model(x)
                pred.append(logits.softmax(dim=-1))
                gtrue.append(y)
                batch_loss = lossfn(logits, y) * batch_size
                loss += batch_loss
        loss /= test_size
        pred = torch.concat(pred)
        gtrue = torch.concat(gtrue)
        # ============= compute error asymmetry =============
        p = pred.softmax(dim=-1)
        mask = gtrue[:, None] == torch.arange(self.args.DATA.CLASSES).to(
            self.args.CL.MODEL.DEVICE
        )
        pi = (p[:, None, :] * mask[:, :, None]).sum(dim=0) / mask.sum(dim=0)[:, None]
        ea = (1 - pi.diag()) / (pi.sum(dim=0) - pi.diag())
        self.tables["ea_global"].add_data(self.global_round, *ea.tolist())  # type: ignore
        # ===================================================
        score_start = time.time()
        score = ib_metrics(
            pred, gtrue, self.args.CL.MODEL.DEVICE, self.args.DATA.CLASSES
        )
        score_end = time.time()
        self.logger.debug(f"\tevaluated score in: {score_end - score_start:.2f}s")
        wandb.log({"V-Loss": loss} | score)
        self.logger.info(f'test loss: {loss:.5f}, acc: {score["V-Acc"]*100:5.2f}%')
        mtacc = MulticlassAccuracy(num_classes=self.args.DATA.CLASSES, average=None).to(
            self.args.CL.MODEL.DEVICE
        )
        acc = mtacc(pred, gtrue).cpu().numpy()
        self.tables["pre_cls_acc"].add_data(self.global_round, *acc)  # type: ignore
        eval_end = time.time()
        self.logger.debug(f"evaluated model in: {eval_end - eval_start:.2f}s")

    def post_eval(self, id):
        eval_start = time.time()
        device = self.args.CL.MODEL.DEVICE
        mtacc = MulticlassAccuracy(num_classes=self.args.DATA.CLASSES, average=None).to(
            device
        )
        lossfn = nn.CrossEntropyLoss()
        self.global_model.eval()
        loss = 0
        test_size = len(self.test_loader.dataset)
        pred = []
        gtrue = []
        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, dict):
                    x, y = batch["image"], batch["label"]
                else:
                    x, y = batch
                batch_size = len(y)
                x, y = x.to(self.args.CL.MODEL.DEVICE), y.to(self.args.CL.MODEL.DEVICE)
                logits = self.global_model(x)
                pred.append(logits.softmax(dim=-1))
                gtrue.append(y)
                batch_loss = lossfn(logits, y) * batch_size
                loss += batch_loss
        loss /= test_size
        pred = torch.concat(pred)
        gtrue = torch.concat(gtrue)
        # ============= compute error asymmetry =============
        p = pred.softmax(dim=-1)
        mask = gtrue[:, None] == torch.arange(self.args.DATA.CLASSES).to(device)
        pi = (p[:, None, :] * mask[:, :, None]).sum(dim=0) / mask.sum(dim=0)[:, None]
        ea = (1 - pi.diag()) / (pi.sum(dim=0) - pi.diag())
        self.tables["ea_local"].add_data(self.global_round, id, *ea.tolist())  # type: ignore
        # ===================================================
        acc = mtacc(pred, gtrue).cpu().numpy()
        self.tables["post_cls_acc"].add_data(self.global_round, id, *acc.tolist())  # type: ignore
        eval_end = time.time()
        self.logger.debug(f"\t\tpost evaluated model in: {eval_end - eval_start:.2f}s")

    def __tb_writer__(self, args):
        if args.FL.NAME is None:
            name = "_".join(
                map(
                    str,
                    [
                        args.FL.ALG,
                        args.DATA.IB.ALPHA,
                        args.CL.MODEL.NET,
                        args.CL.OPTIM.TYPE,
                        args.CL.OPTIM.LR,
                        args.FL.PARAM,
                        args.DATA.SEED,
                        time.time(),
                    ],
                )
            )
        else:
            name = args.FL.NAME
        self.wandb_run = wandb.init(
            name=name,
            mode="offline",
            project=self.args.FL.PROJECT,
            config=self.args.to_dict(),
            settings=wandb.Settings(_disable_stats=True, _disable_machine_info=True),
        )
        logger = get_logger(level=args.FL.LOG)
        cls_str = list(map(str, range(args.DATA.CLASSES)))
        tables = {}
        tables["pre_cls_acc"] = wandb.Table(columns=["round"] + cls_str)
        tables["post_cls_acc"] = wandb.Table(columns=["round", "client"] + cls_str)
        tables["ea_local"] = wandb.Table(columns=["round", "client"] + cls_str)
        tables["ea_global"] = wandb.Table(columns=["round"] + cls_str)
        tables["client_size"] = wandb.Table(
            data=[dl.dataset.label_dist for dl in self.dataloaders], columns=cls_str
        )
        return logger, tables

    def test_update(self, update, id):
        self.global_model + update
        self.post_eval(id)
        self.global_model - update


def count(y, cls):
    """count y for each batch use broadcast

    Arguments:
        y {tensor} -- [description]
        cls {int} -- [description]

    Returns:
        [tensor] -- [description]
    """
    return (1.0 * (torch.arange(cls)[:, None] == y)) @ torch.ones(y.shape[0])


class FedRL(FedAvg):
    def __init__(self, dataloaders, test_loader, model, args) -> None:
        super().__init__(dataloaders, test_loader, model, args)
        self.aux_data = self.get_aux_data()
        # default alpha and beta to 1 and 0.1 as in CLIMB implementation
        # check 'Addressing Class Imbalance in Federated Learning' for details
        self.alpha, self.beta = self.args.FL.PARAM.ALPHA, self.args.FL.PARAM.BETA

    def client_fit(self, selected_clients):
        fit_start = time.time()

        Ra = self.compute_Ra_p(self.aux_data)
        weights = self.alpha + self.beta * Ra

        updates = []
        loss = 0
        for i, client in enumerate(selected_clients):
            client_start = time.time()
            update, client_loss = trainX(
                self.global_model,
                self.dataloaders[client],
                self.args,
                self.global_round,
                client,
                cls_w=weights,
            )
            updates.append(update)
            loss += client_loss
            client_end = time.time()
            self.logger.debug(
                f"\tclient {client:3d} trained in: {client_end - client_start:.2f}s"
            )
            if self.global_round % self.args.FL.CEVAL == 0:
                test_start = time.time()
                self.test_update(update, client)
                test_end = time.time()
                self.logger.debug(
                    f"\tclient {client:3d} tested in: {test_end - test_start:.2f}s"
                )
        loss /= len(selected_clients)
        wandb.log({"T-Loss": loss}, commit=False)
        self.logger.info(
            f"Epoche [{self.global_round+1:4d}/{self.args.FL.EPOCH.GLOBAL}] >> train loss: {loss:.5f}"
        )
        fit_end = time.time()
        self.logger.debug(f"clients trained in: {fit_end - fit_start:.2f}s")
        return updates

    def __tb_writer__(self, args):
        logger, tables = super().__tb_writer__(args)
        tables["Ra_p"] = wandb.Table(
            columns=["round"] + list(map(str, range(args.DATA.CLASSES)))
        )
        return logger, tables

    def compute_Ra_p(self, aux_data):
        device = self.args.CL.MODEL.DEVICE
        model = self.global_model
        n_cls = len(aux_data[1])

        loss_fn = torch.nn.CrossEntropyLoss()
        Delta_W = []

        for c in range(n_cls):
            data, label = aux_data[0][c].to(device), aux_data[1][c].to(device)
            logits = model(data.unsqueeze(0))
            loss = loss_fn(logits, label.unsqueeze(0))
            grad_lw = torch.autograd.grad(loss, model.parameters())
            Delta_W.append(grad_lw[-2])

        Delta_W = torch.stack(Delta_W, dim=0)

        Delta_W_max = Delta_W.max(dim=0, keepdim=True)[0]
        max_nbrs = (Delta_W_max == Delta_W) | (Delta_W / Delta_W_max > 0.8)
        mask = max_nbrs.int() == max_nbrs.sum(dim=0, keepdim=True)

        Ra_p = []
        for i in range(n_cls):
            tem = Delta_W * mask[i][None, :]
            tem = tem.sum(dim=[1, 2])
            Ra_p.append((tem[i] / (tem.sum() - tem[i])).abs())

        Ra_p = torch.stack(Ra_p) * (n_cls - 1)
        Ra_p[Ra_p.isnan()] = 10
        Ra_p[Ra_p > 5000] = 5000

        self.tables["Ra_p"].add_data(self.global_round, *Ra_p.tolist())  # type: ignore

        return Ra_p

    def get_aux_data(self):
        """get auxiliary data for ratio loss

        Returns:
            [list] -- [list of tuple (data, label) for each class]
        """
        n_class = self.args.DATA.CLASSES
        n_client = self.args.FL.CLIENT.TOTAL

        torch_state = torch.get_rng_state()
        torch.manual_seed(123)

        dl_idx = torch.randperm(n_client)

        data = []
        label = []
        j = 0
        for i in range(n_class):
            fund = False
            while not fund:
                dl = dl_idx[j % n_client]
                for batch in self.dataloaders[dl]:
                    if isinstance(batch, dict):
                        x, y = batch["image"], batch["label"]
                    else:
                        x, y = batch
                    match = y == i
                    if match.sum() > 0:
                        data.append(x[match][0])
                        label.append(y[match][0])
                        fund = True
                        break
                j += 1
        data = torch.stack(data)
        label = torch.stack(label)

        torch.set_rng_state(torch_state)

        return data, label


class CLIMB(FedAvg):
    """CLIMB algorithm proposed in the paper:
    "AN AGNOSTIC APPROACH TO FEDERATED LEARNING WITH CLASS IMBALANCE" in ICLR 2022
    """

    def __init__(self, dataloaders, test_loader, model, args) -> None:
        """Init CLIMB algorithm

        Args:
            dataloaders (list[Dataloader]): FL dataloaders
            test_loader (Dataloader): test dataloader
            model (Module): torch model
            args (dict): dict of arguments
        """
        super().__init__(dataloaders, test_loader, model, args)
        print(f"[-->> CLIMB epsilon: {args.FL.PARAM} <<--]")
        self.lambdas = torch.ones(len(dataloaders))
        # epsilon default 1 as in https://github.com/shenzebang/Federated-Learning-Pytorch
        self.epsilon = args.FL.PARAM.EPSILON
        self.lr_dual = args.FL.PARAM.LR_DUAL

    def client_fit(self, selected_clients):
        fit_start = time.time()

        weights = 1.0 + self.lambdas - self.lambdas.mean()
        self.logger.debug(f"round {self.global_round} weights: {weights}")

        clts_loss = torch.zeros(self.args.FL.CLIENT.TOTAL)
        torch_state = torch.get_rng_state()
        for client in range(self.args.FL.CLIENT.TOTAL):
            clts_loss[client] = self.client_loss(client)
        torch.set_rng_state(torch_state)

        updates = []
        loss = 0
        for i, client in enumerate(selected_clients):
            client_start = time.time()
            update, client_loss = trainX(
                self.global_model,
                self.dataloaders[client],
                self.args,
                self.global_round,
                client,
                all_w=int(weights[client].item()),
            )
            updates.append(update)
            loss += client_loss
            client_end = time.time()
            self.logger.debug(
                f"\tclient {client:3d} trained in: {client_end - client_start:.2f}s"
            )
            if self.global_round % self.args.FL.CEVAL == 0:
                test_start = time.time()
                self.test_update(update, client)
                test_end = time.time()
                self.logger.debug(
                    f"\tclient {client:3d} tested in: {test_end - test_start:.2f}s"
                )
        loss /= len(selected_clients)
        wandb.log({"T-Loss": loss}, commit=False)
        self.logger.info(
            f"Epoche [{self.global_round+1:4d}/{self.args.FL.EPOCH.GLOBAL}] >> train loss: {loss:.5f}"
        )
        fit_end = time.time()
        self.logger.debug(f"clients trained in: {fit_end - fit_start:.2f}s")

        factor = self.lr_dual / self.args.FL.CLIENT.TOTAL
        self.lambdas += ((clts_loss - clts_loss.mean()) - self.epsilon) * factor
        self.lambdas.clamp_(min=0.0, max=100.0)

        return updates

    def client_loss(self, id):
        """return the clients loss for compute lambda

        Arguments:
            id {int} -- client id

        Returns:
            float -- clients loss
        """

        device = self.args.CL.MODEL.DEVICE
        train_dl = self.dataloaders[id]
        dataloader = DataLoader(train_dl.dataset, batch_size=256)
        lossfn = nn.CrossEntropyLoss()

        test_size = len(train_dl.dataset)
        loss = 0

        self.global_model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x, y = batch["image"], batch["label"]
                else:
                    x, y = batch
                x, y = x.to(device), y.to(device)
                loss += lossfn(self.global_model(x), y) * len(y)

        return loss / test_size


class FedProx(FedAvg):
    def __init__(self, dataloaders, test_loader, model, args) -> None:
        super().__init__(dataloaders, test_loader, model, args)
        self.prox = args.FL.PARAM


class FedGA(FedAvg):
    def __init__(self, dataloaders, test_loader, model, args) -> None:
        super().__init__(dataloaders, test_loader, model, args)


class FedNTD(FedAvg):
    def __init__(self, dataloaders, test_loader, model, args) -> None:
        super().__init__(dataloaders, test_loader, model, args)
        # both tau and beta default to 1 as in original paper, for details check:
        # 'Preservation of the Global Knowledge by Not-True Distillation in Federated Learning'
        self.tau, self.beta = args.FL.PARAM.TAU, args.FL.PARAM.BETA
        assert self.tau > 0, "tau should be greater than 0"
        assert self.beta > 0, "beta should be greater than 0"


def trainX(
    global_model,
    dataloader,
    args,
    round,
    client,
    cls_w=None,
    all_w=1,
    prox=0,
):
    device = args.CL.MODEL.DEVICE
    optm = args.CL.OPTIM.TYPE
    lr = args.CL.OPTIM.LR
    momt = args.CL.OPTIM.MOMENTUM
    damp = args.CL.OPTIM.get("DAMPENING", 0)

    local_model = deepcopy(global_model)
    local_model.to(device)

    data_size = len(dataloader.dataset)
    onehot = torch.eye(args.DATA.CLASSES).to(device)

    params = (lr, momt, damp) if optm == "sgd" else (lr,)
    optimizer = OPTIM[optm](local_model.parameters(), *params)
    lossfn = nn.CrossEntropyLoss(weight=cls_w)

    loss = 0
    for step in range(args.FL.EPOCH.LOCAL):
        for batch in dataloader:
            if isinstance(batch, dict):
                X, y = batch["image"], batch["label"]
            else:
                X, y = batch
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            logits = local_model(X)

            if "FedGA" == args.FL.ALG:
                dist = count(y.cpu(), args.DATA.CLASSES).to(device)
                p_mask = (dist[:, None] - dist[None, :]) / dist[:, None]
                target = logits.softmax(dim=1).detach() * p_mask[y] + onehot[y]
            else:
                target = onehot[y]
            batch_loss = lossfn(logits, target)

            if prox > 0:
                for pl, pg in zip(local_model.parameters(), global_model.parameters()):
                    batch_loss += prox * (pl - pg.data).norm(2)

            batch_loss *= all_w
            if "FedNTD" == args.FL.ALG:
                ntd_tau, ntd_beta = args.FL.PARAM.TAU, args.FL.PARAM.BETA
                with torch.no_grad():
                    logits_g = global_model(X)
                # NTD loss
                ntd_loss = ntd_loss_fn(logits, logits_g, target, ntd_tau)
                batch_loss += ntd_loss * ntd_beta
            loss += batch_loss * len(y)
            batch_loss.backward()
            clip_grad_norm_(parameters=local_model.parameters(), max_norm=5.0)
            optimizer.step()

    loss /= data_size * args.FL.EPOCH.LOCAL

    local_model - global_model
    return local_model, loss


def ntd_loss_fn(logits: Tensor, logits_g: Tensor, targets: Tensor, tau: float = 1):
    """compute the NTD loss based on the logits and logits_g and targets

    Args:
        logits (Tensor): local model logits
        logits_g (Tensor): global model logits
        targets (Tensor): onehot target
        tau (float, optional): smooth factor. Defaults to 1 as official repo.

    Returns:
        Tensor: NTD loss
    """
    nt_pos = (1 - targets).nonzero()[:, 1].view(targets.size(0), -1)
    nt_log_p = (torch.gather(logits, 1, nt_pos) / tau).log_softmax(dim=1)
    nt_p_g = (torch.gather(logits_g, 1, nt_pos) / tau).softmax(dim=1)
    return kl_div(nt_log_p, nt_p_g, reduction="batchmean")
