import os

import torch
import wandb
import yaml

from fedga.utils.log import get_logger


class Federation:
    def __init__(self, dataloaders, test_loader, model, args) -> None:
        self.dataloaders = dataloaders
        self.test_loader = test_loader
        self.global_model = model
        self.global_round = 0
        self.global_grad = None
        self.momentum_buffer = None
        self.args = args
        self.logger, self.tables = self.__tb_writer__(args)

    def global_step(self):
        selected_clients = torch.randperm(self.args.FL.CLIENT.TOTAL)[
            : self.args.FL.CLIENT.ACTIVE
        ]

        client_updates = self.client_fit(selected_clients)

        weights = torch.tensor(
            [len(self.dataloaders[client].dataset) for client in selected_clients]
        )
        weights = weights / weights.sum()
        self.server_agg(client_updates, weights)
        self.args.CL.OPTIM.LR *= self.args.CL.OPTIM.get("LR_DECAY", 1.0)
        self.global_round += 1

    def train(self):
        print("Start training Fed learners with follow settings ===>:\n" + "-" * 53)
        print(yaml.dump(self.args.to_dict(), default_flow_style=False))
        print("=" * 53)
        for epoch in range(self.args.FL.EPOCH.GLOBAL):
            self.global_step()
            if self.global_round % self.args.FL.EVAL == 0:
                self.eval()
        wandb.log(self.tables)
        wandb.finish()
        os.system(f"wandb sync {os.path.dirname(self.wandb_run.dir)}")

    def client_fit(self, selected_clients):
        raise NotImplementedError

    def server_agg(self, updates, weights):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def __tb_writer__(self, args):
        log_dir = f"./{args.DATA.NAME}/"
        self.wandb_run = wandb.init(
            project=self.__class__.__name__,
            config=self.args.to_dict(),
            tags=self.args.get("TAGS", None),
            name=log_dir,
            mode="offline",
            settings=wandb.Settings(_disable_stats=True, _disable_machine_info=True),
        )
        logger = get_logger(log_dir + "/runtime")
        return logger, None
